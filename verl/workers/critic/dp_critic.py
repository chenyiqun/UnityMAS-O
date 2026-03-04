# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Implement a multiprocess PPOCritic
"""

import logging
import os

import torch
import torch.distributed
from torch import nn, optim
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_functional import masked_mean
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad_and_slice_inputs
from verl.workers.critic import BasePPOCritic

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOCritic(BasePPOCritic):
    def __init__(self, config, critic_module: nn.Module, critic_optimizer: optim.Optimizer):
        super().__init__(config=config)
        self.critic_module = critic_module
        self.critic_optimizer = critic_optimizer
        self.use_remove_padding = self.config.model.get("use_remove_padding", False)
        print(f"Critic use_remove_padding={self.use_remove_padding}")

        self.ulysses_sequence_parallel_size = self.config.get("ulysses_sequence_parallel_size", 1)
        self.device_name = get_device_name()
        self.value_head_cfg = self.config.model.get("value_head", {})
        self.num_value_heads = int(self.value_head_cfg.get("num_outputs", 1))
        self.value_targets_key = self.value_head_cfg.get("target_key", "returns")
        self.value_old_key = self.value_head_cfg.get("old_value_key", "values_multi")
        self.return_multi_values = bool(self.value_head_cfg.get("return_multi_values", self.num_value_heads > 1))
        self.value_head_reduce = self.value_head_cfg.get("reduce_mode_for_advantage", "mean")
        self.value_head_loss_reduce = self.value_head_cfg.get("loss_reduce_mode", "mean")
        self.enable_value_clipping = bool(self.value_head_cfg.get("enable_value_clipping", True))
        self.log_per_head_metrics = bool(self.value_head_cfg.get("log_per_head_metrics", self.num_value_heads <= 16))
        self.value_head_weights = self.value_head_cfg.get("head_weights", None)
        if self.value_head_weights is not None:
            self.value_head_weights = list(self.value_head_weights)
            assert len(self.value_head_weights) == self.num_value_heads, (
                f"value_head.head_weights length ({len(self.value_head_weights)}) must equal "
                f"value_head.num_outputs ({self.num_value_heads})"
            )

    def _aggregate_value_heads(self, values: torch.Tensor, mode: str | None = None) -> torch.Tensor:
        if values.dim() != 3:
            return values
        mode = self.value_head_reduce if mode is None else mode
        if mode == "none":
            return values
        if mode not in ("mean", "weighted_mean"):
            raise ValueError(
                f"Unsupported value head reduce mode={mode}. "
                "Expected one of ['mean', 'weighted_mean', 'none']"
            )

        if mode == "weighted_mean":
            assert self.value_head_weights is not None, (
                "value_head.head_weights must be set when reduce_mode_for_advantage=weighted_mean"
            )
            weights = torch.tensor(self.value_head_weights, device=values.device, dtype=values.dtype).view(1, 1, -1)
            return (values * weights).sum(dim=-1) / (weights.sum() + 1e-8)
        return values.mean(dim=-1)

    def _align_value_shape(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.dim() == 3:
            if source.dim() == 2:
                if source.size(1) == target.size(1):
                    # (bs, response_len) -> (bs, response_len, n_values)
                    return source.unsqueeze(-1).expand(-1, -1, target.size(-1))
                if source.size(1) == target.size(-1):
                    # (bs, n_values) -> (bs, response_len, n_values)
                    return source.unsqueeze(1).expand(-1, target.size(1), -1)
                raise ValueError(
                    f"Cannot align source shape {tuple(source.shape)} to target shape {tuple(target.shape)}. "
                    "Expected source second dim to be response_len or n_values."
                )
            if source.dim() == 3 and source.size(-1) == 1 and target.size(-1) > 1:
                return source.expand(-1, -1, target.size(-1))
        if target.dim() == 2 and source.dim() == 3:
            if source.size(-1) == 1:
                return source.squeeze(-1)
            return self._aggregate_value_heads(source, mode=self.value_head_loss_reduce)
        return source

    def _compute_multi_value_loss(
        self,
        vpreds: torch.Tensor,
        returns: torch.Tensor,
        values: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.enable_value_clipping:
            vpredclipped = torch.maximum(
                torch.minimum(vpreds, values + self.config.cliprange_value),
                values - self.config.cliprange_value,
            )
            vf_losses1 = (vpreds - returns) ** 2
            vf_losses2 = (vpredclipped - returns) ** 2
            clipped_vf_losses = torch.max(vf_losses1, vf_losses2)
            clip_mask = torch.gt(vf_losses2, vf_losses1).float()
        else:
            clipped_vf_losses = (vpreds - returns) ** 2
            clip_mask = torch.zeros_like(clipped_vf_losses)

        token_losses = self._aggregate_value_heads(clipped_vf_losses, mode=self.value_head_loss_reduce)
        token_clip_mask = self._aggregate_value_heads(clip_mask, mode=self.value_head_loss_reduce)

        vf_loss = 0.5 * core_algos.agg_loss(
            loss_mat=token_losses,
            loss_mask=response_mask,
            loss_agg_mode=self.config.loss_agg_mode,
        )
        vf_clipfrac = masked_mean(token_clip_mask, response_mask)

        per_head_loss = (
            (clipped_vf_losses * response_mask.unsqueeze(-1)).sum(dim=(0, 1))
            / (response_mask.unsqueeze(-1).sum(dim=(0, 1)) + 1e-8)
        )
        return vf_loss, vf_clipfrac, per_head_loss

    def _forward_micro_batch(self, micro_batch):
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                # pad and slice the inputs if sp > 1
                if self.ulysses_sequence_parallel_size > 1:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad, position_ids_rmpad, sp_size=self.ulysses_sequence_parallel_size
                    )

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.critic_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                )  # prevent model thinks we are generating

                if hasattr(self.critic_module, "v_head"):
                    # For trl.AutoModelForCausalLMWithValueHead
                    values_rmpad = output[2].squeeze(0).unsqueeze(-1)
                else:
                    values_rmpad = output.logits
                    values_rmpad = values_rmpad.squeeze(0)  # (total_nnz) or (total_nnz, n_values)

                # gather output if sp > 1
                if self.ulysses_sequence_parallel_size > 1:
                    values_rmpad = gather_outputs_and_unpad(
                        values_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                    )

                # pad it back
                values = pad_input(values_rmpad, indices=indices, batch=batch, seqlen=seqlen)
                values = values[:, -response_length - 1 : -1]
            else:
                output = self.critic_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                )  # prevent model thinks we are generating
                if hasattr(self.critic_module, "v_head"):
                    # For trl.AutoModelForCausalLMWithValueHead
                    values = output[2]
                else:
                    values = output.logits
                values = values[:, -response_length - 1 : -1]
            if values.dim() == 3 and values.size(-1) == 1:
                values = values.squeeze(-1)
            return values

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.critic_module, FSDP):
            grad_norm = self.critic_module.clip_grad_norm_(self.config.grad_clip)
        elif isinstance(self.critic_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.critic_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.critic_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm is not finite: {grad_norm}")
            self.critic_optimizer.zero_grad()
        else:
            self.critic_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp critic", logger=logger)
    def compute_values(self, data: DataProto) -> torch.Tensor | dict[str, torch.Tensor]:
        self.critic_module.eval()
        micro_batch_size = data.meta_info["micro_batch_size"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = (
            ["responses", "input_ids", "response_mask", "attention_mask", "position_ids"]
            if "response_mask" in data.batch
            else ["responses", "input_ids", "attention_mask", "position_ids"]
        )
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        values_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                values = self._forward_micro_batch(model_inputs)
            values_lst.append(values)
        values = torch.concat(values_lst, dim=0)

        if use_dynamic_bsz:
            values = restore_dynamic_batch(values, batch_idx_list)

        if "response_mask" in data.batch:
            response_mask = data.batch["response_mask"]
            response_mask = response_mask.to(values.device)
            if values.dim() == 3:
                values = values * response_mask.unsqueeze(-1)
            else:
                values = values * response_mask  # Only action tokens have values

        if values.dim() == 3:
            values_for_adv = self._aggregate_value_heads(values, mode=self.value_head_reduce)
            if values_for_adv.dim() != 2:
                raise ValueError(
                    "value_head.reduce_mode_for_advantage='none' is not compatible with PPO advantage computation. "
                    "Please use 'mean' or 'weighted_mean'."
                )
            if self.return_multi_values:
                return {"values": values_for_adv, self.value_old_key: values}
            return values_for_adv
        return values

    @GPUMemoryLogger(role="dp critic", logger=logger)
    def update_critic(self, data: DataProto):
        # make sure we are in training mode
        self.critic_module.train()
        metrics = {
            "critic/vf_loss": 0.0,
        }

        select_keys = ["input_ids", "responses", "response_mask", "attention_mask", "position_ids", "values"]
        if self.value_targets_key not in select_keys:
            select_keys.append(self.value_targets_key)
        if self.value_old_key in data.batch and self.value_old_key not in select_keys:
            select_keys.append(self.value_old_key)
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.critic_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    if self.value_targets_key not in model_inputs:
                        raise KeyError(
                            f"Critic target key `{self.value_targets_key}` not found in batch keys: "
                            f"{list(model_inputs.keys())}"
                        )

                    values = model_inputs.get(self.value_old_key, model_inputs["values"])
                    returns = model_inputs[self.value_targets_key]

                    vpreds = self._forward_micro_batch(model_inputs)
                    values = self._align_value_shape(values, vpreds)
                    returns = self._align_value_shape(returns, vpreds)

                    if vpreds.dim() == 3 or values.dim() == 3 or returns.dim() == 3:
                        vf_loss, vf_clipfrac, per_head_loss = self._compute_multi_value_loss(
                            vpreds=vpreds,
                            values=values,
                            returns=returns,
                            response_mask=response_mask,
                        )
                    else:
                        vf_loss, vf_clipfrac = core_algos.compute_value_loss(
                            vpreds=vpreds,
                            values=values,
                            returns=returns,
                            response_mask=response_mask,
                            cliprange_value=self.config.cliprange_value,
                            loss_agg_mode=self.config.loss_agg_mode,
                        )
                        per_head_loss = None
                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                        loss = vf_loss * loss_scale_factor
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation
                        loss = vf_loss * loss_scale_factor

                    loss.backward()

                    micro_batch_metrics.update(
                        {
                            "critic/vf_clipfrac": vf_clipfrac.detach().item(),
                            "critic/vpred_mean": masked_mean(
                                self._aggregate_value_heads(vpreds, mode=self.value_head_loss_reduce),
                                response_mask,
                            ).detach().item(),
                        }
                    )
                    if per_head_loss is not None and self.log_per_head_metrics:
                        for head_idx, head_loss in enumerate(per_head_loss):
                            micro_batch_metrics[f"critic/vf_loss_head_{head_idx}"] = head_loss.detach().item()

                    metrics["critic/vf_loss"] += vf_loss.detach().item() * loss_scale_factor
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"critic/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.critic_optimizer.zero_grad()
        return metrics
