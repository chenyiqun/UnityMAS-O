import os
import threading
import time
import uuid

import numpy as np
import torch
from ray.util.collective import collective

from verl import DataProto
from verl.experimental.one_step_off_policy.fsdp_workers import (
    CriticWorker,
    DetachActorWorker,
    DetachAsyncRolloutWorker,
    RewardModelWorker,
)
from verl.experimental.star_ppo.trajectory_buffer import TrajectoryBuffer, TrajectoryEntry
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.utils.device import get_torch_device
from verl.utils.fsdp_utils import load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu
from verl.utils.ray_utils import get_event_loop

__all__ = [
    "StarDetachActorWorker",
    "StarDetachAsyncRolloutWorker",
    "CriticWorker",
    "RewardModelWorker",
]


_LOCAL_PAIR_END = "__star_local_pair_end__"
_LOCAL_PAIR_CHANNELS = {}
_LOCAL_PAIR_CHANNELS_LOCK = threading.Lock()


class _LocalPairChannel:
    def __init__(self):
        self._cond = threading.Condition()
        self._slot = None

    def put(self, item):
        with self._cond:
            while self._slot is not None:
                self._cond.wait()
            self._slot = item
            self._cond.notify_all()

    def get(self):
        with self._cond:
            while self._slot is None:
                self._cond.wait()
            item = self._slot
            self._slot = None
            self._cond.notify_all()
            return item


def _get_local_pair_channel(group_name: str) -> _LocalPairChannel:
    with _LOCAL_PAIR_CHANNELS_LOCK:
        chan = _LOCAL_PAIR_CHANNELS.get(group_name)
        if chan is None:
            chan = _LocalPairChannel()
            _LOCAL_PAIR_CHANNELS[group_name] = chan
        return chan


def _get_vllm_inference_model(rollout):
    """Best-effort fetch of in-proc vLLM model; returns None for server-adapter rollout."""
    inference_engine = getattr(rollout, "inference_engine", None)
    if inference_engine is None:
        return None
    if hasattr(inference_engine, "llm_engine"):
        return inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner.model
    if hasattr(inference_engine, "worker"):
        return inference_engine.worker.model_runner.model
    return None


class StarDetachActorWorker(DetachActorWorker):
    """Actor worker alias for star PPO."""

    def __init__(self, config, role: str):
        super().__init__(config=config, role=role)
        self._weight_sync_group_name = "actor_rollout"
        self._weight_sync_mode = "collective"

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_weight_sync_group_name(self, group_name: str):
        self._weight_sync_group_name = str(group_name)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_weight_sync_mode(self, mode: str):
        self._weight_sync_mode = str(mode).strip().lower()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def sync_rollout_weights(self):
        assert (self._is_actor or self._is_rollout) and not self.config.hybrid_engine
        assert hasattr(self, "_weights_info") and self._weights_info is not None

        if self._is_actor and self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        params = self._get_actor_params() if self._is_actor else None

        rollout_name = self.config.rollout.name
        inference_model = None
        use_vllm_server_adapter = False
        if self._is_rollout:
            if rollout_name == "vllm":
                from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader

                inference_model = _get_vllm_inference_model(self.rollout)
                if inference_model is not None:
                    patch_vllm_moe_model_weight_loader(inference_model)
                elif hasattr(self.rollout, "update_weights"):
                    use_vllm_server_adapter = True
                else:
                    raise AttributeError(
                        f"Unsupported vllm rollout object for weight sync: {type(self.rollout)}"
                    )
            elif rollout_name == "sglang":
                inference_model = self.rollout._engine
            else:
                raise NotImplementedError(f"Unknown rollout name: {rollout_name}")
        loop = get_event_loop()
        group_name = getattr(self, "_weight_sync_group_name", "actor_rollout")
        sync_mode = str(getattr(self, "_weight_sync_mode", os.environ.get("STAR_WEIGHT_SYNC_MODE", "collective"))).lower()
        if sync_mode == "local_pair":
            channel = _get_local_pair_channel(group_name)
            if self._is_actor:
                try:
                    for key, shape, dtype in self._weights_info:
                        assert key in params
                        origin_data = params[key]
                        if hasattr(origin_data, "full_tensor"):
                            origin_data = origin_data.full_tensor()
                        tensor = torch.empty(shape, dtype=dtype, device=get_torch_device().current_device())
                        tensor.copy_(origin_data)
                        channel.put((key, tensor))
                finally:
                    channel.put((_LOCAL_PAIR_END, None))
            else:
                def _iter_local_pair_weights():
                    for expected_key, _, _ in self._weights_info:
                        recv_key, tensor = channel.get()
                        if recv_key != expected_key:
                            raise RuntimeError(
                                f"local_pair weight order mismatch: got {recv_key}, expected {expected_key}"
                            )
                        yield expected_key, tensor
                    end_key, _ = channel.get()
                    if end_key != _LOCAL_PAIR_END:
                        raise RuntimeError(f"local_pair weight stream missing end sentinel, got {end_key}")

                if rollout_name == "vllm" and use_vllm_server_adapter:
                    loop.run_until_complete(self.rollout.update_weights(_iter_local_pair_weights()))
                else:
                    for expected_key, tensor in _iter_local_pair_weights():
                        if rollout_name == "vllm":
                            inference_model.load_weights([(expected_key, tensor)])
                        elif rollout_name == "sglang":
                            if inference_model is not None:
                                loop.run_until_complete(self.update_weights(inference_model, [(expected_key, tensor)]))
            if self._is_actor and self._is_offload_param:
                offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            get_torch_device().empty_cache()
            return

        if self._is_rollout and rollout_name == "vllm" and use_vllm_server_adapter:
            def _iter_collective_weights():
                for key, shape, dtype in self._weights_info:
                    tensor = torch.empty(shape, dtype=dtype, device=get_torch_device().current_device())
                    if hasattr(self, "_weight_sync_group") and self._weight_sync_group is not None:
                        self._weight_sync_group.broadcast(tensor, src=0, stream=get_torch_device().current_stream())
                    else:
                        collective.broadcast(tensor, src_rank=0, group_name=group_name)
                    yield key, tensor

            loop.run_until_complete(self.rollout.update_weights(_iter_collective_weights()))
            if self._is_actor and self._is_offload_param:
                offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            get_torch_device().empty_cache()
            return

        for key, shape, dtype in self._weights_info:
            tensor = torch.empty(shape, dtype=dtype, device=get_torch_device().current_device())
            if self._is_actor:
                assert key in params
                origin_data = params[key]
                if hasattr(origin_data, "full_tensor"):
                    origin_data = origin_data.full_tensor()
                if torch.distributed.get_rank() == 0:
                    tensor.copy_(origin_data)

            if hasattr(self, "_weight_sync_group") and self._weight_sync_group is not None:
                self._weight_sync_group.broadcast(tensor, src=0, stream=get_torch_device().current_stream())
            else:
                collective.broadcast(tensor, src_rank=0, group_name=group_name)

            if self._is_rollout:
                if rollout_name == "vllm":
                    inference_model.load_weights([(key, tensor)])
                elif rollout_name == "sglang":
                    if inference_model is not None:
                        loop.run_until_complete(self.update_weights(inference_model, [(key, tensor)]))

        if self._is_actor and self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
        get_torch_device().empty_cache()


class StarDetachAsyncRolloutWorker(DetachAsyncRolloutWorker):
    """Rollout worker with local fat-data buffer and thin-data return path."""

    def __init__(self, config, role: str):
        super().__init__(config=config, role=role)
        buffer_cfg = config.get("star_buffer", {})
        max_items = int(buffer_cfg.get("max_items", 100000))
        ttl_seconds = int(buffer_cfg.get("ttl_seconds", 7200))
        self._traj_buffer = TrajectoryBuffer(max_items=max_items, ttl_seconds=ttl_seconds)
        self._weight_sync_group_name = "actor_rollout"
        self._weight_sync_mode = "collective"

    def _decode_action_text(self, response_tokens: torch.Tensor) -> str:
        if response_tokens is None:
            return ""
        tokens = response_tokens.detach().cpu().tolist()
        try:
            return self.tokenizer.decode(tokens, skip_special_tokens=True)
        except Exception:
            return ""

    def _build_thin_from_batch(self, full_batch: DataProto) -> DataProto:
        bsz = len(full_batch)
        query_ids = full_batch.non_tensor_batch.get("query_id", np.array(["unknown"] * bsz, dtype=object))
        agent_ids = full_batch.non_tensor_batch.get("agent_id", np.array(["agent_0"] * bsz, dtype=object))
        model_id = str(self.config.get("model_id", "unknown_model"))

        traj_ids = np.empty((bsz,), dtype=object)
        model_ids = np.empty((bsz,), dtype=object)
        action_text = np.empty((bsz,), dtype=object)
        created_ts = np.empty((bsz,), dtype=np.float64)

        responses = full_batch.batch.get("responses", None)
        now = time.time()
        for i in range(bsz):
            traj_id = uuid.uuid4().hex
            traj_ids[i] = traj_id
            model_ids[i] = model_id
            created_ts[i] = now

            response_tokens = responses[i] if responses is not None else None
            action_text[i] = self._decode_action_text(response_tokens)

            fat_item = full_batch[i : i + 1]
            self._traj_buffer.put(
                TrajectoryEntry(
                    traj_id=traj_id,
                    model_id=model_id,
                    query_id=str(query_ids[i]),
                    agent_id=str(agent_ids[i]),
                    fat_data=fat_item,
                )
            )

        return DataProto.from_dict(
            non_tensors={
                "traj_id": traj_ids,
                "query_id": query_ids.astype(object),
                "agent_id": agent_ids.astype(object),
                "model_id": model_ids,
                "action_text": action_text,
                "created_ts": created_ts,
            },
            meta_info={"thin_only": True},
        )

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="rollout"))
    def generate_sequences_thin(self, prompts: DataProto) -> DataProto:
        fat_output = self.generate_sequences(prompts)
        full_batch = prompts.union(fat_output)
        return self._build_thin_from_batch(full_batch)

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="rollout"))
    def build_thin_from_generated(self, full_batch: DataProto) -> DataProto:
        return self._build_thin_from_batch(full_batch)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def commit_rewards(self, rewards: DataProto) -> dict:
        traj_ids = rewards.non_tensor_batch.get("traj_id", np.array([], dtype=object))
        if len(traj_ids) == 0:
            return {"star/committed": 0, "star/reward_in": 0, **self._traj_buffer.stats()}

        reward_vec = rewards.batch.get("reward", None)
        done_vec = rewards.batch.get("done", None)

        if reward_vec is None:
            reward_vec = torch.zeros((len(traj_ids),), dtype=torch.float32)
        if done_vec is None:
            done_vec = torch.ones((len(traj_ids),), dtype=torch.bool)

        committed = 0
        for i, traj_id in enumerate(traj_ids):
            ok = self._traj_buffer.commit_reward(
                str(traj_id),
                reward=reward_vec[i].reshape(()).to(torch.float32),
                done=bool(done_vec[i].item()),
            )
            committed += int(ok)

        return {"star/committed": committed, "star/reward_in": len(traj_ids), **self._traj_buffer.stats()}

    def _empty_batch(self) -> DataProto:
        return DataProto.from_dict(non_tensors={"traj_id": np.array([], dtype=object)})

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def build_ready_train_batch(self, max_items: int = 0) -> DataProto:
        entries = self._traj_buffer.pop_ready(max_items=max_items if max_items and max_items > 0 else None)
        if len(entries) == 0:
            return self._empty_batch()

        fat_list = [e.fat_data for e in entries]
        batch = DataProto.concat(fat_list)

        response_mask = batch.batch.get("response_mask", None)
        responses = batch.batch.get("responses", None)
        if responses is None:
            return batch

        bsz, resp_len = responses.shape[0], responses.shape[1]
        token_level_scores = torch.zeros((bsz, resp_len), dtype=torch.float32)

        reward_scalar = torch.tensor([float(e.reward.item()) if e.reward is not None else 0.0 for e in entries])
        if response_mask is None:
            token_level_scores[:, -1] = reward_scalar
        else:
            last_pos = response_mask.to(torch.long).sum(dim=-1) - 1
            last_pos = torch.clamp(last_pos, min=0)
            token_level_scores[torch.arange(bsz), last_pos] = reward_scalar

        extra = DataProto.from_dict(
            tensors={
                "token_level_scores": token_level_scores,
                "token_level_rewards": token_level_scores.clone(),
                "reward": reward_scalar,
                "done": torch.tensor([e.done for e in entries], dtype=torch.bool),
            },
            non_tensors={
                "traj_id": np.array([e.traj_id for e in entries], dtype=object),
                "query_id": np.array([e.query_id for e in entries], dtype=object),
                "agent_id": np.array([e.agent_id for e in entries], dtype=object),
                "model_id": np.array([e.model_id for e in entries], dtype=object),
            },
        )
        return batch.union(extra)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def train_from_local_batch(self, data: DataProto, do_actor: bool = True, do_critic: bool = True) -> dict:
        bsz = len(data)
        if bsz == 0:
            return {"star/consumed": 0, "star/placeholder_update": 0}

        avg_reward = 0.0
        if data.batch is not None and "reward" in data.batch.keys():
            avg_reward = data.batch["reward"].float().mean().item()

        # V1 skeleton: keep FSDP update call site but avoid forcing full PPO fields.
        return {
            "star/consumed": bsz,
            "star/placeholder_update": 1,
            "star/avg_reward": avg_reward,
            "star/do_actor": int(do_actor),
            "star/do_critic": int(do_critic),
        }

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def train_from_ready_queue(
        self,
        max_items: int = 0,
        drop_last: bool = True,
        world_size_divisor: int = 1,
        do_actor: bool = True,
        do_critic: bool = True,
    ) -> dict:
        batch = self.build_ready_train_batch(max_items=max_items)
        bsz = len(batch)
        if bsz == 0:
            return {"star/consumed": 0, "star/dropped": 0, "star/placeholder_update": 0}

        dropped = 0
        if drop_last and world_size_divisor > 1:
            keep = (bsz // world_size_divisor) * world_size_divisor
            if keep <= 0:
                return {"star/consumed": 0, "star/dropped": bsz, "star/placeholder_update": 0}
            if keep < bsz:
                indices = np.random.permutation(bsz)[:keep].tolist()
                batch = batch.select_idxs(indices)
                dropped = bsz - keep

        metrics = self.train_from_local_batch(batch, do_actor=do_actor, do_critic=do_critic)
        metrics["star/dropped"] = dropped
        return metrics

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_weight_sync_group_name(self, group_name: str):
        self._weight_sync_group_name = str(group_name)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_weight_sync_mode(self, mode: str):
        self._weight_sync_mode = str(mode).strip().lower()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def sync_rollout_weights(self):
        assert (self._is_actor or self._is_rollout) and not self.config.hybrid_engine
        assert hasattr(self, "_weights_info") and self._weights_info is not None

        if self._is_actor and self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        params = self._get_actor_params() if self._is_actor else None

        rollout_name = self.config.rollout.name
        inference_model = None
        use_vllm_server_adapter = False
        if self._is_rollout:
            if rollout_name == "vllm":
                from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader

                inference_model = _get_vllm_inference_model(self.rollout)
                if inference_model is not None:
                    patch_vllm_moe_model_weight_loader(inference_model)
                elif hasattr(self.rollout, "update_weights"):
                    use_vllm_server_adapter = True
                else:
                    raise AttributeError(
                        f"Unsupported vllm rollout object for weight sync: {type(self.rollout)}"
                    )
            elif rollout_name == "sglang":
                inference_model = self.rollout._engine
            else:
                raise NotImplementedError(f"Unknown rollout name: {rollout_name}")
        loop = get_event_loop()
        group_name = getattr(self, "_weight_sync_group_name", "actor_rollout")
        sync_mode = str(getattr(self, "_weight_sync_mode", os.environ.get("STAR_WEIGHT_SYNC_MODE", "collective"))).lower()
        if sync_mode == "local_pair":
            channel = _get_local_pair_channel(group_name)
            if self._is_actor:
                try:
                    for key, shape, dtype in self._weights_info:
                        assert key in params
                        origin_data = params[key]
                        if hasattr(origin_data, "full_tensor"):
                            origin_data = origin_data.full_tensor()
                        tensor = torch.empty(shape, dtype=dtype, device=get_torch_device().current_device())
                        tensor.copy_(origin_data)
                        channel.put((key, tensor))
                finally:
                    channel.put((_LOCAL_PAIR_END, None))
            else:
                def _iter_local_pair_weights():
                    for expected_key, _, _ in self._weights_info:
                        recv_key, tensor = channel.get()
                        if recv_key != expected_key:
                            raise RuntimeError(
                                f"local_pair weight order mismatch: got {recv_key}, expected {expected_key}"
                            )
                        yield expected_key, tensor
                    end_key, _ = channel.get()
                    if end_key != _LOCAL_PAIR_END:
                        raise RuntimeError(f"local_pair weight stream missing end sentinel, got {end_key}")

                if rollout_name == "vllm" and use_vllm_server_adapter:
                    loop.run_until_complete(self.rollout.update_weights(_iter_local_pair_weights()))
                else:
                    for expected_key, tensor in _iter_local_pair_weights():
                        if rollout_name == "vllm":
                            inference_model.load_weights([(expected_key, tensor)])
                        elif rollout_name == "sglang":
                            if inference_model is not None:
                                loop.run_until_complete(self.update_weights(inference_model, [(expected_key, tensor)]))
            if self._is_actor and self._is_offload_param:
                offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            get_torch_device().empty_cache()
            return

        if self._is_rollout and rollout_name == "vllm" and use_vllm_server_adapter:
            def _iter_collective_weights():
                for key, shape, dtype in self._weights_info:
                    tensor = torch.empty(shape, dtype=dtype, device=get_torch_device().current_device())
                    if hasattr(self, "_weight_sync_group") and self._weight_sync_group is not None:
                        self._weight_sync_group.broadcast(tensor, src=0, stream=get_torch_device().current_stream())
                    else:
                        collective.broadcast(tensor, src_rank=0, group_name=group_name)
                    yield key, tensor

            loop.run_until_complete(self.rollout.update_weights(_iter_collective_weights()))
            if self._is_actor and self._is_offload_param:
                offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            get_torch_device().empty_cache()
            return

        for key, shape, dtype in self._weights_info:
            tensor = torch.empty(shape, dtype=dtype, device=get_torch_device().current_device())
            if self._is_actor:
                assert key in params
                origin_data = params[key]
                if hasattr(origin_data, "full_tensor"):
                    origin_data = origin_data.full_tensor()
                if torch.distributed.get_rank() == 0:
                    tensor.copy_(origin_data)

            if hasattr(self, "_weight_sync_group") and self._weight_sync_group is not None:
                self._weight_sync_group.broadcast(tensor, src=0, stream=get_torch_device().current_stream())
            else:
                collective.broadcast(tensor, src_rank=0, group_name=group_name)

            if self._is_rollout:
                if rollout_name == "vllm":
                    inference_model.load_weights([(key, tensor)])
                elif rollout_name == "sglang":
                    if inference_model is not None:
                        loop.run_until_complete(self.update_weights(inference_model, [(key, tensor)]))

        if self._is_actor and self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
        get_torch_device().empty_cache()
