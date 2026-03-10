import asyncio
import json
import os
import uuid
import zlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from ray.util.collective import collective
from torch.utils.data import DataLoader, Dataset, Sampler

from verl import DataProto
from verl.experimental.star_ppo.types import EngineSpec
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.ray_trainer import apply_kl_penalty, compute_advantage, compute_response_mask
from verl.trainer.ppo.utils import Role, need_critic, need_reference_policy
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.device import get_nccl_backend
from verl.utils.import_utils import load_extern_object
from verl.utils import omega_conf_to_dataclass
from verl.utils.metric import reduce_metrics
from verl.utils.tracking import Tracking


@dataclass
class ModelWorkerContext:
    model_id: str
    resource_pool: RayResourcePool
    actor_wg: RayWorkerGroup
    rollout_wg: RayWorkerGroup
    critic_wg: Optional[RayWorkerGroup] = None
    ref_policy_wg: Optional[RayWorkerGroup] = None
    rm_wg: Optional[RayWorkerGroup] = None


class StarRayTrainer:
    """Star-topology PPO skeleton trainer with multi-engine routing."""

    def __init__(
        self,
        config,
        tokenizer,
        engine_specs: list[EngineSpec],
        role_worker_mapping,
        ray_worker_group_cls=RayWorkerGroup,
        processor=None,
        reward_fn=None,
        train_dataset: Dataset | None = None,
        val_dataset: Dataset | None = None,
        collate_fn=None,
        train_sampler: Sampler | None = None,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.reward_fn = reward_fn
        self.engine_specs = engine_specs
        self.role_worker_mapping = role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = self.config.trainer.device

        self.use_critic = need_critic(self.config)
        self.use_reference_policy = need_reference_policy(self.config)
        self.use_rm = Role.RewardModel in self.role_worker_mapping

        self.model_ids = [spec.model_id for spec in self.engine_specs]
        self.engine_cfg_by_model_id = {
            str(engine.model_id): engine for engine in self.config.trainer.get("llm_engines", [])
        }
        self.model_contexts: dict[str, ModelWorkerContext] = {}
        self.kl_ctrl_by_model = {}
        self.query_reward_ledger: dict[str, float] = defaultdict(float)
        self.global_steps = 0
        self._max_parallel_rollouts_per_model = int(self.config.star.workflow.get("max_parallel_rollouts_per_model", 32))
        self._rollout_semaphore_by_model: dict[str, asyncio.Semaphore] = {}
        self.workflow_runner = self._create_workflow_runner()
        if self.config.algorithm.use_kl_in_reward:
            for model_id in self.model_ids:
                self.kl_ctrl_by_model[model_id] = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("train_max_samples", -1),
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("val_max_samples", -1),
            )
        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, train_dataset)

        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=self.config.data.get("dataloader_num_workers", 0),
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        self.val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.config.data.val_batch_size or len(val_dataset),
            num_workers=self.config.data.get("dataloader_num_workers", 0),
            drop_last=False,
            collate_fn=collate_fn,
            shuffle=self.config.data.get("validation_shuffle", True),
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps
        self.total_training_steps = total_training_steps

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception:
            pass

    def _clone_actor_rollout_cfg_for_model(self, model_id: str):
        cfg = OmegaConf.create(OmegaConf.to_container(self.config.actor_rollout_ref, resolve=True))
        cfg.model_id = model_id
        cfg.star_buffer = OmegaConf.to_container(self.config.star.buffer, resolve=True)
        engine_cfg = self.engine_cfg_by_model_id.get(model_id, None)
        if engine_cfg is not None:
            # Optional per-engine model path override for true multi-LLM training.
            engine_model_path = engine_cfg.get("model_path", None)
            if engine_model_path is not None:
                cfg.model.path = str(engine_model_path)
        return cfg

    def init_workers(self):
        for spec in self.engine_specs:
            resource_pool = RayResourcePool(
                process_on_nodes=[spec.n_gpus_per_node] * spec.nnodes,
                use_gpu=True,
                max_colocate_count=3,
                name_prefix=f"star_{spec.model_id}",
                accelerator_type=spec.accelerator_type,
            )

            actor_rollout_cfg = self._clone_actor_rollout_cfg_for_model(spec.model_id)
            class_dict = {
                "actor": RayClassWithInitArgs(
                    cls=self.role_worker_mapping[Role.Actor], config=actor_rollout_cfg, role=str(Role.Actor)
                ),
                "rollout": RayClassWithInitArgs(
                    cls=self.role_worker_mapping[Role.Rollout], config=actor_rollout_cfg, role=str(Role.Rollout)
                ),
            }

            if self.use_critic:
                critic_cfg = omega_conf_to_dataclass(self.config.critic)
                class_dict["critic"] = RayClassWithInitArgs(
                    cls=self.role_worker_mapping[Role.Critic], config=critic_cfg
                )

            if self.use_reference_policy:
                class_dict["ref"] = RayClassWithInitArgs(
                    cls=self.role_worker_mapping[Role.RefPolicy], config=actor_rollout_cfg, role=str(Role.RefPolicy)
                )

            if self.use_rm:
                class_dict["rm"] = RayClassWithInitArgs(
                    cls=self.role_worker_mapping[Role.RewardModel],
                    config=omega_conf_to_dataclass(self.config.reward_model),
                )

            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            # Star stateless weight-sync may need actor/rollout RPCs to run concurrently
            # on the same colocated WorkerDict process.
            worker_max_concurrency = int(os.environ.get("STAR_WORKER_MAX_CONCURRENCY", "4"))
            worker_dict_cls.update_options({"max_concurrency": worker_max_concurrency})
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                device_name=self.device_name,
            )
            spawned = wg_dict.spawn(prefix_set=class_dict.keys())

            actor_wg = spawned["actor"]
            rollout_wg = spawned["rollout"]
            actor_wg.init_model()
            rollout_wg.init_model()

            critic_wg = spawned.get("critic")
            if critic_wg is not None:
                critic_wg.init_model()

            ref_wg = spawned.get("ref")
            if ref_wg is not None:
                ref_wg.init_model()

            rm_wg = spawned.get("rm")
            if rm_wg is not None:
                rm_wg.init_model()

            self.model_contexts[spec.model_id] = ModelWorkerContext(
                model_id=spec.model_id,
                resource_pool=resource_pool,
                actor_wg=actor_wg,
                rollout_wg=rollout_wg,
                critic_wg=critic_wg,
                ref_policy_wg=ref_wg,
                rm_wg=rm_wg,
            )
            self._init_weight_sync_group(spec.model_id, self.model_contexts[spec.model_id])
            self._sync_rollout_weights(spec.model_id, self.model_contexts[spec.model_id])

    def _init_weight_sync_group(self, model_id: str, ctx: ModelWorkerContext):
        weights_info = ctx.actor_wg.get_actor_weights_info()[0]
        ctx.rollout_wg.set_actor_weights_info(weights_info)

        group_name = f"actor_rollout_{model_id}"
        ctx.actor_wg.set_weight_sync_group_name(group_name)
        ctx.rollout_wg.set_weight_sync_group_name(group_name)

        actor_rollout_workers = ctx.actor_wg.workers + ctx.rollout_wg.workers
        n_workers = len(actor_rollout_workers)

        def _to_ref_list(x):
            if x is None:
                return []
            if isinstance(x, list | tuple):
                return list(x)
            return [x]

        # Weight-sync mode:
        # - auto (default): try Ray collective first, fallback to stateless.
        # - collective: use Ray collective only.
        # - stateless: use stateless process group only.
        mode = str(os.environ.get("STAR_WEIGHT_SYNC_MODE", "auto")).strip().lower()
        if mode not in {"auto", "collective", "stateless"}:
            print(f"[star] invalid STAR_WEIGHT_SYNC_MODE={mode}, fallback to auto")
            mode = "auto"

        master_address = ray.get(ctx.actor_wg.workers[0]._get_node_ip.remote()).strip("[]")
        fixed_port = int(os.environ.get("STAR_WEIGHT_SYNC_MASTER_PORT", "0"))
        model_idx = self.model_ids.index(model_id) if model_id in self.model_ids else 0
        max_retries = int(os.environ.get("STAR_WEIGHT_SYNC_RETRIES", "3"))
        retry_stride = int(os.environ.get("STAR_WEIGHT_SYNC_PORT_RETRY_STRIDE", "10"))
        collective_ready = False
        if self.device_name != "npu" and mode in {"auto", "collective"}:
            try:
                collective.create_collective_group(
                    actor_rollout_workers,
                    n_workers,
                    list(range(0, n_workers)),
                    backend=get_nccl_backend(),
                    group_name=group_name,
                )
                collective_ready = True
                print(f"[star] Ray collective group ready model={model_id} group={group_name}")
            except Exception as e:
                print(f"[star] Ray collective unavailable model={model_id}: {e}")
                if mode == "collective":
                    raise

        need_stateless = (self.device_name == "npu") or (mode == "stateless") or (
            mode == "auto" and not collective_ready
        )
        if need_stateless:
            last_err = None
            for attempt in range(max_retries):
                if fixed_port > 0:
                    # Per-model stable port + retry stride to avoid collisions with stale groups.
                    master_port = fixed_port + model_idx + attempt * retry_stride
                else:
                    master_port = ray.get(ctx.actor_wg.workers[0]._get_free_port.remote())
                print(
                    f"[star] init stateless weight sync model={model_id} attempt={attempt + 1}/{max_retries} "
                    f"addr={master_address}:{master_port} workers={n_workers}"
                )
                actor_refs = ctx.actor_wg.create_weight_sync_group(master_address, master_port, 0, n_workers)
                rollout_refs = ctx.rollout_wg.create_weight_sync_group(
                    master_address, master_port, len(ctx.actor_wg.workers), n_workers
                )
                try:
                    ray.get(_to_ref_list(actor_refs) + _to_ref_list(rollout_refs))
                    last_err = None
                    print(f"[star] stateless weight sync ready model={model_id}")
                    break
                except Exception as e:
                    last_err = e
                    print(f"[star] stateless weight sync init failed model={model_id} attempt={attempt + 1}: {e}")
                    time.sleep(2)
            if last_err is not None:
                raise last_err

    @staticmethod
    def _sync_rollout_weights(model_id: str, ctx: ModelWorkerContext):
        ctx.actor_wg.sync_rollout_weights()
        ray.get(ctx.rollout_wg.sync_rollout_weights())

    def _ensure_routing_fields(self, batch: DataProto):
        bsz = len(batch)
        if "query_id" not in batch.non_tensor_batch:
            batch.non_tensor_batch["query_id"] = np.array([uuid.uuid4().hex for _ in range(bsz)], dtype=object)
        if "agent_id" not in batch.non_tensor_batch:
            batch.non_tensor_batch["agent_id"] = np.array(["agent_0"] * bsz, dtype=object)

    def _route_batch(self, batch: DataProto, epoch: int) -> dict[str, list[int]]:
        routed: dict[str, list[int]] = {model_id: [] for model_id in self.model_ids}
        query_ids = batch.non_tensor_batch["query_id"]
        policy = self.config.trainer.routing.policy

        for idx, query_id in enumerate(query_ids):
            base = str(query_id)
            if policy == "epoch_dynamic":
                base = f"{epoch}:{base}"
            slot = zlib.crc32(base.encode("utf-8")) % len(self.model_ids)
            routed[self.model_ids[slot]].append(idx)

        return {k: v for k, v in routed.items() if v}

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        # Skeleton version: keep all fields for easy routing/reward alignment.
        return batch

    async def _rollout_model_async(self, model_id: str, batch: DataProto):
        ctx = self.model_contexts[model_id]
        if model_id not in self._rollout_semaphore_by_model:
            self._rollout_semaphore_by_model[model_id] = asyncio.Semaphore(max(1, self._max_parallel_rollouts_per_model))
        async with self._rollout_semaphore_by_model[model_id]:
            thin = await asyncio.to_thread(ctx.rollout_wg.generate_sequences_thin, self._get_gen_batch(batch))
        return model_id, thin, batch

    @staticmethod
    def _extract_first_json_object(text: str) -> Optional[dict]:
        raw = str(text or "").strip()
        if not raw:
            return None
        try:
            obj = json.loads(raw)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass

        # fallback: find the first balanced json object
        start = raw.find("{")
        while start != -1:
            depth = 0
            for i in range(start, len(raw)):
                ch = raw[i]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        segment = raw[start : i + 1]
                        try:
                            obj = json.loads(segment)
                            if isinstance(obj, dict):
                                return obj
                        except Exception:
                            break
            start = raw.find("{", start + 1)
        return None

    @staticmethod
    def _to_str_list(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, dict):
            if "ground_truth" in value:
                return StarRayTrainer._to_str_list(value["ground_truth"])
            if "target" in value:
                return StarRayTrainer._to_str_list(value["target"])
            return [json.dumps(value, ensure_ascii=False)]
        if isinstance(value, list | tuple):
            return [str(x) for x in value if str(x).strip()]
        if isinstance(value, np.ndarray):
            return [str(x) for x in value.tolist() if str(x).strip()]
        return [str(value)]

    def _extract_string_vector(self, source_batch: DataProto, keys: list[str], default: str = "") -> np.ndarray:
        bsz = len(source_batch)
        for key in keys:
            if key in source_batch.non_tensor_batch:
                values = source_batch.non_tensor_batch[key]
                out = [str(v) if v is not None else default for v in values]
                return np.array(out, dtype=object)
        return np.array([default] * bsz, dtype=object)

    def _extract_ground_truth_lists(self, source_batch: DataProto, keys: list[str]) -> list[list[str]]:
        bsz = len(source_batch)
        out: list[list[str]] = [[] for _ in range(bsz)]
        for i in range(bsz):
            picked = None
            for key in keys:
                if key not in source_batch.non_tensor_batch:
                    continue
                value = source_batch.non_tensor_batch[key][i]
                if value is None:
                    continue
                picked = value
                break
            out[i] = self._to_str_list(picked)
        return out

    def _build_workflow_prompt_batch(
        self,
        source_batch: DataProto,
        raw_prompts: list[list[dict]],
        agent_id: str,
    ) -> DataProto:
        bsz = len(source_batch)
        if source_batch.batch is not None and "dummy_tensor" in source_batch.batch.keys():
            tensors = {"dummy_tensor": source_batch.batch["dummy_tensor"]}
        else:
            tensors = {"dummy_tensor": torch.zeros((bsz, 1), dtype=torch.uint8)}

        non_tensors = {k: v.copy() for k, v in source_batch.non_tensor_batch.items()}
        non_tensors["raw_prompt"] = np.array(raw_prompts, dtype=object)
        non_tensors["agent_id"] = np.array([agent_id] * bsz, dtype=object)
        return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info=dict(source_batch.meta_info))

    def _build_commit_rewards_from_thin(self, thin_batch: DataProto, reward: np.ndarray) -> DataProto:
        done = np.ones((len(thin_batch),), dtype=bool)
        return DataProto.from_dict(
            tensors={
                "reward": torch.tensor(reward, dtype=torch.float32),
                "done": torch.tensor(done, dtype=torch.bool),
            },
            non_tensors={
                "traj_id": thin_batch.non_tensor_batch["traj_id"],
                "model_id": thin_batch.non_tensor_batch["model_id"],
                "query_id": thin_batch.non_tensor_batch["query_id"],
                "agent_id": thin_batch.non_tensor_batch["agent_id"],
            },
        )

    @staticmethod
    def _empty_rewards() -> DataProto:
        return DataProto.from_dict(
            tensors={"reward": torch.zeros((0,), dtype=torch.float32), "done": torch.zeros((0,), dtype=torch.bool)},
            non_tensors={
                "traj_id": np.array([], dtype=object),
                "model_id": np.array([], dtype=object),
                "query_id": np.array([], dtype=object),
                "agent_id": np.array([], dtype=object),
            },
        )

    async def _run_single_agent_workflow(self, batch: DataProto, epoch: int) -> tuple[DataProto, dict[str, float]]:
        routed = self._route_batch(batch, epoch)
        tasks = [self._rollout_model_async(model_id, batch.select_idxs(idxs)) for model_id, idxs in routed.items()]
        rollout_results = await asyncio.gather(*tasks)

        reward_parts = []
        for _, thin_output, source_sub_batch in rollout_results:
            if len(thin_output) == 0:
                continue
            reward_parts.append(self._assemble_rewards(thin_output, source_batch=source_sub_batch))

        if len(reward_parts) == 0:
            return self._empty_rewards(), {}
        rewards = DataProto.concat(reward_parts) if len(reward_parts) > 1 else reward_parts[0]
        return rewards, {}

    def _create_workflow_runner(self):
        workflow_cfg = self.config.star.get("workflow", {})
        runner_cfg = workflow_cfg.get("runner", {})
        if "path" in runner_cfg and "name" in runner_cfg:
            runner_cls = load_extern_object(str(runner_cfg.get("path")), str(runner_cfg.get("name")))
            return runner_cls(trainer=self, config=self.config)

        from verl.experimental.star_ppo.workflows.builtin import BuiltinWorkflowRunner

        return BuiltinWorkflowRunner(trainer=self, config=self.config)

    @staticmethod
    def _extract_optional_vector(source_batch: Optional[DataProto], keys: list[str], default: float, size: int) -> np.ndarray:
        if source_batch is None:
            return np.full((size,), default, dtype=np.float32)
        for key in keys:
            if source_batch.batch is not None and key in source_batch.batch.keys():
                return source_batch.batch[key].detach().cpu().float().reshape(-1).numpy()
            if key in source_batch.non_tensor_batch:
                return np.array(source_batch.non_tensor_batch[key], dtype=np.float32).reshape(-1)
        return np.full((size,), default, dtype=np.float32)

    @staticmethod
    def _extract_bool_vector(source_batch: Optional[DataProto], keys: list[str], default: bool, size: int) -> np.ndarray:
        if source_batch is None:
            return np.full((size,), default, dtype=bool)
        for key in keys:
            if source_batch.batch is not None and key in source_batch.batch.keys():
                return source_batch.batch[key].detach().cpu().bool().reshape(-1).numpy()
            if key in source_batch.non_tensor_batch:
                return np.array(source_batch.non_tensor_batch[key], dtype=bool).reshape(-1)
        return np.full((size,), default, dtype=bool)

    def _assemble_rewards(self, thin_batch: DataProto, source_batch: Optional[DataProto] = None) -> DataProto:
        action_text = thin_batch.non_tensor_batch.get("action_text", np.array([], dtype=object))
        bsz = len(action_text)
        if bsz == 0:
            return DataProto.from_dict(
                tensors={"reward": torch.zeros((0,), dtype=torch.float32), "done": torch.zeros((0,), dtype=torch.bool)},
                non_tensors={"traj_id": np.array([], dtype=object), "model_id": np.array([], dtype=object)},
            )

        format_reward = self._extract_optional_vector(
            source_batch,
            keys=["format_reward", "format_rewards", "step_reward", "rule_reward"],
            default=np.nan,
            size=bsz,
        )
        fallback_reward = np.array([1.0 if len(str(text).strip()) > 0 else 0.0 for text in action_text], dtype=np.float32)
        format_reward = np.where(np.isnan(format_reward), fallback_reward, format_reward).astype(np.float32)

        outcome_reward = self._extract_optional_vector(
            source_batch,
            keys=["outcome_reward", "final_reward", "task_reward"],
            default=0.0,
            size=bsz,
        ).astype(np.float32)
        workflow_done = self._extract_bool_vector(
            source_batch,
            keys=["workflow_done", "query_done", "is_done", "done"],
            default=True,
            size=bsz,
        )

        reward = np.zeros((bsz,), dtype=np.float32)
        reward_mode = self.config.star.reward.get("mode", "streaming")
        emit_intermediate = bool(self.config.star.reward.get("emit_intermediate_format", True))
        query_ids = thin_batch.non_tensor_batch["query_id"]
        for i in range(bsz):
            query_id = str(query_ids[i])
            self.query_reward_ledger[query_id] += float(format_reward[i])

            if reward_mode == "terminal" and not bool(workflow_done[i]):
                reward[i] = float(format_reward[i]) if emit_intermediate else 0.0
            elif reward_mode == "terminal" and bool(workflow_done[i]):
                reward[i] = float(self.query_reward_ledger[query_id] + outcome_reward[i])
                self.query_reward_ledger.pop(query_id, None)
            else:
                reward[i] = float(format_reward[i] + (outcome_reward[i] if workflow_done[i] else 0.0))
                if workflow_done[i]:
                    self.query_reward_ledger.pop(query_id, None)

        # In V3 we always release trajectories to training queue each step.
        done = np.full((bsz,), True, dtype=bool)
        rewards = DataProto.from_dict(
            tensors={"reward": torch.tensor(reward, dtype=torch.float32), "done": torch.tensor(done, dtype=torch.bool)},
            non_tensors={
                "traj_id": thin_batch.non_tensor_batch["traj_id"],
                "model_id": thin_batch.non_tensor_batch["model_id"],
                "query_id": thin_batch.non_tensor_batch["query_id"],
                "agent_id": thin_batch.non_tensor_batch["agent_id"],
            },
            meta_info={"format_reward": format_reward.tolist(), "outcome_reward": outcome_reward.tolist()},
        )
        return rewards

    def _commit_rewards(self, rewards: DataProto) -> dict[str, float]:
        model_ids = rewards.non_tensor_batch["model_id"]
        metrics = {}
        for model_id in self.model_ids:
            indices = [i for i, mid in enumerate(model_ids) if str(mid) == model_id]
            if not indices:
                continue
            sub = rewards.select_idxs(indices)
            worker_outputs = self.model_contexts[model_id].rollout_wg.commit_rewards(sub)
            reduced = self._reduce_worker_metrics(worker_outputs)
            for k, v in reduced.items():
                metrics[f"model/{model_id}/{k}"] = v
        # reward metrics by agent id (for independent curves on tracking backend)
        reward_vec = rewards.batch["reward"].detach().cpu().numpy() if len(rewards) > 0 else np.array([])
        agent_ids = rewards.non_tensor_batch.get("agent_id", np.array([], dtype=object))
        for agent_id in np.unique(agent_ids):
            mask = agent_ids == agent_id
            if mask.sum() > 0:
                metrics[f"agent/{agent_id}/reward_mean"] = float(np.mean(reward_vec[mask]))
                metrics[f"agent/{agent_id}/samples"] = float(np.sum(mask))
        return metrics

    @staticmethod
    def _reduce_worker_metrics(worker_outputs) -> dict[str, float]:
        if isinstance(worker_outputs, dict):
            return {k: float(v) for k, v in worker_outputs.items() if isinstance(v, int | float)}
        if not isinstance(worker_outputs, list) or len(worker_outputs) == 0:
            return {}

        sums: dict[str, float] = {}
        counts: dict[str, int] = {}
        for item in worker_outputs:
            if not isinstance(item, dict):
                continue
            for k, v in item.items():
                if isinstance(v, int | float):
                    sums[k] = sums.get(k, 0.0) + float(v)
                    counts[k] = counts.get(k, 0) + 1
        return {k: sums[k] / max(1, counts[k]) for k in sums}

    def _get_dp_size(self, worker_group, role: str) -> int:
        if role not in worker_group._dispatch_info:
            dp_rank_mapping = worker_group._query_dispatch_info(role)
            worker_group._dispatch_info[role] = dp_rank_mapping
        else:
            dp_rank_mapping = worker_group._dispatch_info[role]
        return max(dp_rank_mapping) + 1

    def _merge_ready_batches(self, ready_parts: list[DataProto]) -> Optional[DataProto]:
        valid = [x for x in ready_parts if isinstance(x, DataProto) and len(x) > 0]
        if not valid:
            return None
        return valid[0] if len(valid) == 1 else DataProto.concat(valid)

    def _maybe_drop_last(self, batch: DataProto, dp_size: int) -> tuple[DataProto, int]:
        enforce_divisible_batch = bool(self.config.star.train.get("enforce_divisible_batch", True))
        if dp_size <= 1:
            return batch, 0
        if not enforce_divisible_batch and not self.config.star.train.drop_last:
            return batch, 0
        bsz = len(batch)
        keep = (bsz // dp_size) * dp_size
        if keep <= 0:
            return batch.select_idxs([]), bsz
        if keep == bsz:
            return batch, 0
        indices = np.random.permutation(bsz)[:keep].tolist()
        return batch.select_idxs(indices), bsz - keep

    def _run_model_ppo_update(self, model_id: str, ctx: ModelWorkerContext, batch: DataProto, global_step: int):
        metrics: dict[str, float] = {}
        if len(batch) == 0:
            return metrics

        if "response_mask" not in batch.batch.keys():
            batch.batch["response_mask"] = compute_response_mask(batch)

        batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

        old_log_prob = ctx.actor_wg.compute_log_prob(batch)
        if "entropys" in old_log_prob.batch.keys():
            old_log_prob.batch.pop("entropys")
        batch = batch.union(old_log_prob)

        if self.use_reference_policy and ctx.ref_policy_wg is not None:
            ref_log_prob = ctx.ref_policy_wg.compute_ref_log_prob(batch)
            batch = batch.union(ref_log_prob)

        if self.use_critic and ctx.critic_wg is not None:
            values = ctx.critic_wg.compute_values(batch)
            batch = batch.union(values)

        if self.config.algorithm.use_kl_in_reward and "ref_log_prob" in batch.batch.keys():
            batch, kl_metrics = apply_kl_penalty(
                batch,
                kl_ctrl=self.kl_ctrl_by_model[model_id],
                kl_penalty=self.config.algorithm.kl_penalty,
            )
            for key, val in kl_metrics.items():
                metrics[f"model/{model_id}/{key}"] = float(val)
        else:
            if "token_level_rewards" not in batch.batch and "token_level_scores" in batch.batch:
                batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

        batch = compute_advantage(
            batch,
            adv_estimator=self.config.algorithm.adv_estimator,
            gamma=self.config.algorithm.gamma,
            lam=self.config.algorithm.lam,
            num_repeat=self.config.actor_rollout_ref.rollout.n,
            norm_adv_by_std_in_grpo=self.config.algorithm.get("norm_adv_by_std_in_grpo", True),
            config=self.config.algorithm,
        )

        if "agent_id" in batch.non_tensor_batch and "reward" in batch.batch.keys():
            agent_ids = batch.non_tensor_batch["agent_id"]
            reward_vec = batch.batch["reward"].detach().cpu().numpy().reshape(-1)
            adv_vec = batch.batch["advantages"].detach().cpu().float().mean(dim=-1).numpy()
            for agent_id in np.unique(agent_ids):
                mask = agent_ids == agent_id
                if mask.sum() == 0:
                    continue
                metrics[f"model/{model_id}/agent/{agent_id}/reward_mean"] = float(np.mean(reward_vec[mask]))
                metrics[f"model/{model_id}/agent/{agent_id}/adv_mean"] = float(np.mean(adv_vec[mask]))
                metrics[f"model/{model_id}/agent/{agent_id}/samples"] = float(np.sum(mask))

        if self.use_critic and ctx.critic_wg is not None:
            critic_output = ctx.critic_wg.update_critic(batch)
            critic_metrics = reduce_metrics(critic_output.meta_info.get("metrics", {}))
            for key, val in critic_metrics.items():
                metrics[f"model/{model_id}/{key}"] = float(val)

        if self.config.trainer.critic_warmup <= global_step:
            rollout_cfg = self.config.actor_rollout_ref.rollout
            batch.meta_info["multi_turn"] = rollout_cfg.multi_turn.enable
            batch.meta_info["temperature"] = rollout_cfg.temperature
            actor_output = ctx.actor_wg.update_actor(batch)
            actor_metrics = reduce_metrics(actor_output.meta_info.get("metrics", {}))
            for key, val in actor_metrics.items():
                metrics[f"model/{model_id}/{key}"] = float(val)
            self._sync_rollout_weights(model_id, ctx)

        metrics[f"model/{model_id}/star/consumed"] = float(len(batch))
        return metrics

    def _global_sync_and_update(self) -> dict[str, float]:
        metrics = {}
        max_ready_items = int(self.config.star.train.get("max_ready_items", 0))

        for model_id, ctx in self.model_contexts.items():
            ready_parts = ctx.rollout_wg.build_ready_train_batch(max_items=max_ready_items)
            ready_batch = self._merge_ready_batches(ready_parts if isinstance(ready_parts, list) else [ready_parts])
            if ready_batch is None:
                metrics[f"model/{model_id}/star/consumed"] = 0.0
                metrics[f"model/{model_id}/star/dropped"] = 0.0
                continue

            actor_dp_size = self._get_dp_size(ctx.actor_wg, "actor")
            metrics[f"model/{model_id}/star/drop_divisor"] = float(actor_dp_size)
            ready_batch, dropped = self._maybe_drop_last(ready_batch, actor_dp_size)
            metrics[f"model/{model_id}/star/dropped"] = float(dropped)
            if len(ready_batch) == 0:
                metrics[f"model/{model_id}/star/consumed"] = 0.0
                continue

            ppo_metrics = self._run_model_ppo_update(
                model_id=model_id,
                ctx=ctx,
                batch=ready_batch,
                global_step=self._global_step,
            )
            metrics.update(ppo_metrics)

        return metrics

    def _get_checkpoint_root(self) -> str:
        checkpoint_root = self.config.trainer.default_local_dir
        if not os.path.isabs(checkpoint_root):
            checkpoint_root = os.path.join(os.getcwd(), checkpoint_root)
        return checkpoint_root

    def _save_checkpoint(self, step: int):
        checkpoint_root = self._get_checkpoint_root()
        global_step_folder = os.path.join(checkpoint_root, f"global_step_{step}")
        os.makedirs(global_step_folder, exist_ok=True)

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        for model_id, ctx in self.model_contexts.items():
            model_folder = os.path.join(global_step_folder, model_id)
            actor_local_path = os.path.join(model_folder, "actor")
            actor_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{step}", model_id, "actor")
            )
            ctx.actor_wg.save_checkpoint(
                actor_local_path,
                actor_remote_path,
                step,
                max_ckpt_to_keep=max_actor_ckpt_to_keep,
            )

            if self.use_critic and ctx.critic_wg is not None:
                critic_local_path = os.path.join(model_folder, str(Role.Critic))
                critic_remote_path = (
                    None
                    if self.config.trainer.default_hdfs_dir is None
                    else os.path.join(
                        self.config.trainer.default_hdfs_dir, f"global_step_{step}", model_id, str(Role.Critic)
                    )
                )
                ctx.critic_wg.save_checkpoint(
                    critic_local_path,
                    critic_remote_path,
                    step,
                    max_ckpt_to_keep=max_critic_ckpt_to_keep,
                )

        star_meta = {"global_step": step, "models": sorted(self.model_contexts.keys())}
        with open(os.path.join(global_step_folder, "star_meta.json"), "w", encoding="utf-8") as f:
            json.dump(star_meta, f, ensure_ascii=False, indent=2)
        with open(os.path.join(checkpoint_root, "latest_checkpointed_iteration.txt"), "w", encoding="utf-8") as f:
            f.write(str(step))

    def _load_checkpoint(self) -> int:
        resume_mode = self.config.trainer.resume_mode
        if resume_mode == "disable":
            self.global_steps = 0
            return 0

        if resume_mode == "resume_path":
            global_step_folder = self.config.trainer.resume_from_path
            if not os.path.isabs(global_step_folder):
                global_step_folder = os.path.join(os.getcwd(), global_step_folder)
        else:
            checkpoint_root = self._get_checkpoint_root()
            global_step_folder = find_latest_ckpt_path(checkpoint_root)
            if global_step_folder is None:
                self.global_steps = 0
                return 0

        if global_step_folder is None or not os.path.exists(global_step_folder):
            self.global_steps = 0
            return 0

        self.global_steps = int(global_step_folder.split("global_step_")[-1])
        for model_id, ctx in self.model_contexts.items():
            model_folder = os.path.join(global_step_folder, model_id)
            actor_path = os.path.join(model_folder, "actor")
            critic_path = os.path.join(model_folder, str(Role.Critic))
            ctx.actor_wg.load_checkpoint(
                actor_path if os.path.exists(actor_path) else None,
                del_local_after_load=self.config.trainer.del_local_ckpt_after_load,
            )
            if self.use_critic and ctx.critic_wg is not None:
                ctx.critic_wg.load_checkpoint(
                    critic_path if os.path.exists(critic_path) else None,
                    del_local_after_load=self.config.trainer.del_local_ckpt_after_load,
                )

            self._sync_rollout_weights(model_id, ctx)
        return self.global_steps

    def _drain_rollout_ready_queues(self):
        # Validation also uses thin->commit flow, so drain ready queue to avoid
        # mixing validation trajectories into subsequent training updates.
        for _, ctx in self.model_contexts.items():
            _ = ctx.rollout_wg.build_ready_train_batch(max_items=0)

    async def _run_validation(self, epoch: int, global_step: int) -> dict[str, float]:
        max_batches = int(self.config.trainer.get("val_max_batches", -1))
        batch_count = 0
        reward_sum = 0.0
        reward_count = 0
        workflow_acc: dict[str, list[float]] = defaultdict(list)

        for batch_idx, batch_dict in enumerate(self.val_dataloader):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            batch_count += 1
            batch = DataProto.from_single_dict(batch_dict)
            self._ensure_routing_fields(batch)
            rewards, workflow_metrics = await self.workflow_runner.run_batch(batch, epoch)

            for key, val in workflow_metrics.items():
                if isinstance(val, int | float):
                    workflow_acc[key].append(float(val))

            if len(rewards) > 0:
                reward_vec = rewards.batch["reward"].detach().cpu().float().reshape(-1).numpy()
                reward_sum += float(np.sum(reward_vec))
                reward_count += int(reward_vec.shape[0])
                # Commit to local buffers so worker-side trajectory states are consistent.
                self._commit_rewards(rewards)
                self._drain_rollout_ready_queues()

        metrics: dict[str, float] = {
            "validation/global_step": float(global_step),
            "validation/epoch": float(epoch),
            "validation/batches": float(batch_count),
            "validation/samples": float(reward_count),
            "validation/reward_mean": float(reward_sum / max(1, reward_count)),
        }
        for key, values in workflow_acc.items():
            if values:
                metrics[f"validation/{key}"] = float(np.mean(values))
        return metrics

    async def fit(self):
        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        global_step = self._load_checkpoint()
        self._global_step = global_step
        start_epoch = global_step // max(1, len(self.train_dataloader))
        val_before_train = bool(self.config.trainer.get("val_before_train", False))
        test_freq = int(self.config.trainer.get("test_freq", -1))

        if val_before_train:
            val_metrics = await self._run_validation(epoch=start_epoch, global_step=global_step)
            logger.log(data=val_metrics, step=global_step)
            print(f"[star] pre-train validation={val_metrics}")

        if bool(self.config.trainer.get("val_only", False)):
            return

        for epoch in range(start_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                global_step += 1
                self._global_step = global_step
                if global_step > self.total_training_steps:
                    break

                batch = DataProto.from_single_dict(batch_dict)
                self._ensure_routing_fields(batch)
                rewards, workflow_metrics = await self.workflow_runner.run_batch(batch, epoch)
                if len(rewards) == 0:
                    continue

                commit_metrics = self._commit_rewards(rewards)

                step_metrics = {
                    "training/global_step": float(global_step),
                    "training/epoch": float(epoch),
                    **workflow_metrics,
                    **commit_metrics,
                }
                logger.log(data=step_metrics, step=global_step)
                if global_step % max(1, self.config.trainer.get("log_freq", 1)) == 0:
                    print(f"[star] step={global_step} commit={step_metrics}")

                is_last_step = global_step >= self.total_training_steps
                if test_freq > 0 and (is_last_step or global_step % test_freq == 0):
                    val_metrics = await self._run_validation(epoch=epoch, global_step=global_step)
                    logger.log(data=val_metrics, step=global_step)
                    print(f"[star] step={global_step} validation={val_metrics}")

                if self.config.trainer.save_freq > 0 and (
                    is_last_step or global_step % self.config.trainer.save_freq == 0
                ):
                    self._save_checkpoint(global_step)

            sync_metrics = self._global_sync_and_update()
            sync_metrics.update({"training/global_step": float(global_step), "training/epoch": float(epoch)})
            logger.log(data=sync_metrics, step=global_step)
            print(f"[star] epoch={epoch} sync_update={sync_metrics}")

            if global_step >= self.total_training_steps:
                break
