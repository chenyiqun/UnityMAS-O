"""STAR PPO worker adapters built on verl's unified engine abstraction.

The STAR layer does not implement FSDP or Megatron directly. Actor/ref/critic
training is delegated to ``TrainingWorker`` and the engine selected by config
(``fsdp``/``fsdp2``/``megatron``). This module only adds STAR-specific detached
actor/rollout coordination, trajectory buffering, and weight sync glue.
"""

import asyncio
import os
import threading
import time
import uuid
from functools import partial
from typing import Any, Optional

import numpy as np
import torch
from ray.util.collective import collective

from verl import DataProto
from verl.experimental.separation.engine_workers import DetachActorWorker
from verl.experimental.star_ppo.trajectory_buffer import TrajectoryBuffer, TrajectoryEntry
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.utils.device import get_torch_device
from verl.utils.model import convert_weight_keys
from verl.utils.ray_utils import get_event_loop
from verl.workers.config.engine import TrainingWorkerConfig
from verl.workers.engine_workers import TrainingWorker

__all__ = [
    "StarDetachActorWorker",
    "StarDetachAsyncRolloutWorker",
    "CriticWorker",
]


_LOCAL_PAIR_END = "__star_local_pair_end__"
_LOCAL_PAIR_ERROR = "__star_local_pair_error__"
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


def _run_coro_blocking(coro_factory):
    loop = get_event_loop()
    if not loop.is_running():
        return loop.run_until_complete(coro_factory())

    result = {}

    def _runner():
        try:
            result["value"] = asyncio.run(coro_factory())
        except BaseException as exc:
            result["error"] = exc

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()
    if "error" in result:
        raise result["error"]
    return result.get("value")


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


def _actor_strategy(worker) -> str:
    return str(getattr(getattr(worker, "config", None), "actor", {}).get("strategy", "")).strip().lower()


def _is_fsdp_strategy(strategy: str) -> bool:
    return str(strategy).strip().lower() in {"fsdp", "fsdp2"}


def _actor_engine(worker):
    actor = getattr(worker, "actor", None)
    return getattr(actor, "engine", None)


def _actor_param_offload_enabled(worker) -> bool:
    engine = _actor_engine(worker)
    return bool(getattr(engine, "is_param_offload_enabled", False))


def _offload_actor_params_after_weight_sync(worker) -> None:
    engine = _actor_engine(worker)
    if engine is not None and _actor_param_offload_enabled(worker):
        engine.to("cpu", model=True, optimizer=False, grad=False)


def _get_actor_weights_info_from_state_dict(actor_worker):
    """Return HF-keyed weight metadata without materializing full FSDP/DTensor params."""
    if not _is_fsdp_strategy(_actor_strategy(actor_worker)):
        # Megatron weight export is handled by MegatronEngine.get_per_tensor_param().
        return None

    engine = getattr(actor_worker.actor, "engine", None)
    module = getattr(engine, "module", None)
    if module is None:
        return None

    peft_model = getattr(module, "_fsdp_wrapped_module", module)
    model_config = getattr(engine, "model_config", None)
    lora_config = getattr(model_config, "lora", {}) or {}
    if hasattr(peft_model, "peft_config") and not lora_config.get("merge", False):
        return None

    params = convert_weight_keys(module.state_dict(), peft_model)
    return [
        (
            key,
            tensor.size(),
            # FSDP get_per_tensor_param materializes DTensor weights as bf16.
            torch.bfloat16 if hasattr(tensor, "full_tensor") else tensor.dtype,
        )
        for key, tensor in params.items()
    ]


def _rollout_base_sync_done(worker) -> bool:
    if hasattr(worker, "base_sync_done"):
        return bool(worker.base_sync_done)
    load_format = getattr(getattr(worker.config, "rollout", {}), "load_format", "")
    return "dummy" not in str(load_format)


def _first_not_none(*values):
    for value in values:
        if value is not None:
            return value
    return None


def _set_rollout_base_sync_done(worker, done: bool) -> None:
    worker.base_sync_done = bool(done)
    if done:
        worker._weight_sync_needs_lora_base_sync = False


def _actor_per_tensor_param(worker, base_sync_done: bool = True):
    kwargs = {"base_sync_done": base_sync_done}
    rollout_cfg = getattr(worker.config, "rollout", {})
    kwargs["layered_summon"] = getattr(worker, "layered_summon", rollout_cfg.get("layered_summon", False))
    return worker.actor.engine.get_per_tensor_param(**kwargs)


def _weights_info_from_params(params_iter):
    return [(key, tensor.size(), tensor.dtype) for key, tensor in params_iter]


def _build_actor_weights_info_payload(actor_worker):
    state_dict_info = _get_actor_weights_info_from_state_dict(actor_worker)
    if state_dict_info is not None:
        return {"weights_info": state_dict_info, "base_sync_done": True, "peft_config": None}

    params_iter, peft_config = _actor_per_tensor_param(actor_worker, base_sync_done=True)
    payload = {
        "weights_info": _weights_info_from_params(params_iter),
        "base_sync_done": True,
        "peft_config": peft_config,
    }
    if peft_config is not None and not _rollout_base_sync_done(actor_worker):
        base_params_iter, base_peft_config = _actor_per_tensor_param(actor_worker, base_sync_done=False)
        payload.update(
            {
                "base_weights_info": _weights_info_from_params(base_params_iter),
                "base_sync_done": False,
                "needs_lora_base_sync": True,
                "peft_config": base_peft_config or peft_config,
            }
        )
    return payload


def _apply_actor_weights_info(worker, weights_info):
    if isinstance(weights_info, dict) and "weights_info" in weights_info:
        worker._weights_info = weights_info["weights_info"]
        worker._weight_sync_base_weights_info = weights_info.get("base_weights_info", None)
        worker._weight_sync_peft_config = weights_info.get("peft_config", None)
        worker._weight_sync_needs_lora_base_sync = bool(weights_info.get("needs_lora_base_sync", False))
        if "base_sync_done" in weights_info:
            worker.base_sync_done = bool(weights_info["base_sync_done"])
    else:
        worker._weights_info = weights_info
        worker._weight_sync_base_weights_info = None
        worker._weight_sync_peft_config = None
        worker._weight_sync_needs_lora_base_sync = False

    rollout = getattr(worker, "rollout", None)
    if rollout is not None and getattr(worker, "_weight_sync_peft_config", None) is not None:
        rollout.sleep_level = 1


def _weight_sync_phases(worker):
    peft_config = getattr(worker, "_weight_sync_peft_config", None)
    base_info = getattr(worker, "_weight_sync_base_weights_info", None)
    needs_base = (
        peft_config is not None
        and base_info is not None
        and (getattr(worker, "_weight_sync_needs_lora_base_sync", False) or not _rollout_base_sync_done(worker))
    )
    if needs_base:
        return [
            ("base", base_info, False),
            ("adapter", worker._weights_info, True),
        ]
    return [("weights", worker._weights_info, True)]


def _weight_update_kwargs(worker, base_sync_done: bool):
    peft_config = getattr(worker, "_weight_sync_peft_config", None)
    if peft_config is None:
        return {}
    return {"peft_config": peft_config, "base_sync_done": base_sync_done}


def _rollout_can_update_weights(worker) -> bool:
    return getattr(worker, "_is_rollout", False) and hasattr(getattr(worker, "rollout", None), "update_weights")


def _sync_rollout_weights_impl(worker):
    assert (worker._is_actor or worker._is_rollout) and not worker.config.hybrid_engine
    assert hasattr(worker, "_weights_info") and worker._weights_info is not None

    rollout_name = worker.config.rollout.name
    inference_model = None
    use_rollout_update = False
    if worker._is_rollout:
        if rollout_name == "vllm":
            from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader

            inference_model = _get_vllm_inference_model(worker.rollout)
            if inference_model is not None:
                patch_vllm_moe_model_weight_loader(inference_model)
            elif _rollout_can_update_weights(worker):
                use_rollout_update = True
            else:
                raise AttributeError(f"Unsupported vllm rollout object for weight sync: {type(worker.rollout)}")
        elif rollout_name == "sglang":
            if _rollout_can_update_weights(worker):
                use_rollout_update = True
            else:
                inference_model = getattr(worker.rollout, "_engine", None)
                if inference_model is None:
                    raise AttributeError(f"Unsupported sglang rollout object for weight sync: {type(worker.rollout)}")
        else:
            raise NotImplementedError(f"Unknown rollout name: {rollout_name}")

    group_name = getattr(worker, "_weight_sync_group_name", "actor_rollout")
    sync_mode = str(getattr(worker, "_weight_sync_mode", os.environ.get("STAR_WEIGHT_SYNC_MODE", "collective"))).lower()
    phases = _weight_sync_phases(worker)

    def _finish_actor_side(mark_base_done: bool = True):
        if worker._is_actor:
            _offload_actor_params_after_weight_sync(worker)
        if mark_base_done and getattr(worker, "_weight_sync_peft_config", None) is not None:
            _set_rollout_base_sync_done(worker, True)
        get_torch_device().empty_cache()

    if sync_mode == "local_pair":
        channel = _get_local_pair_channel(group_name)
        if worker._is_actor:
            def _produce_local_pair_weights():
                success = False
                try:
                    for _, weights_info, base_sync_done in phases:
                        params_iter = worker._iter_actor_params_for_sync(base_sync_done)
                        for key, _, dtype in weights_info:
                            tensor = worker._next_actor_param(params_iter, key, dtype)
                            channel.put((key, tensor))
                    success = True
                except BaseException as exc:
                    channel.put((_LOCAL_PAIR_ERROR, repr(exc)))
                finally:
                    channel.put((_LOCAL_PAIR_END, None))
                    _finish_actor_side(mark_base_done=success)

            threading.Thread(target=_produce_local_pair_weights, daemon=True).start()
            return

        def _iter_local_pair_weights(weights_info):
            for expected_key, _, _ in weights_info:
                recv_key, tensor = channel.get()
                if recv_key == _LOCAL_PAIR_ERROR:
                    raise RuntimeError(f"local_pair actor producer failed: {tensor}")
                if recv_key == _LOCAL_PAIR_END:
                    raise RuntimeError(f"local_pair weight stream ended before {expected_key}")
                if recv_key != expected_key:
                    raise RuntimeError(f"local_pair weight order mismatch: got {recv_key}, expected {expected_key}")
                yield expected_key, tensor

        for _, weights_info, base_sync_done in phases:
            if use_rollout_update:
                kwargs = _weight_update_kwargs(worker, base_sync_done)
                _run_coro_blocking(
                    lambda weights_info=weights_info, kwargs=kwargs: worker.rollout.update_weights(
                        _iter_local_pair_weights(weights_info), **kwargs
                    )
                )
            else:
                for expected_key, tensor in _iter_local_pair_weights(weights_info):
                    if rollout_name == "vllm":
                        inference_model.load_weights([(expected_key, tensor)])
                    elif rollout_name == "sglang":
                        raise AttributeError(
                            "SGLang rollout does not expose update_weights(); cannot sync STAR weights safely"
                        )
        end_key, value = channel.get()
        if end_key == _LOCAL_PAIR_ERROR:
            raise RuntimeError(f"local_pair actor producer failed after all expected weights: {value}")
        if end_key != _LOCAL_PAIR_END:
            raise RuntimeError(f"local_pair weight stream missing end sentinel, got {end_key}")
        if getattr(worker, "_weight_sync_peft_config", None) is not None:
            _set_rollout_base_sync_done(worker, True)
        get_torch_device().empty_cache()
        return

    if worker._is_rollout and use_rollout_update:
        for _, weights_info, base_sync_done in phases:
            def _iter_collective_weights(weights_info=weights_info):
                for key, shape, dtype in weights_info:
                    tensor = torch.empty(shape, dtype=dtype, device=get_torch_device().current_device())
                    if hasattr(worker, "_weight_sync_group") and worker._weight_sync_group is not None:
                        worker._weight_sync_group.broadcast(
                            tensor, src=0, stream=get_torch_device().current_stream()
                        )
                    else:
                        collective.broadcast(tensor, src_rank=0, group_name=group_name)
                    yield key, tensor

            kwargs = _weight_update_kwargs(worker, base_sync_done)
            _run_coro_blocking(
                lambda kwargs=kwargs: worker.rollout.update_weights(_iter_collective_weights(), **kwargs)
            )
        if getattr(worker, "_weight_sync_peft_config", None) is not None:
            _set_rollout_base_sync_done(worker, True)
        get_torch_device().empty_cache()
        return

    for _, weights_info, base_sync_done in phases:
        params_iter = worker._iter_actor_params_for_sync(base_sync_done) if worker._is_actor else None
        for key, shape, dtype in weights_info:
            if worker._is_actor:
                tensor = worker._next_actor_param(params_iter, key, dtype)
            else:
                tensor = torch.empty(shape, dtype=dtype, device=get_torch_device().current_device())

            if hasattr(worker, "_weight_sync_group") and worker._weight_sync_group is not None:
                worker._weight_sync_group.broadcast(tensor, src=0, stream=get_torch_device().current_stream())
            else:
                collective.broadcast(tensor, src_rank=0, group_name=group_name)

            if worker._is_rollout:
                if rollout_name == "vllm":
                    inference_model.load_weights([(key, tensor)])
                elif rollout_name == "sglang":
                    raise AttributeError(
                        "SGLang rollout does not expose update_weights(); cannot sync STAR weights safely"
                    )

    if getattr(worker, "_weight_sync_peft_config", None) is not None:
        _set_rollout_base_sync_done(worker, True)
    _finish_actor_side()


def _prepare_rollout_memory_for_weight_sync(worker):
    if not getattr(worker, "_is_rollout", False):
        return
    rollout = getattr(worker, "rollout", None)
    if rollout is None or not getattr(worker.config.rollout, "free_cache_engine", False):
        return
    if hasattr(rollout, "release"):
        _run_coro_blocking(lambda: rollout.release())
    if hasattr(rollout, "resume"):
        _run_coro_blocking(lambda: rollout.resume(tags=["weights"]))


def _restore_rollout_memory_after_weight_sync(worker):
    if not getattr(worker, "_is_rollout", False):
        return
    rollout = getattr(worker, "rollout", None)
    if rollout is None or not getattr(worker.config.rollout, "free_cache_engine", False):
        return
    if hasattr(rollout, "resume"):
        _run_coro_blocking(lambda: rollout.resume(tags=["kv_cache"]))


DetachAsyncRolloutWorker = DetachActorWorker


class CriticWorker(TrainingWorker):
    """Compatibility wrapper exposing the old STAR critic worker surface."""

    def __init__(self, config):
        self.critic_config = config
        engine_config = config.engine
        engine_config.use_dynamic_bsz = config.use_dynamic_bsz
        engine_config.infer_max_token_len_per_gpu = _first_not_none(
            config.ppo_infer_max_token_len_per_gpu,
            getattr(config, "forward_max_token_len_per_gpu", None),
            config.ppo_max_token_len_per_gpu,
        )
        engine_config.infer_micro_batch_size_per_gpu = _first_not_none(
            config.ppo_infer_micro_batch_size_per_gpu,
            getattr(config, "forward_micro_batch_size_per_gpu", None),
            getattr(config, "forward_micro_batch_size", None),
            config.ppo_micro_batch_size_per_gpu,
            config.ppo_micro_batch_size,
            1,
        )
        engine_config.max_token_len_per_gpu = config.ppo_max_token_len_per_gpu
        engine_config.micro_batch_size_per_gpu = _first_not_none(
            config.ppo_micro_batch_size_per_gpu,
            config.ppo_micro_batch_size,
            1,
        )
        worker_config = TrainingWorkerConfig(
            model_type="value_model",
            model_config=config.model,
            engine_config=engine_config,
            optimizer_config=config.optim,
            checkpoint_config=config.checkpoint,
            profiler_config=getattr(config, "profiler", None),
        )
        super().__init__(config=worker_config)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        result = self.reset()
        from verl.workers.utils.losses import value_loss

        self.set_loss_fn(partial(value_loss, config=self.critic_config))
        return result

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="train"), blocking=False)
    def compute_values(self, data):
        return self.infer_batch(data)

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="train"), blocking=False)
    def update_critic(self, data):
        import verl.utils.tensordict_utils as tu

        # Newer PPO call sites pass these train_mini_batch controls explicitly,
        # e.g. scaling mini-batch sizes by rollout.n. Preserve those values and
        # only provide defaults for legacy STAR callers.
        defaults = {}
        if tu.get(data, "global_batch_size") is None:
            defaults["global_batch_size"] = self.critic_config.ppo_mini_batch_size
        if tu.get(data, "mini_batch_size") is None:
            defaults["mini_batch_size"] = self.critic_config.ppo_mini_batch_size
        if tu.get(data, "epochs") is None:
            defaults["epochs"] = self.critic_config.ppo_epochs
        if tu.get(data, "seed") is None:
            defaults["seed"] = self.critic_config.data_loader_seed
        if tu.get(data, "dataloader_kwargs") is None:
            defaults["dataloader_kwargs"] = {"shuffle": self.critic_config.shuffle}
        if defaults:
            tu.assign_non_tensor(data, **defaults)
        return self.train_mini_batch(data)


class StarDetachActorWorker(DetachActorWorker):
    """Actor worker alias for star PPO."""

    def __init__(self, config, role: str):
        super().__init__(config=config, role=role)
        self._weight_sync_group_name = "actor_rollout"
        self._weight_sync_mode = "collective"

    def _get_actor_params(self):
        assert self._is_actor
        per_tensor_param, _ = _actor_per_tensor_param(self, base_sync_done=True)
        return dict(per_tensor_param)

    def _iter_actor_params(self):
        assert self._is_actor
        per_tensor_param, _ = _actor_per_tensor_param(self, base_sync_done=True)
        return iter(per_tensor_param)

    def _iter_actor_params_for_sync(self, base_sync_done: bool):
        assert self._is_actor
        per_tensor_param, _ = _actor_per_tensor_param(self, base_sync_done=base_sync_done)
        return iter(per_tensor_param)

    def _next_actor_param(self, params_iter, expected_key, expected_dtype):
        try:
            key, tensor = next(params_iter)
        except StopIteration as exc:
            raise RuntimeError(f"actor weight stream ended before {expected_key}") from exc
        if key != expected_key:
            raise RuntimeError(f"actor weight stream order mismatch: got {key}, expected {expected_key}")
        if tensor.dtype != expected_dtype:
            tensor = tensor.to(expected_dtype, non_blocking=True)
        return tensor

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_actor_weights_info(self):
        assert self._is_actor
        if hasattr(self, "_weights_info"):
            payload = {
                "weights_info": self._weights_info,
                "base_weights_info": getattr(self, "_weight_sync_base_weights_info", None),
                "peft_config": getattr(self, "_weight_sync_peft_config", None),
                "needs_lora_base_sync": getattr(self, "_weight_sync_needs_lora_base_sync", False),
                "base_sync_done": _rollout_base_sync_done(self),
            }
            return payload
        payload = _build_actor_weights_info_payload(self)
        _apply_actor_weights_info(self, payload)
        return payload

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_actor_weights_info(self, weights_info):
        _apply_actor_weights_info(self, weights_info)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_weight_sync_group_name(self, group_name: str):
        self._weight_sync_group_name = str(group_name)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_weight_sync_mode(self, mode: str):
        self._weight_sync_mode = str(mode).strip().lower()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def prepare_rollout_weight_sync(self):
        _prepare_rollout_memory_for_weight_sync(self)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def finish_rollout_weight_sync(self):
        _restore_rollout_memory_after_weight_sync(self)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def sync_rollout_weights(self):
        _sync_rollout_weights_impl(self)


class StarDetachAsyncRolloutWorker(DetachAsyncRolloutWorker):
    """Rollout worker with local fat-data buffer and thin-data return path."""

    def __init__(self, config, role: str):
        super().__init__(config=config, role=role)
        buffer_cfg = config.get("star_buffer", {})
        max_items = int(buffer_cfg.get("max_items", 100000))
        ttl_seconds = int(buffer_cfg.get("ttl_seconds", 7200))
        dropped_query_ttl_seconds = int(buffer_cfg.get("dropped_query_ttl_seconds", 120))
        shuffle_ready = buffer_cfg.get("shuffle_ready", True)
        self._shuffle_ready_buffer = (
            shuffle_ready.strip().lower() in {"1", "true", "yes", "on"}
            if isinstance(shuffle_ready, str)
            else bool(shuffle_ready)
        )
        self._traj_buffer = TrajectoryBuffer(
            max_items=max_items,
            ttl_seconds=ttl_seconds,
            dropped_query_ttl_seconds=dropped_query_ttl_seconds,
        )
        self._weight_sync_group_name = "actor_rollout"
        self._weight_sync_mode = "collective"

    def _register_star_rollout_dispatch_info(self):
        rollout = getattr(self, "rollout", None)
        if rollout is not None and hasattr(rollout, "replica_rank") and hasattr(rollout, "rollout_rank"):
            dp_rank = int(rollout.replica_rank)
            is_collect = int(rollout.rollout_rank) == 0
        else:
            rollout_cfg = self.config.rollout
            infer_tp = int(rollout_cfg.tensor_model_parallel_size) * int(rollout_cfg.data_parallel_size)
            infer_pp = int(rollout_cfg.pipeline_model_parallel_size)
            infer_world_size = infer_tp * infer_pp
            dp_rank = int(torch.distributed.get_rank()) // infer_world_size
            is_collect = int(torch.distributed.get_rank()) % infer_world_size == 0

        try:
            self._register_dispatch_collect_info("rollout", dp_rank=dp_rank, is_collect=is_collect)
        except ValueError as exc:
            if "has been registered" not in str(exc):
                raise

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        result = super().init_model()
        if getattr(self, "_is_rollout", False):
            self._register_star_rollout_dispatch_info()
        return result

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_actor_weights_info(self, weights_info):
        _apply_actor_weights_info(self, weights_info)

    def _decode_action_text(self, response_tokens: torch.Tensor) -> str:
        if response_tokens is None:
            return ""
        tokens = response_tokens.detach().cpu().tolist()
        try:
            return self.tokenizer.decode(tokens, skip_special_tokens=True)
        except Exception:
            return ""

    @staticmethod
    def _strip_concat_volatile_meta(data: DataProto) -> DataProto:
        meta_info = dict(data.meta_info or {})
        meta_info.pop("timing", None)
        meta_info.pop("metrics", None)
        return DataProto(batch=data.batch, non_tensor_batch=data.non_tensor_batch, meta_info=meta_info)

    @staticmethod
    def _meta_values_equal(left: Any, right: Any) -> bool:
        if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
            if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
                return False
            return bool(torch.equal(left.detach().cpu(), right.detach().cpu()))
        if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
            if not isinstance(left, np.ndarray) or not isinstance(right, np.ndarray):
                return False
            return bool(np.array_equal(left, right))
        try:
            return bool(left == right)
        except Exception:
            return False

    @classmethod
    def _align_concat_meta(cls, parts: list[DataProto]) -> list[DataProto]:
        if len(parts) <= 1:
            return parts
        common_meta = dict(parts[0].meta_info or {})
        for part in parts[1:]:
            meta_info = dict(part.meta_info or {})
            for key in list(common_meta.keys()):
                if key not in meta_info or not cls._meta_values_equal(common_meta[key], meta_info[key]):
                    common_meta.pop(key, None)
        return [
            DataProto(batch=part.batch, non_tensor_batch=part.non_tensor_batch, meta_info=dict(common_meta))
            for part in parts
        ]

    @classmethod
    def _concat_data_proto_safe(cls, parts: list[DataProto]) -> DataProto:
        cleaned = [cls._strip_concat_volatile_meta(part) for part in parts]
        if len(cleaned) <= 1:
            return cleaned[0]
        try:
            return DataProto.concat(cleaned)
        except AssertionError:
            return DataProto.concat(cls._align_concat_meta(cleaned))

    def _extract_inner_rollout_timing(self, full_batch: DataProto) -> dict[str, float]:
        timing: dict[str, float] = {}
        meta_info = full_batch.meta_info or {}

        inner_timing = meta_info.get("timing", None)
        if isinstance(inner_timing, dict):
            generate_s = inner_timing.get("generate_sequences", None)
            if isinstance(generate_s, int | float | np.integer | np.floating):
                timing["engine_generate_s"] = float(generate_s)
            timing_aliases = {
                "agent_loop/generate_sequences/mean": "engine_generate_s",
                "agent_loop/generate_sequences/max": "engine_generate_max_s",
                "agent_loop/tool_calls/mean": "agent_loop_tool_calls_s",
                "agent_loop/tool_calls/max": "agent_loop_tool_calls_max_s",
                "agent_loop/server_rpc_roundtrip/mean": "agent_server_rpc_roundtrip_s",
                "agent_loop/server_rpc_roundtrip/max": "agent_server_rpc_roundtrip_max_s",
                "agent_loop/server_total/mean": "agent_server_total_s",
                "agent_loop/server_total/max": "agent_server_total_max_s",
                "agent_loop/server_rpc_overhead/mean": "agent_server_rpc_overhead_s",
                "agent_loop/server_rpc_overhead/max": "agent_server_rpc_overhead_max_s",
                "agent_loop/server_first_token/mean": "agent_server_first_token_s",
                "agent_loop/server_first_token/max": "agent_server_first_token_max_s",
                "agent_loop/server_decode_tail/mean": "agent_server_decode_tail_s",
                "agent_loop/server_decode_tail/max": "agent_server_decode_tail_max_s",
                "agent_loop/worker/start_lag/mean": "agent_worker_start_lag_s",
                "agent_loop/worker/start_lag/max": "agent_worker_start_lag_max_s",
                "agent_loop/worker/prep/mean": "agent_worker_prep_s",
                "agent_loop/worker/prep/max": "agent_worker_prep_max_s",
                "agent_loop/worker/run_loops/mean": "agent_worker_run_loops_s",
                "agent_loop/worker/run_loops/max": "agent_worker_run_loops_max_s",
                "agent_loop/worker/postprocess/mean": "agent_worker_postprocess_s",
                "agent_loop/worker/postprocess/max": "agent_worker_postprocess_max_s",
                "agent_loop/worker/total/mean": "agent_worker_total_s",
                "agent_loop/worker/total/max": "agent_worker_total_max_s",
                "agent_loop/worker/non_loop_overhead/mean": "agent_worker_non_loop_overhead_s",
                "agent_loop/worker/non_loop_overhead/max": "agent_worker_non_loop_overhead_max_s",
                "agent_loop/manager/prep": "agent_loop_manager_prep_s",
                "agent_loop/manager/worker_rpc_wait": "agent_loop_manager_worker_rpc_wait_s",
                "agent_loop/manager/worker_rpc_mean": "agent_loop_manager_worker_rpc_mean_s",
                "agent_loop/manager/worker_rpc_max": "agent_loop_manager_worker_rpc_max_s",
                "agent_loop/manager/concat": "agent_loop_manager_concat_s",
                "agent_loop/manager/metrics_reduce": "agent_loop_manager_metrics_reduce_s",
                "agent_loop/manager/total": "agent_loop_manager_total_s",
                "agent_loop/manager/overhead": "agent_loop_manager_overhead_s",
            }
            for src_key, dst_key in timing_aliases.items():
                value = inner_timing.get(src_key, None)
                if isinstance(value, int | float | np.integer | np.floating):
                    timing[dst_key] = float(value)

        metric_list = meta_info.get("metrics", None)
        if isinstance(metric_list, list):
            metric_acc: dict[str, list[float]] = {}
            for item in metric_list:
                if not isinstance(item, dict):
                    continue
                for key, value in item.items():
                    if not isinstance(value, int | float | np.integer | np.floating):
                        continue
                    metric_acc.setdefault(str(key), []).append(float(value))
            if metric_acc.get("generate_sequences") and "engine_generate_s" not in timing:
                timing["engine_generate_s"] = float(np.mean(metric_acc["generate_sequences"]))
            if metric_acc.get("tool_calls"):
                timing["agent_loop_tool_calls_s"] = float(np.mean(metric_acc["tool_calls"]))

        for key, value in full_batch.non_tensor_batch.items():
            if not str(key).startswith("__star_timing_"):
                continue
            if not isinstance(value, np.ndarray) or value.size == 0:
                continue
            flat = value.reshape(-1)
            if np.issubdtype(flat.dtype, np.number):
                timing[str(key).replace("__star_timing_", "", 1)] = float(np.mean(flat.astype(np.float64)))

        return timing

    def _attach_rollout_timing(self, batch: DataProto, timing: dict[str, Any]) -> None:
        merged: dict[str, float] = {}
        for key, value in batch.non_tensor_batch.items():
            if not str(key).startswith("__star_timing_"):
                continue
            if not isinstance(value, np.ndarray) or value.size == 0:
                continue
            flat = value.reshape(-1)
            if np.issubdtype(flat.dtype, np.number):
                merged[str(key).replace("__star_timing_", "", 1)] = float(np.mean(flat.astype(np.float64)))
        for key, value in timing.items():
            if isinstance(value, int | float | np.integer | np.floating):
                merged[str(key)] = float(value)
        bsz = len(batch)
        for key, value in merged.items():
            batch.non_tensor_batch[f"__star_timing_{key}"] = np.full((bsz,), float(value), dtype=np.float64)

    def _build_thin_from_batch(self, full_batch: DataProto) -> DataProto:
        build_start = time.perf_counter()
        bsz = len(full_batch)
        query_ids = full_batch.non_tensor_batch.get("query_id", np.array(["unknown"] * bsz, dtype=object))
        agent_ids = full_batch.non_tensor_batch.get("agent_id", np.array(["agent_0"] * bsz, dtype=object))
        keep_mask = full_batch.non_tensor_batch.get("__star_keep_in_buffer__", None)
        if keep_mask is None:
            keep_mask = np.ones((bsz,), dtype=bool)
        else:
            keep_mask = np.array(keep_mask, dtype=bool).reshape(-1)
            if keep_mask.shape[0] != bsz:
                keep_mask = np.ones((bsz,), dtype=bool)
        model_id = str(self.config.get("model_id", "unknown_model"))

        traj_ids = np.empty((bsz,), dtype=object)
        model_ids = np.empty((bsz,), dtype=object)
        action_text = np.empty((bsz,), dtype=object)
        created_ts = np.empty((bsz,), dtype=np.float64)

        responses = full_batch.batch.get("responses", None)
        now = time.time()
        decode_action_text_s = 0.0
        buffer_put_s = 0.0
        for i in range(bsz):
            traj_id = uuid.uuid4().hex if bool(keep_mask[i]) else ""
            traj_ids[i] = traj_id
            model_ids[i] = model_id
            created_ts[i] = now

            response_tokens = responses[i] if responses is not None else None
            decode_start = time.perf_counter()
            action_text[i] = self._decode_action_text(response_tokens)
            decode_action_text_s += float(time.perf_counter() - decode_start)

            if bool(keep_mask[i]):
                fat_item = full_batch[i : i + 1]
                put_start = time.perf_counter()
                self._traj_buffer.put(
                    TrajectoryEntry(
                        traj_id=traj_id,
                        model_id=model_id,
                        query_id=str(query_ids[i]),
                        agent_id=str(agent_ids[i]),
                        fat_data=fat_item,
                    )
                )
                buffer_put_s += float(time.perf_counter() - put_start)

        thin = DataProto.from_dict(
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
        thin_build_s = float(time.perf_counter() - build_start)
        inherited_timing = self._extract_inner_rollout_timing(full_batch)
        inherited_timing.update(
            {
                "worker_thin_build_s": thin_build_s,
                "worker_decode_action_text_s": float(decode_action_text_s),
                "worker_buffer_put_s": float(buffer_put_s),
                "worker_build_overhead_s": float(max(thin_build_s - decode_action_text_s - buffer_put_s, 0.0)),
            }
        )
        self._attach_rollout_timing(thin, inherited_timing)
        return thin

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="rollout"))
    def generate_sequences_thin(self, prompts: DataProto) -> DataProto:
        worker_start = time.perf_counter()
        generate_start = time.perf_counter()
        fat_output = self.generate_sequences(prompts)
        worker_generate_call_s = float(time.perf_counter() - generate_start)
        # Avoid DataProto.union() conflicts on object-typed fields (e.g. raw_prompt).
        full_batch = fat_output
        for key in ("query_id", "agent_id"):
            if key not in full_batch.non_tensor_batch and key in prompts.non_tensor_batch:
                full_batch.non_tensor_batch[key] = prompts.non_tensor_batch[key]
        thin = self._build_thin_from_batch(full_batch)
        timing = self._extract_inner_rollout_timing(full_batch)
        timing.update(
            {
                "worker_generate_call_s": worker_generate_call_s,
                "worker_total_s": float(time.perf_counter() - worker_start),
            }
        )
        engine_generate_s = timing.get("engine_generate_s", None)
        if isinstance(engine_generate_s, int | float | np.integer | np.floating):
            timing["worker_generate_overhead_s"] = float(max(worker_generate_call_s - float(engine_generate_s), 0.0))
        self._attach_rollout_timing(thin, timing)
        return thin

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="rollout"))
    def build_thin_from_generated(self, full_batch: DataProto) -> DataProto:
        worker_start = time.perf_counter()
        thin = self._build_thin_from_batch(full_batch)
        self._attach_rollout_timing(
            thin,
            {
                "worker_total_s": float(time.perf_counter() - worker_start),
            },
        )
        return thin

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

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def drop_queries(self, query_ids: list[str]) -> dict:
        if not isinstance(query_ids, list | tuple) or len(query_ids) == 0:
            return {"star/dropped_queries": 0, "star/purged_traj": 0, **self._traj_buffer.stats()}
        normalized = [str(q).strip() for q in query_ids if str(q).strip()]
        if len(normalized) == 0:
            return {"star/dropped_queries": 0, "star/purged_traj": 0, **self._traj_buffer.stats()}
        purge_stats = self._traj_buffer.drop_queries(normalized)
        return {
            "star/dropped_queries": purge_stats.get("dropped_queries", len(set(normalized))),
            "star/purged_traj": purge_stats.get("purged_traj", 0),
            **self._traj_buffer.stats(),
        }

    def _empty_batch(self) -> DataProto:
        return DataProto.from_dict(non_tensors={"traj_id": np.array([], dtype=object)})

    @staticmethod
    def _pad_fill_value_for_key(key: str, tensor: torch.Tensor):
        key_lower = str(key).lower()
        if tensor.dtype == torch.bool:
            return False
        if "label" in key_lower:
            return -100
        if tensor.is_floating_point():
            return 0.0
        return 0

    @classmethod
    def _pad_tensor_to_shape(cls, key: str, tensor: torch.Tensor, target_shape: tuple[int, ...]) -> torch.Tensor:
        fill_value = cls._pad_fill_value_for_key(key, tensor)
        padded = torch.full(target_shape, fill_value=fill_value, dtype=tensor.dtype, device=tensor.device)
        copy_slices = tuple(slice(0, min(src, dst)) for src, dst in zip(tensor.shape, target_shape))
        padded[copy_slices] = tensor[copy_slices]
        return padded

    def _align_fat_batch_shapes_for_concat(self, fat_list: list[DataProto]) -> list[DataProto]:
        if len(fat_list) <= 1:
            return fat_list

        target_shapes: dict[str, tuple[int, ...]] = {}
        prototypes: dict[str, torch.Tensor] = {}
        for fat in fat_list:
            if fat.batch is None:
                continue
            for key in fat.batch.keys():
                tensor = fat.batch[key]
                if not isinstance(tensor, torch.Tensor):
                    continue
                key = str(key)
                shape = tuple(int(x) for x in tensor.shape)
                if key not in target_shapes:
                    target_shapes[key] = shape
                    prototypes[key] = tensor
                    continue
                prev = target_shapes[key]
                if len(prev) != len(shape):
                    raise RuntimeError(
                        f"Inconsistent tensor rank for key={key}: shape={shape} vs prev_shape={prev}"
                    )
                target_shapes[key] = tuple(max(a, b) for a, b in zip(prev, shape))

        if not target_shapes:
            return fat_list

        aligned: list[DataProto] = []
        for fat in fat_list:
            if fat.batch is None:
                aligned.append(fat)
                continue

            tensors: dict[str, torch.Tensor] = {}
            changed = False
            bsz = int(fat.batch.batch_size[0])

            for key, target_shape in target_shapes.items():
                if key in fat.batch.keys():
                    tensor = fat.batch[key]
                    if not isinstance(tensor, torch.Tensor):
                        continue
                else:
                    proto = prototypes[key]
                    cur_shape = list(target_shape)
                    cur_shape[0] = bsz
                    tensors[key] = torch.full(
                        tuple(cur_shape),
                        fill_value=self._pad_fill_value_for_key(key, proto),
                        dtype=proto.dtype,
                        device=proto.device,
                    )
                    changed = True
                    continue

                cur_target = list(target_shape)
                cur_target[0] = int(tensor.shape[0])
                cur_target_t = tuple(cur_target)
                if tuple(int(x) for x in tensor.shape) != cur_target_t:
                    tensors[key] = self._pad_tensor_to_shape(key, tensor, cur_target_t)
                    changed = True
                else:
                    tensors[key] = tensor

            if changed:
                aligned.append(
                    DataProto.from_dict(
                        tensors=tensors,
                        non_tensors=fat.non_tensor_batch,
                        meta_info=self._strip_concat_volatile_meta(fat).meta_info,
                    )
                )
            else:
                aligned.append(self._strip_concat_volatile_meta(fat))

        return aligned

    @staticmethod
    def _summarize_fat_shapes(fat_list: list[DataProto], max_items: int = 8) -> str:
        items = []
        for i, fat in enumerate(fat_list[:max_items]):
            if fat.batch is None:
                items.append(f"{i}:<none>")
                continue
            key_shapes = []
            for key in sorted(fat.batch.keys()):
                tensor = fat.batch[key]
                if isinstance(tensor, torch.Tensor):
                    key_shapes.append(f"{key}:{tuple(int(x) for x in tensor.shape)}")
            items.append(f"{i}:{{{', '.join(key_shapes[:6])}}}")
        suffix = f", ...+{len(fat_list) - max_items}" if len(fat_list) > max_items else ""
        return "[" + "; ".join(items) + suffix + "]"

    @staticmethod
    def _ready_loss_mask_tensor(
        batch: DataProto, response_mask: Optional[torch.Tensor], responses: torch.Tensor
    ) -> Optional[torch.Tensor]:
        if batch.batch is not None and "loss_mask" in batch.batch.keys():
            if response_mask is not None and tuple(batch.batch["loss_mask"].shape) != tuple(response_mask.shape):
                loss_mask = batch.batch["loss_mask"]
                if loss_mask.dim() == response_mask.dim() and loss_mask.shape[0] == response_mask.shape[0]:
                    batch.batch["loss_mask"] = loss_mask[..., -response_mask.shape[-1] :]
                else:
                    raise RuntimeError(
                        "STAR ready batch has incompatible response/loss mask ranks: "
                        f"response_mask={tuple(response_mask.shape)} "
                        f"loss_mask={tuple(loss_mask.shape)}"
                    )
            return None
        if response_mask is not None:
            return response_mask.clone()
        return torch.ones_like(responses, dtype=torch.bool)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def build_ready_train_batch(self, max_items: int = 0) -> DataProto:
        entries = self._traj_buffer.pop_ready(
            max_items=max_items if max_items and max_items > 0 else None,
            shuffle=self._shuffle_ready_buffer,
        )
        if len(entries) == 0:
            return self._empty_batch()

        fat_list = []
        for e in entries:
            # Each trajectory may carry per-call timing/metrics in meta_info.
            # Those fields are expected to differ and would make DataProto.concat
            # raise on conflicting meta values, so sanitize before concatenation.
            fat = e.fat_data
            if fat.meta_info is None:
                fat.meta_info = {}
            else:
                fat.meta_info.pop("timing", None)
                fat.meta_info.pop("metrics", None)
            fat_list.append(fat)
        fat_list = self._align_fat_batch_shapes_for_concat(fat_list)
        try:
            batch = self._concat_data_proto_safe(fat_list)
        except RuntimeError as exc:
            shape_summary = self._summarize_fat_shapes(fat_list)
            raise RuntimeError(f"Failed to concat ready fat batches. shapes={shape_summary}") from exc
        # Keep rollout-ready batch meta deterministic for downstream consumers.
        batch.meta_info = {}

        response_mask = batch.batch.get("response_mask", None)
        responses = batch.batch.get("responses", None)
        if responses is None:
            return batch

        bsz, resp_len = responses.shape[0], responses.shape[1]
        token_level_scores = torch.zeros((bsz, resp_len), dtype=torch.float32, device=responses.device)

        reward_scalar = torch.tensor(
            [float(e.reward.item()) if e.reward is not None else 0.0 for e in entries],
            dtype=torch.float32,
            device=responses.device,
        )
        if response_mask is None:
            token_level_scores[:, -1] = reward_scalar
        else:
            last_pos = response_mask.to(device=responses.device, dtype=torch.long).sum(dim=-1) - 1
            last_pos = torch.clamp(last_pos, min=0)
            token_level_scores[torch.arange(bsz, device=responses.device), last_pos] = reward_scalar

        extra_tensors = {
            "token_level_scores": token_level_scores,
            "token_level_rewards": token_level_scores.clone(),
            "reward": reward_scalar,
            "done": torch.tensor([e.done for e in entries], dtype=torch.bool, device=responses.device),
        }
        loss_mask = self._ready_loss_mask_tensor(batch, response_mask, responses)
        if loss_mask is not None:
            extra_tensors["loss_mask"] = loss_mask

        for key, value in extra_tensors.items():
            batch.batch[key] = value
        batch.non_tensor_batch["traj_id"] = np.array([e.traj_id for e in entries], dtype=object)
        batch.non_tensor_batch["query_id"] = np.array([e.query_id for e in entries], dtype=object)
        batch.non_tensor_batch["agent_id"] = np.array([e.agent_id for e in entries], dtype=object)
        batch.non_tensor_batch["model_id"] = np.array([e.model_id for e in entries], dtype=object)
        return batch

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def train_from_local_batch(self, data: DataProto, do_actor: bool = True, do_critic: bool = True) -> dict:
        bsz = len(data)
        if bsz == 0:
            return {"star/consumed": 0}
        raise NotImplementedError(
            "train_from_local_batch is a legacy STAR rollout-worker API and no longer performs PPO updates. "
            "Use StarRayTrainer._global_sync_and_update(), which builds ready batches on the driver and calls "
            "actor/critic worker update APIs."
        )

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
            return {"star/consumed": 0, "star/dropped": 0}

        dropped = 0
        if drop_last and world_size_divisor > 1:
            keep = (bsz // world_size_divisor) * world_size_divisor
            if keep <= 0:
                return {"star/consumed": 0, "star/dropped": bsz}
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

    def _iter_actor_params(self):
        assert self._is_actor
        per_tensor_param, _ = _actor_per_tensor_param(self, base_sync_done=True)
        return iter(per_tensor_param)

    def _iter_actor_params_for_sync(self, base_sync_done: bool):
        assert self._is_actor
        per_tensor_param, _ = _actor_per_tensor_param(self, base_sync_done=base_sync_done)
        return iter(per_tensor_param)

    def _next_actor_param(self, params_iter, expected_key, expected_dtype):
        try:
            key, tensor = next(params_iter)
        except StopIteration as exc:
            raise RuntimeError(f"actor weight stream ended before {expected_key}") from exc
        if key != expected_key:
            raise RuntimeError(f"actor weight stream order mismatch: got {key}, expected {expected_key}")
        if tensor.dtype != expected_dtype:
            tensor = tensor.to(expected_dtype, non_blocking=True)
        return tensor

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def prepare_rollout_weight_sync(self):
        _prepare_rollout_memory_for_weight_sync(self)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def finish_rollout_weight_sync(self):
        _restore_rollout_memory_after_weight_sync(self)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def sync_rollout_weights(self):
        _sync_rollout_weights_impl(self)
