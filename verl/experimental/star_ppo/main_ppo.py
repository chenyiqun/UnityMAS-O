"""Entry point for star-topology multi-engine PPO skeleton."""

import asyncio
import os
import socket

import hydra
import ray
from omegaconf import OmegaConf

from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.experimental.star_ppo.ray_trainer import StarRayTrainer
from verl.experimental.star_ppo.types import EngineSpec
from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler, run_ppo
from verl.trainer.ppo.reward import load_reward_manager
from verl.trainer.ppo.utils import Role, need_critic, need_reference_policy, need_reward_model
from verl.utils.config import validate_config
from verl.utils.device import auto_set_device


_SUPPORTED_TRAINING_STRATEGIES = {"fsdp", "fsdp2", "megatron"}


def _normalize_training_strategy(strategy: str) -> str:
    return str(strategy or "fsdp").strip().lower()


def _strategy_family(strategy: str) -> str:
    strategy = _normalize_training_strategy(strategy)
    if strategy in {"fsdp", "fsdp2"}:
        return "fsdp"
    if strategy == "megatron":
        return "megatron"
    return strategy


def _assert_supported_strategy(strategy: str, source: str) -> str:
    strategy = _normalize_training_strategy(strategy)
    if strategy not in _SUPPORTED_TRAINING_STRATEGIES:
        raise NotImplementedError(
            f"STAR PPO supports training strategies {sorted(_SUPPORTED_TRAINING_STRATEGIES)}, "
            f"got {strategy!r} from {source}"
        )
    return strategy


def create_engine_specs(config) -> list[EngineSpec]:
    specs = []
    default_strategy = _assert_supported_strategy(
        OmegaConf.select(config, "actor_rollout_ref.actor.strategy") or "fsdp",
        "actor_rollout_ref.actor.strategy",
    )
    for engine in config.trainer.llm_engines:
        strategy = _assert_supported_strategy(
            engine.get("strategy", default_strategy),
            f"trainer.llm_engines.{engine.model_id}.strategy",
        )
        specs.append(
            EngineSpec(
                model_id=str(engine.model_id),
                nnodes=int(engine.nnodes),
                n_gpus_per_node=int(engine.n_gpus_per_node),
                accelerator_type=engine.get("accelerator_type", None),
                strategy=strategy,
            )
        )
    return specs


def create_role_worker_mapping(config):
    strategy = _assert_supported_strategy(
        config.actor_rollout_ref.actor.strategy,
        "actor_rollout_ref.actor.strategy",
    )
    lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
    if lora_rank <= 0:
        lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
    ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None

    if need_critic(config):
        critic_strategy = _assert_supported_strategy(config.critic.strategy, "critic.strategy")
        if _strategy_family(strategy) != _strategy_family(critic_strategy):
            raise ValueError(
                "actor strategy and critic strategy must use the same backend family: "
                f"actor={strategy}, critic={critic_strategy}"
            )

    for engine in config.trainer.get("llm_engines", []):
        engine_strategy = _assert_supported_strategy(
            engine.get("strategy", strategy),
            f"trainer.llm_engines.{engine.model_id}.strategy",
        )
        if _strategy_family(strategy) != _strategy_family(engine_strategy):
            raise ValueError(
                "STAR model-engine specs must match the actor backend family: "
                f"actor={strategy}, engine={engine.model_id}:{engine_strategy}. "
                "For Megatron, launch with STAR_OPTIMIZATION_STRATEGY=megatron "
                "or pass model_engine=megatron explicitly."
            )

    from verl.experimental.star_ppo.star_fsdp_workers import (
        CriticWorker,
        StarDetachActorWorker,
        StarDetachAsyncRolloutWorker,
    )

    role_worker_mapping = {
        Role.Actor: ray.remote(StarDetachActorWorker),
        Role.Rollout: ray.remote(StarDetachAsyncRolloutWorker),
    }

    if need_critic(config):
        role_worker_mapping[Role.Critic] = ray.remote(CriticWorker)

    if need_reward_model(config):
        raise NotImplementedError(
            "STAR PPO does not support config.reward.reward_model.enable=True yet. "
            "Use workflow reward allocators/custom reward logic, or add a STAR-specific RewardLoopManager path."
        )

    if need_reference_policy(config) and not ref_in_actor:
        role_worker_mapping[Role.RefPolicy] = ray.remote(StarDetachActorWorker)

    return role_worker_mapping


@ray.remote(num_cpus=10, max_concurrency=100)
class StarTaskRunner:
    def run(self, config):
        from pprint import pprint

        from verl.utils import hf_processor, hf_tokenizer
        from verl.utils.dataset.rl_dataset import collate_fn
        from verl.utils.fs import copy_to_local

        print(f"StarTaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        role_worker_mapping = create_role_worker_mapping(config)
        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(config),
            use_critic=need_critic(config),
        )

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )
        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        reward_fn = load_reward_manager(config, tokenizer)

        train_dataset = create_rl_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            max_samples=config.data.get("train_max_samples", -1),
        )
        val_dataset = create_rl_dataset(
            config.data.val_files,
            config.data,
            tokenizer,
            processor,
            max_samples=config.data.get("val_max_samples", -1),
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)

        trainer = StarRayTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            engine_specs=create_engine_specs(config),
            role_worker_mapping=role_worker_mapping,
            reward_fn=reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        trainer.init_workers()
        asyncio.run(trainer.fit())


@hydra.main(config_path="config", config_name="star_ppo_trainer", version_base=None)
def main(config):
    from time import time

    start_time = time()
    auto_set_device(config)
    config = migrate_legacy_reward_impl(config)
    run_ppo(config, task_runner_class=StarTaskRunner)
    print(f"total time: {time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
