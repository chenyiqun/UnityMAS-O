import pytest
from omegaconf import OmegaConf

from verl.experimental.star_ppo.main_ppo import create_engine_specs, create_role_worker_mapping
from verl.trainer.ppo.utils import Role


def _minimal_star_config(actor_strategy="fsdp", critic_strategy="fsdp", engine_strategy=None):
    if engine_strategy is None:
        engine_strategy = actor_strategy
    return OmegaConf.create(
        {
            "trainer": {
                "llm_engines": [
                    {
                        "model_id": "planner_llm",
                        "nnodes": 1,
                        "n_gpus_per_node": 8,
                        "accelerator_type": None,
                        "strategy": engine_strategy,
                    }
                ]
            },
            "actor_rollout_ref": {
                "model": {"lora": {"rank": 0}, "lora_adapter_path": None},
                "actor": {"strategy": actor_strategy, "use_kl_loss": False},
            },
            "critic": {"strategy": critic_strategy, "enable": True},
            "algorithm": {"use_kl_in_reward": False},
            "reward": {"reward_model": {"enable": False}},
        }
    )


def test_create_engine_specs_uses_configured_strategy():
    cfg = _minimal_star_config(actor_strategy="megatron", critic_strategy="megatron")

    specs = create_engine_specs(cfg)

    assert len(specs) == 1
    assert specs[0].strategy == "megatron"


def test_fsdp_and_fsdp2_engine_specs_are_same_backend_family():
    cfg = _minimal_star_config(actor_strategy="fsdp", critic_strategy="fsdp", engine_strategy="fsdp2")

    mapping = create_role_worker_mapping(cfg)

    assert Role.Actor in mapping
    assert Role.Rollout in mapping
    assert Role.Critic in mapping


def test_megatron_engine_requires_megatron_actor_backend():
    cfg = _minimal_star_config(actor_strategy="fsdp", critic_strategy="fsdp", engine_strategy="megatron")

    with pytest.raises(ValueError, match="must match the actor backend family"):
        create_role_worker_mapping(cfg)
