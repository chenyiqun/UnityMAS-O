from __future__ import annotations

from abc import ABC, abstractmethod

from verl import DataProto


class WorkflowRunner(ABC):
    """Workflow runner plugin interface.

    A runner encapsulates query-level workflow orchestration and returns reward
    trajectories ready for commit into per-model rollout buffers.
    """

    def __init__(self, trainer, config):
        self.trainer = trainer
        self.config = config

    @staticmethod
    def _rollout_prompt_token_budget(config, workflow_cfg) -> int:
        rollout_cfg = config.actor_rollout_ref.rollout
        prompt_len_cfg = int(rollout_cfg.get("prompt_length", 4096))
        response_len_cfg = max(1, int(rollout_cfg.get("response_length", 1024)))
        configured_default = max(1, prompt_len_cfg)
        requested_budget = int(workflow_cfg.get("per_infer_prompt_max_tokens", configured_default))
        trunc_margin = max(0, int(workflow_cfg.get("prompt_truncation_margin", 128)))

        max_model_len_cfg = rollout_cfg.get("max_model_len", None)
        if max_model_len_cfg is None:
            context_budget = prompt_len_cfg
        else:
            max_model_len_cfg = int(max_model_len_cfg)
            reserved_response = min(response_len_cfg, max(1, max_model_len_cfg - 1))
            context_budget = max(1, max_model_len_cfg - reserved_response - trunc_margin)

        return max(1, min(prompt_len_cfg, requested_budget, context_budget))

    @abstractmethod
    async def run_batch(
        self,
        batch: DataProto,
        epoch: int,
        stage: str = "train",
    ) -> tuple[DataProto, dict[str, float]]:
        raise NotImplementedError
