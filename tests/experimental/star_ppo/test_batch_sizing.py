# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

import pytest

from verl.experimental.star_ppo.ray_trainer import StarRayTrainer
from verl.workers.engine_workers import _temporary_optimizer_lr_scale


class _FakeOptimizer:
    def __init__(self, *lrs: float):
        self.param_groups = [{"lr": lr} for lr in lrs]


class _FakeScheduler:
    def __init__(self, *lrs: float):
        self._lrs = list(lrs)

    def get_last_lr(self):
        return self._lrs


class _FakeEngine:
    def __init__(self, optimizer, scheduler=None):
        self.optimizer = optimizer
        self.lr_scheduler = scheduler


def test_aligns_closed_workflow_batch_to_configured_mini_batch():
    divisor = StarRayTrainer._configured_mini_batch_drop_divisor(
        actor_mini_batch_size=64,
        actor_update_divisor=16,
        critic_mini_batch_size=64,
        critic_update_divisor=16,
        use_critic=True,
    )

    aligned_batch_size = (368 // divisor) * divisor

    assert divisor == 64
    assert aligned_batch_size == 320
    assert StarRayTrainer._effective_global_mini_batch_size(64, aligned_batch_size, 16, 1) == 64


def test_actor_and_critic_alignment_uses_common_multiple():
    assert (
        StarRayTrainer._configured_mini_batch_drop_divisor(
            actor_mini_batch_size=64,
            actor_update_divisor=16,
            critic_mini_batch_size=96,
            critic_update_divisor=16,
            use_critic=True,
        )
        == 192
    )


def test_rejects_configured_mini_batch_incompatible_with_distributed_update():
    with pytest.raises(ValueError, match="Configured actor PPO mini-batch"):
        StarRayTrainer._configured_mini_batch_drop_divisor(
            actor_mini_batch_size=64,
            actor_update_divisor=48,
        )


@pytest.mark.parametrize(
    ("effective_size", "expected_scale"),
    [(16, 0.25), (32, 0.5), (48, 0.75), (64, 1.0)],
)
def test_dynamic_mini_batch_scales_lr_linearly(effective_size, expected_scale):
    assert StarRayTrainer._mini_batch_lr_scale(64, effective_size) == expected_scale


def test_optimizer_lr_scale_is_temporary_and_preserves_scheduler_lr():
    optimizer = _FakeOptimizer(1e-6, 2e-6)
    engine = _FakeEngine(optimizer, _FakeScheduler(9e-7, 1.8e-6))

    with _temporary_optimizer_lr_scale(engine, 0.25):
        assert [group["lr"] for group in optimizer.param_groups] == [2.5e-7, 5e-7]

    assert [group["lr"] for group in optimizer.param_groups] == [9e-7, 1.8e-6]


def test_optimizer_lr_scale_restores_lr_after_failure():
    optimizer = _FakeOptimizer(1e-6)
    engine = _FakeEngine(optimizer, _FakeScheduler(9e-7))

    with pytest.raises(RuntimeError, match="failed update"):
        with _temporary_optimizer_lr_scale(engine, 0.25):
            raise RuntimeError("failed update")

    assert optimizer.param_groups[0]["lr"] == 1e-6
