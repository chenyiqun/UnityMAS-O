# Copyright 2025 Meituan Ltd. and/or its affiliates
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
import asyncio
import logging
import math
import os
import time

import numpy as np
import ray

from verl.experimental.agent_loop.agent_loop import AgentLoopManager
from verl.protocol import DataProto

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class OneStepOffAgentLoopManager(AgentLoopManager):
    async def generate_sequences_async(self, prompts: DataProto) -> DataProto:
        """Split input batch and dispatch to agent loop workers (async version).

        Args:
            prompts (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
        """

        if len(self.agent_loop_workers) == 0:
            raise RuntimeError("No agent loop workers available for async generation.")

        manager_start = time.perf_counter()

        # Validation/workflow may issue tiny batches (e.g. size=1).
        # DataProto.chunk requires equal split when padding is disabled, so fallback to
        # split() for non-divisible cases to avoid assertion failures.
        target_workers = min(len(self.agent_loop_workers), max(1, len(prompts)))
        if len(prompts) % target_workers == 0:
            chunks = prompts.chunk(target_workers)
        else:
            split_size = math.ceil(len(prompts) / target_workers)
            chunks = prompts.split(split_size)

        # Use round-robin worker selection so tiny batches (especially len=1 in STAR
        # per-node workflow execution) do not always hit worker-0.
        if not hasattr(self, "_rr_worker_cursor"):
            self._rr_worker_cursor = 0
        start = int(self._rr_worker_cursor % len(self.agent_loop_workers))
        workers = [self.agent_loop_workers[(start + i) % len(self.agent_loop_workers)] for i in range(len(chunks))]
        self._rr_worker_cursor += len(chunks)

        manager_prep_s = time.perf_counter() - manager_start

        async def _run_chunk(worker, chunk):
            rpc_start = time.perf_counter()
            output = await asyncio.to_thread(ray.get, worker.generate_sequences.remote(chunk))
            return output, time.perf_counter() - rpc_start

        worker_rpc_start = time.perf_counter()
        chunk_results = await asyncio.gather(
            *[_run_chunk(worker, chunk) for worker, chunk in zip(workers, chunks, strict=True)]
        )
        worker_rpc_elapsed_s = time.perf_counter() - worker_rpc_start
        outputs = [output for output, _ in chunk_results]
        worker_rpc_times = [elapsed_s for _, elapsed_s in chunk_results]

        concat_start = time.perf_counter()
        output = DataProto.concat(outputs)
        manager_concat_s = time.perf_counter() - concat_start

        # calculate performance metrics
        metrics_start = time.perf_counter()
        metrics = [output.meta_info.pop("metrics") for output in outputs]  # List[List[Dict[str, str]]]
        timing = self._performance_metrics(metrics, output)
        manager_metrics_reduce_s = time.perf_counter() - metrics_start

        worker_rpc_times_np = np.array(worker_rpc_times, dtype=np.float64)
        manager_total_s = time.perf_counter() - manager_start
        timing["agent_loop/manager/prep"] = manager_prep_s
        timing["agent_loop/manager/worker_rpc_wait"] = worker_rpc_elapsed_s
        timing["agent_loop/manager/worker_rpc_mean"] = worker_rpc_times_np.mean()
        timing["agent_loop/manager/worker_rpc_max"] = worker_rpc_times_np.max()
        timing["agent_loop/manager/concat"] = manager_concat_s
        timing["agent_loop/manager/metrics_reduce"] = manager_metrics_reduce_s
        timing["agent_loop/manager/total"] = manager_total_s
        timing["agent_loop/manager/overhead"] = max(
            manager_total_s - float(worker_rpc_times_np.max()),
            0.0,
        )

        output.meta_info = {"timing": timing, **outputs[0].meta_info}
        return output

    async def wake_up(self):
        await asyncio.gather(*[replica.wake_up() for replica in self.rollout_replicas])

    async def sleep(self):
        await asyncio.gather(*[replica.sleep() for replica in self.rollout_replicas])

    async def clear_kv_cache(self):
        await asyncio.gather(*[replica.clear_kv_cache() for replica in self.rollout_replicas])
