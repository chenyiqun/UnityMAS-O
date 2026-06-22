#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"

cd "${PROJECT_ROOT}"

RANK="${RANK:-0}"
WORLD_SIZE="${WORLD_SIZE:-4}"
HEAD_IP="${HEAD_IP:-}"
RAY_PORT="${RAY_PORT:-6379}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8265}"
CPUS_PER_NODE="${CPUS_PER_NODE:-64}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
CONFIG_NAME="${CONFIG_NAME:?CONFIG_NAME is required}"
PROJECT_NAME="${PROJECT_NAME:-}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-}"
WANDB_API_KEY_VALUE="${WANDB_API_KEY:-}"
WANDB_ENTITY_VALUE="${WANDB_ENTITY:-}"

if [[ -z "${HEAD_IP}" ]]; then
  echo "[common/run_per_node] ERROR: HEAD_IP is required"
  exit 1
fi

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-0}"
export STAR_WEIGHT_SYNC_MODE="${STAR_WEIGHT_SYNC_MODE:-local_pair}"
export STAR_WORKER_MAX_CONCURRENCY="${STAR_WORKER_MAX_CONCURRENCY:-8}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_HOST_IP="${VLLM_HOST_IP:-${HEAD_IP}}"
export ROLLOUT_FREE_CACHE_ENGINE="${ROLLOUT_FREE_CACHE_ENGINE:-true}"
export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.15}"
export STAR_CODE_ROLLOUT_GPU_MEMORY_UTILIZATION_CAP="${STAR_CODE_ROLLOUT_GPU_MEMORY_UTILIZATION_CAP:-0.15}"
export ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE="${ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE:-4}"
export ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-8192}"
export ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-16}"
export ROLLOUT_ENFORCE_EAGER="${ROLLOUT_ENFORCE_EAGER:-true}"
export ROLLOUT_ENABLE_CHUNKED_PREFILL="${ROLLOUT_ENABLE_CHUNKED_PREFILL:-false}"
export ROLLOUT_DISABLE_CUSTOM_ALL_REDUCE="${ROLLOUT_DISABLE_CUSTOM_ALL_REDUCE:-true}"
export STAR_MAX_INFLIGHT_QUERIES="${STAR_MAX_INFLIGHT_QUERIES:-64}"
export STAR_MAX_PARALLEL_ROLLOUTS_PER_MODEL="${STAR_MAX_PARALLEL_ROLLOUTS_PER_MODEL:-16}"
export STAR_QUERY_TIMEOUT_SECONDS="${STAR_QUERY_TIMEOUT_SECONDS:-600}"
export STAR_TOOL_TIMEOUT_SECONDS="${STAR_TOOL_TIMEOUT_SECONDS:-30}"
export STAR_WORKFLOW_BATCH_TIMEOUT_SECONDS="${STAR_WORKFLOW_BATCH_TIMEOUT_SECONDS:-900}"
export STAR_RAY_GET_TIMEOUT_SECONDS="${STAR_RAY_GET_TIMEOUT_SECONDS:-300}"
export STAR_WORKER_CALL_TIMEOUT_SECONDS="${STAR_WORKER_CALL_TIMEOUT_SECONDS:-${STAR_RAY_GET_TIMEOUT_SECONDS}}"
export STAR_WEIGHT_SYNC_TIMEOUT_SECONDS="${STAR_WEIGHT_SYNC_TIMEOUT_SECONDS:-600}"
export STAR_STALL_DETECT_SECONDS="${STAR_STALL_DETECT_SECONDS:-180}"
export STAR_STALL_HEARTBEAT_SECONDS="${STAR_STALL_HEARTBEAT_SECONDS:-30}"
export VERL_VLLM_FORCE_SHM_WEIGHT_SYNC="${VERL_VLLM_FORCE_SHM_WEIGHT_SYNC:-1}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"

if [[ "${CONFIG_NAME}" == star_code_* ]]; then
  export ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-16384}"
  if python3 -c 'import os, sys; sys.exit(0 if float(os.environ["ROLLOUT_GPU_MEMORY_UTILIZATION"]) > float(os.environ["STAR_CODE_ROLLOUT_GPU_MEMORY_UTILIZATION_CAP"]) else 1)' >/dev/null 2>&1; then
    echo "[common/run_per_node] lowering ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION} to STAR_CODE_ROLLOUT_GPU_MEMORY_UTILIZATION_CAP=${STAR_CODE_ROLLOUT_GPU_MEMORY_UTILIZATION_CAP} for code workflow"
    export ROLLOUT_GPU_MEMORY_UTILIZATION="${STAR_CODE_ROLLOUT_GPU_MEMORY_UTILIZATION_CAP}"
  fi
  if python3 -c 'import os, sys; paths = " ".join(os.environ.get(k, "") for k in ("AGENT_MODEL_PATH", "ACTOR_MODEL_PATH", "PLANNER_MODEL_PATH", "CODER_MODEL_PATH", "REFLECTION_MODEL_PATH", "SHARED_MODEL_PATH")); allow = os.environ.get("STAR_CODE_QWEN35_9B_ALLOW_TP4", "").lower() in {"1", "true", "yes", "on"}; sys.exit(0 if "Qwen3.5-9B" in paths and int(os.environ["ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE"]) >= 4 and not allow else 1)' >/dev/null 2>&1; then
    echo "[common/run_per_node] lowering ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=${ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE} to 2 for Qwen3.5-9B code rollout stability; set STAR_CODE_QWEN35_9B_ALLOW_TP4=true to override"
    export ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=2
  fi
  if python3 -c 'import os, sys; paths = " ".join(os.environ.get(k, "") for k in ("AGENT_MODEL_PATH", "ACTOR_MODEL_PATH", "PLANNER_MODEL_PATH", "CODER_MODEL_PATH", "REFLECTION_MODEL_PATH", "SHARED_MODEL_PATH")); allow = os.environ.get("STAR_CODE_QWEN35_9B_ALLOW_CUMEM", "").lower() in {"1", "true", "yes", "on"}; sys.exit(0 if "Qwen3.5-9B" in paths and os.environ.get("NCCL_CUMEM_ENABLE", "1") != "0" and not allow else 1)' >/dev/null 2>&1; then
    echo "[common/run_per_node] setting NCCL_CUMEM_ENABLE=0 for Qwen3.5-9B code FSDP stability; set STAR_CODE_QWEN35_9B_ALLOW_CUMEM=true to override"
    export NCCL_CUMEM_ENABLE=0
  fi
  if python3 -c 'import os, sys; paths = " ".join(os.environ.get(k, "") for k in ("AGENT_MODEL_PATH", "ACTOR_MODEL_PATH", "PLANNER_MODEL_PATH", "CODER_MODEL_PATH", "REFLECTION_MODEL_PATH", "SHARED_MODEL_PATH")); allow = os.environ.get("STAR_CODE_QWEN35_9B_ALLOW_MICRO2", "").lower() in {"1", "true", "yes", "on"}; sys.exit(0 if "Qwen3.5-9B" in paths and not allow else 1)' >/dev/null 2>&1; then
    echo "[common/run_per_node] lowering Qwen3.5-9B code update/logprob micro batches to 1 for FSDP stability; set STAR_CODE_QWEN35_9B_ALLOW_MICRO2=true to override"
    export ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU=1
    export CRITIC_PPO_MICRO_BATCH_SIZE_PER_GPU=1
    export ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU=1
    export REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU=1
  fi
  if python3 -c 'import os, sys; sys.exit(0 if os.environ["ROLLOUT_ENABLE_CHUNKED_PREFILL"].lower() in {"0", "false", "no", "off"} and int(os.environ["ROLLOUT_MAX_NUM_BATCHED_TOKENS"]) < int(os.environ["ROLLOUT_MAX_MODEL_LEN"]) else 1)' >/dev/null 2>&1; then
    echo "[common/run_per_node] raising ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS} to ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN} because chunked prefill is disabled"
    export ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_MODEL_LEN}"
  fi
fi

if [[ -n "${WANDB_API_KEY_VALUE}" ]]; then
  export WANDB_API_KEY="${WANDB_API_KEY_VALUE}"
fi
if [[ -n "${WANDB_ENTITY_VALUE}" ]]; then
  export WANDB_ENTITY="${WANDB_ENTITY_VALUE}"
fi

if command -v gcc-12 >/dev/null 2>&1 && command -v g++-12 >/dev/null 2>&1; then
  export CC="${CC:-$(command -v gcc-12)}"
  export CXX="${CXX:-$(command -v g++-12)}"
  export CUDAHOSTCXX="${CUDAHOSTCXX:-${CXX}}"
else
  export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:--allow-unsupported-compiler}"
  export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
fi

if [[ "${PYTORCH_CUDA_ALLOC_CONF:-}" == *"expandable_segments:True"* ]]; then
  echo "[common/run_per_node] Detected incompatible PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}, unsetting it."
  unset PYTORCH_CUDA_ALLOC_CONF
fi

ray stop -f >/dev/null 2>&1 || true

echo "[common/run_per_node] RANK=${RANK} WORLD_SIZE=${WORLD_SIZE} HEAD_IP=${HEAD_IP} CONFIG_NAME=${CONFIG_NAME}"
echo "[common/run_per_node] wandb entity=${WANDB_ENTITY_VALUE:-<unset>} project=${PROJECT_NAME:-<config>} experiment=${EXPERIMENT_NAME:-<config>}"

if [[ "${RANK}" == "0" ]]; then
  if [[ -n "${WANDB_API_KEY_VALUE}" ]] && command -v wandb >/dev/null 2>&1; then
    echo "[common/run_per_node] refreshing wandb login from WANDB_API_KEY"
    wandb login --relogin "${WANDB_API_KEY_VALUE}" >/dev/null 2>&1 || true
  fi
  echo "[common/run_per_node] starting ray head at ${HEAD_IP}:${RAY_PORT}"
  ray start --head \
    --node-ip-address="${HEAD_IP}" \
    --port="${RAY_PORT}" \
    --dashboard-host=0.0.0.0 \
    --dashboard-port="${DASHBOARD_PORT}" \
    --num-cpus="${CPUS_PER_NODE}" \
    --num-gpus="${GPUS_PER_NODE}" \
    --disable-usage-stats

  if [[ "${WORLD_SIZE}" != "1" ]]; then
    echo "[common/run_per_node] waiting for ${WORLD_SIZE} alive nodes"
    WORLD_SIZE="${WORLD_SIZE}" python3 - <<'PY'
import os
import time
import ray

expected = int(os.environ.get("WORLD_SIZE", "4"))
ray.init(address="auto")
for _ in range(120):
    alive = sum(1 for n in ray.nodes() if n.get("Alive", False))
    print(f"[common/run_per_node] alive nodes: {alive}/{expected}")
    if alive >= expected:
        break
    time.sleep(5)
else:
    raise RuntimeError(f"Timed out waiting for {expected} nodes")
ray.shutdown()
PY
  fi

  echo "[common/run_per_node] launching training"
  EXTRA_OVERRIDES=()
  if [[ -n "${PROJECT_NAME}" ]]; then
    EXTRA_OVERRIDES+=("trainer.project_name=${PROJECT_NAME}")
  fi
  if [[ -n "${EXPERIMENT_NAME}" ]]; then
    EXTRA_OVERRIDES+=("trainer.experiment_name=${EXPERIMENT_NAME}")
  fi
  python3 -m verl.experimental.star_ppo.main_ppo \
    --config-name "${CONFIG_NAME}" \
    trainer.nnodes="${WORLD_SIZE}" \
    trainer.n_gpus_per_node="${GPUS_PER_NODE}" \
    trainer.logger='["console","wandb"]' \
    "${EXTRA_OVERRIDES[@]}" \
    "$@"
else
  echo "[common/run_per_node] starting ray worker at ${HEAD_IP}:${RAY_PORT}"
  ray start \
    --address="${HEAD_IP}:${RAY_PORT}" \
    --num-cpus="${CPUS_PER_NODE}" \
    --num-gpus="${GPUS_PER_NODE}" \
    --disable-usage-stats \
    --block
fi
