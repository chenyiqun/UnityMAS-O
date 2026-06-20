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
STAR_OPTIMIZATION_STRATEGY="${STAR_OPTIMIZATION_STRATEGY:-fsdp}"
STAR_OPTIMIZATION_STRATEGY="$(printf '%s' "${STAR_OPTIMIZATION_STRATEGY}" | tr '[:upper:]' '[:lower:]')"
export STAR_OPTIMIZATION_STRATEGY

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

STRATEGY_OVERRIDES=()
case "${STAR_OPTIMIZATION_STRATEGY}" in
  fsdp)
    STRATEGY_OVERRIDES+=("model_engine=dp" "trainer.optimization_strategy=fsdp")
    ;;
  fsdp2)
    STRATEGY_OVERRIDES+=(
      "model_engine=dp"
      "trainer.optimization_strategy=fsdp2"
      "actor_rollout_ref.actor.strategy=fsdp2"
      "actor_rollout_ref.actor.fsdp_config.strategy=fsdp2"
      "critic.strategy=fsdp2"
      "critic.fsdp.strategy=fsdp2"
    )
    ;;
  megatron)
    export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
    MEGATRON_TP="${STAR_MEGATRON_TENSOR_MODEL_PARALLEL_SIZE:-${MEGATRON_TENSOR_MODEL_PARALLEL_SIZE:-1}}"
    MEGATRON_PP="${STAR_MEGATRON_PIPELINE_MODEL_PARALLEL_SIZE:-${MEGATRON_PIPELINE_MODEL_PARALLEL_SIZE:-1}}"
    MEGATRON_CP="${STAR_MEGATRON_CONTEXT_PARALLEL_SIZE:-${MEGATRON_CONTEXT_PARALLEL_SIZE:-1}}"
    MEGATRON_EP="${STAR_MEGATRON_EXPERT_MODEL_PARALLEL_SIZE:-${MEGATRON_EXPERT_MODEL_PARALLEL_SIZE:-1}}"
    MEGATRON_ETP="${STAR_MEGATRON_EXPERT_TENSOR_PARALLEL_SIZE:-${MEGATRON_EXPERT_TENSOR_PARALLEL_SIZE:-null}}"
    MEGATRON_PARAM_OFFLOAD="${STAR_MEGATRON_PARAM_OFFLOAD:-${MEGATRON_PARAM_OFFLOAD:-False}}"
    MEGATRON_GRAD_OFFLOAD="${STAR_MEGATRON_GRAD_OFFLOAD:-${MEGATRON_GRAD_OFFLOAD:-False}}"
    MEGATRON_OPTIMIZER_OFFLOAD="${STAR_MEGATRON_OPTIMIZER_OFFLOAD:-${MEGATRON_OPTIMIZER_OFFLOAD:-False}}"
    MEGATRON_USE_MBRIDGE="${STAR_MEGATRON_USE_MBRIDGE:-${MEGATRON_USE_MBRIDGE:-True}}"
    MEGATRON_VANILLA_MBRIDGE="${STAR_MEGATRON_VANILLA_MBRIDGE:-${MEGATRON_VANILLA_MBRIDGE:-True}}"
    MEGATRON_USE_DIST_CKPT="${STAR_MEGATRON_USE_DIST_CHECKPOINTING:-${MEGATRON_USE_DIST_CHECKPOINTING:-False}}"
    MEGATRON_DIST_CKPT_PATH="${STAR_MEGATRON_DIST_CHECKPOINTING_PATH:-${MEGATRON_DIST_CHECKPOINTING_PATH:-null}}"
    MEGATRON_SEQUENCE_PARALLEL="${STAR_MEGATRON_SEQUENCE_PARALLEL:-${MEGATRON_SEQUENCE_PARALLEL:-True}}"
    MEGATRON_DTYPE="${STAR_MEGATRON_DTYPE:-${MEGATRON_DTYPE:-bfloat16}}"
    STRATEGY_OVERRIDES+=(
      "model_engine=megatron"
      "trainer.optimization_strategy=megatron"
      "actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${MEGATRON_TP}"
      "actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${MEGATRON_PP}"
      "actor_rollout_ref.actor.megatron.context_parallel_size=${MEGATRON_CP}"
      "actor_rollout_ref.actor.megatron.expert_model_parallel_size=${MEGATRON_EP}"
      "actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${MEGATRON_ETP}"
      "actor_rollout_ref.actor.megatron.param_offload=${MEGATRON_PARAM_OFFLOAD}"
      "actor_rollout_ref.actor.megatron.grad_offload=${MEGATRON_GRAD_OFFLOAD}"
      "actor_rollout_ref.actor.megatron.optimizer_offload=${MEGATRON_OPTIMIZER_OFFLOAD}"
      "actor_rollout_ref.actor.megatron.use_mbridge=${MEGATRON_USE_MBRIDGE}"
      "actor_rollout_ref.actor.megatron.vanilla_mbridge=${MEGATRON_VANILLA_MBRIDGE}"
      "actor_rollout_ref.actor.megatron.use_dist_checkpointing=${MEGATRON_USE_DIST_CKPT}"
      "actor_rollout_ref.actor.megatron.dist_checkpointing_path=${MEGATRON_DIST_CKPT_PATH}"
      "actor_rollout_ref.actor.megatron.sequence_parallel=${MEGATRON_SEQUENCE_PARALLEL}"
      "actor_rollout_ref.actor.megatron.dtype=${MEGATRON_DTYPE}"
      "actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${MEGATRON_TP}"
      "actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${MEGATRON_PP}"
      "actor_rollout_ref.ref.megatron.context_parallel_size=${MEGATRON_CP}"
      "actor_rollout_ref.ref.megatron.expert_model_parallel_size=${MEGATRON_EP}"
      "actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${MEGATRON_ETP}"
      "actor_rollout_ref.ref.megatron.param_offload=${MEGATRON_PARAM_OFFLOAD}"
      "actor_rollout_ref.ref.megatron.use_mbridge=${MEGATRON_USE_MBRIDGE}"
      "actor_rollout_ref.ref.megatron.vanilla_mbridge=${MEGATRON_VANILLA_MBRIDGE}"
      "actor_rollout_ref.ref.megatron.use_dist_checkpointing=${MEGATRON_USE_DIST_CKPT}"
      "actor_rollout_ref.ref.megatron.dist_checkpointing_path=${MEGATRON_DIST_CKPT_PATH}"
      "actor_rollout_ref.ref.megatron.sequence_parallel=${MEGATRON_SEQUENCE_PARALLEL}"
      "actor_rollout_ref.ref.megatron.dtype=${MEGATRON_DTYPE}"
      "critic.megatron.tensor_model_parallel_size=${MEGATRON_TP}"
      "critic.megatron.pipeline_model_parallel_size=${MEGATRON_PP}"
      "critic.megatron.context_parallel_size=${MEGATRON_CP}"
      "critic.megatron.expert_model_parallel_size=${MEGATRON_EP}"
      "critic.megatron.expert_tensor_parallel_size=${MEGATRON_ETP}"
      "critic.megatron.param_offload=${MEGATRON_PARAM_OFFLOAD}"
      "critic.megatron.grad_offload=${MEGATRON_GRAD_OFFLOAD}"
      "critic.megatron.optimizer_offload=${MEGATRON_OPTIMIZER_OFFLOAD}"
      "critic.megatron.use_mbridge=${MEGATRON_USE_MBRIDGE}"
      "critic.megatron.vanilla_mbridge=${MEGATRON_VANILLA_MBRIDGE}"
      "critic.megatron.use_dist_checkpointing=${MEGATRON_USE_DIST_CKPT}"
      "critic.megatron.dist_checkpointing_path=${MEGATRON_DIST_CKPT_PATH}"
      "critic.megatron.sequence_parallel=${MEGATRON_SEQUENCE_PARALLEL}"
      "critic.megatron.dtype=${MEGATRON_DTYPE}"
    )
    ;;
  *)
    echo "[common/run_per_node] ERROR: STAR_OPTIMIZATION_STRATEGY must be fsdp, fsdp2, or megatron; got ${STAR_OPTIMIZATION_STRATEGY}"
    exit 1
    ;;
esac

if [[ "${CONFIG_NAME}" == star_code_* ]]; then
  if python3 -c 'import os, sys; sys.exit(0 if float(os.environ["ROLLOUT_GPU_MEMORY_UTILIZATION"]) > float(os.environ["STAR_CODE_ROLLOUT_GPU_MEMORY_UTILIZATION_CAP"]) else 1)' >/dev/null 2>&1; then
    echo "[common/run_per_node] lowering ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION} to STAR_CODE_ROLLOUT_GPU_MEMORY_UTILIZATION_CAP=${STAR_CODE_ROLLOUT_GPU_MEMORY_UTILIZATION_CAP} for code workflow"
    export ROLLOUT_GPU_MEMORY_UTILIZATION="${STAR_CODE_ROLLOUT_GPU_MEMORY_UTILIZATION_CAP}"
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
echo "[common/run_per_node] optimization strategy=${STAR_OPTIMIZATION_STRATEGY}"
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
    "${STRATEGY_OVERRIDES[@]}" \
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
