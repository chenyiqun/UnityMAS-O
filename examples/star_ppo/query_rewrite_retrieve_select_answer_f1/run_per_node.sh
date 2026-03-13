#!/usr/bin/env bash
# =============================================================================
# Star PPO 多节点运行脚本
# 在每个 node 上分别执行此脚本即可启动训练。
#
# 使用方法：
#   Node 0 (head):  RANK=0 HEAD_IP=<node0的IP> WORLD_SIZE=4 bash run_per_node.sh
#   Node 1:         RANK=1 HEAD_IP=<node0的IP> WORLD_SIZE=4 bash run_per_node.sh
#   Node 2:         RANK=2 HEAD_IP=<node0的IP> WORLD_SIZE=4 bash run_per_node.sh
#   Node 3:         RANK=3 HEAD_IP=<node0的IP> WORLD_SIZE=4 bash run_per_node.sh
#
# 若在项目根目录执行，需先 cd 到项目根，或设置 PROJECT_ROOT 环境变量。
#
# 后台运行示例（带日志）：
#   TS=$(date +%Y%m%d_%H%M%S)
#   RANK=0 HEAD_IP=10.146.231.133 WORLD_SIZE=4 nohup stdbuf -oL -eL bash run_per_node.sh \
#     > "logs/star_ppo/run_rank0_${TS}.log" 2>&1 < /dev/null &
#   echo "PID=$! LOG=logs/star_ppo/run_rank0_${TS}.log"
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"

cd "${PROJECT_ROOT}"
mkdir -p logs/star_ppo

# --- 必须由用户在各自 node 上设置 ---
RANK="${RANK:-0}"
WORLD_SIZE="${WORLD_SIZE:-4}"
HEAD_IP="${HEAD_IP:-}"

# --- 若未设置 HEAD_IP，rank0 会自检 IP；其他 rank 必须设置 ---
if [[ "${RANK}" != "0" && -z "${HEAD_IP}" ]]; then
  echo "[run_per_node] ERROR: RANK!=0 时必须设置 HEAD_IP (node0 的 IP)"
  exit 1
fi

# --- 共享目录：用于 rank0 写入 MASTER_ADDR，worker 读取（当 HEAD_IP 未设置时）---
MASTER_ADDR_FILE="${MASTER_ADDR_FILE:-$(pwd)/.star_master_addr}"

# --- Star PPO / vLLM 相关环境变量 ---
# 若出现 "Engine core initialization failed"，可尝试：
#   1) 进一步降低 ROLLOUT_GPU_MEMORY_UTILIZATION (如 0.2)
#   2) 设置 VLLM_USE_V1=0 使用 v0 引擎
export HEAD_IP
export RANK
export WORLD_SIZE
export MASTER_ADDR_FILE
export STAR_WEIGHT_SYNC_MODE="${STAR_WEIGHT_SYNC_MODE:-auto}"
export STAR_WORKER_MAX_CONCURRENCY="${STAR_WORKER_MAX_CONCURRENCY:-8}"
export STAR_WEIGHT_SYNC_RETRIES="${STAR_WEIGHT_SYNC_RETRIES:-5}"
export STAR_WEIGHT_SYNC_PORT_RETRY_STRIDE="${STAR_WEIGHT_SYNC_PORT_RETRY_STRIDE:-20}"
export ROLLOUT_UPDATE_WEIGHTS_BUCKET_MB="${ROLLOUT_UPDATE_WEIGHTS_BUCKET_MB:-4096}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export PYTHONUNBUFFERED=1
export RAY_DEDUP_LOGS=0
export ROLLOUT_FREE_CACHE_ENGINE="${ROLLOUT_FREE_CACHE_ENGINE:-true}"
export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.6}"
export ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-1024}"
export ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-128}"

# vLLM v1 memory pool is incompatible with expandable_segments:True.
# See: https://github.com/pytorch/pytorch/issues/147851
if [[ "${PYTORCH_CUDA_ALLOC_CONF:-}" == *"expandable_segments:True"* ]]; then
  echo "[run_per_node] Detected incompatible PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}, unsetting it."
  unset PYTORCH_CUDA_ALLOC_CONF
fi

# --- 清理残留 Ray 进程（可选，首次运行建议保留）---
(ray stop -f >/dev/null 2>&1 || true)
(pkill -9 -f "ray::|raylet|gcs_server|dashboard|runtime_env_agent|log_monitor" >/dev/null 2>&1 || true)

echo "[run_per_node] RANK=${RANK} WORLD_SIZE=${WORLD_SIZE} HEAD_IP=${HEAD_IP} PROJECT_ROOT=${PROJECT_ROOT}"
bash "${SCRIPT_DIR}/run_star_pytorchjob_oneclick.sh"
