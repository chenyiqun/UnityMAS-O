#!/usr/bin/env bash
set -euo pipefail

# One-command entry for PyTorchJob:
# - Fresh node: auto setup env + run
# - Reused node: skip setup and run directly
#
# You can run the same script every time.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-verl}"
FORCE_ENV_SETUP="${FORCE_ENV_SETUP:-false}"
HEAD_IP="${HEAD_IP:-}"

# Your defaults (can still be overridden by env vars from job config).
TRAIN_PARQUET="${TRAIN_PARQUET:-/mnt/tidal-alsh01/usr/chenyiqun/research_project/MARL_Framework/verl/data/gsm8k/train.parquet}"
VAL_PARQUET="${VAL_PARQUET:-/mnt/tidal-alsh01/usr/chenyiqun/research_project/MARL_Framework/verl/data/gsm8k/test.parquet}"
REWRITE_MODEL_PATH="${REWRITE_MODEL_PATH:-/mnt/tidal-alsh01/usr/chenyiqun/base_models/Qwen/Qwen2.5-7B-Instruct}"
SELECT_MODEL_PATH="${SELECT_MODEL_PATH:-/mnt/tidal-alsh01/usr/chenyiqun/base_models/Qwen/Qwen2.5-7B-Instruct}"
ANSWER_MODEL_PATH="${ANSWER_MODEL_PATH:-/mnt/tidal-alsh01/usr/chenyiqun/base_models/Qwen/Qwen2.5-14B-Instruct}"

# Replace with your real retriever endpoint pool if needed.
RETRIEVAL_API_URLS_JSON="${RETRIEVAL_API_URLS_JSON:-[\"http://127.0.0.1:8000/retrieve\"]}"

VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-true}"
TEST_FREQ="${TEST_FREQ:-50}"
SAVE_FREQ="${SAVE_FREQ:-50}"
ROLLOUT_NAME="${ROLLOUT_NAME:-vllm}"
ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU="${ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
CRITIC_PPO_MICRO_BATCH_SIZE_PER_GPU="${CRITIC_PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU="${ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU:-1}"
REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU="${REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU:-1}"
STAR_WEIGHT_SYNC_MASTER_PORT="${STAR_WEIGHT_SYNC_MASTER_PORT:-29600}"
STAR_WEIGHT_SYNC_TIMEOUT_SEC="${STAR_WEIGHT_SYNC_TIMEOUT_SEC:-900}"
STAR_WEIGHT_SYNC_RETRIES="${STAR_WEIGHT_SYNC_RETRIES:-3}"
STAR_WEIGHT_SYNC_PORT_RETRY_STRIDE="${STAR_WEIGHT_SYNC_PORT_RETRY_STRIDE:-10}"
STAR_WEIGHT_SYNC_MODE="${STAR_WEIGHT_SYNC_MODE:-auto}"
STAR_WORKER_MAX_CONCURRENCY="${STAR_WORKER_MAX_CONCURRENCY:-4}"
VLLM_USE_V1="${VLLM_USE_V1:-1}"

# IP convenience:
# 1) set HEAD_IP once, script maps it to MASTER_ADDR
# 2) or keep MASTER_ADDR from PyTorchJob env
# 3) if still empty, rank0 writes detected IP to MASTER_ADDR_FILE and workers read it
MASTER_ADDR="${MASTER_ADDR:-${HEAD_IP}}"
MASTER_ADDR_FILE="${MASTER_ADDR_FILE:-$(pwd)/.star_master_addr}"

need_setup="false"
if [[ "${FORCE_ENV_SETUP}" == "true" ]]; then
  need_setup="true"
elif [[ ! -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
  need_setup="true"
else
  # shellcheck disable=SC1090
  source "${CONDA_ROOT}/etc/profile.d/conda.sh"
  if ! conda env list | awk '{print $1}' | grep -Fxq "${CONDA_ENV_NAME}"; then
    need_setup="true"
  fi
fi

if [[ "${need_setup}" == "true" ]]; then
  echo "[oneclick] env not ready (or FORCE_ENV_SETUP=true), will run setup."
  export DO_ENV_SETUP=true
else
  echo "[oneclick] detected existing env '${CONDA_ENV_NAME}', skip setup."
  export DO_ENV_SETUP=false
fi

export CONDA_ROOT CONDA_ENV_NAME
export TRAIN_PARQUET VAL_PARQUET
export REWRITE_MODEL_PATH SELECT_MODEL_PATH ANSWER_MODEL_PATH
export RETRIEVAL_API_URLS_JSON
export VAL_BEFORE_TRAIN TEST_FREQ SAVE_FREQ
export ROLLOUT_NAME
export ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU CRITIC_PPO_MICRO_BATCH_SIZE_PER_GPU
export ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU
export STAR_WEIGHT_SYNC_MASTER_PORT STAR_WEIGHT_SYNC_TIMEOUT_SEC
export STAR_WEIGHT_SYNC_RETRIES STAR_WEIGHT_SYNC_PORT_RETRY_STRIDE STAR_WEIGHT_SYNC_MODE
export STAR_WORKER_MAX_CONCURRENCY
export VLLM_USE_V1
export MASTER_ADDR MASTER_ADDR_FILE

bash "${SCRIPT_DIR}/run_star_pytorchjob_bootstrap.sh"
