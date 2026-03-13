#!/usr/bin/env bash
set -euo pipefail

# Use PyTorchJob env vars to bootstrap a Ray cluster:
# - RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT
# Rank 0 starts Ray head and launches training.
# Other ranks join as Ray workers and block.

RANK="${RANK:-0}"
WORLD_SIZE="${WORLD_SIZE:-4}"
MASTER_ADDR="${MASTER_ADDR:-${HEAD_IP:-}}"
MASTER_PORT="${MASTER_PORT:-6379}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8265}"
CPUS_PER_NODE="${CPUS_PER_NODE:-64}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
MASTER_ADDR_FILE="${MASTER_ADDR_FILE:-}"

TRAIN_PARQUET="${TRAIN_PARQUET:-/mnt/tidal-alsh01/usr/chenyiqun/datasets/data/verl_format_data/hotpotqa/train_verl.parquet}"
VAL_PARQUET="${VAL_PARQUET:-/mnt/tidal-alsh01/usr/chenyiqun/datasets/data/verl_format_data/hotpotqa/test_verl.parquet}"
REWRITE_MODEL_PATH="${REWRITE_MODEL_PATH:-/path/to/rewrite_7b}"
SELECT_MODEL_PATH="${SELECT_MODEL_PATH:-/path/to/select_7b}"
ANSWER_MODEL_PATH="${ANSWER_MODEL_PATH:-/path/to/answer_14b}"
RETRIEVAL_API_URLS_JSON="${RETRIEVAL_API_URLS_JSON:-[\"http://10.158.147.72:8000/retrieve\"]}"
ROLLOUT_NAME="${ROLLOUT_NAME:-vllm}"
VLLM_USE_V1="${VLLM_USE_V1:-1}"
WANDB_API_KEY="${WANDB_API_KEY:-5235f681e1a2a0ef6fe3a1f4686280daad738532}"

VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-true}"
TEST_FREQ="${TEST_FREQ:-50}"
SAVE_FREQ="${SAVE_FREQ:-50}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-128}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-128}"
ROLLOUT_FREE_CACHE_ENGINE="${ROLLOUT_FREE_CACHE_ENGINE:-true}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.6}"
ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-1024}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-128}"

ray stop -f >/dev/null 2>&1 || true

if [[ -z "${MASTER_ADDR}" ]]; then
  if [[ "${RANK}" == "0" ]]; then
    MASTER_ADDR="$(hostname -I 2>/dev/null | awk '{print $1}')"
    if [[ -z "${MASTER_ADDR}" ]]; then
      MASTER_ADDR="$(hostname -i 2>/dev/null | awk '{print $1}')"
    fi
    if [[ -z "${MASTER_ADDR}" ]]; then
      echo "[star-pytorchjob] failed to auto-detect master IP on rank0"
      exit 1
    fi
    if [[ -n "${MASTER_ADDR_FILE}" ]]; then
      echo "${MASTER_ADDR}" > "${MASTER_ADDR_FILE}"
      echo "[star-pytorchjob] wrote MASTER_ADDR=${MASTER_ADDR} to ${MASTER_ADDR_FILE}"
    fi
  else
    if [[ -z "${MASTER_ADDR_FILE}" ]]; then
      echo "[star-pytorchjob] MASTER_ADDR is empty and MASTER_ADDR_FILE is not set on rank${RANK}"
      exit 1
    fi
    echo "[star-pytorchjob] rank${RANK} waiting for ${MASTER_ADDR_FILE}"
    for _ in $(seq 1 180); do
      if [[ -s "${MASTER_ADDR_FILE}" ]]; then
        MASTER_ADDR="$(cat "${MASTER_ADDR_FILE}")"
        break
      fi
      sleep 2
    done
    if [[ -z "${MASTER_ADDR}" ]]; then
      echo "[star-pytorchjob] timeout waiting MASTER_ADDR_FILE=${MASTER_ADDR_FILE}"
      exit 1
    fi
  fi
fi
export MASTER_ADDR
export VLLM_USE_V1
export WANDB_API_KEY
export ROLLOUT_FREE_CACHE_ENGINE
export ROLLOUT_GPU_MEMORY_UTILIZATION
export ROLLOUT_MAX_NUM_BATCHED_TOKENS
export ROLLOUT_MAX_NUM_SEQS

if [[ "${RANK}" == "0" ]]; then
  echo "[star-pytorchjob] rank0 starts Ray head at ${MASTER_ADDR}:${MASTER_PORT}"
  ray start --head \
    --node-ip-address="${MASTER_ADDR}" \
    --port="${MASTER_PORT}" \
    --dashboard-host=0.0.0.0 \
    --dashboard-port="${DASHBOARD_PORT}" \
    --num-cpus="${CPUS_PER_NODE}" \
    --num-gpus="${GPUS_PER_NODE}" \
    --disable-usage-stats

  echo "[star-pytorchjob] waiting for ${WORLD_SIZE} Ray nodes"
  if [[ "${WORLD_SIZE}" == "1" ]]; then
    echo "[star-pytorchjob] WORLD_SIZE=1, skip waiting for workers"
  else
    python3 - <<PY
import os
import time
import ray

expected = int(os.environ.get("WORLD_SIZE", "4"))
ray.init(address="auto")
timeout = int(os.environ.get("STAR_NODE_WAIT_TIMEOUT", "180"))  # 默认 15 分钟
for i in range(timeout):
    alive = sum(1 for n in ray.nodes() if n.get("Alive", False))
    print(f"[star-pytorchjob] alive nodes: {alive}/{expected}")
    if alive >= expected:
        break
    if i > 0 and i % 6 == 0:
        print("[star-pytorchjob] 提示: 若一直卡住，请确保已在所有 worker 节点执行相同命令")
    time.sleep(5)
else:
    raise RuntimeError(f"Timed out waiting for {expected} nodes (waited {timeout*5}s)")
ray.shutdown()
PY
  fi

  echo "[star-pytorchjob] launching training on rank0"
  python3 -m verl.experimental.star_ppo.main_ppo \
    --config-name star_query_rewrite_retrieve_select_answer_f1_trainer \
    data.train_files="${TRAIN_PARQUET}" \
    data.val_files="${VAL_PARQUET}" \
    trainer.nnodes="${WORLD_SIZE}" \
    trainer.n_gpus_per_node="${GPUS_PER_NODE}" \
    actor_rollout_ref.model.path="${REWRITE_MODEL_PATH}" \
    actor_rollout_ref.rollout.name="${ROLLOUT_NAME}" \
    star.workflow.tools.retriever.api_urls="${RETRIEVAL_API_URLS_JSON}" \
    data.gen_batch_size="${GEN_BATCH_SIZE}" \
    data.train_batch_size="${GEN_BATCH_SIZE}" \
    data.val_batch_size="${VAL_BATCH_SIZE}" \
    trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.logger='["console","wandb"]'
else
  echo "[star-pytorchjob] rank${RANK} waits for Ray head ${MASTER_ADDR}:${MASTER_PORT}"
  python3 - <<'PY'
import os
import socket
import time

host = os.environ["MASTER_ADDR"]
port = int(os.environ.get("MASTER_PORT", "6379"))
for _ in range(180):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2)
    try:
        sock.connect((host, port))
        sock.close()
        print(f"[star-pytorchjob] head reachable at {host}:{port}")
        break
    except Exception:
        time.sleep(2)
else:
    raise RuntimeError(f"Cannot reach Ray head {host}:{port}")
PY

  echo "[star-pytorchjob] rank${RANK} starts Ray worker"
  ray start \
    --address="${MASTER_ADDR}:${MASTER_PORT}" \
    --num-cpus="${CPUS_PER_NODE}" \
    --num-gpus="${GPUS_PER_NODE}" \
    --disable-usage-stats \
    --block
fi
