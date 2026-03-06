#!/usr/bin/env bash
set -euo pipefail

# Use PyTorchJob env vars to bootstrap a Ray cluster:
# - RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT
# Rank 0 starts Ray head and launches training.
# Other ranks join as Ray workers and block.

RANK="${RANK:-0}"
WORLD_SIZE="${WORLD_SIZE:-4}"
MASTER_ADDR="${MASTER_ADDR:?MASTER_ADDR is required}"
MASTER_PORT="${MASTER_PORT:-6379}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8265}"
CPUS_PER_NODE="${CPUS_PER_NODE:-64}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"

TRAIN_PARQUET="${TRAIN_PARQUET:-/path/to/train.parquet}"
VAL_PARQUET="${VAL_PARQUET:-/path/to/val.parquet}"
REWRITE_MODEL_PATH="${REWRITE_MODEL_PATH:-/path/to/rewrite_7b}"
SELECT_MODEL_PATH="${SELECT_MODEL_PATH:-/path/to/select_7b}"
ANSWER_MODEL_PATH="${ANSWER_MODEL_PATH:-/path/to/answer_14b}"
RETRIEVAL_API_URLS_JSON="${RETRIEVAL_API_URLS_JSON:-[\"http://127.0.0.1:8000/retrieve\"]}"

VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-true}"
TEST_FREQ="${TEST_FREQ:-50}"
SAVE_FREQ="${SAVE_FREQ:-50}"

ray stop -f >/dev/null 2>&1 || true

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
  python3 - <<'PY'
import os
import time
import ray

expected = int(os.environ.get("WORLD_SIZE", "4"))
ray.init(address="auto")
for _ in range(180):
    alive = sum(1 for n in ray.nodes() if n.get("Alive", False))
    print(f"[star-pytorchjob] alive nodes: {alive}/{expected}")
    if alive >= expected:
        break
    time.sleep(5)
else:
    raise RuntimeError(f"Timed out waiting for {expected} nodes")
ray.shutdown()
PY

  echo "[star-pytorchjob] launching training on rank0"
  python3 -m verl.experimental.star_ppo.main_ppo \
    --config-name star_query_rewrite_retrieve_select_answer_f1_trainer \
    ray_kwargs.ray_init.address=auto \
    data.train_files="${TRAIN_PARQUET}" \
    data.val_files="${VAL_PARQUET}" \
    trainer.nnodes="${WORLD_SIZE}" \
    trainer.n_gpus_per_node="${GPUS_PER_NODE}" \
    actor_rollout_ref.model.path="${REWRITE_MODEL_PATH}" \
    trainer.llm_engines[0].model_path="${REWRITE_MODEL_PATH}" \
    trainer.llm_engines[1].model_path="${SELECT_MODEL_PATH}" \
    trainer.llm_engines[2].model_path="${ANSWER_MODEL_PATH}" \
    star.workflow.tools.retriever.api_urls="${RETRIEVAL_API_URLS_JSON}" \
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
