# Star PPO Query-Retrieve-Select-Answer (F1)

This setup trains three LLM agents in one graph workflow:

- `rewrite` (7B, 1 node)
- `select_docs` (7B, 1 node)
- `answer` (14B, 2 nodes)

Retrieval is a tool call to an external API that supports:

- input: `{"questions": [question], "N": N}`
- output: `data[0]["top_k_docs"]`

Reward:

- global reward: final answer vs ground-truth token-level F1
- format reward: invalid tag format gets `-1` for each LLM node

## Dataset

Parquet rows like:

- `extra_info.question`
- `extra_info.answer`
- `reward_model.ground_truth`

are supported directly by the workflow runner.

## Run on 4 nodes / 32 GPUs

1. On 3 worker nodes:

```bash
HEAD_IP=<head_ip> \
bash examples/star_ppo/query_rewrite_retrieve_select_answer_f1/run_star_worker.sh
```

2. On head node:

```bash
HEAD_IP=<head_ip> \
WORLD_SIZE=4 \
TRAIN_PARQUET=/path/to/train.parquet \
VAL_PARQUET=/path/to/val.parquet \
REWRITE_MODEL_PATH=/path/to/rewrite_7b \
SELECT_MODEL_PATH=/path/to/select_7b \
ANSWER_MODEL_PATH=/path/to/answer_14b \
RETRIEVAL_API_URLS_JSON='["http://api1/retrieve","http://api2/retrieve"]' \
VAL_BEFORE_TRAIN=true \
TEST_FREQ=50 \
SAVE_FREQ=50 \
bash examples/star_ppo/query_rewrite_retrieve_select_answer_f1/run_star_head_train.sh
```

3. Smoke test command:

```bash
HEAD_IP=<head_ip> \
WORLD_SIZE=4 \
TRAIN_PARQUET=/path/to/train.parquet \
VAL_PARQUET=/path/to/val.parquet \
REWRITE_MODEL_PATH=/path/to/rewrite_7b \
SELECT_MODEL_PATH=/path/to/select_7b \
ANSWER_MODEL_PATH=/path/to/answer_14b \
bash examples/star_ppo/query_rewrite_retrieve_select_answer_f1/run_star_head_test.sh
```

## Run with PyTorchJob

If your cluster does not support RayJob CRD, use PyTorchJob to start pods and run the same
entry script on every pod:

- rank 0 pod starts Ray head and launches training
- other pods join Ray as workers and block

Script:

- `examples/star_ppo/query_rewrite_retrieve_select_answer_f1/run_star_pytorchjob_entry.sh`

Required env (typically provided by PyTorchJob):

- `RANK`
- `WORLD_SIZE`
- `MASTER_ADDR` (optional if you set `HEAD_IP`, or use shared `MASTER_ADDR_FILE`)
- `MASTER_PORT`

Additional env you should set:

- `TRAIN_PARQUET`
- `VAL_PARQUET`
- `REWRITE_MODEL_PATH`
- `SELECT_MODEL_PATH`
- `ANSWER_MODEL_PATH`
- `RETRIEVAL_API_URLS_JSON`
- `VAL_BEFORE_TRAIN`
- `TEST_FREQ`
- `SAVE_FREQ`

### One-line command template for PyTorchJob

Use the same command on all pods (rank is auto-routed by `RANK`).
Default behavior is **run only** (no environment reinstall):

```bash
bash -lc 'cd /mnt/tidal-alsh01/usr/chenyiqun/research_project/adaptive_joint_optim/rl/verl && bash examples/star_ppo/query_rewrite_retrieve_select_answer_f1/run_star_pytorchjob_bootstrap.sh'
```

If you need to install environment on fresh nodes, run once with:

```bash
bash -lc 'cd /mnt/tidal-alsh01/usr/chenyiqun/research_project/adaptive_joint_optim/rl/verl && DO_ENV_SETUP=true bash examples/star_ppo/query_rewrite_retrieve_select_answer_f1/run_star_pytorchjob_bootstrap.sh'
```

The bootstrap script handles:

- optional environment setup (`DO_ENV_SETUP=true` -> `setup_verl_env.sh`)
- conda activation (existing env)
- rank-based head/worker launch (`run_star_pytorchjob_entry.sh`)

Files:

- `examples/star_ppo/query_rewrite_retrieve_select_answer_f1/setup_verl_env.sh`
- `examples/star_ppo/query_rewrite_retrieve_select_answer_f1/run_star_pytorchjob_bootstrap.sh`
- `examples/star_ppo/query_rewrite_retrieve_select_answer_f1/run_star_pytorchjob_entry.sh`

### Recommended single script (auto detect setup vs run)

Use this same command every time:

```bash
bash -lc 'cd /mnt/tidal-alsh01/usr/chenyiqun/research_project/adaptive_joint_optim/rl/verl && bash examples/star_ppo/query_rewrite_retrieve_select_answer_f1/run_star_pytorchjob_oneclick.sh'
```

Behavior:

- fresh nodes: auto install env then run
- existing nodes: skip install and run directly

Script:

- `examples/star_ppo/query_rewrite_retrieve_select_answer_f1/run_star_pytorchjob_oneclick.sh`

### Easy IP configuration

You can choose one of these methods:

1) Set head IP once (recommended):

```bash
HEAD_IP=10.146.231.133 \
bash examples/star_ppo/query_rewrite_retrieve_select_answer_f1/run_star_pytorchjob_oneclick.sh
```

2) Use standard PyTorchJob env directly:

```bash
MASTER_ADDR=10.146.231.133 MASTER_PORT=6379 \
bash examples/star_ppo/query_rewrite_retrieve_select_answer_f1/run_star_pytorchjob_oneclick.sh
```

3) Auto-sync IP through shared file (no manual worker IP setup):

```bash
MASTER_ADDR_FILE=/mnt/tidal-alsh01/usr/chenyiqun/research_project/MARL_Framework/.star_master_addr \
bash examples/star_ppo/query_rewrite_retrieve_select_answer_f1/run_star_pytorchjob_oneclick.sh
```

Notes:

- rank0 auto-detects its local IP and writes to `MASTER_ADDR_FILE`
- workers wait for that file and read `MASTER_ADDR`
