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
