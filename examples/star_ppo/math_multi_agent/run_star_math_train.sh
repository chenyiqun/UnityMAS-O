#!/usr/bin/env bash
set -euo pipefail

CONFIG_NAME="${CONFIG_NAME:-star_math_solver_verifier_refiner_finalizer_trainer}"

PROJECT_NAME="${PROJECT_NAME:-star_math_multi_agent}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-solver_verifier_refiner_finalizer_qwen2_5_7b}"

AGENT_MODEL_PATH="${AGENT_MODEL_PATH:-/mnt/tidal-alsh01/usr/chenyiqun/base_models/Qwen/Qwen2.5-7B-Instruct}"
TRAIN_JSONL="${TRAIN_JSONL:-/mnt/tidal-alsh01/usr/chenyiqun/datasets/Math/DAPO-Math-17k/data/dapo-math-17k.question.jsonl}"
VAL_FILES="${VAL_FILES:-${VAL_JSONL:-}}"
VAL_FILES="${VAL_FILES:-/mnt/tidal-alsh01/usr/chenyiqun/datasets/Math/MATH-500/test.jsonl}"

NNODES="${NNODES:-4}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
AGENT_GPUS_PER_NODE="${AGENT_GPUS_PER_NODE:-8}"

GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-128}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-128}"
ACTOR_PPO_MINI_BATCH_SIZE="${ACTOR_PPO_MINI_BATCH_SIZE:-64}"
ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU="${ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU:-2}"
CRITIC_PPO_MICRO_BATCH_SIZE_PER_GPU="${CRITIC_PPO_MICRO_BATCH_SIZE_PER_GPU:-2}"

ROLLOUT_PROMPT_LENGTH="${ROLLOUT_PROMPT_LENGTH:-8192}"
ROLLOUT_RESPONSE_LENGTH="${ROLLOUT_RESPONSE_LENGTH:-4096}"
ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-16384}"
ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE="${ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE:-2}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.70}"

VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-true}"
TEST_FREQ="${TEST_FREQ:-50}"
SAVE_FREQ="${SAVE_FREQ:-300}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-10}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-}"
CKPTS_DIR="${CKPTS_DIR:-/mnt/tidal-alsh01/usr/chenyiqun/ckpts/${PROJECT_NAME}/${EXPERIMENT_NAME}}"

cmd=(
  python3 -m verl.experimental.star_ppo.main_ppo
  --config-name "${CONFIG_NAME}"
  "trainer.project_name=${PROJECT_NAME}"
  "trainer.experiment_name=${EXPERIMENT_NAME}"
  "trainer.nnodes=${NNODES}"
  "trainer.n_gpus_per_node=${GPUS_PER_NODE}"
  "trainer.logger=[console,wandb]"
  "trainer.val_before_train=${VAL_BEFORE_TRAIN}"
  "trainer.test_freq=${TEST_FREQ}"
  "trainer.save_freq=${SAVE_FREQ}"
  "trainer.total_epochs=${TOTAL_EPOCHS}"
  "trainer.default_local_dir=${CKPTS_DIR}"
  "trainer.llm_engines.0.model_path=${SOLVER_MODEL_PATH:-${AGENT_MODEL_PATH}}"
  "trainer.llm_engines.1.model_path=${VERIFIER_MODEL_PATH:-${AGENT_MODEL_PATH}}"
  "trainer.llm_engines.2.model_path=${REFINER_MODEL_PATH:-${AGENT_MODEL_PATH}}"
  "trainer.llm_engines.3.model_path=${FINALIZER_MODEL_PATH:-${AGENT_MODEL_PATH}}"
  "trainer.llm_engines.0.n_gpus_per_node=${SOLVER_GPUS_PER_NODE:-${AGENT_GPUS_PER_NODE}}"
  "trainer.llm_engines.1.n_gpus_per_node=${VERIFIER_GPUS_PER_NODE:-${AGENT_GPUS_PER_NODE}}"
  "trainer.llm_engines.2.n_gpus_per_node=${REFINER_GPUS_PER_NODE:-${AGENT_GPUS_PER_NODE}}"
  "trainer.llm_engines.3.n_gpus_per_node=${FINALIZER_GPUS_PER_NODE:-${AGENT_GPUS_PER_NODE}}"
  "actor_rollout_ref.model.path=${SOLVER_MODEL_PATH:-${AGENT_MODEL_PATH}}"
  "data.train_files=${TRAIN_JSONL}"
  "data.val_files=${VAL_FILES}"
  "data.gen_batch_size=${GEN_BATCH_SIZE}"
  "data.train_batch_size=${GEN_BATCH_SIZE}"
  "data.val_batch_size=${VAL_BATCH_SIZE}"
  "actor_rollout_ref.actor.ppo_mini_batch_size=${ACTOR_PPO_MINI_BATCH_SIZE}"
  "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU}"
  "critic.ppo_micro_batch_size_per_gpu=${CRITIC_PPO_MICRO_BATCH_SIZE_PER_GPU}"
  "actor_rollout_ref.rollout.prompt_length=${ROLLOUT_PROMPT_LENGTH}"
  "actor_rollout_ref.rollout.response_length=${ROLLOUT_RESPONSE_LENGTH}"
  "actor_rollout_ref.rollout.max_model_len=${ROLLOUT_MAX_MODEL_LEN}"
  "actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE}"
  "actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION}"
)

if [[ -n "${TOTAL_TRAINING_STEPS}" ]]; then
  cmd+=("trainer.total_training_steps=${TOTAL_TRAINING_STEPS}")
fi

"${cmd[@]}" "$@"
