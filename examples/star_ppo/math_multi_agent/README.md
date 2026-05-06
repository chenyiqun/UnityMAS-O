# STAR PPO Math Multi-Agent Workflow

This workflow trains four non-shared math agents:

- `solver_agent` on `solver_llm`
- `verifier_agent` on `verifier_llm`
- `refiner_agent` on `refiner_llm`
- `finalizer_agent` on `finalizer_llm`

The default model path for all four agents is:

```bash
/mnt/tidal-alsh01/usr/chenyiqun/base_models/Qwen/Qwen2.5-7B-Instruct
```

The workflow is:

```text
Problem -> Solver -> Verifier -> Refiner -> Finalizer -> Final Answer
```

Reward for every trainable agent is:

```text
final answer accuracy in {0, 1} + format penalty in {0, -1}
```

Validation logs overall accuracy at:

```text
validation/workflow/math/final_acc
```

and per-dataset accuracy at paths like:

```text
validation/workflow/math/dataset/math-500/acc
validation/workflow/math/dataset/aime24/acc
validation/workflow/math/dataset/aime25/acc
validation/workflow/math/dataset/aime26/acc
validation/workflow/math/dataset/amc23/acc
```

Run training:

```bash
bash examples/star_ppo/math_multi_agent/run_star_math_train.sh
```

Run validation-only evaluation over multiple datasets:

```bash
VAL_FILES='[/path/to/MATH-500.jsonl,/path/to/aime24.jsonl,/path/to/aime25.jsonl,/path/to/aime26.jsonl,/path/to/amc23.jsonl]' \
bash examples/star_ppo/math_multi_agent/run_star_math_test.sh
```

Useful overrides:

- `AGENT_MODEL_PATH`
- `SOLVER_MODEL_PATH`, `VERIFIER_MODEL_PATH`, `REFINER_MODEL_PATH`, `FINALIZER_MODEL_PATH`
- `TRAIN_JSONL`
- `VAL_FILES`
- `NNODES`, `GPUS_PER_NODE`, `AGENT_GPUS_PER_NODE`
- `GEN_BATCH_SIZE`, `VAL_BATCH_SIZE`
- `TEST_FREQ`, `SAVE_FREQ`

Print complete validation examples:

```bash
STAR_VAL_DEBUG=true \
STAR_VAL_DEBUG_SAMPLE_COUNT=2 \
STAR_VAL_DEBUG_EVERY_N_BATCHES=1 \
STAR_VAL_DEBUG_MAX_CHARS=0 \
bash examples/star_ppo/math_multi_agent/run_star_math_test.sh
```

`STAR_VAL_DEBUG_MAX_CHARS=0` means no truncation. The printed block includes the
problem, ground truth, every agent's raw/parsed output, final answer, and final acc.
