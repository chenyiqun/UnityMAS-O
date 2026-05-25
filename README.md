# UnityMAS-O

UnityMAS-O 是一个基于 [verl](https://github.com/verl-project/verl) 改造的 LLM 多智能体强化学习优化框架。它把传统单策略 RL post-training 扩展到可配置的 multi-agent workflow：用户定义逻辑 agent、workflow 执行图、agent 到物理 LLM 的映射关系，以及面向节点、轮次或整条轨迹的奖励分配规则；框架负责异步执行 workflow、收集结构化轨迹、把奖励归因到对应 agent，再用 PPO 风格的训练流程更新每个物理 LLM。

这份 README 是本仓库的使用入口。原始 Verl 能力仍然保留；UnityMAS-O 的新增代码主要位于 `verl/experimental/star_ppo/` 和 `examples/star_ppo/`。

<a href="docs/assets/unitymas-o/unity-framework.pdf">
  <img src="docs/assets/unitymas-o/unity-framework.png" alt="UnityMAS-O agent framework" width="100%">
</a>

## 核心思想

UnityMAS-O 的目标不是只训练一个最终回答模型，而是优化整个 LLM-based multi-agent system。一个任务样本会被展开成多步结构化轨迹，例如：

```text
QA/search:  plan -> search -> retrieve(tool) -> summarize -> update -> answer
code:       planner -> coder -> verifier(tool) -> reflector -> planner -> ...
math:       solver -> verifier -> refiner -> finalizer
```

框架把多智能体训练拆成四个显式对象：

- **Logical agents**：workflow 中的角色，例如 planner、searcher、summarizer、coder、reflector、answerer。
- **Agent-LLM mapping**：逻辑 agent 到物理模型的映射。可以全共享、全分离，也可以部分共享。
- **Workflow trace**：每个样本执行时产生的结构化轨迹，包括 agent 输出、工具结果、状态更新、控制流和调试信息。
- **Reward allocator**：把最终指标、局部格式奖励、轮次增益或工具反馈分配回具体 agent invocation。

这种设计允许同一个 workflow 在不同参数共享方案下训练。例如 M-ASK 可以用 4 个独立模型组训练，也可以让所有角色共享一个 `shared_agent_llm`；代码 workflow 可以让 planner、coder、reflector 使用三个独立模型组，也可以切换到 shared LLM 配置。

## 系统架构

<a href="docs/assets/unitymas-o/system.pdf">
  <img src="docs/assets/unitymas-o/system.png" alt="UnityMAS-O distributed training architecture" width="100%">
</a>

运行时采用 Ray star topology：

- 中央 controller 负责 workflow 调度、工具调用、状态转移、reward assembly 和训练协调。
- 每个物理 LLM 对应一个 model-local worker group，负责 rollout、fat tensor 缓存、ready batch 构造、advantage/logprob/value 计算和 PPO update。
- controller 只传递轻量的 action/output/metadata；大 tensor 保留在生成它的 worker group 内，降低跨节点通信成本。
- `phi: logical agent -> model_id` 决定奖励和 rollout 数据最终进入哪个物理模型的训练 buffer。

## 代码结构

```text
verl/experimental/star_ppo/
  main_ppo.py                         # UnityMAS-O / STAR PPO 入口
  ray_trainer.py                      # 多 engine Ray trainer、workflow 执行、reward commit、PPO update
  star_fsdp_workers.py                # detach actor / async rollout / critic / reward worker
  trajectory_buffer.py                # model-local trajectory buffer
  types.py                            # engine spec 等基础类型

  config/                             # Hydra 配置
    star_ppo_trainer.yaml             # 通用 STAR PPO 基础配置
    star_code_iterative_plan_code_reflect_trainer.yaml
    star_code_iterative_plan_code_reflect_shared_llm_trainer.yaml
    star_iterative_plan_search_summary_update_answer_f1_trainer.yaml
    star_iterative_plan_search_summary_update_answer_f1_shared_llm_trainer.yaml
    star_math_solver_verifier_refiner_finalizer_*.yaml
    star_query_decompose_retrieve*_trainer.yaml

  workflows/                          # workflow runner 插件
    base.py                           # WorkflowRunner 接口
    schema.py                         # WorkflowTrace / WorkflowExecutionRecord / RewardAssignment
    mask_iterative_workflow.py        # M-ASK iterative search workflow
    code_iterative_workflow.py        # plan-code-reflect code workflow
    math_multi_agent_workflow.py      # math multi-agent workflow
    graph_workflow.py                 # graph-style workflow 支持

  reward_allocators/                  # reward 分配插件
    base.py
    mask_turn_level.py
    code_turn_level.py
    math_final_answer.py

  tools/                              # 工具接口
    retriever.py                      # retrieval API pool
    code_verifier.py                  # 本地代码执行/verifier
    math_answer.py
    prompt_builders.py

  datasets/
    code_jsonl_dataset.py             # code JSON/JSONL/Parquet adapter
    math_jsonl_dataset.py             # math JSON/JSONL/Parquet adapter

examples/star_ppo/
  common/
    run_per_node.sh                   # 每个节点启动 Ray head/worker，并在 rank0 启动训练
    run_per_node_background.sh        # 后台启动，日志写入 logs/star_ppo/
    run_ip_list.sh                    # 按 IP 列表启动
    launch_ip_list_background.sh
    launch_kubectl_exec_background.sh
  code_iterative_workflow/README.md
  mask_iterative_workflow/README.md
  math_multi_agent/README.md
```

## 环境准备

推荐使用独立的 `verl` conda 环境。下面是当前实验环境使用过的一套安装流程：

```bash
cd /path/to/UnityMAS-O

# 创建 Python 3.10 环境。前面的 printf 用于自动回答 conda 的交互式确认。
printf 'a\na\nyes\n' | conda create -n verl python=3.10
conda activate verl

# 安装 vLLM / SGLang / Megatron-Core 相关依赖。
bash scripts/install_vllm_sglang_mcore_0.7.sh

# 以 editable 方式安装本仓库，便于直接修改代码后运行。
pip install --no-deps -e .

# 版本固定。numpy 2.x、Transformers/TRL 的不同版本可能影响 Verl/vLLM 兼容性。
pip install "numpy<2.0"
pip uninstall transformers -y
pip install transformers==4.57 --no-cache-dir
pip uninstall -y trl
pip install "trl==0.26.2"

# 可选：远程调试用。
pip install debugpy==1.8.0
```

常用依赖来自 Verl、PyTorch、Ray、vLLM/SGLang、Transformers、Hydra/OmegaConf、datasets 等。不同集群镜像可能已经内置部分依赖；如果你在已有环境上安装，建议仍然检查上面几个关键版本 pin。

启动前建议清理旧 Ray 进程和旧 Python worker：

```bash
ray stop --force >/dev/null 2>&1 || true
pkill -9 -f "/miniconda3/envs/verl/bin/python3.10" || true
```

如果使用 wandb，请通过环境变量传入：

```bash
export WANDB_API_KEY="..."
export WANDB_ENTITY="..."
```

## 多节点启动方式

通用脚本是 `examples/star_ppo/common/run_per_node_background.sh`。需要在每个节点执行一次：

- `HEAD_IP`：rank0 节点 IP，所有节点保持一致。
- `WORLD_SIZE`：总节点数。
- `RANK`：当前节点 rank。head 节点为 `0`，其他节点依次为 `1..WORLD_SIZE-1`。
- `CONFIG_NAME`：选择要运行的 workflow 配置。
- 其他环境变量用于指定模型、数据、batch size、rollout、timeout、debug 等。

rank0 会启动 Ray head，等待所有节点加入后启动训练；非 rank0 节点只启动 Ray worker 并 block。

最小形态：

```bash
RANK=0 HEAD_IP=10.0.0.1 WORLD_SIZE=4 \
CONFIG_NAME=star_iterative_plan_search_summary_update_answer_f1_trainer \
bash examples/star_ppo/common/run_per_node_background.sh
```

worker 节点：

```bash
RANK=1 HEAD_IP=10.0.0.1 WORLD_SIZE=4 \
CONFIG_NAME=star_iterative_plan_search_summary_update_answer_f1_trainer \
bash examples/star_ppo/common/run_per_node_background.sh
```

日志默认写到：

```text
logs/star_ppo/run_rank<rank>_<timestamp>.log
```

## 支持的主要 workflow

<a href="docs/assets/unitymas-o/workflow.pdf">
  <img src="docs/assets/unitymas-o/workflow.png" alt="UnityMAS-O workflow templates" width="100%">
</a>

| Workflow | 配置 | 逻辑 agent | 典型 reward |
| --- | --- | --- | --- |
| Reflective Code | `star_code_iterative_plan_code_reflect_trainer` | planner, coder, reflector；verifier 是工具 | 第 0 轮使用 verifier pass score，后续轮次使用 pass-score delta；叠加格式奖励 |
| Reflective Code shared | `star_code_iterative_plan_code_reflect_shared_llm_trainer` | planner/coder/reflector 共享一个物理 LLM | 同上 |
| M-ASK iterative search | `star_iterative_plan_search_summary_update_answer_f1_trainer` | planning/answer 共享 reasoning LLM，search/summary/update 独立 | planning/answer 使用 absolute F1；search/summary/update 使用 F1 delta |
| M-ASK shared | `star_iterative_plan_search_summary_update_answer_f1_shared_llm_trainer` | 所有 search workflow 角色共享一个物理 LLM | 同上 |
| Math multi-agent | `star_math_solver_verifier_refiner_finalizer_trainer` | solver, verifier, refiner, finalizer | final-answer accuracy + format reward |
| Query decomposition RAG | `star_query_decompose_retrieve_answer_f1_trainer` 等 | query decomposer, answerer, evidence/summarizer 可选 | final-answer F1 + node-level format reward |

## 示例 1：代码生成 reflective workflow

这个配置训练三个非共享 LLM agent：

- `planner_agent` -> `planner_llm`
- `coder_agent` -> `coder_llm`
- `reflection_agent` -> `reflection_llm`

数据可以是 JSON、JSONL 或 Parquet。常见字段：

```json
{
  "uid": "example/0",
  "source": "codeforces",
  "problem": "problem statement ...",
  "starter_code": "",
  "tests": "[{\"input\":\"1\\n2 1\",\"output\":\"1 2\"}]"
}
```

三节点示例：在三个节点上分别把 `RANK` 改成 `0`、`1`、`2`，其余参数保持一致。

```bash
cd /path/to/UnityMAS-O
conda activate verl
ray stop --force >/dev/null 2>&1 || true
pkill -9 -f "/miniconda3/envs/verl/bin/python3.10" || true

RANK=0 HEAD_IP=10.0.0.1 WORLD_SIZE=3 \
CONFIG_NAME=star_code_iterative_plan_code_reflect_trainer \
PROJECT_NAME="STAR-Code" \
EXPERIMENT_NAME="deepcoder_marti_iterative_plan_code_reflect_3xQwen3_4B_no_think_sp4" \
TRAIN_JSONL="/path/to/train.jsonl" \
VAL_JSONL="/path/to/test.jsonl" \
AGENT_MODEL_PATH="/path/to/Qwen3-4B" \
PLANNER_MODEL_PATH="/path/to/Qwen3-4B" \
CODER_MODEL_PATH="/path/to/Qwen3-4B" \
REFLECTION_MODEL_PATH="/path/to/Qwen3-4B" \
ACTOR_MODEL_PATH="/path/to/Qwen3-4B" \
QWEN_ENABLE_THINKING=false \
GEN_BATCH_SIZE=64 \
VAL_BATCH_SIZE=64 \
VAL_MAX_BATCHES=-1 \
VAL_BEFORE_TRAIN=true \
STAR_MAX_INFLIGHT_QUERIES=64 \
STAR_MAX_PARALLEL_ROLLOUTS_PER_MODEL=64 \
STAR_LLM_MICROBATCH_MAX_SIZE=64 \
STAR_LLM_MICROBATCH_MAX_WAIT_MS=1000 \
ACTOR_PPO_MINI_BATCH_SIZE=64 \
ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU=1 \
CRITIC_PPO_MICRO_BATCH_SIZE_PER_GPU=1 \
ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU=1 \
REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU=1 \
ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=4 \
ROLLOUT_GPU_MEMORY_UTILIZATION=0.40 \
ROLLOUT_PROMPT_LENGTH=8192 \
ROLLOUT_RESPONSE_LENGTH=2048 \
ROLLOUT_MAX_MODEL_LEN=10240 \
ROLLOUT_MAX_NUM_SEQS=64 \
DATA_MAX_PROMPT_LENGTH=8192 \
STAR_PER_INFER_PROMPT_MAX_TOKENS=7680 \
CODE_MAX_TURNS=3 \
CODE_STOP_ON_ALL_PASSED=true \
CODE_VERIFY_TIMEOUT_SECONDS=1.0 \
CODE_VERIFY_DEFAULT_CHECKER_TYPE=auto \
CODE_VERIFY_MAX_TESTS_PER_EXAMPLE=8 \
CODE_VERIFIER_FAIL_OPEN=false \
STAR_QUERY_TIMEOUT_SECONDS=420 \
STAR_WORKFLOW_BATCH_TIMEOUT_SECONDS=900 \
STAR_RAY_GET_TIMEOUT_SECONDS=600 \
STAR_WORKER_CALL_TIMEOUT_SECONDS=600 \
STAR_LLM_TIMEOUT_SECONDS=900 \
STAR_VAL_PROGRESS_EVERY=1 \
STAR_WORKFLOW_DEBUG=true \
STAR_WORKFLOW_DEBUG_EVERY_N_BATCHES=10 \
STAR_WORKFLOW_DEBUG_SAMPLE_INDEX=0 \
STAR_WORKFLOW_DEBUG_MAX_CHARS=4000 \
STAR_VAL_DEBUG=true \
STAR_VAL_DEBUG_MAX_CHARS=4000 \
STAR_TOOL_TIMEOUT_SECONDS=0 \
bash examples/star_ppo/common/run_per_node_background.sh \
  actor_rollout_ref.model.use_remove_padding=true \
  critic.model.use_remove_padding=true \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=4 \
  actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=4 \
  actor_rollout_ref.ref.ulysses_sequence_parallel_size=4 \
  actor_rollout_ref.ref.fsdp_config.ulysses_sequence_parallel_size=4 \
  critic.ulysses_sequence_parallel_size=4 \
  critic.model.fsdp_config.ulysses_sequence_parallel_size=4
```

关键开关：

- `CODE_MAX_TURNS`：最多 plan-code-verify-reflect 轮数。
- `CODE_STOP_ON_ALL_PASSED`：所有 verifier tests 通过后提前停止。
- `CODE_VERIFY_TIMEOUT_SECONDS`：单次代码执行超时。
- `CODE_VERIFIER_FAIL_OPEN=false`：verifier 异常时是否放行。训练代码任务通常建议保持 `false`。
- `STAR_PER_INFER_PROMPT_MAX_TOKENS`：单次 agent prompt 的截断上限。

## 示例 2：M-ASK iterative search，4 个模型组

这个配置包含 5 个逻辑 agent、4 个物理 LLM：

- `planning_agent` 和 `answer_agent` 共享 `reasoning_agent_llm`
- `search_agent` 使用独立 LLM
- `summary_agent` 使用独立 LLM
- `update_agent` 使用独立 LLM

数据默认来自：

```text
DATASET_ROOT/<DATASET_NAME>/train_verl.parquet
DATASET_ROOT/<DATASET_NAME>/test_verl.parquet
```

可通过 `TRAIN_PARQUET` 和 `VAL_PARQUET` 覆盖。

四节点示例：在四个节点上分别把 `RANK` 改成 `0`、`1`、`2`、`3`，其余参数保持一致。

```bash
cd /path/to/UnityMAS-O
conda activate verl
ray stop --force >/dev/null 2>&1 || true
pkill -9 -f "/miniconda3/envs/verl/bin/python3.10" || true

RANK=0 HEAD_IP=10.0.0.1 WORLD_SIZE=4 \
CONFIG_NAME=star_iterative_plan_search_summary_update_answer_f1_trainer \
DATASET_NAME="hotpotqa" \
STAR_RETRIEVER_RANDOM_ENDPOINT=true \
RETRIEVAL_API_URLS_JSON='["http://host0:8000/retrieve","http://host0:8001/retrieve"]' \
PROJECT_NAME="M-ASK" \
EXPERIMENT_NAME="hotpotqa_M-ASK_f1_4x7B" \
REASONING_MODEL_PATH="/path/to/Qwen2.5-7B-Instruct" \
SEARCH_MODEL_PATH="/path/to/Qwen2.5-7B-Instruct" \
SUMMARY_MODEL_PATH="/path/to/Qwen2.5-7B-Instruct" \
UPDATE_MODEL_PATH="/path/to/Qwen2.5-7B-Instruct" \
GEN_BATCH_SIZE=128 \
STAR_MAX_INFLIGHT_QUERIES=128 \
STAR_MAX_PARALLEL_ROLLOUTS_PER_MODEL=32 \
ACTOR_PPO_MINI_BATCH_SIZE=128 \
ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU=1 \
CRITIC_PPO_MICRO_BATCH_SIZE_PER_GPU=1 \
ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU=1 \
REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU=1 \
ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=1 \
ROLLOUT_GPU_MEMORY_UTILIZATION=0.20 \
MASK_MAX_TURNS=3 \
MASK_STOP_ON_SEARCH_END=true \
STAR_QUERY_TIMEOUT_SECONDS=600 \
STAR_WORKFLOW_BATCH_TIMEOUT_SECONDS=900 \
STAR_RAY_GET_TIMEOUT_SECONDS=300 \
STAR_WORKER_CALL_TIMEOUT_SECONDS=300 \
STAR_LLM_TIMEOUT_SECONDS=300 \
STAR_VAL_PROGRESS_EVERY=1 \
STAR_WORKFLOW_DEBUG=true \
STAR_WORKFLOW_DEBUG_EVERY_N_BATCHES=10 \
STAR_WORKFLOW_DEBUG_SAMPLE_INDEX=0 \
STAR_WORKFLOW_DEBUG_MAX_CHARS=160 \
bash examples/star_ppo/common/run_per_node_background.sh
```

M-ASK reward 分配：

- planning agent：初始答案 `a0` 的 absolute F1。
- answer agent：每轮临时答案 `at` 的 absolute F1。
- search/summary/update：共享 `F1(at) - F1(at-1)` 的 marginal improvement。
- search 输出 `<end>` 时，该 search step 的 task reward 为 0。

## 示例 3：M-ASK shared LLM，单模型组

该配置把 planning/search/summary/update/answer 都映射到一个 `shared_agent_llm`，适合研究参数共享、节省资源或做小模型快速实验。

```bash
cd /path/to/UnityMAS-O
conda activate verl
ray stop --force >/dev/null 2>&1 || true
pkill -9 -f "/miniconda3/envs/verl/bin/python3.10" || true

RANK=0 HEAD_IP=10.0.0.1 WORLD_SIZE=1 \
CONFIG_NAME=star_iterative_plan_search_summary_update_answer_f1_shared_llm_trainer \
DATASET_NAME="hotpotqa" \
STAR_RETRIEVER_RANDOM_ENDPOINT=true \
RETRIEVAL_API_URLS_JSON='["http://host0:8000/retrieve","http://host0:8001/retrieve"]' \
PROJECT_NAME="M-ASK" \
EXPERIMENT_NAME="hotpotqa_M-ASK_f1_3B_shared" \
SHARED_MODEL_PATH="/path/to/Qwen2.5-3B-Instruct" \
GEN_BATCH_SIZE=128 \
STAR_MAX_INFLIGHT_QUERIES=128 \
STAR_MAX_PARALLEL_ROLLOUTS_PER_MODEL=32 \
ACTOR_PPO_MINI_BATCH_SIZE=48 \
ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU=6 \
CRITIC_PPO_MICRO_BATCH_SIZE_PER_GPU=6 \
ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU=6 \
REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU=6 \
ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=1 \
ROLLOUT_GPU_MEMORY_UTILIZATION=0.20 \
MASK_MAX_TURNS=3 \
MASK_STOP_ON_SEARCH_END=true \
STAR_QUERY_TIMEOUT_SECONDS=600 \
STAR_WORKFLOW_BATCH_TIMEOUT_SECONDS=900 \
STAR_RAY_GET_TIMEOUT_SECONDS=300 \
STAR_WORKER_CALL_TIMEOUT_SECONDS=300 \
STAR_LLM_TIMEOUT_SECONDS=300 \
STAR_VAL_PROGRESS_EVERY=1 \
STAR_WORKFLOW_DEBUG=true \
STAR_WORKFLOW_DEBUG_EVERY_N_BATCHES=10 \
STAR_WORKFLOW_DEBUG_SAMPLE_INDEX=0 \
STAR_WORKFLOW_DEBUG_MAX_CHARS=160 \
bash examples/star_ppo/common/run_per_node_background.sh
```

## 常用环境变量

| 变量 | 作用 |
| --- | --- |
| `CONFIG_NAME` | Hydra 配置名，不带 `.yaml` |
| `PROJECT_NAME`, `EXPERIMENT_NAME` | wandb/console tracking 名称 |
| `RANK`, `HEAD_IP`, `WORLD_SIZE` | 多节点 Ray 启动参数 |
| `GPUS_PER_NODE`, `CPUS_PER_NODE` | 每节点资源声明 |
| `AGENT_MODEL_PATH` | 多数配置的通用模型路径 fallback |
| `ACTOR_MODEL_PATH`, `ACTOR_TOKENIZER_PATH` | Verl actor/ref/critic 的基础模型与 tokenizer |
| `PLANNER_MODEL_PATH`, `CODER_MODEL_PATH`, `REFLECTION_MODEL_PATH` | code workflow 的三模型路径 |
| `REASONING_MODEL_PATH`, `SEARCH_MODEL_PATH`, `SUMMARY_MODEL_PATH`, `UPDATE_MODEL_PATH` | M-ASK 非共享配置的模型路径 |
| `SHARED_MODEL_PATH` | shared LLM 配置的模型路径 |
| `TRAIN_JSONL`, `VAL_JSONL` | code/math JSONL 数据路径 |
| `TRAIN_PARQUET`, `VAL_PARQUET`, `DATASET_ROOT`, `DATASET_NAME` | QA/search Verl-format parquet 数据路径 |
| `GEN_BATCH_SIZE`, `VAL_BATCH_SIZE` | rollout generation batch 和 validation batch |
| `ACTOR_PPO_MINI_BATCH_SIZE` | PPO mini-batch size |
| `ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU` | actor micro-batch size |
| `ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE` | vLLM tensor parallel size |
| `ROLLOUT_GPU_MEMORY_UTILIZATION` | vLLM 显存比例 |
| `ROLLOUT_PROMPT_LENGTH`, `ROLLOUT_RESPONSE_LENGTH`, `ROLLOUT_MAX_MODEL_LEN` | rollout 长度控制 |
| `STAR_MAX_INFLIGHT_QUERIES` | controller 并发执行的 query 数 |
| `STAR_MAX_PARALLEL_ROLLOUTS_PER_MODEL` | 每个 model_id 的并发 rollout 数 |
| `STAR_LLM_MICROBATCH_MAX_SIZE`, `STAR_LLM_MICROBATCH_MAX_WAIT_MS` | LLM 请求 microbatch 合并 |
| `STAR_QUERY_TIMEOUT_SECONDS` | 单 query workflow 超时 |
| `STAR_WORKFLOW_BATCH_TIMEOUT_SECONDS` | 一个 workflow batch 超时 |
| `STAR_RAY_GET_TIMEOUT_SECONDS`, `STAR_WORKER_CALL_TIMEOUT_SECONDS` | Ray/worker 调用超时 |
| `STAR_LLM_TIMEOUT_SECONDS`, `STAR_TOOL_TIMEOUT_SECONDS` | LLM/tool 调用超时 |
| `STAR_WORKFLOW_DEBUG`, `STAR_VAL_DEBUG` | 打印 workflow trace 调试信息 |

## 数据格式

### Code JSONL

`CodeJsonlDataset` 会读取 `problem/question/query` 作为题面，读取 `tests/test_cases/answer/label/reward_model/extra_info.*` 作为测试用例，读取 `starter_code/extra_info.starter_code` 作为 starter code。

最小示例：

```json
{"uid":"code/0","problem":"Write a function ...","starter_code":"","tests":[{"input":"1\n","output":"1\n"}]}
```

也支持把 `tests` 存成 JSON string。

### QA / Search Parquet

QA/search 配置默认使用 Verl-format parquet，常用字段包括：

- `question` / `query` / `problem` / `extra_info.question`
- `answer` / `ground_truth` / `extra_info.answer` / `reward_model.ground_truth`

检索工具通过 `RETRIEVAL_API_URLS_JSON` 提供一个或多个 HTTP endpoint。每个 endpoint 需要暴露 `/retrieve` 接口，返回 workflow runner 能消费的候选文档。

### Math JSONL

`MathJsonlDataset` 支持 JSON、JSONL、Parquet，读取 `question/problem/query` 作为题目，读取 `answer/ground_truth/target/reward_model.ground_truth/solution` 作为答案，并会自动推断 `data_source` 以便 validation 分数据集统计。

## 如何新增一个 workflow

新增任务通常只需要动三类文件：

1. 在 `verl/experimental/star_ppo/workflows/` 下实现一个 `WorkflowRunner`。
2. 在 `verl/experimental/star_ppo/reward_allocators/` 下实现一个 `RewardAllocator`。
3. 在 `verl/experimental/star_ppo/config/` 下增加一个 Hydra YAML，声明 `trainer.llm_engines`、agent 到 `model_id` 的映射、runner、reward allocator、工具和数据路径。

`WorkflowRunner` 的核心接口：

```python
class WorkflowRunner:
    async def run_batch(self, batch: DataProto, epoch: int) -> tuple[DataProto, dict[str, float]]:
        ...
```

`RewardAllocator` 的核心接口：

```python
class RewardAllocator:
    def allocate(self, trace: WorkflowTrace) -> tuple[list[RewardAssignment], dict[str, float]]:
        ...
```

关键约定：

- 每次 trainable LLM 调用都应产生一个 `WorkflowExecutionRecord`，并保留对应 thin/fat trajectory id。
- tool node 可以进入 trace，但不需要进入 PPO 训练 buffer。
- reward allocator 最终把 scalar reward 绑定到具体 `WorkflowExecutionRecord`。
- 只要 reward 能通过 `traj_id` commit 回对应 buffer，PPO trainer 不需要理解具体 workflow 语义。

## 调试与排障

查看后台日志：

```bash
tail -f logs/star_ppo/run_rank0_*.log
```

检查 Ray 集群：

```bash
ray status
```

常见问题：

- **非 head 节点未加入**：确认所有节点 `HEAD_IP` 一致，`RANK` 唯一，`WORLD_SIZE` 正确，端口 `6379/8265` 可达。
- **训练前卡住 waiting alive nodes**：某个 worker 没启动成功，先看对应 rank 日志。
- **vLLM OOM**：降低 `ROLLOUT_GPU_MEMORY_UTILIZATION`、`ROLLOUT_MAX_NUM_SEQS`、`ROLLOUT_MAX_NUM_BATCHED_TOKENS` 或 `STAR_MAX_PARALLEL_ROLLOUTS_PER_MODEL`。
- **prompt 过长**：降低 `STAR_PER_INFER_PROMPT_MAX_TOKENS`、`DATA_MAX_PROMPT_LENGTH`，或打开/调整 workflow 的 state truncation。
- **verifier 太慢**：调小 `CODE_VERIFY_MAX_TESTS_PER_EXAMPLE`，调大 `CODE_VERIFY_TIMEOUT_SECONDS`，检查测试用例大小限制。
- **检索不稳定**：增加 `RETRIEVAL_API_URLS_JSON` endpoint 数量，设置 `STAR_RETRIEVER_RANDOM_ENDPOINT=true`，并检查 retrieval server 超时。
- **debug 输出太多**：关闭 `STAR_WORKFLOW_DEBUG` / `STAR_VAL_DEBUG`，或调小 `STAR_WORKFLOW_DEBUG_MAX_CHARS`。

## 与 Verl 的关系

UnityMAS-O 复用了 Verl 的核心训练基础设施，包括 Ray 分布式执行、FSDP/FSDP2 worker、actor/ref/critic、vLLM rollout、PPO update、tracking 和 checkpoint 机制。在此基础上，本仓库新增了面向 multi-agent workflow 的 controller、routing、trace、reward allocation、model-local trajectory buffer 和多 LLM engine 配置。

如果只需要原始 Verl 单策略 PPO/GRPO/SFT 功能，仍可使用 `verl/trainer/` 和 `examples/ppo_trainer/` 等原始入口；如果要训练多 agent workflow，请使用 `verl.experimental.star_ppo.main_ppo` 和 `examples/star_ppo/` 下的脚本。

## 技术报告

本仓库对应的技术报告是：

```text
UnityMAS-O: A General RL Optimization Framework for LLM-Based Multi-Agent Systems
```

报告中的主要结论包括：UnityMAS-O 能把 QA/search、M-ASK iterative search、reflective code generation 等手写 workflow 转换成可训练的 MARL 问题；训练后在 QA F1、代码 all-passed rate 和代码验证轮数上均有明显改进，并支持参数共享与独立多模型组之间的可控对比。

README 中展示的预览图均由技术报告 LaTeX 中的 `\includegraphics` 原始 PDF 文件导出；点击图片可打开对应原始 PDF。

<a href="docs/assets/unitymas-o/qa_training_gains_dumbbell.pdf">
  <img src="docs/assets/unitymas-o/qa_training_gains_dumbbell.png" alt="QA training gains" width="100%">
</a>

<a href="docs/assets/unitymas-o/mask_3b_shared_vs_independent.pdf">
  <img src="docs/assets/unitymas-o/mask_3b_shared_vs_independent.png" alt="HotpotQA M-ASK shared vs independent" width="100%">
</a>

<a href="docs/assets/unitymas-o/code_train_test_curves.pdf">
  <img src="docs/assets/unitymas-o/code_train_test_curves.png" alt="Code training and held-out test curves" width="100%">
</a>

<a href="docs/assets/unitymas-o/code_test_used_turns.pdf">
  <img src="docs/assets/unitymas-o/code_test_used_turns.png" alt="Average verification turns on held-out code tasks" width="100%">
</a>
