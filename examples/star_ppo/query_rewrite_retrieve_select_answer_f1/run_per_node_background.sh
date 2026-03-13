#!/usr/bin/env bash
# =============================================================================
# Star PPO 多节点后台运行脚本
# 在每个 node 上分别执行此脚本，以后台方式启动训练并写入日志。
#
# 使用方法（统一命令，必须在所有 4 个 node 上分别执行）：
#   cd 项目根目录 && bash run_per_node_background.sh
# 注意：master 会等待 worker 加入，若一直显示 "alive nodes: 1/4"，说明 worker 未执行脚本
# 单节点测试：WORLD_SIZE=1 bash run_per_node_background.sh（仅在 master 执行）
#
# IP 与 RANK 对应关系（见下方 RANK0_IP ~ RANK3_IP）：
#   RANK0: 10.146.231.133 (master-0)
#   RANK1: 10.146.230.176 (worker-0)
#   RANK2: 10.144.172.97  (worker-1)
#   RANK3: 10.146.233.190 (worker-2)
#
# 或手动指定 RANK：
#   RANK=0 HEAD_IP=10.146.231.133 WORLD_SIZE=4 bash run_per_node_background.sh
#
# 执行完毕后会打印 PID 和日志路径，可用 tail -f <LOG> 查看输出。
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"

cd "${PROJECT_ROOT}"
mkdir -p logs/star_ppo

# --- IP -> RANK 映射（master-0=0, worker-0=1, worker-1=2, worker-2=3）---
RANK0_IP="10.146.231.133"   # master-0
RANK1_IP="10.146.230.176"   # worker-0
RANK2_IP="10.144.172.97"    # worker-1
RANK3_IP="10.146.233.190"   # worker-2
HEAD_IP="${RANK0_IP}"
WORLD_SIZE="${WORLD_SIZE:-4}"

if [[ -z "${RANK:-}" ]]; then
  MY_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
  [[ -z "${MY_IP}" ]] && MY_IP=$(hostname -i 2>/dev/null | awk '{print $1}')
  RANK=0
  found=false
  for r in 0 1 2 3; do
    eval "node_ip=\${RANK${r}_IP}"
    if [[ "${node_ip}" == "${MY_IP}" ]]; then
      RANK="${r}"
      found=true
      break
    fi
  done
  [[ "${found}" != "true" ]] && echo "[run_per_node_background] WARN: MY_IP=${MY_IP} 未匹配到 RANK0~3，使用 RANK=0"
  export RANK HEAD_IP WORLD_SIZE
  echo "[run_per_node_background] auto-detect: MY_IP=${MY_IP} -> RANK=${RANK}"
fi

RANK="${RANK:-0}"
TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_FILE:-logs/star_ppo/run_rank${RANK}_${TS}.log}"

echo "[run_per_node_background] Starting RANK=${RANK} in background, log=${LOG_FILE}"
nohup stdbuf -oL -eL bash "${SCRIPT_DIR}/run_per_node.sh" \
  > "${LOG_FILE}" 2>&1 < /dev/null &
PID=$!
echo "PID=${PID} LOG=${LOG_FILE}"
echo "  tail -f ${LOG_FILE}"
