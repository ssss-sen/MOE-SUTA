#!/bin/bash
# run_autodl.sh -- 在 AutoDL 上跑官方规模的 md_easy / md_hard（每域 500 条，共 2500 条）
# 用法:
#   bash run_autodl.sh md_easy            # 跑 md_easy 全矩阵
#   bash run_autodl.sh md_hard            # 跑 md_hard 全矩阵
#   REP=1 bash run_autodl.sh md_easy      # moe 只跑 1 遍（默认 REP=2）
# 在仓库根目录（含 run_benchmark.py）运行。
set -u
TASK=${1:-md_easy}
TAG=${2:-$(date +%m%d)}
REP=${REP:-2}
LOG="log_${TASK}_${TAG}.txt"

echo "TASK=$TASK TAG=$TAG REP=$REP LOG=$LOG" | tee -a "$LOG"

run () {
  echo -e "\n########## $* ##########" | tee -a "$LOG"
  python md_eval.py "$@" 2>&1 | tee -a "$LOG"
}

# ---------- 1) 基线（快，先把时间基准跑出来） ----------
run -s none  -t "$TASK" -n "${TAG}-none"
run -s csuta -t "$TASK" -n "${TAG}-csuta" --strategy_config config/strategy/1step.yaml
run -s dsuta -t "$TASK" -n "${TAG}-dsuta" --strategy_config config/strategy/dsuta.yaml config/strategy/5step.yaml
run -s suta  -t "$TASK" -n "${TAG}-suta"  --strategy_config config/strategy/10step.yaml

# ---------- 2) MoE-SUTA：lb_coef 0（= 无路由学习信号，EXP-F） vs 0.01（EXP-K 修复后） ----------
for LB in 0.0 0.01; do
  run -s moe_suta -t "$TASK" -n "${TAG}-moe-lb${LB}" \
      --config config/system/moe-suta.yaml \
      --strategy_config config/strategy/5step.yaml \
      --set lb_coef="$LB" --repeat "$REP"
done

# ---------- 3) 归因：随机路由 / 单专家（都在 lb=0.01 下） ----------
echo -e "\n########## moe + lb=0.01 + 随机路由 ##########" | tee -a "$LOG"
MOE_FORCE_RANDOM=1 python md_eval.py -s moe_suta -t "$TASK" -n "${TAG}-moe-lb0.01-rand" \
    --config config/system/moe-suta.yaml \
    --strategy_config config/strategy/5step.yaml \
    --set lb_coef=0.01 --repeat "$REP" 2>&1 | tee -a "$LOG"

run -s moe_suta -t "$TASK" -n "${TAG}-moe-lb0.01-E1" \
    --config config/system/moe-suta.yaml \
    --strategy_config config/strategy/5step.yaml \
    --set lb_coef=0.01 --set-moetta num_expert=1 --repeat "$REP"

echo -e "\n########## DONE: 结果见 results/md_eval/ 与 $LOG ##########" | tee -a "$LOG"
