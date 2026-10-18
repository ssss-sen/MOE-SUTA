#!/bin/bash
# 样本级实现上线后的验证 + 补种子（标准口径：样本级路由 + 无窗口 + lb_coef=0 + em_coef=0.5）
cd "$(dirname "$0")/.." || exit 1
LOG=sl_batch.log; : > $LOG
F='WER=|per-domain|Traceback|Error|SAMPLE_LEVEL|FUSE_PER_MODEL|FAST_MASK|\[RG\]'
say() { echo "$@" | tee -a $LOG; }
exec 2>/tmp/run_sl.lock; flock -n 2 || { echo "[SL] already running"; exit 1; }

BASE="MOE_FUSE_MODE MOE_FAST_FUSE MOE_FAST_TOPK1 MOE_DYN_RESET MOE_DYN_K MOE_DYN_WIN MOE_DYN_PERSIST MOE_DYN_DEV MOE_DYN_DEV_LO MOE_DYN_DWELL MOE_DYN_TRACE MOE_DYN_PARTIAL"
BAD=""
for v in $(grep -oE "MOE_[A-Z0-9_]+=" "$0" | tr -d '=' | sort -u); do
  case " $BASE " in *" $v "*) ;; *) BAD="$BAD $v";; esac
done
[ -n "$BAD" ] && { say "[SL] 静态闸门失败：未放行 env:$BAD"; exit 2; }
say "[SL] 静态闸门通过（只有基准 env；无窗口/无 LB/无慢路 lr）"

N15=51,56,59,64,67,72,75,80,83,88,91,96,99,104,107
ES9="expert_schedule=1-8:9,11-48:4 normal_layer=$N15 route_granularity=sample"
FUSE="MOE_FUSE_MODE=prob MOE_FAST_FUSE=topk MOE_FAST_TOPK1=1"
S1="MOE_DYN_RESET=1 MOE_DYN_K=100 MOE_DYN_WIN=6 MOE_DYN_PERSIST=4 MOE_DYN_DEV=1.3 MOE_DYN_DEV_LO=1.0 MOE_DYN_DWELL=0 MOE_DYN_TRACE=1"
SC="config/strategy/dsuta.yaml config/strategy/5step.yaml"
EXP1="MoE Optimizable Params:  42131056"
EXP2="[FUSE_PER_MODEL] fast=topk | slow=prob"
EXP3="[FAST_MASK] enable: fast path updates SHARED + top-1 expert only"
EXP4="[SLOWCOMMIT] enable win=0 uf=2 after=5 slide=0 sbatch=1"
EXP5="[RG] route_granularity=sample"
EM50="'em_coef': 0.5, 'temp': 2.5, 'blank_coef': 0.5, 'nb_threshold': 0.2, 'lb_coef': 0.0"

say "[SL] START $(date '+%F %T')"
while pgrep -f 'md_eva[l]' >/dev/null; do say "[SL] waiting ... $(date '+%T')"; sleep 30; done

done_already() { [ -f "results/md_eval/moe_dsuta/$1/$2/summary.json" ]; }

run_task() {   # $1 name  $2 task  $3 seed
  local n0=$(wc -l < "$LOG") seg n_sl
  say ""
  say "########## $1 | $2 | seed=$3 | em_coef=0.5 + 样本级路由 | $(date '+%F %T') ##########"
  env $FUSE $S1 MOE_DYN_PARTIAL=shared \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      flock -x /tmp/ourmethod.lock \
      python md_eval.py -s moe_dsuta -t "$2" -n "$1" \
        --config config/system/moe-conv6.yaml --strategy_config $SC \
        --set-moetta $ES9 --set lb_coef=0.0 --set em_coef=0.5 --seed $3 2>&1 | tee -a "$LOG" | grep -E "$F"
  seg=$(tail -n +$((n0+1)) "$LOG"); echo "$seg" > /tmp/_sl_seg.txt
  n_sl=$(echo "$seg" | grep -c '\[SAMPLE_LEVEL\]')
  say "[SL] $1 结束 $(date '+%T')"
  say "$(echo "$seg" | grep -E 'WER=|per-domain' | tail -n 4)"
  say "[SL] $1 | $(echo "$seg" | grep -o 'time=[0-9.]*s' | tail -1) | $(echo "$seg" | grep -o 'adapt=[0-9]*' | tail -1) | SAMPLE_LEVEL=$n_sl | OOM=$(echo "$seg" | grep -ci 'out of memory')"
  for s in "$EXP1" "$EXP2" "$EXP3" "$EXP4" "$EXP5" "$EM50"; do
    grep -qF "$s" /tmp/_sl_seg.txt || { say "[SL] $1 指纹/生效值缺失: $s => 停"; exit 3; }
  done
  [ "$n_sl" -ge 1 ] || { say "[SL] $1 没有 [SAMPLE_LEVEL] => 新路径没生效 => 停"; exit 3; }
  echo "$seg" | grep -qi 'out of memory' && { say "[SL] $1 OOM => 停"; exit 3; }
  say "[SL] $1 通过（新路径 $n_sl 个层 + 口径指纹全对）"
}

run_or_skip() {   # $1 name  $2 task  $3 seed
  if done_already "$1" "$2"; then say "[SL] $1 ($2) 已有 summary.json，跳过"; else run_task "$1" "$2" "$3"; fi
}

run_or_skip A_sl_em50s_hard      md_hard 666
run_or_skip A_sl_em50s_easy      md_easy 666
run_or_skip A_sl_em50s_easy_s668 md_easy 668
run_or_skip A_sl_em50s_hard_s667 md_hard 667

say ""
say "[SL] DONE $(date '+%F %T')"
grep -E 'WER=|\[SL\]' $LOG | tail -20
