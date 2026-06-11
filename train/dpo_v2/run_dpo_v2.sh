#!/bin/bash
# run_dpo_v2.sh  —  DPO v2 (hard-mining) 训练入口
#
# 用法：
#   bash train/dpo_v2/run_dpo_v2.sh           # 前台运行
#   bash train/dpo_v2/run_dpo_v2.sh --bg      # 后台运行（nohup）
#
# 监控：
#   tail -f train/logs/qwen_dpo_v2/train_rank0.log
#   tail -f train/logs/qwen_dpo_v2/main.log         # nohup 输出

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="$REPO_ROOT/train/dpo_v2/dpo_v2_config.yaml"
LOG_DIR="$REPO_ROOT/train/logs/qwen_dpo_v2"

mkdir -p "$LOG_DIR"

CMD="torchrun \
    --nproc_per_node=2 \
    --master_port=29503 \
    $REPO_ROOT/train/dpo/dpo_train.py \
    --config $CONFIG"

echo "========================================"
echo "  DPO v2 Training  $(date '+%Y-%m-%d %H:%M:%S')"
echo "  config : $CONFIG"
echo "  logs   : $LOG_DIR/train_rank{0,1}.log"
echo "  tb     : tensorboard --logdir $LOG_DIR"
echo "========================================"

if [[ "$1" == "--bg" ]]; then
    nohup bash -c "$CMD" > "$LOG_DIR/main.log" 2>&1 &
    BGPID=$!
    echo "bash PID: $BGPID"
    sleep 2
    echo "训练进程（用这些 PID 来 kill）："
    pgrep -f "dpo_train.py.*dpo_v2" | while read pid; do echo "  $pid"; done
    echo ""
    echo "停止训练:  pkill -f 'dpo_train.py.*dpo_v2'"
    echo "查看进度:"
    echo "  tail -f $LOG_DIR/train_rank0.log"
    echo "  tail -f $LOG_DIR/main.log"
else
    eval "$CMD"
fi
