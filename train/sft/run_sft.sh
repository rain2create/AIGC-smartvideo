#!/bin/bash
# run_sft.sh  —  SFT 训练入口脚本
#
# 用法：
#   bash train/sft/run_sft.sh           # 前台运行
#   bash train/sft/run_sft.sh --bg      # 后台运行（nohup）
#
# 监控：
#   tail -f train/logs/qwen_sft/train_rank0.log
#   tail -f train/logs/qwen_sft/train_rank1.log
#   tensorboard --logdir train/logs/qwen_sft

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="$REPO_ROOT/train/sft/sft_config.yaml"
LOG_DIR="$REPO_ROOT/train/logs/qwen_sft"

mkdir -p "$LOG_DIR"

CMD="torchrun \
    --nproc_per_node=2 \
    --master_port=29502 \
    $REPO_ROOT/train/sft/sft_train.py \
    --config $CONFIG"

echo "========================================"
echo "  SFT Training  $(date '+%Y-%m-%d %H:%M:%S')"
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
    pgrep -f "sft_train.py" | while read pid; do echo "  $pid"; done
    echo ""
    echo "停止训练:  pkill -f sft_train.py"
    echo "查看进度:"
    echo "  tail -f $LOG_DIR/train_rank0.log"
    echo "  tail -f $LOG_DIR/train_rank1.log"
else
    eval "$CMD"
fi
