#!/bin/bash
# run_awr_v2.sh  —  AWR v2 训练入口（从 DPO v1 继续，Linear AWR）
#
# 用法：
#   bash train/listwise_awr/run_awr_v2.sh           # 前台运行
#   bash train/listwise_awr/run_awr_v2.sh --bg      # 后台运行（nohup）
#
# 监控：
#   tail -f train/logs/qwen_awr_v2/train_rank0.log
#   tensorboard --logdir train/logs/qwen_awr_v2

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="$REPO_ROOT/train/listwise_awr/awr_v2_config.yaml"
LOG_DIR="$REPO_ROOT/train/logs/qwen_awr_v2"

mkdir -p "$LOG_DIR"

CMD="torchrun \
    --nproc_per_node=2 \
    --master_port=29502 \
    $REPO_ROOT/train/listwise_awr/awr_train.py \
    --config $CONFIG"

echo "========================================"
echo "  AWR v2 Training (Linear, from DPO v1)"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "  config : $CONFIG"
echo "  logs   : $LOG_DIR/train_rank{0,1}.log"
echo "  tb     : tensorboard --logdir $LOG_DIR"
echo "========================================"

if [[ "$1" == "--bg" ]]; then
    MAIN_LOG="$LOG_DIR/main_$(date +%Y%m%d_%H%M%S).log"
    nohup bash -c "$CMD" > "$MAIN_LOG" 2>&1 &
    BGPID=$!
    echo "bash PID: $BGPID"
    echo "主日志: $MAIN_LOG"
    sleep 2
    echo "训练进程（用这些 PID 来 kill）："
    pgrep -f "awr_train.py" | while read pid; do echo "  $pid"; done
    echo ""
    echo "停止训练:  pkill -f awr_train.py"
    echo "查看进度:"
    echo "  tail -f $LOG_DIR/train_rank0.log"
    echo "  tail -f $LOG_DIR/train_rank1.log"
else
    eval "$CMD"
fi
