#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$BASE_DIR/logs"
PYTHON_BIN="${QWEN_PYTHON:-python}"

# 可直接在这个脚本里手改的默认参数
DEFAULT_TOP_K="${QWEN_TOP_K:-5}"
DEFAULT_QUERY_VERSION="${QWEN_QUERY_VERSION:-Q1}"
DEFAULT_MAX_SEGMENTS="${QWEN_MAX_SEGMENTS:-0}"
DEFAULT_VIDEO_INDEX_VERSION="${QWEN_VIDEO_INDEX_VERSION:-V1}"
DEFAULT_RERANK_VERSION="${QWEN_RERANK_VERSION:-}"

# 解析 --video-index-version / --rerank-version
VIDEO_INDEX_VERSION="$DEFAULT_VIDEO_INDEX_VERSION"
RERANK_VERSION="$DEFAULT_RERANK_VERSION"
PARSED_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --video-index-version)
      VIDEO_INDEX_VERSION="${2:-}"
      shift 2
      ;;
    --rerank-version)
      RERANK_VERSION="${2:-}"
      shift 2
      ;;
    *)
      PARSED_ARGS+=("$1")
      shift
      ;;
  esac
done
set -- "${PARSED_ARGS[@]}"

mkdir -p "$LOG_DIR"
cd "$BASE_DIR"

if [[ $# -lt 1 ]]; then
  cat <<'EOF'
用法:
1. 单剧模式
   ./run_qwen.sh [--foreground] <dj> <start> <end> [output] [top_k] [query_version] [max_segments]

   例:
   ./run_qwen.sh chongfan80 1 10 chongfan80_q1_final.mp4
   ./run_qwen.sh --foreground chongfan80 1 10 chongfan80_q1_final.mp4
   ./run_qwen.sh chongfan80 1 10 chongfan80_q3_final.mp4 5 Q3 0
   ./run_qwen.sh chongfan80 1 10 chongfan80_b1_rerank_q1_final.mp4 20 Q3 0 --video-index-version B1 --rerank-version Q1

2. 批量模式
   ./run_qwen.sh [--foreground] --batch tasks_qwen.txt [top_k] [query_version] [max_segments] --video-index-version B1

环境变量:
  QWEN_PYTHON   指定运行 python，默认使用当前环境里的 python
  QWEN_TOP_K    默认 top_k，未显式传参时生效
  QWEN_QUERY_VERSION   默认 query 版本，未显式传参时生效
  QWEN_MAX_SEGMENTS    默认 max_segments，未显式传参时生效
  QWEN_VIDEO_INDEX_VERSION  视频索引版本(V1/B1)，默认 V1
  QWEN_RERANK_VERSION   Reranker 版本(Q1/Q2)，默认空=不启用
  QWEN_GPUS            批量模式按 GPU 队列限流，例如 0,1 表示两张卡各串行一个任务
  QWEN_BACKGROUND=0    强制前台运行；默认后台
EOF
  exit 1
fi

BACKGROUND=1
if [[ "${QWEN_BACKGROUND:-}" == "0" ]]; then
  BACKGROUND=0
fi

if [[ "${1:-}" == "--background" ]]; then
  BACKGROUND=1
  shift
fi

if [[ "${1:-}" == "--foreground" ]]; then
  BACKGROUND=0
  shift
fi

TS="$(date +%m%d_%H%M%S)"
if [[ "$1" == "--batch" ]]; then
  LOG_FILE="$LOG_DIR/run_qwen_batch_${TS}.log"
else
  DJ="$1"
  LOG_FILE="$LOG_DIR/run_qwen_${DJ}_${TS}.log"
fi

if [[ "$1" == "--batch" ]]; then
  TASK_FILE="$2"
  TOP_K="${3:-$DEFAULT_TOP_K}"
  QUERY_VERSION="${4:-$DEFAULT_QUERY_VERSION}"
  MAX_SEGMENTS="${5:-$DEFAULT_MAX_SEGMENTS}"
  CMD=("$PYTHON_BIN" run_mul_qwen.py --batch "$TASK_FILE" "$TOP_K" "$QUERY_VERSION" "$MAX_SEGMENTS" "$VIDEO_INDEX_VERSION")
else
  DJ="$1"
  START="${2:-1}"
  END="${3:-20}"
  QUERY_VERSION="${6:-$DEFAULT_QUERY_VERSION}"
  OUTPUT="${4:-${DJ}_$(echo "$QUERY_VERSION" | tr '[:upper:]' '[:lower:]')_final.mp4}"
  TOP_K="${5:-$DEFAULT_TOP_K}"
  MAX_SEGMENTS="${7:-$DEFAULT_MAX_SEGMENTS}"
  CMD=("$PYTHON_BIN" run_mul_qwen.py "$DJ" "$START" "$END" "$OUTPUT" "$TOP_K" "$QUERY_VERSION" "$MAX_SEGMENTS" "$VIDEO_INDEX_VERSION" "$RERANK_VERSION")
fi

if [[ "$BACKGROUND" == "1" ]]; then
  {
    echo "[启动] Qwen pipeline"
    echo "Run mode: background"
    echo "Python: $PYTHON_BIN"
    echo "Log: $LOG_FILE"
    echo "Default top_k: $DEFAULT_TOP_K"
    echo "Default query_version: $DEFAULT_QUERY_VERSION"
    echo "Default max_segments: $DEFAULT_MAX_SEGMENTS"
    echo "QWEN_GPUS: ${QWEN_GPUS:-}"
    echo "Rerank version: ${RERANK_VERSION:-}"
    echo "Cmd: ${CMD[*]}"
    echo "Timestamp: $(date '+%F %T')"
  } >> "$LOG_FILE"
  nohup env PYTHONUNBUFFERED=1 "${CMD[@]}" >> "$LOG_FILE" 2>&1 &
  PID=$!
  echo "[已启动] qwen pipeline 后台运行中"
  echo "PID: $PID"
  echo "日志: $LOG_FILE"
  # 批量模式：等 Python 写出 PID 文件后打印
  if [[ "$1" == "--batch" ]]; then
    PID_FILE="$(dirname "$TASK_FILE")/pids_$(basename "$TASK_FILE" .txt).txt"
    KILL_SCRIPT="$(dirname "$TASK_FILE")/kill_$(basename "$TASK_FILE" .txt).sh"
    for _ in $(seq 1 30); do
      if [[ -s "$PID_FILE" ]]; then
        echo ""
        echo "  === Python PID ==="
        while IFS= read -r p; do
          echo "    $p"
        done < "$PID_FILE"
        echo ""
        echo "  === 终止命令 ==="
        echo "  bash $(realpath "$KILL_SCRIPT")"
        break
      fi
      sleep 0.5
    done
  fi
  exit 0
fi

{
  echo "[启动] Qwen pipeline"
  echo "Run mode: foreground"
  echo "Python: $PYTHON_BIN"
  echo "Log: $LOG_FILE"
  echo "Default top_k: $DEFAULT_TOP_K"
  echo "Default query_version: $DEFAULT_QUERY_VERSION"
  echo "Default max_segments: $DEFAULT_MAX_SEGMENTS"
  echo "QWEN_GPUS: ${QWEN_GPUS:-}"
  echo "Rerank version: ${RERANK_VERSION:-}"
  echo "Cmd: ${CMD[*]}"
  echo "Timestamp: $(date '+%F %T')"
  env PYTHONUNBUFFERED=1 "${CMD[@]}"
} 2>&1 | tee "$LOG_FILE"
