#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
TASK_FILE="${1:-task2.txt}"
MAX_SEGMENTS="${MAX_SEGMENTS:-0}"
MODEL="${MODEL:-google/gemini-3-pro-preview}"
VARIANT="${JUDGE_VARIANT:-v4}"
LOG_DIR="$ROOT_DIR/logs"

mkdir -p "$LOG_DIR"
cd "$ROOT_DIR"

if [[ "${RUN_BATCH_JUDGE_BG:-0}" != "1" ]]; then
  TS="$(date +%m%d_%H%M%S)"
  LOG_FILE="$LOG_DIR/batch_judge_${TS}.log"
  PID_FILE="$LOG_DIR/batch_judge_${TS}.pid"

  RUN_BATCH_JUDGE_BG=1 \
  LOG_FILE="$LOG_FILE" \
  PID_FILE="$PID_FILE" \
  TASK_FILE="$TASK_FILE" \
  MAX_SEGMENTS="$MAX_SEGMENTS" \
  MODEL="$MODEL" \
  bash "$0" "$TASK_FILE" >/dev/null 2>&1 &

  WRAPPER_PID=$!
  PYTHON_PID=""
  for _ in $(seq 1 40); do
    if [[ -s "$PID_FILE" ]]; then
      PYTHON_PID="$(tr -d '[:space:]' < "$PID_FILE")"
      [[ -n "$PYTHON_PID" ]] && break
    fi
    if ! kill -0 "$WRAPPER_PID" 2>/dev/null; then
      break
    fi
    sleep 0.25
  done
  : "${PYTHON_PID:=$WRAPPER_PID}"

  echo "[已启动] batch_judge 后台运行中"
  echo "PID: $PYTHON_PID"
  echo "Wrapper PID: $WRAPPER_PID"
  echo "日志: $LOG_FILE"
  exit 0
fi

LOG_FILE="${LOG_FILE:?LOG_FILE is required}"
PID_FILE="${PID_FILE:-}"

exec >>"$LOG_FILE" 2>&1

echo "[$(date '+%F %T')] ROOT_DIR=$ROOT_DIR"
echo "[$(date '+%F %T')] TASK_FILE=$TASK_FILE"
echo "[$(date '+%F %T')] MODEL=$MODEL"
echo "[$(date '+%F %T')] MAX_SEGMENTS=$MAX_SEGMENTS"
echo "[$(date '+%F %T')] VARIANT=$VARIANT"
echo "[$(date '+%F %T')] WRAPPER_PID=$$"

CMD=(python judge/batch_judge.py "$TASK_FILE" --model "$MODEL" --variant "$VARIANT")
if [[ "$MAX_SEGMENTS" != "0" ]]; then
  CMD+=(--max-segments "$MAX_SEGMENTS")
fi

echo "[$(date '+%F %T')] CMD=${CMD[*]}"
CHILD_PID=""

cleanup_pid_file() {
  if [[ -n "${PID_FILE:-}" && -f "$PID_FILE" ]]; then
    rm -f "$PID_FILE"
  fi
}

forward_signal() {
  local sig="$1"
  if [[ -n "$CHILD_PID" ]] && kill -0 "$CHILD_PID" 2>/dev/null; then
    kill "-$sig" "$CHILD_PID" 2>/dev/null || true
  fi
}

trap cleanup_pid_file EXIT
trap 'forward_signal TERM' TERM
trap 'forward_signal INT' INT

"${CMD[@]}" &
CHILD_PID=$!
echo "$CHILD_PID" > "$PID_FILE"
echo "[$(date '+%F %T')] PYTHON_PID=$CHILD_PID"
wait "$CHILD_PID"
STATUS=$?
echo "[$(date '+%F %T')] batch_judge finished exit_code=$STATUS"
exit "$STATUS"
