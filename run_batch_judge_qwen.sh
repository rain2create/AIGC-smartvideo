#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$ROOT_DIR/logs"
PYTHON_BIN="${JUDGE_PYTHON:-python}"
MODEL_DEFAULT="${MODEL:-google/gemini-3-pro-preview}"

mkdir -p "$LOG_DIR"
cd "$ROOT_DIR"

usage() {
  cat <<'EOF'
用法：
  bash run_batch_judge_qwen.sh --dj chongfan80 --query-version Q1
  bash run_batch_judge_qwen.sh --dj chongfan80 --query-version q3 --max-segments 3
  bash run_batch_judge_qwen.sh --dj chongfan80 --query-version Q1 --report-only --run-id qwen_q1_20260424_101500
  bash run_batch_judge_qwen.sh --dj chongfan80 --query-version Q1 --structured-json text_video_match/cache/chongfan80/xxx_structured_qwen_q1.json

参数：
  --dj                剧名（必填）
  --query-version     Q1/Q3/...（必填，会自动转成 qwen_q1 / qwen_q3 这类 variant）
  --structured-json   显式指定待评估的输入 json；可传 final.json 或 *_structured*.json
  --run-id            显式指定 judge run_id
  --run-id-prefix     自动生成 run_id 时的前缀，默认使用 variant
  --model             Judge 模型
  --max-segments      最多处理几条文案
  --cache-dir         显式指定 cache/{dj} 目录
  --report-only       只生成报告，不重新评估
  --foreground        前台运行；默认后台运行并输出日志路径
EOF
}

DJ=""
QUERY_VERSION=""
STRUCTURED_JSON=""
RUN_ID=""
RUN_ID_PREFIX=""
MAX_SEGMENTS="0"
MODEL_NAME="$MODEL_DEFAULT"
CACHE_DIR=""
REPORT_ONLY="0"
FOREGROUND="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dj)
      DJ="${2:-}"
      shift 2
      ;;
    --query-version)
      QUERY_VERSION="${2:-}"
      shift 2
      ;;
    --structured-json)
      STRUCTURED_JSON="${2:-}"
      shift 2
      ;;
    --run-id)
      RUN_ID="${2:-}"
      shift 2
      ;;
    --run-id-prefix)
      RUN_ID_PREFIX="${2:-}"
      shift 2
      ;;
    --model)
      MODEL_NAME="${2:-}"
      shift 2
      ;;
    --max-segments)
      MAX_SEGMENTS="${2:-}"
      shift 2
      ;;
    --cache-dir)
      CACHE_DIR="${2:-}"
      shift 2
      ;;
    --report-only)
      REPORT_ONLY="1"
      shift
      ;;
    --foreground)
      FOREGROUND="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "未知参数: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$DJ" || -z "$QUERY_VERSION" ]]; then
  echo "--dj 和 --query-version 为必填项" >&2
  usage
  exit 1
fi

QUERY_VERSION_LOWER="$(echo "$QUERY_VERSION" | tr '[:upper:]' '[:lower:]')"
if [[ "$QUERY_VERSION_LOWER" != q* ]]; then
  QUERY_VERSION_LOWER="q${QUERY_VERSION_LOWER}"
fi
VARIANT="qwen_${QUERY_VERSION_LOWER}"

if [[ -z "$CACHE_DIR" ]]; then
  CACHE_DIR="$ROOT_DIR/text_video_match/cache/$DJ"
fi

if [[ -n "$STRUCTURED_JSON" ]]; then
  if [[ "$STRUCTURED_JSON" != /* ]]; then
    STRUCTURED_JSON="$ROOT_DIR/$STRUCTURED_JSON"
  fi
else
  STRUCTURED_JSON="$(
    find "$CACHE_DIR" -maxdepth 1 -type f -name "*_structured_${VARIANT}.json" -printf '%T@ %p\n' \
      | sort -nr \
      | head -n 1 \
      | cut -d' ' -f2-
  )"
  if [[ -z "$STRUCTURED_JSON" ]]; then
    FINAL_JSON="$CACHE_DIR/${DJ}_final_${VARIANT}.json"
    if [[ -f "$FINAL_JSON" ]]; then
      STRUCTURED_JSON="$FINAL_JSON"
    fi
  fi
fi

if [[ -z "$STRUCTURED_JSON" || ! -f "$STRUCTURED_JSON" ]]; then
  echo "未找到 ${VARIANT} 对应的 structured/final json" >&2
  echo "查找目录: $CACHE_DIR" >&2
  echo "优先查找: *_structured_${VARIANT}.json" >&2
  echo "回退模式: ${DJ}_final_${VARIANT}.json" >&2
  exit 1
fi

if [[ "$REPORT_ONLY" == "1" && -z "$RUN_ID" ]]; then
  echo "--report-only 模式必须显式传 --run-id" >&2
  exit 1
fi

if [[ -z "$RUN_ID" ]]; then
  if [[ -z "$RUN_ID_PREFIX" ]]; then
    RUN_ID_PREFIX="$VARIANT"
  fi
  RUN_ID="${RUN_ID_PREFIX}_$(date +%Y%m%d_%H%M%S)"
fi

TS="$(date +%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/batch_judge_${DJ}_${VARIANT}_${TS}.log"
PID_FILE="$LOG_DIR/batch_judge_${DJ}_${VARIANT}_${TS}.pid"
LOG_FILE_DEFAULT="$LOG_FILE"
PID_FILE_DEFAULT="$PID_FILE"

if [[ "${RUN_BATCH_JUDGE_QWEN_BG:-0}" != "1" && "$FOREGROUND" != "1" ]]; then
  BG_ARGS=(
    --dj "$DJ"
    --query-version "$QUERY_VERSION"
    --structured-json "$STRUCTURED_JSON"
    --run-id "$RUN_ID"
    --model "$MODEL_NAME"
    --max-segments "$MAX_SEGMENTS"
    --cache-dir "$CACHE_DIR"
  )
  if [[ "$REPORT_ONLY" == "1" ]]; then
    BG_ARGS+=(--report-only)
  fi

  RUN_BATCH_JUDGE_QWEN_BG=1 \
  DJ="$DJ" \
  QUERY_VERSION="$QUERY_VERSION" \
  STRUCTURED_JSON="$STRUCTURED_JSON" \
  RUN_ID="$RUN_ID" \
  RUN_ID_PREFIX="$RUN_ID_PREFIX" \
  MAX_SEGMENTS="$MAX_SEGMENTS" \
  MODEL_NAME="$MODEL_NAME" \
  CACHE_DIR="$CACHE_DIR" \
  REPORT_ONLY="$REPORT_ONLY" \
  LOG_FILE="$LOG_FILE" \
  PID_FILE="$PID_FILE" \
  PYTHON_BIN="$PYTHON_BIN" \
  bash "$0" "${BG_ARGS[@]}" >/dev/null 2>&1 &

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
  echo "[已启动] qwen batch_judge 后台运行中"
  echo "DJ: $DJ"
  echo "Variant: $VARIANT"
  echo "Run ID: $RUN_ID"
  echo "Structured JSON: $STRUCTURED_JSON"
  echo "PID: $PYTHON_PID"
  echo "Wrapper PID: $WRAPPER_PID"
  echo "日志: $LOG_FILE"
  exit 0
fi

: "${LOG_FILE:=$LOG_FILE_DEFAULT}"
: "${PID_FILE:=$PID_FILE_DEFAULT}"
exec >>"$LOG_FILE" 2>&1

echo "[$(date '+%F %T')] ROOT_DIR=$ROOT_DIR"
echo "[$(date '+%F %T')] DJ=$DJ"
echo "[$(date '+%F %T')] QUERY_VERSION=$QUERY_VERSION"
echo "[$(date '+%F %T')] VARIANT=$VARIANT"
echo "[$(date '+%F %T')] STRUCTURED_JSON=$STRUCTURED_JSON"
echo "[$(date '+%F %T')] RUN_ID=$RUN_ID"
echo "[$(date '+%F %T')] MODEL=$MODEL_NAME"
echo "[$(date '+%F %T')] MAX_SEGMENTS=$MAX_SEGMENTS"
echo "[$(date '+%F %T')] REPORT_ONLY=$REPORT_ONLY"
echo "[$(date '+%F %T')] WRAPPER_PID=$$"

CMD=(
  "$PYTHON_BIN"
  judge/batch_judge.py
  --dj "$DJ"
  --model "$MODEL_NAME"
  --run-id "$RUN_ID"
  --cache-dir "$CACHE_DIR"
  --variant "$VARIANT"
  --structured-json "$STRUCTURED_JSON"
)

if [[ "$MAX_SEGMENTS" != "0" ]]; then
  CMD+=(--max-segments "$MAX_SEGMENTS")
fi
if [[ "$REPORT_ONLY" == "1" ]]; then
  CMD+=(--report-only)
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
echo "[$(date '+%F %T')] qwen batch_judge finished exit_code=$STATUS"
exit "$STATUS"
