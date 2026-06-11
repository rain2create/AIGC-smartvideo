#!/bin/bash
# 影视解说-画面匹配Pipeline
# 用法:
#   单剧: ./run.sh <dj> <start> <end> [output] [top_k]
#   批量: ./run.sh --batch tasks.txt [top_k]
# 所有输出写到 logs/ 日志文件，终端只打印启动和结果
# 查看进度: tail -f logs/buxi5_*.log

set -euo pipefail
export TZ="Asia/Shanghai"

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
COMM_DIR="$BASE_DIR/commentary"
MATCH_DIR="$BASE_DIR/text_video_match"
FINISH_DIR="$BASE_DIR/finish"
LOG_DIR="$BASE_DIR/logs"

# MeloTTS 默认配置；如外部已设置则保持外部值。
export MELO_TTS_PYTHON="${MELO_TTS_PYTHON:-/mnt_liuhuan/miniconda3/envs/melo_tts/bin/python}"
export MELO_TTS_DEVICE="${MELO_TTS_DEVICE:-cuda}"
export MELO_TTS_HTTP_PROXY="${MELO_TTS_HTTP_PROXY:-http://public-proxy.qihoo.net:3128}"
export MELO_TTS_HTTPS_PROXY="${MELO_TTS_HTTPS_PROXY:-http://public-proxy.qihoo.net:3128}"

mkdir -p "$FINISH_DIR" "$LOG_DIR"

# 检查 JSON 中是否已有成功匹配
has_match_result() {
    local json_path="$1"
    [ ! -f "$json_path" ] && return 1
    python3 -c "
import json, sys
try:
    data = json.load(open('$json_path', encoding='utf8'))
    sys.exit(0 if any(seg.get('match_result',{}).get('status')=='success' for seg in data) else 1)
except: sys.exit(1)
" 2>/dev/null
}

# 如果依赖文件比目标文件更新，则说明目标已过期
is_stale_against_inputs() {
    local target="$1"
    shift
    [ ! -f "$target" ] && return 0
    local dep
    for dep in "$@"; do
        [ -f "$dep" ] && [ "$dep" -nt "$target" ] && return 0
    done
    return 1
}

# ========== 单剧 Pipeline ==========
process_single() {
    local DJ="$1"
    local START="$2"
    local END="$3"
    local OUT_FILE="${4:-${DJ}_final.mp4}"
    local TOP_K="${5:-5}"

    local CACHE_DIR="$MATCH_DIR/cache/$DJ"
    local RES_DIR="$COMM_DIR/results/$DJ"
    local LOG_FILE="$LOG_DIR/${DJ}_$(date '+%m%d_%H%M%S').log"

    # 所有输出重定向到日志文件
    {
        echo "[$(date '+%H:%M:%S')] ============================================================"
        echo "[$(date '+%H:%M:%S')] 开始处理: $DJ | 集数: $START-$END | top_k: $TOP_K"
        echo "[$(date '+%H:%M:%S')] ============================================================"

        # ---------- 1. 确保拿到 txt ----------
        TXT_FILE=""
        MD_FILE=""
        for f in $(ls -t "$RES_DIR"/*.md 2>/dev/null); do
            name=$(basename "$f")
            [[ "$name" != *"prompt"* && "$name" != *"system"* && "$name" != *"patched_commentary"* ]] && { MD_FILE="$f"; break; }
        done

        if [ -n "$MD_FILE" ]; then
            echo "[$(date '+%H:%M:%S')] ▶ [从 md 提取最终文案]"
            python3 -u "$COMM_DIR/extract_final_script_to_text.py" "$MD_FILE"
            TXT_FILE="${MD_FILE%.md}.txt"
        else
            echo "[$(date '+%H:%M:%S')] ▶ [生成短剧文案(text0204)]"
            python3 -u "$COMM_DIR/text0204.py" \
                --dj "$DJ" --start "$START" --end "$END" \
                --output-dir "$RES_DIR"

            for f in $(ls -t "$RES_DIR"/*.md 2>/dev/null); do
                name=$(basename "$f")
                [[ "$name" != *"prompt"* && "$name" != *"system"* && "$name" != *"patched_commentary"* ]] && { MD_FILE="$f"; break; }
            done
            if [ -n "$MD_FILE" ]; then
                echo "[$(date '+%H:%M:%S')] ▶ [从 md 提取最终文案]"
                python3 -u "$COMM_DIR/extract_final_script_to_text.py" "$MD_FILE"
                TXT_FILE="${MD_FILE%.md}.txt"
            fi
        fi

        if [ -z "$TXT_FILE" ]; then
            for f in $(ls -t "$RES_DIR"/*.txt 2>/dev/null); do TXT_FILE="$f"; break; done
        fi

        if [ -z "$TXT_FILE" ]; then
            echo "[$(date '+%H:%M:%S')] [ERROR] 最终仍拿不到 txt 文案"
            exit 1
        fi
        echo "[$(date '+%H:%M:%S')] ✓ 文案文件: $TXT_FILE"

        # ---------- S1: 视频集数索引化 ----------
        S1_JSON="$CACHE_DIR/structured_description.json"
        if [ -f "$S1_JSON" ]; then
            echo "[$(date '+%H:%M:%S')] ⏭  跳过 [视频集数索引化]（JSON已存在）"
        else
            echo "[$(date '+%H:%M:%S')] ▶ [视频集数索引化]"
            python3 -u "$MATCH_DIR/skill_video_episode_indexer.py" "$DJ" "$START" "$END"
        fi

        # ---------- S2: 结构化信息索引化 ----------
        S2_PKL="$CACHE_DIR/embeddings.pkl"
        if [ -f "$S2_PKL" ]; then
            echo "[$(date '+%H:%M:%S')] ⏭  跳过 [结构化信息索引化]（pkl已存在）"
        else
            echo "[$(date '+%H:%M:%S')] ▶ [结构化信息索引化]"
            python3 -u "$MATCH_DIR/skill_struct_info_emb.py" "$DJ"
        fi

        # ---------- S3: 解说文案拆分 ----------
        TXT_STEM=$(basename "${TXT_FILE%.*}")
        S3_JSON="$CACHE_DIR/${TXT_STEM}_structured.json"
        if [ -f "$S3_JSON" ]; then
            echo "[$(date '+%H:%M:%S')] ⏭  跳过 [解说文案拆分]（JSON已存在）"
        else
            echo "[$(date '+%H:%M:%S')] ▶ [解说文案拆分]"
            python3 -u "$MATCH_DIR/skill_text_structured_description.py" "$DJ" "$TXT_FILE" "$START" "$END"
        fi

        # ---------- S4: 核心匹配与精排 ----------
        if has_match_result "$S3_JSON"; then
            if is_stale_against_inputs "$S3_JSON" "$S1_JSON" "$S2_PKL"; then
                echo "[$(date '+%H:%M:%S')] ↻ [核心匹配与精排] 检测到索引已更新，强制重跑"
                python3 -u "$MATCH_DIR/skill_match_and_rank.py" "$DJ" "$S3_JSON" "$TOP_K"
            else
                echo "[$(date '+%H:%M:%S')] ⏭  跳过 [核心匹配与精排]（已含成功匹配）"
            fi
        else
            echo "[$(date '+%H:%M:%S')] ▶ [核心匹配与精排]"
            python3 -u "$MATCH_DIR/skill_match_and_rank.py" "$DJ" "$S3_JSON" "$TOP_K"
        fi

        # ---------- S5: 视频合成 ----------
        FINAL_MP4="$FINISH_DIR/$OUT_FILE"
        if is_stale_against_inputs "$FINAL_MP4" "$S3_JSON"; then
            echo "[$(date '+%H:%M:%S')] ▶ [视频合成]"
            python3 -u "$MATCH_DIR/skill_postprocess.py" "$DJ" "$S3_JSON" "$OUT_FILE"
        else
            echo "[$(date '+%H:%M:%S')] ⏭  跳过 [视频合成]（最终视频已存在）"
        fi

        # ---------- S6: 生成 final.json ----------
        FINAL_JSON="$CACHE_DIR/${DJ}_final.json"
        if is_stale_against_inputs "$FINAL_JSON" "$S3_JSON"; then
            echo "[$(date '+%H:%M:%S')] ▶ [生成final.json]"
            python3 -u -c "
import json
from pathlib import Path
s3 = Path('$S3_JSON')
data = json.loads(s3.read_text(encoding='utf8'))
out = {'dj': '$DJ', 'total_segments': len(data), 'segments': []}
for seg in data:
    mr = seg.get('match_result', {})
    tops = mr.get('top_matches', [])
    out['segments'].append({
        'segment_sequence': seg.get('segment_sequence'),
        'script_text': seg.get('script_text'),
        'episode_target': seg.get('episode_target'),
        'audio_path': seg.get('audio_path'),
        'audio_duration': seg.get('audio_duration'),
        'visual_query': seg.get('visual_query', {}),
        'match_result': {
            'status': mr.get('status', 'unknown'),
            'top1_match': tops[0] if tops else None,
            'match_breakdown': mr.get('match_breakdown', {})
        } if mr.get('status') == 'success' else None
    })
Path('$FINAL_JSON').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf8')
print(f'final.json 生成完成: {len(out[\"segments\"])} 个segments')
"
        else
            echo "[$(date '+%H:%M:%S')] ⏭  跳过 [生成final.json]（已存在）"
        fi

        echo "[$(date '+%H:%M:%S')] ============================================================"
        echo "[$(date '+%H:%M:%S')] ✓ $DJ 全部完成！"
        echo "[$(date '+%H:%M:%S')] 视频: $FINAL_MP4"
        echo "[$(date '+%H:%M:%S')] JSON:  $FINAL_JSON"
        echo "[$(date '+%H:%M:%S')] ============================================================"

    } > "$LOG_FILE" 2>&1

    local EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "[完成] $DJ ✓"
    else
        echo "[失败] $DJ ✗ 查看日志: tail -50 $LOG_FILE"
    fi
    return $EXIT_CODE
}

# ========== 批量并行模式 ==========
process_batch() {
    local BATCH_FILE="$1"
    local TOP_K="${2:-5}"

    if [ ! -f "$BATCH_FILE" ]; then
        echo "[ERROR] 任务文件不存在: $BATCH_FILE"; exit 1
    fi

    TASKS=()
    while IFS= read -r line; do
        line=$(echo "$line" | xargs)
        [[ -z "$line" || "$line" == \#* ]] && continue
        TASKS+=("$line")
    done < "$BATCH_FILE"

    echo "============================================================"
    echo "准备并行启动 ${#TASKS[@]} 个任务（后台运行）"
    echo "============================================================"

    for task in "${TASKS[@]}"; do
        IFS=',' read -r dj start end out topk <<< "$task"
        out="${out:-${dj}_final.mp4}"
        topk="${topk:-$TOP_K}"

        LOG_FILE="$LOG_DIR/${dj}_$(date '+%m%d_%H%M%S').log"
        process_single "$dj" "$start" "$end" "$out" "$topk" > /dev/null 2>&1 &
        PID=$!
        disown
        # 等待 python3 子进程启动
        PY_PID=""
        for i in $(seq 1 10); do
            PY_PID=$(pgrep -P $PID python3 2>/dev/null | head -1) || true
            [ -n "$PY_PID" ] && break
            sleep 1
        done
        : ${PY_PID:=$PID}

        echo "[启动] $dj | 集数 $start-$end | python3 PID: $PY_PID | 日志: $LOG_FILE"
        echo "  kill $PY_PID                   # 停止python3"
        echo "  pkill -P $PID                  # 停止所有子进程"
        echo "  tail -f $LOG_FILE              # 查看日志"
        echo ""
    done

    echo "============================================================"
    echo "所有任务已后台启动，终端可关闭"
    echo "查看进度: tail -f logs/<剧名>_*.log"
    echo "============================================================"
}

# ========== 主入口 ==========
if [ $# -lt 1 ]; then
    echo "用法:"
    echo "  单剧: bash run.sh <dj> <start> <end> [output] [top_k]"
    echo "  批量: bash run.sh --batch tasks.txt [top_k]"
    echo ""
    echo "查看进度: tail -f logs/<剧名>_*.log"
    exit 1
fi

if [ "$1" == "--batch" ]; then
    process_batch "${2:-tasks.txt}" "${3:-5}"
else
    if [ $# -lt 3 ]; then
        echo "用法: bash run.sh <dj> <start> <end> [output] [top_k]"
        echo "      bash run.sh --batch tasks.txt [top_k]"
        echo ""
        echo "查看进度: tail -f logs/<剧名>_*.log"
        exit 1
    fi
    LOG_FILE="$LOG_DIR/${1}_$(date '+%m%d_%H%M%S').log"
    process_single "$1" "$2" "$3" "${4:-}" "${5:-5}" &
    PID=$!
    disown
    # 等待 python3 子进程启动（最多10秒）
    PY_PID=""
    for i in $(seq 1 10); do
        PY_PID=$(pgrep -P $PID python3 2>/dev/null | head -1) || true
        [ -n "$PY_PID" ] && break
        sleep 1
    done
    : ${PY_PID:=$PID}
    echo "[启动] $1 | 集数 $2-$3 | python3 PID: $PY_PID | 日志: $LOG_FILE"
    echo "  kill $PY_PID                   # 停止python3"
    echo "  pkill -P $PID                  # 停止所有子进程"
fi
