#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen 多模态检索完整 Pipeline（单剧 / 并行批量版）

流程：
1. S1 视频集数索引化
2. S3 文案结构化
3. Qwen-S2 视频 visual embedding 索引
4. Qwen-S4 裸检索（query -> video emb）
5. S5 视频合成
6. S6 生成 final.json

说明：
- 不复用旧的 skill_match_and_rank.py
- 产物按 variant 分路径，避免和 V4 / 其他 Qwen 版本互相覆盖
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from multiprocessing import Process
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# ─── 批量进度追踪 ───

_PROGRESS_FILE: Optional[Path] = None


def set_progress_file(path: Path) -> None:
    global _PROGRESS_FILE
    _PROGRESS_FILE = path


def _read_progress() -> Dict[str, Any]:
    if _PROGRESS_FILE and _PROGRESS_FILE.exists():
        try:
            return json.loads(_PROGRESS_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {"tasks": {}}


def _write_progress(data: Dict[str, Any]) -> None:
    if not _PROGRESS_FILE:
        return
    lock_path = Path(str(_PROGRESS_FILE) + ".lock")
    with open(lock_path, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            tmp = Path(str(_PROGRESS_FILE) + ".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.rename(_PROGRESS_FILE)
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


def init_progress(task_list: List[Tuple[str, str, str, str, str, str, str, str, str, str, str]], progress_path: Path) -> None:
    set_progress_file(progress_path)
    tasks = {}
    for t in task_list:
        dj = t[0]
        tasks[dj] = {"dj": dj, "status": "pending", "step": None}
    data = {
        "total": len(tasks),
        "pending": len(tasks),
        "running": 0,
        "completed": 0,
        "failed": 0,
        "started_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "tasks": tasks,
    }
    _write_progress(data)


def update_progress(dj: str, status: str, step: Optional[str] = None, error: Optional[str] = None) -> None:
    if not _PROGRESS_FILE:
        return
    lock_path = Path(str(_PROGRESS_FILE) + ".lock")
    with open(lock_path, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            data = _read_progress()
            tasks = data.get("tasks", {})
            if dj not in tasks:
                tasks[dj] = {"dj": dj}
            task = tasks[dj]
            old_status = task.get("status")
            task["status"] = status
            task["updated_at"] = datetime.now().isoformat()
            if step:
                task["step"] = step
            if error:
                task["error"] = str(error)[:200]
            if status == "running" and old_status not in ("running", None) and "started_at" not in task:
                task["started_at"] = datetime.now().isoformat()
            if status in ("completed", "failed"):
                task["completed_at"] = datetime.now().isoformat()
                started = task.get("started_at")
                if started:
                    try:
                        t0 = datetime.fromisoformat(started)
                        task["duration_sec"] = round((datetime.now() - t0).total_seconds(), 1)
                    except (ValueError, TypeError):
                        pass
            counts = {"pending": 0, "running": 0, "completed": 0, "failed": 0}
            for t in tasks.values():
                s = t.get("status", "pending")
                if s in counts:
                    counts[s] += 1
            data.update(counts)
            data["updated_at"] = datetime.now().isoformat()
            tmp = Path(str(_PROGRESS_FILE) + ".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.rename(_PROGRESS_FILE)
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


def setup_task_logger(task_name: str):
    logger = logging.getLogger(task_name)
    logger.setLevel(logging.INFO)
    logger.handlers = []
    logger.propagate = False

    fh = logging.FileHandler(LOG_DIR / f"{task_name}_{datetime.now().strftime('%m%d_%H%M%S')}.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] %(message)s", "%H:%M:%S"))
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(f"[%(asctime)s][{task_name}][%(levelname)s] %(message)s", "%H:%M:%S"))
    logger.addHandler(ch)
    return logger


def run_step(cmd: List[str], step_name: str, logger):
    logger.info("▶ [%s]", step_name)
    logger.info("  cwd: %s", Path(__file__).parent.absolute())
    logger.info("  cmd: %s", shlex.join(cmd))

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    started_at = time.time()
    merged_lines: List[str] = []
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    assert process.stdout is not None
    for raw_line in process.stdout:
        line = raw_line.rstrip()
        merged_lines.append(line)
        logger.info("  [%s] %s", step_name, line)

    return_code = process.wait()
    duration = time.time() - started_at

    if return_code != 0:
        logger.error("✗ [%s] 失败", step_name)
        logger.error("  return_code: %s", return_code)
        logger.error("  duration_sec: %.2f", duration)
        if merged_lines:
            logger.error("  tail: %s", " | ".join(merged_lines[-5:]))
        raise RuntimeError(f"{step_name} 失败")

    logger.info("✓ [%s] 完成", step_name)
    logger.info("  duration_sec: %.2f", duration)


def log_artifact(path: Path, label: str, logger):
    if not path.exists():
        raise RuntimeError(f"{label} 未生成: {path}")
    stat = path.stat()
    logger.info(
        "  artifact[%s]: %s | size=%s bytes | mtime=%s",
        label,
        path,
        stat.st_size,
        datetime.fromtimestamp(stat.st_mtime).strftime("%F %T"),
    )


def has_match_result(json_path: Path) -> bool:
    if not json_path.exists():
        return False
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        return any(seg.get("match_result", {}).get("status") == "success" for seg in data)
    except Exception:
        return False


def is_stale_against_inputs(target: Path, *deps: Path) -> bool:
    if not target.exists():
        return True
    return any(dep.exists() and dep.stat().st_mtime > target.stat().st_mtime for dep in deps)


def generate_final_json(dj: str, structured_json: Path, output_path: Path, variant: str, logger):
    data = json.loads(structured_json.read_text(encoding="utf-8"))
    final_output = {
        "dj": dj,
        "variant": variant,
        "generated_at": str(datetime.now()),
        "total_segments": len(data),
        "segments": [],
    }

    for seg in data:
        match_result = seg.get("match_result", {})
        top_matches = match_result.get("top_matches", [])
        final_output["segments"].append(
            {
                "segment_sequence": seg.get("segment_sequence"),
                "script_text": seg.get("script_text"),
                "episode_target": seg.get("episode_target"),
                "audio_path": seg.get("audio_path"),
                "audio_duration": seg.get("audio_duration"),
                "visual_query": seg.get("visual_query", {}),
                "match_result": {
                    "status": match_result.get("status", "unknown"),
                    "matched_episode": match_result.get("episode_target"),
                    "top1_match": top_matches[0] if top_matches else None,
                    "match_breakdown": match_result.get("match_breakdown", {}),
                }
                if match_result.get("status") == "success"
                else None,
            }
        )

    output_path.write_text(json.dumps(final_output, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("✓ final.json 生成完成: %s", output_path)


def find_or_create_txt(dj: str, start: str, end: str, logger, python_bin: str) -> Path:
    base = Path(__file__).parent.absolute()
    comm_dir = base / "commentary"
    res_dir = comm_dir / "results" / dj
    res_dir.mkdir(parents=True, exist_ok=True)

    txt_file = None
    md_file = None

    for f in sorted(res_dir.iterdir(), reverse=True) if res_dir.exists() else []:
        if f.suffix == ".txt":
            txt_file = f
            break

    if txt_file:
        logger.info("⏭ 跳过 [提取/生成短剧文案]（txt 已存在）: %s", txt_file.name)
        return txt_file

    if res_dir.exists():
        for f in sorted(res_dir.iterdir(), reverse=True):
            if f.suffix == ".md" and "prompt" not in f.name and "system" not in f.name:
                md_file = f
                break

    if md_file:
        run_step([python_bin, str(comm_dir / "extract_final_script_to_text.py"), str(md_file)], "从 md 提取最终文案", logger)
    else:
        run_step(
            [
                python_bin,
                str(comm_dir / "text0204.py"),
                "--dj",
                dj,
                "--start",
                start,
                "--end",
                end,
                "--output-dir",
                str(res_dir),
            ],
            "生成短剧文案(text0204)",
            logger,
        )
        for f in sorted(res_dir.iterdir(), reverse=True):
            if f.suffix == ".md" and "prompt" not in f.name and "system" not in f.name:
                md_file = f
                break
        if md_file:
            run_step([python_bin, str(comm_dir / "extract_final_script_to_text.py"), str(md_file)], "从 md 提取最终文案", logger)

    for f in sorted(res_dir.iterdir(), reverse=True):
        if f.suffix == ".txt":
            txt_file = f
            break

    if not txt_file:
        raise RuntimeError("最终仍拿不到 txt 文案")
    return txt_file


def _lora_tag(lora_path: str) -> str:
    """从 lora_path 提取短标识用于 variant 命名。

    Examples:
        train/checkpoints/qwen_dpo/checkpoint-2000  → dpo2000
        train/checkpoints/qwen_dpo/checkpoint-1500  → dpo1500
        train/checkpoints/qwen_dpo/final            → dpo_final
        /abs/path/my_adapter                        → my_adapter
    """
    p = Path(lora_path.rstrip("/"))
    name = p.name                     # checkpoint-2000 / final
    parent = p.parent.name            # qwen_dpo

    # checkpoint-{N} → 取数字
    if name.startswith("checkpoint-"):
        num = name.split("-", 1)[1]
        prefix = parent.replace("qwen_", "") if parent.startswith("qwen_") else parent
        return f"{prefix}{num}"       # dpo2000
    # final / 其他
    prefix = parent.replace("qwen_", "") if parent.startswith("qwen_") else parent
    return f"{prefix}_{name}"         # dpo_final


def process_single_dj(
    dj: str,
    start: str,
    end: str,
    out_file: str,
    top_k: str,
    query_version: str,
    max_segments: str,
    video_index_version: str = "V1",
    rerank_version: str = "",
    rerank_model_path: str | None = None,
    lora_path: str = "",
    init_lora_path: str = "",
    model_path: str | None = None,
):
    if rerank_version:
        if video_index_version != "V1":
            variant = f"qwen_{video_index_version.lower()}_{query_version.lower()}_rerank_{rerank_version.lower()}"
        else:
            variant = f"qwen_{query_version.lower()}_rerank_{rerank_version.lower()}"
    elif video_index_version != "V1":
        variant = f"qwen_{video_index_version.lower()}_{query_version.lower()}"
    else:
        variant = f"qwen_{query_version.lower()}"

    # LoRA adapter → variant 后缀，避免不同 checkpoint 产物互相覆盖
    if lora_path:
        lora_tag = _lora_tag(lora_path)
        variant = f"{variant}_{lora_tag}"

    task_name = f"{dj}_{variant}"
    logger = setup_task_logger(task_name)

    base = Path(__file__).parent.absolute()
    python_bin = sys.executable
    match_dir = base / "text_video_match"
    cache_dir = match_dir / "cache" / dj
    finish_dir = base / "finish"
    finish_dir.mkdir(exist_ok=True)

    logger.info("=" * 60)
    logger.info("开始处理: dj=%s start=%s end=%s top_k=%s variant=%s", dj, start, end, top_k, variant)
    logger.info("=" * 60)
    update_progress(dj, "running", step="文案准备")

    try:
        txt_file = find_or_create_txt(dj, start, end, logger, python_bin)
        txt_stem = txt_file.stem

        s1_json = cache_dir / "structured_description.json"
        update_progress(dj, "running", step="S1 视频索引化")
        if s1_json.exists():
            logger.info("⏭ 跳过 [S1 视频集数索引化]（JSON 已存在）")
        else:
            run_step([python_bin, str(match_dir / "skill_video_episode_indexer.py"), dj, start, end], "S1 视频集数索引化", logger)
        log_artifact(s1_json, "S1 structured_description", logger)

        s3_json = cache_dir / f"{txt_stem}_structured.json"
        update_progress(dj, "running", step="S3 文案结构化")
        if s3_json.exists():
            logger.info("⏭ 跳过 [S3 解说文案拆分]（JSON 已存在）")
        else:
            run_step(
                [python_bin, str(match_dir / "skill_text_structured_description.py"), dj, str(txt_file), start, end],
                "S3 解说文案拆分",
                logger,
            )
        log_artifact(s3_json, "S3 structured_text", logger)

        # B2/B3: 需要 V1 视觉 embedding + B2 文本 embedding
        update_progress(dj, "running", step="S2 embedding索引")
        if video_index_version in ("B2", "B3"):
            vis_emb_pkl = cache_dir / "embeddings_vl.pkl"
            if vis_emb_pkl.exists():
                logger.info("⏭ 跳过 [Qwen-S2 V1 视觉 embedding]（pkl 已存在）")
            else:
                run_step(
                    [python_bin, str(match_dir / "qwen" / "index_builder.py"), dj, "--overwrite", "--video-index-version", "V1"],
                    "Qwen-S2 V1 视觉 embedding 索引", logger,
                )
            log_artifact(vis_emb_pkl, "Qwen-S2 V1 embeddings_vl.pkl", logger)

            txt_emb_pkl = cache_dir / "text_embeddings_b2.pkl"
            if txt_emb_pkl.exists():
                logger.info("⏭ 跳过 [Qwen-S2 B2 文本 embedding]（pkl 已存在）")
            else:
                run_step(
                    [python_bin, str(match_dir / "qwen" / "index_builder.py"), dj, "--overwrite", "--video-index-version", "B2"],
                    "Qwen-S2 B2 文本 embedding 索引", logger,
                )
            log_artifact(txt_emb_pkl, "Qwen-S2 B2 text_embeddings_b2.pkl", logger)
        else:
            embedding_filename = "embeddings_vl.pkl" if video_index_version == "V1" else f"embeddings_vl_{video_index_version.lower()}.pkl"
            qwen_embedding_pkl = cache_dir / embedding_filename
            if qwen_embedding_pkl.exists():
                logger.info("⏭ 跳过 [Qwen-S2 视频 visual embedding 索引]（pkl 已存在）")
            else:
                cmd = [python_bin, str(match_dir / "qwen" / "index_builder.py"), dj, "--overwrite", "--video-index-version", video_index_version]
                if model_path:
                    cmd.extend(["--model-path", model_path])
                run_step(cmd, "Qwen-S2 视频 visual embedding 索引", logger)
            log_artifact(qwen_embedding_pkl, "Qwen-S2 embeddings_vl.pkl", logger)

        qwen_s4_json = cache_dir / f"{txt_stem}_structured_{variant}.json"
        update_progress(dj, "running", step="S4 检索匹配")
        if video_index_version in ("B2", "B3"):
            b2_inputs = [s3_json, vis_emb_pkl, txt_emb_pkl]
            skip_s4 = has_match_result(qwen_s4_json) and not is_stale_against_inputs(qwen_s4_json, *b2_inputs)
        else:
            skip_s4 = has_match_result(qwen_s4_json) and not is_stale_against_inputs(qwen_s4_json, s3_json, qwen_embedding_pkl)
        if skip_s4:
            logger.info("⏭ 跳过 [Qwen-S4 检索]（已存在成功匹配且未过期）")
        else:
            if video_index_version == "B2":
                cmd = [
                    python_bin,
                    str(match_dir / "qwen" / "retriever_b2.py"),
                    dj,
                    str(s3_json),
                    "--output-path",
                    str(qwen_s4_json),
                    "--top-k",
                    str(top_k),
                    "--query-version",
                    query_version,
                    "--overwrite",
                ]
            elif video_index_version == "B3":
                cmd = [
                    python_bin,
                    str(match_dir / "qwen" / "retriever_b3.py"),
                    dj,
                    str(s3_json),
                    "--output-path",
                    str(qwen_s4_json),
                    "--top-k",
                    str(top_k),
                    "--alpha", "0.8",
                    "--query-version",
                    query_version,
                    "--overwrite",
                ]
            else:
                cmd = [
                    python_bin,
                    str(match_dir / "qwen" / "retriever.py"),
                    dj,
                    str(s3_json),
                    "--output-path",
                    str(qwen_s4_json),
                    "--top-k",
                    str(top_k),
                    "--query-version",
                    query_version,
                    "--video-index-version",
                    video_index_version,
                    "--overwrite",
                ]
            if max_segments and max_segments != "0":
                cmd.extend(["--max-segments", str(max_segments)])
            if model_path:
                cmd.extend(["--model-path", model_path])
            run_step(cmd, "Qwen-S4 检索", logger)
        log_artifact(qwen_s4_json, "Qwen-S4 structured_qwen", logger)

        # S4.5 Reranker 重排序（可选）
        if rerank_version:
            final_mp4_path = finish_dir / variant / out_file
            if final_mp4_path.exists():
                logger.info("⏭ 跳过 [S4.5/S5/S6]（final mp4 已存在）")
                log_artifact(final_mp4_path, "S5 final_mp4", logger)
                update_progress(dj, "completed", step="已完成(跳过)")
                return (task_name, True, f"最终视频已存在，跳过")

            update_progress(dj, "running", step="S4.5 Reranker")
            s45_input = qwen_s4_json
            s45_output = cache_dir / f"{txt_stem}_structured_{variant}.json"
            s45_cmd = [
                python_bin,
                str(match_dir / "qwen" / "reranker.py"),
                dj,
                str(s45_input),
                "--output-path",
                str(s45_output),
                "--doc-version",
                rerank_version,
                "--overwrite",
            ]
            if rerank_model_path:
                s45_cmd.extend(["--model-path", rerank_model_path])
            if lora_path:
                s45_cmd.extend(["--lora-path", lora_path])
            if init_lora_path:
                s45_cmd.extend(["--init-lora-path", init_lora_path])
            run_step(s45_cmd, "S4.5 Reranker 重排序", logger)
            log_artifact(s45_output, "S4.5 reranker output", logger)
            qwen_s4_json = s45_output

        if max_segments and max_segments != "0":
            logger.info("⏹ 检测到 max_segments=%s，本次仅做小样本检索验证，跳过 S5/S6", max_segments)
            logger.info("中间结果: %s", qwen_s4_json)
            update_progress(dj, "completed", step="S4检索完成(max_segments)")
            return (task_name, True, f"小样本检索完成，输出={qwen_s4_json}")

        finish_subdir = finish_dir / variant
        finish_subdir.mkdir(exist_ok=True)
        out_name = out_file if "/" in out_file else f"{variant}/{out_file}"
        final_mp4 = finish_dir / out_name

        update_progress(dj, "running", step="S5 视频合成")
        if is_stale_against_inputs(final_mp4, qwen_s4_json):
            run_step([python_bin, str(match_dir / "skill_postprocess.py"), dj, str(qwen_s4_json), out_name], "S5 视频合成", logger)
            log_artifact(final_mp4, "S5 final_mp4", logger)
        else:
            logger.info("⏭ 跳过 [S5 视频合成]（最终视频已存在且未过期）")
            log_artifact(final_mp4, "S5 final_mp4", logger)

        final_json = cache_dir / f"{dj}_final_{variant}.json"
        if is_stale_against_inputs(final_json, qwen_s4_json):
            logger.info("▶ [S6 生成 final.json]")
            generate_final_json(dj, qwen_s4_json, final_json, variant, logger)
        else:
            logger.info("⏭ 跳过 [S6 生成 final.json]（已存在且未过期）")
        log_artifact(final_json, "S6 final_json", logger)

        logger.info("=" * 60)
        logger.info("🎉 %s 全部完成", task_name)
        logger.info("retrieval json: %s", qwen_s4_json)
        logger.info("final mp4: %s", final_mp4)
        logger.info("final json: %s", final_json)
        logger.info("=" * 60)
        update_progress(dj, "completed")
        return (task_name, True, "成功")

    except Exception as exc:
        update_progress(dj, "failed", error=str(exc))
        logger.exception("💥 %s 处理失败: %s", task_name, exc)
        return (task_name, False, str(exc))


def process_gpu_queue(gpu_id: str, tasks: List[Tuple[str, ...]]):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    failed = 0

    for task in tasks:
        print(f"[GPU {gpu_id}] 开始 dj={task[0]} query_version={task[5]}", flush=True)
        result = process_single_dj(*task)
        if not result[1]:
            failed += 1
            print(f"[GPU {gpu_id}] 失败 dj={task[0]} error={result[2]}", flush=True)
        else:
            print(f"[GPU {gpu_id}] 完成 dj={task[0]}", flush=True)

    if failed:
        sys.exit(1)


def main():
    if len(sys.argv) < 2:
        print(
            """用法:
1. 单剧模式:
   python run_mul_qwen.py <dj> <start> <end> [output] [top_k] [query_version] [max_segments] [video_index_version] [rerank_version]
   例:
   python run_mul_qwen.py chongfan80 1 10 chongfan80_q1_final.mp4 10 Q1 0 V1
   python run_mul_qwen.py chongfan80 1 10 chongfan80_b1_final.mp4 20 Q3 0 B1
   python run_mul_qwen.py chongfan80 1 10 chongfan80_b2_final.mp4 20 Q3 0 B2
   python run_mul_qwen.py chongfan80 1 10 chongfan80_b3_final.mp4 20 Q3 0 B3
   python run_mul_qwen.py chongfan80 1 10 chongfan80_b1_rerank_q1_final.mp4 20 Q3 0 B1 Q1

2. 批量并行模式:
   python run_mul_qwen.py --batch tasks_qwen.txt [top_k] [query_version] [max_segments] [video_index_version]

tasks_qwen.txt 每行格式:
dj,start,end,output,top_k,query_version,max_segments,video_index_version,rerank_version,rerank_model_path,lora_path,init_lora_path
chongfan80,1,10,chongfan80_q1_final.mp4,10,Q1,0,V1,,
chongfan80,1,10,chongfan80_b1_rerank_q1_final.mp4,20,Q3,0,B1,Q1,,
chongfan80,1,10,chongfan80_b1_rerank_q1_dpo2000_final.mp4,20,Q3,0,B1,Q1,,train/checkpoints/qwen_dpo/checkpoint-2000
"""
        )
        sys.exit(1)

    if sys.argv[1] == "--batch":
        batch_file = sys.argv[2]
        default_top_k = sys.argv[3] if len(sys.argv) > 3 else "10"
        default_query_version = sys.argv[4] if len(sys.argv) > 4 else "Q1"
        default_max_segments = sys.argv[5] if len(sys.argv) > 5 else "0"
        default_video_index_version = sys.argv[6] if len(sys.argv) > 6 else "V1"

        batch_path = Path(batch_file)
        if not batch_path.exists():
            print(f"错误: 任务文件不存在: {batch_file}")
            sys.exit(1)

        tasks: List[Tuple[str, ...]] = []
        for line in batch_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [part.strip() for part in line.split(",")]
            while len(parts) < 12:
                parts.append("")
            dj, start, end, out_file, top_k, query_version, max_segments, video_index_version, rerank_version, rerank_model_path, lora_path, init_lora_path = parts[:12]
            tasks.append(
                (
                    dj,
                    start,
                    end,
                    out_file or f"{dj}_{(query_version or default_query_version).lower()}_final.mp4",
                    top_k or default_top_k,
                    query_version or default_query_version,
                    max_segments or default_max_segments,
                    video_index_version or default_video_index_version,
                    rerank_version or "",
                    rerank_model_path or "",
                    lora_path or "",
                    init_lora_path or "",
                )
            )

        progress_path = batch_path.parent / f"progress_{batch_path.stem}.json"
        init_progress(tasks, progress_path)
        print(f"[进度] {progress_path}", flush=True)

        gpu_env = os.getenv("QWEN_GPUS", "").strip()
        if gpu_env:
            gpu_ids = [gpu.strip() for gpu in gpu_env.split(",") if gpu.strip()]
            if not gpu_ids:
                print("错误: QWEN_GPUS 已设置但为空")
                sys.exit(1)

            slots = [(gpu_ids[i], []) for i in range(len(gpu_ids))]
            for index, task in enumerate(tasks):
                slots[index % len(slots)][1].append(task)

            processes: List[Process] = []
            print(f"[GPU队列] QWEN_GPUS={','.join(gpu_ids)}，每张卡并行槽位，总并发={len(gpu_ids)}")
            for slot_idx, (gpu_id, gpu_tasks) in enumerate(slots):
                if not gpu_tasks:
                    continue
                p = Process(target=process_gpu_queue, args=(gpu_id, gpu_tasks), name=f"GPU-{gpu_id}-slot{slot_idx}")
                p.start()
                processes.append(p)
                task_names = ",".join(task[0] for task in gpu_tasks)
                print(f"[启动] gpu={gpu_id} slot={slot_idx} pid={p.pid} tasks={task_names}")

            # 写 PID 文件和终止脚本
            pid_path = batch_path.parent / f"pids_{batch_path.stem}.txt"
            kill_path = batch_path.parent / f"kill_{batch_path.stem}.sh"
            pid_list = [str(p.pid) for p in processes]
            pid_path.write_text("\n".join(pid_list))
            kill_path.write_text(
                "#!/bin/bash\n"
                "echo '=== 杀主进程 ==='\n"
                f"kill {' '.join(pid_list)} 2>/dev/null\n"
                "sleep 2\n"
                "echo '=== 清孤儿进程 ==='\n"
                "ps aux | grep 'qwen3vl/bin/python' | grep -v grep | awk '{print $2}' | xargs -r kill 2>/dev/null\n"
                "sleep 2\n"
                "echo '=== 确认 ==='\n"
                "pgrep -af run_mul_qwen || echo '无残留进程'\n"
                "ps aux | grep qwen3vl | grep -v grep || echo '无 qwen3vl 残留'\n"
            )
            kill_path.chmod(0o755)
            print(f"[PID文件] {pid_path}")
            print(f"[终止脚本] {kill_path}")

            for p in processes:
                p.join()
            failed = [p for p in processes if p.exitcode != 0]
            if failed:
                sys.exit(1)
            return

        processes: List[Process] = []
        for task in tasks:
            p = Process(target=process_single_dj, args=task)
            p.start()
            processes.append(p)
            print(f"[启动] dj={task[0]} query_version={task[5]} pid={p.pid}")

        pid_path = batch_path.parent / f"pids_{batch_path.stem}.txt"
        kill_path = batch_path.parent / f"kill_{batch_path.stem}.sh"
        pid_list = [str(p.pid) for p in processes]
        pid_path.write_text("\n".join(pid_list))
        kill_path.write_text(
            "#!/bin/bash\n"
            "echo '=== 杀主进程 ==='\n"
            f"kill {' '.join(pid_list)} 2>/dev/null\n"
            "sleep 2\n"
            "echo '=== 清孤儿进程 ==='\n"
            "ps aux | grep 'qwen3vl/bin/python' | grep -v grep | awk '{print $2}' | xargs -r kill 2>/dev/null\n"
            "sleep 2\n"
            "echo '=== 确认 ==='\n"
            "pgrep -af run_mul_qwen || echo '无残留进程'\n"
            "ps aux | grep qwen3vl | grep -v grep || echo '无 qwen3vl 残留'\n"
        )
        kill_path.chmod(0o755)
        print(f"[PID文件] {pid_path}")
        print(f"[终止脚本] {kill_path}")

        for p in processes:
            p.join()
        return

    dj = sys.argv[1]
    start = sys.argv[2] if len(sys.argv) > 2 else "1"
    end = sys.argv[3] if len(sys.argv) > 3 else "20"
    out_file = sys.argv[4] if len(sys.argv) > 4 else f"{dj}_q1_final.mp4"
    top_k = sys.argv[5] if len(sys.argv) > 5 else "10"
    query_version = sys.argv[6] if len(sys.argv) > 6 else "Q1"
    max_segments = sys.argv[7] if len(sys.argv) > 7 else "0"
    video_index_version = sys.argv[8] if len(sys.argv) > 8 else "V1"
    rerank_version = sys.argv[9] if len(sys.argv) > 9 else ""

    process_single_dj(dj, start, end, out_file, top_k, query_version, max_segments, video_index_version, rerank_version=rerank_version)


if __name__ == "__main__":
    main()
