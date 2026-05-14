#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
影视解说-画面匹配Pipeline（并行版 - 四阶段匹配）
特点：
- 每个剧独立进程，各跑各的流水线
- 使用新的四阶段匹配 + LLM精排
- 每个剧独立日志文件，方便查看进度
"""
import sys
import subprocess
import json
from pathlib import Path
import logging
import os
from datetime import datetime
from multiprocessing import Process
import time

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

def setup_task_logger(dj: str):
    """为每个剧创建独立日志文件"""
    logger = logging.getLogger(dj)
    logger.setLevel(logging.INFO)
    logger.handlers = []
    
    fh = logging.FileHandler(LOG_DIR / f"{dj}_{datetime.now().strftime('%m%d_%H%M%S')}.log", encoding='utf-8')
    fh.setFormatter(logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", "%H:%M:%S"))
    logger.addHandler(fh)
    
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(f"[%(asctime)s][{dj}][%(levelname)s] %(message)s", "%H:%M:%S"))
    logger.addHandler(ch)
    
    return logger

def run_step(cmd, step_name, logger):
    """单步执行"""
    logger.info(f"▶ [{step_name}]")
    logger.info(f"  cmd: {' '.join(cmd)}")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    started_at = time.time()
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    merged_lines = []
    for raw_line in process.stdout:
        line = raw_line.rstrip()
        merged_lines.append(line)
        logger.info(f"  [{step_name}] {line}")

    return_code = process.wait()
    duration = time.time() - started_at

    if return_code != 0:
        logger.error(f"✗ [{step_name}] 失败 (return_code={return_code}, duration={duration:.1f}s)")
        if merged_lines:
            logger.error(f"  tail: {' | '.join(merged_lines[-5:])}")
        raise RuntimeError(f"{step_name} 失败 (return_code={return_code})")

    logger.info(f"✓ [{step_name}] 完成 (duration={duration:.1f}s)")
    return True

def has_match_result(json_path: Path) -> bool:
    """JSON 存在且至少一个片段已含 match_result.status == success"""
    if not json_path.exists():
        return False
    try:
        data = json.loads(json_path.read_text(encoding="utf8"))
        return any(seg.get("match_result", {}).get("status") == "success" for seg in data)
    except Exception:
        return False

def generate_final_json(dj: str, structured_json: Path, output_path: Path, logger):
    """生成final.json"""
    try:
        data = json.loads(structured_json.read_text(encoding="utf8"))
        
        final_output = {
            "dj": dj,
            "generated_at": str(Path(__file__).stat().st_mtime),
            "total_segments": len(data),
            "segments": []
        }
        
        for seg in data:
            match_result = seg.get("match_result", {})
            top_matches = match_result.get("top_matches", [])
            
            final_segment = {
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
                    "match_breakdown": match_result.get("match_breakdown", {})
                } if match_result.get("status") == "success" else None
            }
            final_output["segments"].append(final_segment)
        
        output_path.write_text(json.dumps(final_output, ensure_ascii=False, indent=2), encoding="utf8")
        logger.info(f"✓ final.json 生成完成: {len(final_output['segments'])} 个segments")
        
    except Exception as e:
        logger.error(f"生成final.json失败: {e}")
        raise

def process_single_dj(dj: str, start: str, end: str, out_file: str, top_k: str):
    """
    单个剧的完整流水线（独立进程中运行）
    S1: 视频集数索引化
    S2: 结构化信息索引化
    S3: 解说文案拆分
    S4: 核心匹配与精排（四阶段）
    S5: 视频合成
    S6: 生成final.json
    """
    logger = setup_task_logger(dj)
    base = Path(__file__).parent.absolute()
    comm_dir = base / "video_commentary_text"
    match_dir = base / "text_video_match"
    finish_dir = base / "finish"
    finish_dir.mkdir(exist_ok=True)
    cache_dir = match_dir / "cache" / dj
    
    logger.info("=" * 60)
    logger.info(f"开始处理: {dj} | 集数: {start}-{end} | top_k: {top_k}")
    logger.info("=" * 60)
    
    try:
        # ---------- 1. 确保拿到 txt ----------
        res_dir = comm_dir / "results" / dj
        txt_file = None
        md_file = None
        
        if res_dir.exists():
            for f in sorted(res_dir.iterdir(), reverse=True):
                if f.suffix == ".txt":
                    txt_file = f
                    break
        
        if txt_file:
            logger.info("⏭ 跳过 [提取/生成短剧文案]（txt已存在）")
        else:
            if res_dir.exists():
                for f in sorted(res_dir.iterdir(), reverse=True):
                    if f.suffix == ".md" and "prompt" not in f.name and "system" not in f.name:
                        md_file = f
                        logger.info(f"选中 md 文件: {f.name}")
                        break
            
            if md_file:
                run_step([
                    "python", str(comm_dir / "extract_final_script_to_text.py"),
                    str(md_file)
                ], "从 md 提取最终文案", logger)
                for f in sorted(res_dir.iterdir(), reverse=True):
                    if f.suffix == ".txt":
                        txt_file = f
                        break
            else:
                logger.info("▶ [生成短剧文案(text0204)]")
                run_step([
                    "python", str(comm_dir / "text0204.py"),
                    "--dj", dj,
                    "--start", start,
                    "--end", end,
                    "--output-dir", str(res_dir),
                ], "生成短剧文案(text0204)", logger)
                
                for f in sorted(res_dir.iterdir(), reverse=True):
                    if f.suffix == ".md" and "prompt" not in f.name and "system" not in f.name:
                        md_file = f
                        break
                
                if md_file:
                    run_step([
                        "python", str(comm_dir / "extract_final_script_to_text.py"),
                        str(md_file)
                    ], "从 md 提取最终文案", logger)
                    for f in sorted(res_dir.iterdir(), reverse=True):
                        if f.suffix == ".txt":
                            txt_file = f
                            break
        
        if not txt_file:
            raise RuntimeError("最终仍拿不到 txt 文案")
        
        # ---------- S1: 视频集数索引化 ----------
        s1_json = cache_dir / "structured_description.json"
        if s1_json.exists():
            logger.info("⏭ 跳过 [视频集数索引化]（JSON已存在）")
        else:
            cmd = ["python", str(match_dir / "skill_video_episode_indexer.py"), dj, start, end]
            run_step(cmd, "视频集数索引化", logger)
        
        # ---------- S2: 结构化信息索引化 ----------
        s2_pkl = cache_dir / "embeddings.pkl"
        if s2_pkl.exists():
            logger.info("⏭ 跳过 [结构化信息索引化]（pkl已存在）")
        else:
            run_step(["python", str(match_dir / "skill_struct_info_emb.py"), dj], "结构化信息索引化", logger)
        
        # ---------- S3: 解说文案拆分 ----------
        s3_json = cache_dir / f"{txt_file.stem}_structured.json"
        if s3_json.exists():
            logger.info("⏭ 跳过 [解说文案拆分]（JSON已存在）")
        else:
            cmd = ["python", str(match_dir / "skill_text_structured_description.py"), dj, str(txt_file), start, end]
            run_step(cmd, "解说文案拆分（集数索引版）", logger)
        
        # ---------- S4: 核心匹配与精排（四阶段） ----------
        if has_match_result(s3_json):
            logger.info("⏭ 跳过 [核心匹配与精排]（已含成功匹配）")
        else:
            run_step(["python", str(match_dir / "skill_match_and_rank.py"), dj, str(s3_json), top_k], "核心匹配与精排（四阶段）", logger)
        
        # ---------- S5: 视频合成 ----------
        final_mp4 = finish_dir / out_file
        if final_mp4.exists():
            logger.info("⏭ 跳过 [视频合成]（最终视频已存在）")
        else:
            run_step(["python", str(match_dir / "skill_postprocess.py"), dj, str(s3_json), out_file], "视频合成", logger)
        
        # ---------- S6: 生成final.json ----------
        final_json = cache_dir / f"{dj}_final.json"
        if final_json.exists():
            logger.info("⏭ 跳过 [生成final.json]（已存在）")
        else:
            logger.info("▶ [生成final.json]")
            generate_final_json(dj, s3_json, final_json, logger)
        
        logger.info("=" * 60)
        logger.info(f"🎉 {dj} 全部完成！")
        logger.info(f"最终视频: {final_mp4}")
        logger.info(f"final.json: {final_json}")
        logger.info("=" * 60)
        return (dj, True, "成功")
        
    except Exception as e:
        logger.error(f"💥 {dj} 处理失败: {e}")
        return (dj, False, str(e))

def main():
    if len(sys.argv) < 2:
        print("""用法:
1. 单剧模式: python run_mul.py <dj> <start> <end> [output] [top_k]
   例: python run_mul.py mamadechangshou5 1 15 mamadechangshou5.mp4 5

2. 批量并行模式: python run_mul.py --batch tasks.txt [top_k]
   tasks.txt 格式（每行一个剧，逗号分隔）:
   mamadechangshou5,1,15,mamadechangshou5.mp4
   another_drama,1,20,another.mp4
   third_drama,5,10
""")
        sys.exit(1)
    
    # 批量并行模式
    if sys.argv[1] == '--batch':
        batch_file = sys.argv[2]
        top_k = sys.argv[3] if len(sys.argv) > 3 else "5"
        
        if not Path(batch_file).exists():
            print(f"错误: 任务文件不存在: {batch_file}")
            sys.exit(1)
        
        tasks = []
        with open(batch_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 3:
                    dj, start, end = parts[0], parts[1], parts[2]
                    out = parts[3] if len(parts) > 3 else f"v4/{dj}_final.mp4"
                    tasks.append((dj, start, end, out, top_k))
        
        if not tasks:
            print("没有有效任务")
            sys.exit(1)
        
        print(f"🚀 准备并行启动 {len(tasks)} 个任务（四阶段匹配 + LLM精排）")
        print(f"📁 日志目录: {LOG_DIR}")
        print("-" * 60)
        
        processes = []
        for task in tasks:
            dj = task[0]
            p = Process(target=process_single_dj, args=task, name=f"Worker-{dj}")
            p.start()
            processes.append((dj, p))
            print(f"✅ 已启动 [{dj}] PID:{p.pid}")
            time.sleep(0.5)
        
        print("-" * 60)
        print("所有任务已启动，正在并行运行...")
        print("💡 提示: 可以另开终端查看各剧日志: tail -f logs/剧名_*.log")
        print("-" * 60)
        
        results = []
        for dj, p in processes:
            p.join()
            results.append((dj, p.exitcode == 0))
        
        success = sum(1 for _, ok in results if ok)
        print("=" * 60)
        print(f"全部完成: {success}/{len(results)} 成功")
        for dj, ok in results:
            status = "🎉" if ok else "💥"
            print(f"{status} {dj}")
        print("=" * 60)
        
        sys.exit(0 if success == len(results) else 1)
    
    # 单剧模式
    else:
        if len(sys.argv) < 4:
            print("用法: python run_mul.py <dj> <start> <end> [output] [top_k]")
            sys.exit(1)
        
        dj, start, end = sys.argv[1], sys.argv[2], sys.argv[3]
        out_file = sys.argv[4] if len(sys.argv) > 4 else f"v4/{dj}_final.mp4"
        top_k = sys.argv[5] if len(sys.argv) > 5 else "5"
        
        result = process_single_dj(dj, start, end, out_file, top_k)
        sys.exit(0 if result[1] else 1)

if __name__ == "__main__":
    main()
