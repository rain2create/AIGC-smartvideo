#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
影视解说-画面匹配Pipeline（真·并行版）
特点：
- 每个剧独立进程，各跑各的流水线（S0->S1->S2...不等待其他剧）
- 快的剧先跑完，慢的剧慢慢跑，互不阻塞
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

# 配置
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

def setup_task_logger(dj: str):
    """为每个剧创建独立日志文件"""
    logger = logging.getLogger(dj)
    logger.setLevel(logging.INFO)
    
    # 清除旧handler（避免重复）
    logger.handlers = []
    
    # 文件handler（每个剧独立文件）
    fh = logging.FileHandler(LOG_DIR / f"{dj}_{datetime.now().strftime('%m%d_%H%M%S')}.log", encoding='utf-8')
    fh.setFormatter(logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", "%H:%M:%S"))
    logger.addHandler(fh)
    
    # 控制台handler（带剧名前缀，方便一眼区分）
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(f"[%(asctime)s][{dj}][%(levelname)s] %(message)s", "%H:%M:%S"))
    logger.addHandler(ch)
    
    return logger

def run_step(cmd, step_name, logger):
    """单步执行"""
    logger.info(f"▶ [{step_name}]")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info(f"✓ [{step_name}] 完成")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ [{step_name}] 失败: {e}")
        logger.error(f"stderr: {e.stderr[:500]}")
        raise

def process_single_dj(dj: str, start: str, end: str, out_file: str, top_k: str):
    """
    单个剧的完整流水线（独立进程中运行）
    各步骤严格顺序执行，但与其他剧互不等待
    """
    logger = setup_task_logger(dj)
    base = Path(__file__).parent.absolute()
    comm_dir = base / "video_commentary_text"
    match_dir = base / "text_video_match"
    finish_dir = base / "finish"
    finish_dir.mkdir(exist_ok=True)
    cache_dir = match_dir / "cache" / dj
    
    logger.info("="*60)
    logger.info(f"开始处理: {dj} | 集数: {start}-{end}")
    logger.info("="*60)
    
    try:
        # S0: 合并视频
        merged_path = Path(f"/mnt_liuhuan/search_public/xby/smart_video/clip/merge/{dj}_merged.mp4")
        if merged_path.exists():
            logger.info("⏭ 跳过 [合并视频]（已存在）")
        else:
            run_step([
                "python", str(base / "clip" / "merge_videos_by_episodes.py"),
                dj, start, end
            ], "合并视频", logger)
        
     # S1: 获取文案
        res_dir = comm_dir / "results" / dj
        txt_file = None
        
        if res_dir.exists():
            txt_file = next((f for f in sorted(res_dir.iterdir(), reverse=True) if f.suffix == ".txt"), None)
        
        if txt_file:
            logger.info("⏭ 跳过 [获取文案]（txt已存在）")
        else:
            # 没有txt，找md（只有目录存在时才找）
            md_file = None
            if res_dir.exists():
                md_file = next((f for f in sorted(res_dir.iterdir(), reverse=True) 
                            if f.suffix == ".md" and "prompt" not in f.name), None)
            
            if md_file:
                run_step(["python", str(comm_dir / "extract_final_script_to_text.py"), str(md_file)], 
                        "MD提取文案", logger)
            else:
                # 目录不存在 或 没有md → 直接生成
                logger.info("▶ [生成文案(text0204)]")
                
                run_step([
                    "python", str(comm_dir / "text0204.py"),
                    "--dj", dj,
                    "--start", start,
                    "--end", end,
                    "--output-dir", str(res_dir),
                ], "生成文案(text0204)", logger)
                
                # 提取生成的md为txt
                if res_dir.exists():
                    md_file = next((f for f in sorted(res_dir.iterdir(), reverse=True) 
                                 if f.suffix == ".md" and "prompt" not in f.name and "system" not in f.name), None)
                    if md_file:
                        run_step(["python", str(comm_dir / "extract_final_script_to_text.py"), str(md_file)],
                                "MD提取文案", logger)
            
            # 重新找txt
            txt_file = next((f for f in sorted(res_dir.iterdir(), reverse=True) if f.suffix == ".txt"), None) if res_dir.exists() else None
        
        if not txt_file:
            raise RuntimeError("无法获取文案")
        
        # S2: 视频结构化
        s1_json = cache_dir / "structured_description.json"
        if s1_json.exists():
            logger.info("⏭ 跳过 [视频结构化]（JSON已存在）")
        else:
            run_step(["python", str(match_dir / "skill_video_structured_description.py"), dj],
                    "视频结构化", logger)
        
        # S3: 索引化
        s2_pkl = cache_dir / "embeddings.pkl"
        if s2_pkl.exists():
            logger.info("⏭ 跳过 [索引化]（pkl已存在）")
        else:
            run_step(["python", str(match_dir / "skill_struct_info_emb.py"), dj],
                    "索引化", logger)
        
        # S4: 文案拆分
        s3_json = cache_dir / f"{txt_file.stem}_structured.json"
        if s3_json.exists():
            logger.info("⏭ 跳过 [文案拆分]（JSON已存在）")
        else:
            run_step(["python", str(match_dir / "skill_text_structured_description.py"), dj, str(txt_file)],
                    "文案拆分", logger)
        
        # S5: 匹配与精排
        if s3_json.exists():
            data = json.loads(s3_json.read_text(encoding="utf8"))
            has_match = any(seg.get("match_result", {}).get("status") == "success" for seg in data)
            if has_match:
                logger.info("⏭ 跳过 [匹配精排]（已含匹配）")
            else:
                run_step(["python", str(match_dir / "skill_match_and_rank.py"), dj, str(s3_json), top_k],
                        "匹配精排", logger)
        
        # S6: 视频合成
        final_mp4 = finish_dir / out_file
        if final_mp4.exists():
            logger.info("⏭ 跳过 [视频合成]（已存在）")
        else:
            run_step(["python", str(match_dir / "skill_postprocess.py"), dj, str(s3_json), out_file],
                    "视频合成", logger)
        
        logger.info(f"🎉 {dj} 全部完成！输出: {final_mp4}")
        return (dj, True, "成功")
        
    except Exception as e:
        logger.error(f"💥 {dj} 处理失败: {e}")
        return (dj, False, str(e))

def main():
    if len(sys.argv) < 2:
        print("""用法:
1. 单剧模式: python run.py <dj> <start> <end> [output] [top_k]
   例: python run.py mamadechangshou5 1 15 mamadechangshou5.mp4 10

2. 批量并行模式: python run.py --batch tasks.txt [top_k]
   tasks.txt 格式（每行一个剧，逗号分隔）:
   mamadechangshou5,1,15,mamadechangshou5.mp4
   another_drama,1,20,another.mp4
   third_drama,5,10
""")
        sys.exit(1)
    
    # 批量并行模式
    if sys.argv[1] == '--batch':
        batch_file = sys.argv[2]
        top_k = sys.argv[3] if len(sys.argv) > 3 else "10"
        
        if not Path(batch_file).exists():
            print(f"错误: 任务文件不存在: {batch_file}")
            sys.exit(1)
        
        # 读取任务
        tasks = []
        with open(batch_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 3:
                    dj, start, end = parts[0], parts[1], parts[2]
                    out = parts[3] if len(parts) > 3 else f"{dj}_final.mp4"
                    tasks.append((dj, start, end, out, top_k))
        
        if not tasks:
            print("没有有效任务")
            sys.exit(1)
        
        print(f"🚀 准备并行启动 {len(tasks)} 个任务（每个剧独立进程，互不等待）")
        print(f"📁 日志目录: {LOG_DIR}")
        print("-" * 60)
        
        # 启动所有进程（真正的并行，各跑各的）
        processes = []
        for task in tasks:
            dj = task[0]
            p = Process(target=process_single_dj, args=task, name=f"Worker-{dj}")
            p.start()
            processes.append((dj, p))
            print(f"✅ 已启动 [{dj}] PID:{p.pid}")
            time.sleep(0.5)  # 稍微错开启动，避免同时抢资源
        
        print("-" * 60)
        print("所有任务已启动，正在并行运行...")
        print("💡 提示: 可以另开终端查看各剧日志: tail -f logs/剧名_*.log")
        print("-" * 60)
        
        # 等待所有进程完成（各自跑各自的，互不等待）
        results = []
        for dj, p in processes:
            p.join()
            results.append((dj, p.exitcode == 0))
        
        # 汇总
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
            print("用法: python run.py <dj> <start> <end> [output] [top_k]")
            sys.exit(1)
        
        dj, start, end = sys.argv[1], sys.argv[2], sys.argv[3]
        out_file = sys.argv[4] if len(sys.argv) > 4 else f"{dj}_final.mp4"
        top_k = sys.argv[5] if len(sys.argv) > 5 else "10"
        
        result = process_single_dj(dj, start, end, out_file, top_k)
        sys.exit(0 if result[1] else 1)

if __name__ == "__main__":
    main()