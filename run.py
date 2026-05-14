#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
影视解说-画面匹配完整Pipeline
规则：
1. S3 只判断 JSON 存在即跳过
2. S4 判断 JSON 里**已含 match_result** 才跳过
3. 任何一步失败立即 exit
用法：python run.py <dj> <start> <end> [output_filename] [top_k]
"""
import sys
import subprocess
import json
from pathlib import Path
import logging
import coloredlogs  # 可选

logger = logging.getLogger("Pipeline")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(name)s] %(message)s", datefmt="%H:%M:%S")
try: coloredlogs.install(logger=logger)
except Exception: pass

def run_step(cmd, step_name):
    logger.info(f"▶  开始步骤 【{step_name}】")
    logger.info(f"执行: {' '.join(cmd)}")
    subprocess.run(cmd, stdout=None, stderr=None, check=True)
    logger.info(f"✓ {step_name} 完成")

def has_match_result(json_path: Path) -> bool:
    """JSON 存在且**至少一个片段**已含 match_result.status == success"""
    if not json_path.exists():
        return False
    try:
        data = json.loads(json_path.read_text(encoding="utf8"))
        # 任意一个片段有成功匹配即可
        return any(seg.get("match_result", {}).get("status") == "success" for seg in data)
    except Exception:
        return False
    
def run_merge_step(dj: str, start: int, end: int):
    logger.info(f"▶ 开始步骤 【合并视频】")
    cmd = [
        "python",
        str(Path(__file__).parent / "clip" / "merge_videos_by_episodes.py"),
        dj,
        str(start),
        str(end)
    ]
    logger.info(f"执行: {' '.join(cmd)}")
    subprocess.run(cmd, stdout=None, stderr=None, check=True)
    logger.info(f"✓ 合并视频完成")



# -------------------- 主流程 --------------------
def main():
    if len(sys.argv) < 4:
        print("用法: python run.py <dj> <start> <end> [output_filename] [top_k]")
        sys.exit(1)

    dj = sys.argv[1]
    start = sys.argv[2] if len(sys.argv) > 2 else "1"
    end   = sys.argv[3] if len(sys.argv) > 3 else "20"
    out_file = sys.argv[4] if len(sys.argv) > 4 else f"{dj}_final.mp4"
    top_k    = sys.argv[5] if len(sys.argv) > 5 else "10"

    base = Path(__file__).parent.absolute()
    clip_dir   = base / "clip"
    comm_dir   = base / "video_commentary_text"
    match_dir  = base / "text_video_match"
    finish_dir = base / "finish"
    finish_dir.mkdir(exist_ok=True)
    cache_dir  = match_dir / "cache" / dj

    logger.info("*" * 80)
    logger.info("影视解说-画面匹配完整Pipeline（S4 内容校验 & 失败即停）")
    logger.info(f"dj={dj}  start={start}  end={end}  output={out_file}  top_k={top_k}")
    logger.info("*" * 80)

       # ---------- 0. 合并视频（第一步） ----------
    merged_video_path = Path(f"/mnt_liuhuan/search_public/xby/smart_video/clip/merge/{dj}_merged.mp4")
    if merged_video_path.exists():
        logger.info("⏭ 跳过 【合并视频】（已存在）")
    else:
        run_merge_step(dj, int(start), int(end))

    try:
        # ---------- 1. 确保拿到 txt ----------
        res_dir  = comm_dir / "results" / dj
        txt_file = None
        md_file  = None

        # ① 先找现成 txt（任意非 eval）
        if res_dir.exists():
            for f in sorted(res_dir.iterdir(), reverse=True):
                if f.suffix == ".txt":
                    txt_file = f
                    break

        if txt_file:
            logger.info("⏭  跳过 【提取/生成短剧文案】（txt 已存在）")
        else:
            # ② 没有 txt → 找最新 md（非 eval）
            if res_dir.exists():
                for f in sorted(res_dir.iterdir(), reverse=True):
                    if f.suffix == ".md" and "prompt" not in f.name and "system" not in f.name:
                        md_file = f
                        logger.info(f"选中 md 文件: {f.name}")
                        break

            if md_file:                       # 有 md 就提取
                logger.info("▶  开始步骤 【从 md 提取最终文案】")
                run_step([
                    "python", str(comm_dir / "extract_final_script_to_text.py"),
                    str(md_file)
                ], "从 md 提取最终文案")
                # 提取完再捡 txt
                for f in sorted(res_dir.iterdir(), reverse=True):
                    if f.suffix == ".txt" :
                        txt_file = f
                        break
            else:                             # 连 md 都没有 → 生成
                logger.info("▶  开始步骤 【生成短剧文案】")
                    # 计算输出文件名（用于提取最终文案）
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                base_filename = f"{start}_{end}_{timestamp}"
                
                run_step([
                    "python", str(comm_dir / "text0204.py"),
                    "--dj", dj,
                    "--start", start,
                    "--end", end,
                    "--output-dir", str(res_dir),  # 确保输出到 results/{dj} 目录
                ], "生成短剧文案(text0204)")
                
                # text0204 生成的是 .md 文件，需要提取为 .txt
                # 找最新生成的 md 文件（非 eval）
                md_file = None
                for f in sorted(res_dir.iterdir(), reverse=True):
                   if f.suffix == ".md" and "prompt" not in f.name and "system" not in f.name:
                        md_file = f
                        break
                
                if md_file:
                    logger.info("▶  开始步骤 【从 md 提取最终文案】")
                    run_step([
                        "python", str(comm_dir / "extract_final_script_to_text.py"),
                        str(md_file)
                    ], "从 md 提取最终文案")
                    # 再捡 txt
                    for f in sorted(res_dir.iterdir(), reverse=True):
                        if f.suffix == ".txt" :
                            txt_file = f
                            break

        if not txt_file:
            raise RuntimeError("最终仍拿不到 txt 文案")

        # ---------- S1: 视频结构化理解 ----------
        s1_json = cache_dir / "structured_description.json"
        if s1_json.exists():
            logger.info("⏭  跳过 【视频结构化理解】（JSON已存在）")
        else:
            run_step(["python", str(match_dir / "skill_video_structured_description.py"), dj], "视频结构化理解")

        # ---------- S2: 结构化信息索引化 ----------
        s2_pkl = cache_dir / "embeddings.pkl"
        if s2_pkl.exists():
            logger.info("⏭  跳过 【结构化信息索引化】（pkl已存在）")
        else:
            run_step(["python", str(match_dir / "skill_struct_info_emb.py"), dj], "结构化信息索引化")

        # ---------- S3: 解说文案拆分 ----------
        s3_json = cache_dir / f"{txt_file.stem}_structured.json"
        if s3_json.exists():
            logger.info("⏭  跳过 【解说文案拆分】（JSON已存在）")
        else:
            run_step(["python", str(match_dir / "skill_text_structured_description.py"), dj, str(txt_file)], "解说文案拆分")

        # ---------- S4: 核心匹配与精排 ----------
        # 关键：必须已含 match_result 才跳过
        if has_match_result(s3_json):
            logger.info("⏭  跳过 【核心匹配与精排】（已含成功匹配）")
        else:
            run_step(["python", str(match_dir / "skill_match_and_rank.py"), dj, str(s3_json), top_k], "核心匹配与精排")

        # ---------- S5: 视频合成 ----------
        final_mp4 = finish_dir / out_file
        if final_mp4.exists():
            logger.info("⏭  跳过 【视频合成】（最终视频已存在）")
        else:
            run_step(["python", str(match_dir / "skill_postprocess.py"), dj, str(s3_json), out_file], "视频合成")

        logger.info("*" * 80)
        logger.info("✓ Pipeline 真正执行成功！")
        logger.info(f"最终视频: {final_mp4}")
        logger.info("*" * 80)

    except Exception as e:
        logger.error(f"✗ Pipeline 中断: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()