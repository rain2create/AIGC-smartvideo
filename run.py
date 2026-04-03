#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
影视解说-画面匹配完整Pipeline（索引化版本）
规则：
1. S1 视频索引化：按集数处理，不再合并
2. S2 索引化embedding：按集数分组
3. S3 文案拆分：融合suoying，按集数边界分段
4. S4 核心匹配：基于episode_target在对应集内匹配
5. S5 视频合成：基于全局时间戳合成
用法：python run.py <dj> [start] [end] [output_filename] [top_k]
"""
import sys
import subprocess
import json
from pathlib import Path
import logging
import coloredlogs

logger = logging.getLogger("Pipeline")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(name)s] %(message)s", datefmt="%H:%M:%S")
try: 
    coloredlogs.install(logger=logger)
except Exception: 
    pass

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
        return any(seg.get("match_result", {}).get("status") == "success" for seg in data)
    except Exception:
        return False


# -------------------- 主流程 --------------------
def main():
    if len(sys.argv) < 2:
        print("用法: python run.py <dj> [start] [end] [output_filename] [top_k]")
        print("示例: python run.py buxi5 1 15")
        sys.exit(1)

    dj = sys.argv[1]
    start = sys.argv[2] if len(sys.argv) > 2 else None  # 可选，默认扫描全部
    end   = sys.argv[3] if len(sys.argv) > 3 else None
    out_file = sys.argv[4] if len(sys.argv) > 4 else f"{dj}_final.mp4"
    top_k    = sys.argv[5] if len(sys.argv) > 5 else "10"

    base = Path(__file__).parent.absolute()
    comm_dir   = base / "video_commentary_text"
    match_dir  = base / "text_video_match"
    finish_dir = base / "finish"
    finish_dir.mkdir(exist_ok=True)
    cache_dir  = match_dir / "cache" / dj

    logger.info("*" * 80)
    logger.info("影视解说-画面匹配Pipeline（索引化版本）")
    logger.info(f"dj={dj}  start={start}  end={end}  output={out_file}  top_k={top_k}")
    logger.info("*" * 80)

    try:
        # ---------- 1. 确保拿到 txt ----------
        res_dir  = comm_dir / "results" / dj
        txt_file = None
        md_file  = None

        # ① 先找现成 txt
        if res_dir.exists():
            for f in sorted(res_dir.iterdir(), reverse=True):
                if f.suffix == ".txt":
                    txt_file = f
                    break

        if txt_file:
            logger.info("⏭  跳过 【提取/生成短剧文案】（txt 已存在）")
        else:
            # ② 没有 txt → 找最新 md
            if res_dir.exists():
                for f in sorted(res_dir.iterdir(), reverse=True):
                    if f.suffix == ".md" and "prompt" not in f.name and "system" not in f.name:
                        md_file = f
                        logger.info(f"选中 md 文件: {f.name}")
                        break

            if md_file:
                logger.info("▶  开始步骤 【从 md 提取最终文案】")
                run_step([
                    "python", str(comm_dir / "extract_final_script_to_text.py"),
                    str(md_file)
                ], "从 md 提取最终文案")
                for f in sorted(res_dir.iterdir(), reverse=True):
                    if f.suffix == ".txt":
                        txt_file = f
                        break
            else:
                # ③ 生成文案
                logger.info("▶  开始步骤 【生成短剧文案】")
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                run_step([
                    "python", str(comm_dir / "text0204.py"),
                    "--dj", dj,
                    "--start", start or "1",
                    "--end", end or "20",
                    "--output-dir", str(res_dir),
                ], "生成短剧文案(text0204)")
                
                # 提取为txt
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
                    for f in sorted(res_dir.iterdir(), reverse=True):
                        if f.suffix == ".txt":
                            txt_file = f
                            break

        if not txt_file:
            raise RuntimeError("最终仍拿不到 txt 文案")

        # ---------- S1: 视频集数索引化（替代原来的合并+切割）----------
        s1_json = cache_dir / "structured_description.json"
        if s1_json.exists():
            logger.info("⏭  跳过 【视频集数索引化】（JSON已存在）")
        else:
            # 新模块：按集数索引化，输入start/end可选
            cmd = ["python", str(match_dir / "skill_video_episode_indexer.py"), dj]
            if start:
                cmd.append(start)
            if end:
                cmd.append(end)
            run_step(cmd, "视频集数索引化")

        # ---------- S2: 结构化信息索引化（按集数分组）----------
        s2_pkl = cache_dir / "embeddings.pkl"
        if s2_pkl.exists():
            logger.info("⏭  跳过 【结构化信息索引化】（pkl已存在）")
        else:
            run_step(["python", str(match_dir / "skill_struct_info_emb.py"), dj], "结构化信息索引化")

        # ---------- S3: 解说文案拆分（融合suoying，按集数边界分段）----------
        s3_json = cache_dir / f"{txt_file.stem}_structured.json"
        if s3_json.exists():
            logger.info("⏭  跳过 【解说文案拆分】（JSON已存在）")
        else:
            # 新参数：传入start/end用于suoying映射
            cmd = ["python", str(match_dir / "skill_text_structured_description.py"), dj, str(txt_file)]
            if start:
                cmd.append(start)
            if end:
                cmd.append(end)
            run_step(cmd, "解说文案拆分（集数索引版）")

        # ---------- S4: 核心匹配与精排（基于episode_target在对应集内匹配）----------
        if has_match_result(s3_json):
            logger.info("⏭  跳过 【核心匹配与精排】（已含成功匹配）")
        else:
            run_step(["python", str(match_dir / "skill_match_and_rank.py"), dj, str(s3_json), top_k], "核心匹配与精排（集数索引版）")

        # ---------- S5: 视频合成（基于全局时间戳）----------
        final_mp4 = finish_dir / out_file
        if final_mp4.exists():
            logger.info("⏭  跳过 【视频合成】（最终视频已存在）")
        else:
            run_step(["python", str(match_dir / "skill_postprocess.py"), dj, str(s3_json), out_file], "视频合成")

        # ---------- S6: 生成final.json（根据memory要求）----------
        final_json = cache_dir / f"{dj}_final.json"
        if final_json.exists():
            logger.info("⏭  跳过 【生成final.json】（已存在）")
        else:
            logger.info("▶  开始步骤 【生成final.json】")
            # 从s3_json读取并生成final.json
            generate_final_json(dj, s3_json, final_json)
            logger.info(f"✓ final.json 已生成: {final_json}")

        logger.info("*" * 80)
        logger.info("✓ Pipeline 真正执行成功！")
        logger.info(f"最终视频: {final_mp4}")
        logger.info(f"final.json: {final_json}")
        logger.info("*" * 80)

    except Exception as e:
        logger.error(f"✗ Pipeline 中断: {e}")
        sys.exit(1)


def generate_final_json(dj: str, structured_json: Path, output_path: Path):
    """
    生成final.json：基于structured_description.json，只保留每个segment的top1匹配
    """
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
        logger.info(f"final.json 生成完成: {len(final_output['segments'])} 个segments")
        
    except Exception as e:
        logger.error(f"生成final.json失败: {e}")
        raise


if __name__ == "__main__":
    main()