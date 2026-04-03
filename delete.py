#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
影视解说-指定剧清理脚本（只保留 structured_description.json / embeddings / segment）
功能：删除 S2~S5 流程中指定 dj 的中间文件和最终视频
用法: 
    直接修改下方 DJ_LIST 后运行: python delete.py
"""
import sys
import logging
from pathlib import Path
import shutil

# ============ 在这里填写要清理的短剧名 ============
DJ_LIST = [
     "luxiansheng",
     "chongfan80",
     "duibuqinidexiaohaishishebao"
]
# ================================================

logger = logging.getLogger("CleanPipeline")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(name)s] %(message)s",
    datefmt="%H:%M:%S"
)


def remove_file(path: Path):
    if path.exists():
        try:
            path.unlink()
            logger.info(f"删除文件: {path}")
        except Exception as e:
            logger.error(f"删除文件失败: {path} -> {e}")


def remove_dir(path: Path):
    if path.exists() and path.is_dir():
        try:
            shutil.rmtree(path)
            logger.info(f"删除目录: {path}")
        except Exception as e:
            logger.error(f"删除目录失败: {path} -> {e}")


def load_tasks(file_path: Path):
    """从 tasks.txt 加载 dj 列表，支持格式: dj,1,15 或 dj"""
    djs = []
    if not file_path.exists():
        return djs
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 解析格式: buxi5,1,15 -> 取第一部分
            dj = line.split(',')[0].strip()
            if dj:
                djs.append(dj)
    return djs


def process_dj(dj):
    """处理单个剧的清理逻辑"""
    base = Path(__file__).parent.absolute()
    match_dir = base / "text_video_match"
    cache_dir = match_dir / "cache" / dj
    finish_dir = base / "finish"
    
    # 文案侧 results 目录
    text_results_dir = base / "video_commentary_text" / "results" / dj

    logger.info(f"开始清理 pipeline 文件，dj={dj}（保留 structured / emb / segment）")

    KEEP_FILES = {
       "structured_description.json",
       "embeddings.pkl",
    }

    KEEP_DIR_PREFIX = (
       "segment",   # segment / segments / segment_xxx
    )

    # ---------- 遍历 cache_dir ----------
    if cache_dir.exists() and cache_dir.is_dir():
        for f in cache_dir.iterdir():
            # 保留指定文件
            if f.is_file() and f.name in KEEP_FILES:
                logger.info(f"保留文件: {f}")
                continue

            # 保留 segment 相关目录
            if f.is_dir() and f.name.startswith(KEEP_DIR_PREFIX):
                logger.info(f"保留目录: {f}")
                continue

            # 其余全部删除
            remove_file(f) if f.is_file() else remove_dir(f)

    # ---------- 删除 finish_dir 下的视频 ----------
    for f in finish_dir.glob(f"{dj}*"):
        remove_file(f)

    # ---------- 删除最终合成视频 ----------
    merged_video_path = Path(f"/mnt_liuhuan/search_public/xby/smart_video/finish/{dj}_final.mp4")
    remove_file(merged_video_path)
    
    logger.info(f"✓ {dj} 清理完成（segment / emb / structured 已保留）")

        # ---------- 删除文案侧 results 目录 ----------
    if text_results_dir.exists() and text_results_dir.is_dir():
        remove_dir(text_results_dir)
    logger.info(f"✓ {dj} 清理完成（segment / emb / structured 已保留，文案侧已清理）")


def main():
    args = DJ_LIST if DJ_LIST else sys.argv[1:]
    
    if not args:
        print("请在脚本开头的 DJ_LIST 中填写要清理的短剧名，或通过命令行参数传入")
        print("用法: python delete.py <dj> [dj2 ...]")
        sys.exit(1)
    
    logger.info(f"准备清理 {len(args)} 个短剧: {args}")
    
    success_count = 0
    for dj in args:
        try:
            process_dj(dj)
            success_count += 1
            print()
        except Exception as e:
            logger.error(f"处理 {dj} 时出错: {e}")
    
    logger.info(f"清理完成，共 {len(args)} 个，成功 {success_count} 个")


if __name__ == "__main__":
    main()