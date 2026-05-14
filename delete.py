#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量清理指定短剧的匹配缓存与产物，保留文案侧 results。

默认来源：
  1. task.txt
  2. task2.txt
  3. task3.txt
  4. DJ_LIST（可选补充）

清理范围：
  - text_video_match/cache/<dj>/ 整个目录
  - finish/ 下所有匹配 <dj>* 的文件（递归）
  - 兼容删除旧路径 finish 成品视频

保留范围：
  - video_commentary_text/results/<dj>/ 文案结果
"""

import logging
import shutil
import sys
from pathlib import Path


TASK_FILES = [
    "task.txt"
]

# 可手工补充。默认留空，主要走 task 文件。
DJ_LIST = []


logger = logging.getLogger("CleanPipeline")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)


def remove_file(path: Path):
    if path.exists():
        try:
            path.unlink()
            logger.info(f"删除文件: {path}")
        except Exception as exc:
            logger.error(f"删除文件失败: {path} -> {exc}")


def remove_dir(path: Path):
    if path.exists() and path.is_dir():
        try:
            shutil.rmtree(path)
            logger.info(f"删除目录: {path}")
        except Exception as exc:
            logger.error(f"删除目录失败: {path} -> {exc}")


def load_tasks(file_path: Path):
    """从任务文件加载 dj 列表，支持格式: dj,1,10 或 dj。"""
    djs = []
    if not file_path.exists():
        return djs

    for line in file_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        dj = line.split(",")[0].strip()
        if dj:
            djs.append(dj)
    return djs


def dedupe_keep_order(items):
    seen = set()
    out = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def resolve_default_djs(base: Path):
    task_djs = []
    for name in TASK_FILES:
        file_path = base / name
        loaded = load_tasks(file_path)
        if loaded:
            logger.info(f"任务文件 {name}: 加载 {len(loaded)} 个短剧")
            task_djs.extend(loaded)
    return dedupe_keep_order(task_djs + DJ_LIST)


def process_dj(base: Path, dj: str):
    match_dir = base / "text_video_match"
    cache_dir = match_dir / "cache" / dj
    finish_dir = base / "finish"
    text_results_dir = base / "video_commentary_text" / "results" / dj

    logger.info(f"开始清理，dj={dj}（删除 cache 与 finish，保留文案 results）")

    remove_dir(cache_dir)

    if finish_dir.exists():
        for path in finish_dir.rglob(f"{dj}*"):
            if path.is_file():
                remove_file(path)

    legacy_finish_paths = [
        Path(f"/mnt_liuhuan/search_public/xby/smart_video/finish/{dj}_final.mp4"),
        Path(f"/home/jovyan/data03/search_public/xby/smart_videoV4/finish/{dj}_final.mp4"),
        Path(f"/home/jovyan/data03/search_public/xby/smart_videoV6/finish/{dj}_final.mp4"),
    ]
    for path in legacy_finish_paths:
        remove_file(path)

    if text_results_dir.exists():
        logger.info(f"保留文案目录: {text_results_dir}")
    else:
        logger.info(f"文案目录不存在，无需保留: {text_results_dir}")

    logger.info(f"✓ {dj} 清理完成")


def main():
    base = Path(__file__).parent.absolute()
    args = sys.argv[1:] or resolve_default_djs(base)

    if not args:
        print("未找到待清理短剧。")
        print("可直接传参: python delete.py <dj> [dj2 ...]")
        print(f"默认会读取: {', '.join(TASK_FILES)}")
        sys.exit(1)

    djs = dedupe_keep_order(args)
    logger.info(f"准备清理 {len(djs)} 个短剧: {djs}")

    success_count = 0
    for dj in djs:
        try:
            process_dj(base, dj)
            success_count += 1
            print()
        except Exception as exc:
            logger.error(f"处理 {dj} 时出错: {exc}")

    logger.info(f"清理完成，共 {len(djs)} 个，成功 {success_count} 个")


if __name__ == "__main__":
    main()
