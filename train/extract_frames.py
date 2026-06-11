"""
extract_frames.py  —  预抽帧脚本

遍历 splits/ 中所有 pairs/groups JSONL，提取唯一视频路径，
以 fps=1.0 / max_frames=8 抽帧，存为 JPEG：
  {frame_cache_dir}/{video_stem}/frame_{i:04d}.jpg

用法：
  python train/extract_frames.py \\
      --video-root /nlp_liuhuan_data03 \\
      --frame-cache-dir judge/training_data/frame_cache \\
      [--splits-dir judge/training_data/v3/splits] \\
      [--workers 8] \\
      [--dry-run]

  视频路径 = video_root / relative_path_in_jsonl
"""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


# ── 默认配置 ────────────────────────────────────────────────────────────────
DEFAULT_FPS = 1.0
DEFAULT_MAX_FRAMES = 8
DEFAULT_SPLITS_DIR = "judge/training_data/v3/splits"
DEFAULT_FRAME_CACHE_DIR = "judge/training_data/frame_cache"
DEFAULT_WORKERS = 8

PAIRS_FILES = ["train_pairs.jsonl", "val_pairs.jsonl", "test_pairs.jsonl"]
GROUPS_FILES = ["train_groups.jsonl", "val_groups.jsonl", "test_groups.jsonl"]


# ── 路径收集 ─────────────────────────────────────────────────────────────────
def collect_video_paths(splits_dir: Path, extra_jsonl: list[str] = None) -> set[str]:
    """返回所有 JSONL 中出现的唯一相对视频路径。"""
    paths: set[str] = set()
    for fname in PAIRS_FILES:
        fp = splits_dir / fname
        if not fp.exists():
            continue
        for line in fp.read_text().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            paths.add(d["chosen"]["video_segment_path"])
            paths.add(d["rejected"]["video_segment_path"])

    for fname in GROUPS_FILES:
        fp = splits_dir / fname
        if not fp.exists():
            continue
        for line in fp.read_text().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            for c in d["candidates"]:
                paths.add(c["video_segment_path"])

    # 额外 JSONL 文件（如 SFT 的 train_sft.jsonl）
    for extra_path in (extra_jsonl or []):
        fp = Path(extra_path)
        if not fp.exists():
            print(f"  [warn] extra jsonl not found: {fp}")
            continue
        for line in fp.read_text().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            if "video_segment_path" in d:
                paths.add(d["video_segment_path"])
            elif "chosen" in d:
                paths.add(d["chosen"]["video_segment_path"])
                paths.add(d["rejected"]["video_segment_path"])
            elif "candidates" in d:
                for c in d["candidates"]:
                    paths.add(c["video_segment_path"])

    return paths


# ── 抽帧核心 ─────────────────────────────────────────────────────────────────
def video_stem(rel_path: str) -> str:
    return Path(rel_path).stem


def frame_dir(cache_root: Path, rel_path: str) -> Path:
    return cache_root / video_stem(rel_path)


def extract_one(
    rel_path: str,
    video_root: Path,
    cache_root: Path,
    fps: float,
    max_frames: int,
    dry_run: bool,
) -> tuple[str, str]:
    """
    返回 (rel_path, status)，status 为 'ok' / 'skip' / 'error: ...'
    """
    out_dir = frame_dir(cache_root, rel_path)

    # 已抽帧且数量足够 → 跳过
    if out_dir.exists():
        existing = sorted(out_dir.glob("frame_*.jpg"))
        if len(existing) >= 1:
            return rel_path, "skip"

    video_path = video_root / rel_path
    if not video_path.exists():
        return rel_path, f"error: video not found ({video_path})"

    if dry_run:
        return rel_path, "dry-run"

    out_dir.mkdir(parents=True, exist_ok=True)

    # 用 ffmpeg 抽帧：先按 fps 抽，再截断到 max_frames
    # vf: fps滤镜 + 限帧数
    vf = f"fps={fps}"
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", vf,
        "-vframes", str(max_frames),
        "-q:v", "2",            # JPEG 质量（2=高质量）
        str(out_dir / "frame_%04d.jpg"),
    ]
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=60,
        )
        if result.returncode != 0:
            err = result.stderr.decode(errors="replace").strip().splitlines()[-1]
            return rel_path, f"error: ffmpeg {err}"

        frames = sorted(out_dir.glob("frame_*.jpg"))
        if not frames:
            return rel_path, "error: ffmpeg produced no frames"
        return rel_path, f"ok ({len(frames)} frames)"

    except subprocess.TimeoutExpired:
        return rel_path, "error: timeout"
    except Exception as e:
        return rel_path, f"error: {e}"


# ── 主流程 ───────────────────────────────────────────────────────────────────
def main():
    repo_root = Path(__file__).parent.parent

    parser = argparse.ArgumentParser(description="Pre-extract video frames to JPEG cache")
    parser.add_argument("--video-root", required=True,
                        help="data03 挂载根目录，如 /nlp_liuhuan_data03 或 /home/jovyan/data03")
    parser.add_argument("--frame-cache-dir", default=DEFAULT_FRAME_CACHE_DIR,
                        help=f"帧缓存目录（相对项目根或绝对路径，默认 {DEFAULT_FRAME_CACHE_DIR}）")
    parser.add_argument("--splits-dir", default=DEFAULT_SPLITS_DIR,
                        help=f"splits 目录（默认 {DEFAULT_SPLITS_DIR}）")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help="并发线程数")
    parser.add_argument("--dry-run", action="store_true",
                        help="只统计路径，不实际抽帧")
    parser.add_argument("--extra-jsonl", nargs="*", default=[],
                        help="额外的 JSONL 文件路径（如 train/sft/train_sft.jsonl）")
    args = parser.parse_args()

    video_root = Path(args.video_root)
    cache_root = Path(args.frame_cache_dir) if Path(args.frame_cache_dir).is_absolute() \
                 else repo_root / args.frame_cache_dir
    splits_dir = Path(args.splits_dir) if Path(args.splits_dir).is_absolute() \
                 else repo_root / args.splits_dir

    print(f"video_root      : {video_root}")
    print(f"frame_cache_dir : {cache_root}")
    print(f"splits_dir      : {splits_dir}")
    print(f"fps={args.fps}  max_frames={args.max_frames}  workers={args.workers}")
    if args.dry_run:
        print("[dry-run] 不写入任何文件")
    print()

    # 1. 收集路径
    all_paths = sorted(collect_video_paths(splits_dir, extra_jsonl=args.extra_jsonl))
    print(f"唯一视频路径数: {len(all_paths)}")

    if not all_paths:
        print("未找到视频路径，退出。")
        sys.exit(1)

    # 2. 并发抽帧
    counters = {"ok": 0, "skip": 0, "dry-run": 0, "error": 0}
    errors = []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(extract_one, p, video_root, cache_root,
                        args.fps, args.max_frames, args.dry_run): p
            for p in all_paths
        }
        done = 0
        total = len(futures)
        for fut in as_completed(futures):
            done += 1
            rel, status = fut.result()
            if status.startswith("error"):
                counters["error"] += 1
                errors.append((rel, status))
            elif status == "skip":
                counters["skip"] += 1
            elif status == "dry-run":
                counters["dry-run"] += 1
            else:
                counters["ok"] += 1

            if done % 200 == 0 or done == total:
                print(f"  [{done}/{total}] ok={counters['ok']} skip={counters['skip']} "
                      f"error={counters['error']}", flush=True)

    # 3. 汇报
    print("\n=== 完成 ===")
    for k, v in counters.items():
        if v:
            print(f"  {k}: {v}")

    if errors:
        print(f"\n错误详情（共 {len(errors)} 条）：")
        for rel, msg in errors[:20]:
            print(f"  {Path(rel).name}: {msg}")
        if len(errors) > 20:
            print(f"  ... 及 {len(errors) - 20} 条更多")
        sys.exit(1)


if __name__ == "__main__":
    main()
