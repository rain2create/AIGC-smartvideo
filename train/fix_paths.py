"""
fix_paths.py  —  将 splits JSONL 中的绝对视频路径改为相对路径

各机器 data03 挂载点不同（/home/jovyan/data03 vs /nlp_liuhuan_data03 等），
统一截取 data03/ 后的部分作为相对路径，例如：
  /home/jovyan/data03/search_public/xby/...
  -> search_public/xby/...

训练脚本通过 --video-root 指定本机的 data03 根目录来还原完整路径。
"""
import json
import re
import sys
from pathlib import Path

SPLITS_DIR = Path(__file__).parent.parent / "judge" / "training_data" / "v3" / "splits"
JSONL_FILES = [
    "train_pairs.jsonl", "val_pairs.jsonl", "test_pairs.jsonl",
    "train_groups.jsonl", "val_groups.jsonl", "test_groups.jsonl",
    "train_samples.jsonl", "val_samples.jsonl", "test_samples.jsonl",
]

# 匹配任意 data03/ 前缀
_PREFIX_RE = re.compile(r"^.*?data03/")


def to_relative(path: str) -> str:
    """把绝对路径截取为 data03/ 后的相对路径，已经是相对路径则原样返回。"""
    return _PREFIX_RE.sub("", path)


def fix_record(record: dict) -> dict:
    """递归替换 record 中所有 video_segment_path 字段。"""
    if isinstance(record, dict):
        return {
            k: (to_relative(v) if k == "video_segment_path" and isinstance(v, str) else fix_record(v))
            for k, v in record.items()
        }
    if isinstance(record, list):
        return [fix_record(item) for item in record]
    return record


def fix_file(path: Path, dry_run: bool = False) -> int:
    lines = path.read_text().splitlines()
    fixed = []
    changed = 0
    for line in lines:
        if not line.strip():
            fixed.append(line)
            continue
        record = json.loads(line)
        new_record = fix_record(record)
        fixed.append(json.dumps(new_record, ensure_ascii=False))
        if new_record != record:
            changed += 1
    if not dry_run:
        path.write_text("\n".join(fixed) + "\n")
    return changed


def main():
    dry_run = "--dry-run" in sys.argv
    if dry_run:
        print("[dry-run] 不写入文件，仅统计变更行数")

    total_changed = 0
    for name in JSONL_FILES:
        p = SPLITS_DIR / name
        if not p.exists():
            print(f"  skip (not found): {name}")
            continue
        changed = fix_file(p, dry_run=dry_run)
        total_changed += changed
        print(f"  {name}: {changed} lines updated")

    print(f"\n合计修改行数: {total_changed}")
    if dry_run:
        print("重新运行（去掉 --dry-run）以实际写入。")


if __name__ == "__main__":
    main()
