#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将旧命名统一到新命名规范。
旧 → 新：
  qwen_b1            → qwen_b1_q3
  qwen_b2            → qwen_b2_q3
  qwen_b1_rerank_q1  → qwen_b1_q3_rerank_q1
  qwen_b1_rerank_q2  → qwen_b1_q3_rerank_q2

改名范围：
  1. text_video_match/cache/{dj}/*_structured_*.json
  2. text_video_match/cache/{dj}/*_final_*.json
  3. judge/outputs/{variant}/
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_ROOT = PROJECT_ROOT / "text_video_match" / "cache"
JUDGE_OUTPUTS = PROJECT_ROOT / "judge" / "outputs"

# 旧名 → 新名 映射（注意：rerank 后缀的要放前面，避免 _b1 误匹配 _b1_rerank）
RENAME_MAP = [
    ("_qwen_b1_rerank_q1", "_qwen_b1_q3_rerank_q1"),
    ("_qwen_b1_rerank_q2", "_qwen_b1_q3_rerank_q2"),
    ("_qwen_b1", "_qwen_b1_q3"),
    ("_qwen_b2", "_qwen_b2_q3"),
]


def _update_variant_field(json_path: Path, old_name: str, new_name: str):
    """更新 JSON 中的 variant 字段。"""
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return
    changed = False
    if isinstance(data, dict):
        if data.get("variant") == old_name:
            data["variant"] = new_name
            changed = True
        if data.get("variant_structured_json"):
            vsj = str(data["variant_structured_json"])
            if old_name in vsj:
                data["variant_structured_json"] = vsj.replace(old_name, new_name)
                changed = True
    if changed:
        json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  updated {json_path}")


def rename_variant_files():
    """重命名 S4 JSON 和 final JSON。"""
    for dj_dir in sorted(CACHE_ROOT.iterdir()):
        if not dj_dir.is_dir():
            continue

        for old_sfx, new_sfx in RENAME_MAP:
            for f in sorted(dj_dir.glob(f"*{old_sfx}.json")):
                new_name = f.name.replace(old_sfx, new_sfx)
                new_path = f.parent / new_name
                if new_path.exists():
                    print(f"  SKIP {f.name} → {new_name} (目标已存在)")
                    continue
                f.rename(new_path)
                print(f"  {f.name} → {new_name}")


def rename_judge_dirs():
    """重命名 judge/outputs 下的目录并更新内部 JSON。"""
    for old_name, new_name in [("qwen_b1_rerank_q1", "qwen_b1_q3_rerank_q1"),
                                ("qwen_b1_rerank_q2", "qwen_b1_q3_rerank_q2"),
                                ("qwen_b1", "qwen_b1_q3"),
                                ("qwen_b2", "qwen_b2_q3")]:
        old_dir = JUDGE_OUTPUTS / old_name
        new_dir = JUDGE_OUTPUTS / new_name
        if not old_dir.is_dir():
            continue
        if new_dir.exists():
            print(f"  SKIP {old_name}/ → {new_name}/ (目标已存在)")
            continue
        shutil.move(str(old_dir), str(new_dir))
        print(f"  judge/outputs/{old_name}/ → {new_name}/")

        # 递归更新目录内所有 JSON 的 variant 字段
        for json_f in new_dir.rglob("*.json"):
            _update_variant_field(json_f, old_name, new_name)


def main():
    print("=" * 60)
    print("1/2 重命名 S4 JSON / final JSON")
    print("=" * 60)
    rename_variant_files()

    print()
    print("=" * 60)
    print("2/2 重命名 judge/outputs 目录")
    print("=" * 60)
    rename_judge_dirs()

    print()
    print("完成。")


if __name__ == "__main__":
    main()
