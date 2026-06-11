#!/usr/bin/env python3
"""批量跑 100 部 canonical 剧的 builder stats，汇总全局分布。"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
JUDGE_DIR = Path(__file__).resolve().parent
CANONICAL_DIR = JUDGE_DIR / "outputs" / "canonical"

# 复用 builder 的核心函数
sys.path.insert(0, str(JUDGE_DIR))
from rl_data_builder import (
    build_samples,
    BUILDER_VERSION,
)


def main():
    drama_dirs = sorted(d for d in CANONICAL_DIR.iterdir() if d.is_dir())

    all_stats = {}
    global_counts = {
        "strong_positive": 0,
        "fallback_positive": 0,
        "dimension_negative": 0,
        "hard_negative": 0,
        "easy_negative": 0,
        "unusable": 0,
        "total_samples": 0,
        "total_dropped": 0,
        "dramas_ok": 0,
        "dramas_skip": 0,
        "dramas_fail": 0,
    }
    skipped = []
    failed = []

    for dj_dir in drama_dirs:
        dj = dj_dir.name
        # 跳过非剧目录（如 compare 输出）
        if dj.startswith(".") or dj in ("compare",):
            continue

        # 找最新 annotation 目录
        runs = sorted(d for d in dj_dir.iterdir() if d.is_dir() and d.name.startswith("annotation_"))
        if not runs:
            print(f"⏭ {dj}: 无 annotation 目录")
            skipped.append(dj)
            global_counts["dramas_skip"] += 1
            continue

        run_dir = runs[-1]
        run_id = run_dir.name

        # 检查必要文件
        results_path = run_dir / "results.jsonl"
        config_path = run_dir / "config.json"
        if not results_path.exists() or not config_path.exists():
            print(f"⏭ {dj}: 缺少 results.jsonl 或 config.json")
            skipped.append(dj)
            global_counts["dramas_skip"] += 1
            continue

        try:
            print(f"📊 {dj} ({run_id}) ...", end=" ", flush=True)
            out_dir = build_samples(
                dj=dj,
                run_id=run_id,
                dataset_version="v3",
                cache_dir=None,
                judge_output_dir=str(CANONICAL_DIR),
                output_dir=None,
            )
            stats = json.loads((out_dir / "stats.json").read_text())
            all_stats[dj] = stats

            dist = stats.get("labels", {}).get("bucket_counts", {})
            for bucket in global_counts:
                if bucket in dist:
                    global_counts[bucket] += dist[bucket]
                elif bucket in ["total_samples", "total_dropped", "dramas_ok", "dramas_skip", "dramas_fail"]:
                    pass  # skip meta keys
            global_counts["total_samples"] += stats.get("kept_samples", 0)
            global_counts["total_dropped"] += stats.get("dropped_samples", 0)
            global_counts["dramas_ok"] += 1
            print(f"✅ {stats.get('kept_samples', 0)} samples")
        except Exception as e:
            print(f"❌ {e}")
            failed.append((dj, str(e)))
            global_counts["dramas_fail"] += 1

    # ── 汇总 ──
    total = global_counts["total_samples"]
    total_w_drop = total + global_counts["total_dropped"]

    print("\n" + "=" * 60)
    print("100 剧全量 Stats 汇总")
    print("=" * 60)
    print(f"OK: {global_counts['dramas_ok']} | Skip: {global_counts['dramas_skip']} | Fail: {global_counts['dramas_fail']}")
    print(f"总样本: {total} | Dropped: {global_counts['total_dropped']} | 合计: {total_w_drop}")

    if total:
        print(f"\n{'Bucket':<25} {'数量':>8} {'占比':>8}")
        print("-" * 42)
        for bucket in ["strong_positive", "fallback_positive", "hard_negative", "dimension_negative", "easy_negative", "unusable"]:
            c = global_counts[bucket]
            pct = c / total * 100 if total else 0
            print(f"{bucket:<25} {c:>8} {pct:>7.1f}%")

        pos_total = global_counts["strong_positive"] + global_counts["fallback_positive"]
        neg_total = global_counts["hard_negative"] + global_counts["dimension_negative"] + global_counts["easy_negative"]
        print(f"\n正样本: {pos_total} ({pos_total/total*100:.1f}%)")
        print(f"负样本: {neg_total} ({neg_total/total*100:.1f}%)")
        print(f"Unusable: {global_counts['unusable']} ({global_counts['unusable']/total*100:.1f}%)")

    if skipped:
        print(f"\n⏭ 跳过 ({len(skipped)}): {', '.join(skipped)}")
    if failed:
        print(f"\n❌ 失败 ({len(failed)}):")
        for dj, err in failed:
            print(f"  {dj}: {err}")

    # 保存汇总
    summary_path = PROJECT_ROOT / "judge" / "training_data" / "v3" / "all_stats_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "builder_version": BUILDER_VERSION,
        "source": "canonical Judge 100 dramas",
        "global_counts": global_counts,
        "skipped": skipped,
        "failed": [{"dj": dj, "error": err} for dj, err in failed],
        "per_drama": all_stats,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n📄 汇总已保存: {summary_path}")


if __name__ == "__main__":
    main()
