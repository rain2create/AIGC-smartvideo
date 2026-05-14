#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 1 Recall@20 天花板分析：各 emb 变体 top-20 的 Judge 质量精细对比。

在 R@20 之外补充两个指标：
- Oracle 均分：每条 query 在系统 top-20 里能拿到的最高 Judge 分，取均值
- 交集均分：系统 top-20 ∩ Judge top-20 交集中候选的 Judge 均分
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_ROOT = PROJECT_ROOT / "text_video_match" / "cache"
JUDGE_DIR = Path(__file__).resolve().parent
CANONICAL_DIR = JUDGE_DIR / "outputs" / "canonical"

EMB_VARIANTS = ["qwen_q1", "qwen_q2", "qwen_q3", "qwen_b1", "qwen_b2"]
V4_MAP = {"v4": None}  # V4 uses default structured json (no suffix)

DJ_MAP: Dict[str, str] = {}  # dj -> canonical run dir


def _discover_data() -> Dict[str, Dict[str, str]]:
    """Discover canonical runs and variant S4 JSONs per drama.

    Returns: {dj: {"canonical_run": "annotation_xxx", "variants": {v_name: json_path}}}
    """
    data: Dict[str, Dict[str, Any]] = defaultdict(dict)

    # Discover canonical runs
    for dj_dir in sorted(CANONICAL_DIR.iterdir()):
        if not dj_dir.is_dir():
            continue
        runs = sorted(dj_dir.iterdir())
        if runs:
            data[dj_dir.name]["canonical_run"] = runs[-1].name

    # Discover variant S4 JSONs
    # V4: structured.json (no variant suffix)
    for dj in list(data.keys()):
        cache_dj = CACHE_ROOT / dj
        if not cache_dj.exists():
            continue
        variants_found: Dict[str, str] = {}
        # Find V4 (default, no variant suffix)
        for f in sorted(cache_dj.glob("*_structured.json")):
            name = f.stem
            # Skip files with known variant suffixes
            if any(
                name.endswith(sfx)
                for sfx in [
                    "_qwen_b1",
                    "_qwen_b2",
                    "_qwen_b3",
                    "_qwen_q1",
                    "_qwen_q2",
                    "_qwen_q3",
                    "_qwen_q4",
                    "_qwen_b1_rerank_q1",
                    "_qwen_b1_rerank_q2",
                ]
            ):
                continue
            variants_found["v4"] = str(f)

        for v in EMB_VARIANTS:
            pattern = f"*_structured_{v}.json"
            matches = sorted(cache_dj.glob(pattern))
            if matches:
                variants_found[v] = str(matches[-1])

        data[dj]["variants"] = variants_found

    return data


def load_judge_scores(
    canonical_run_dir: Path,
) -> Dict[Tuple[int, int, int], Dict[str, float]]:
    """Load canonical judge evaluations.

    Returns: {(segment_sequence, episode, segment_id) -> {"overall": ..., "entity_match": ...}}
    """
    results_file = canonical_run_dir / "results.jsonl"
    if not results_file.exists():
        return {}
    scores: Dict[Tuple[int, int, int], Dict[str, float]] = {}
    with open(results_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            q = d["query"]
            c = d["candidate"]
            key = (q["segment_sequence"], c["episode"], c["segment_id"])
            scores[key] = d["scores"]
    return scores


def compute_per_query(
    judge_scores: Dict[Tuple[int, int, int], Dict[str, float]],
    variant_json_path: str,
) -> List[Dict[str, Any]]:
    """Compute per-query ceiling metrics for one variant on one drama."""
    with open(variant_json_path) as f:
        variant_data = json.load(f)

    results: List[Dict[str, Any]] = []

    for item in variant_data:
        seq = item["segment_sequence"]
        mr = item.get("match_result")
        if mr is None or mr.get("status") not in ("ok", "success"):
            continue

        top_matches = mr.get("top_matches", [])
        if not top_matches:
            continue

        # Collect judge scores for all top_matches
        tm_scores: List[float] = []
        for tm in top_matches:
            key = (seq, tm["episode"], tm["segment_id"])
            js = judge_scores.get(key, {})
            overall = js.get("overall")
            if overall is not None:
                tm_scores.append(overall)

        if not tm_scores:
            continue

        # Build judge top-20 ranking for this query
        query_scores = [
            (key, s["overall"])
            for key, s in judge_scores.items()
            if key[0] == seq
        ]
        query_scores.sort(key=lambda x: x[1], reverse=True)
        judge_top20 = {key for key, _ in query_scores[:20]}

        # System top-20 keys
        sys_top20 = {
            (seq, tm["episode"], tm["segment_id"]) for tm in top_matches[:20]
        }

        # Intersection: candidates in both system top-20 and judge top-20
        intersection_keys = sys_top20 & judge_top20
        intersection_scores = [
            judge_scores[k]["overall"] for k in intersection_keys
        ]

        # Oracle: best judge score in system top-20
        oracle_max = max(tm_scores)

        # Count >=4.0 / >=4.5 in system top-20
        ge40_count = sum(1 for s in tm_scores if s >= 4.0)
        ge45_count = sum(1 for s in tm_scores if s >= 4.5)

        results.append(
            {
                "segment_sequence": seq,
                "n_top_matches": len(top_matches),
                "n_with_judge_score": len(tm_scores),
                "oracle_max": oracle_max,
                "mean_top20_score": sum(tm_scores) / len(tm_scores),
                "intersection_count": len(intersection_keys),
                "intersection_mean_score": (
                    sum(intersection_scores) / len(intersection_scores)
                    if intersection_scores
                    else 0.0
                ),
                "ge40_count": ge40_count,
                "ge45_count": ge45_count,
            }
        )

    return results


def main():
    data = _discover_data()
    # Filter to dramas with known canonical judge results (the 6 from compare_all)
    known_dramas = [
        "buxi5",
        "chongfan80",
        "duibuqinidexiaohaishishebao",
        "luxiansheng",
        "mamadechangshou5",
        "weinishiguit4",
    ]

    all_results: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(dict)
    # all_results[variant][dj] = [per_query_stats, ...]

    for dj in known_dramas:
        if dj not in data:
            print(f"[WARN] {dj}: no data found")
            continue
        canonical_run = data[dj].get("canonical_run")
        if not canonical_run:
            print(f"[WARN] {dj}: no canonical run")
            continue
        canonical_dir = CANONICAL_DIR / dj / canonical_run
        judge_scores = load_judge_scores(canonical_dir)
        if not judge_scores:
            print(f"[WARN] {dj}: no judge scores loaded")
            continue

        print(f"\n{dj}: {len(judge_scores)} judge evaluations loaded")

        for v_name, v_path in data[dj].get("variants", {}).items():
            results = compute_per_query(judge_scores, v_path)
            all_results[v_name][dj] = results
            n_q = len(results)
            oracle_mean = (
                sum(r["oracle_max"] for r in results) / n_q if n_q else 0
            )
            inter_mean = (
                sum(r["intersection_mean_score"] for r in results) / n_q
                if n_q
                else 0
            )
            ge40_total = sum(r["ge40_count"] for r in results)
            ge45_total = sum(r["ge45_count"] for r in results)
            print(
                f"  {v_name:20s}: {n_q} queries, "
                f"oracle_max_avg={oracle_mean:.2f}, "
                f"intersect_mean={inter_mean:.2f}, "
                f"ge40_total={ge40_total}, ge45_total={ge45_total}"
            )

    # Aggregate across all dramas
    print("\n" + "=" * 80)
    print("6-DRAMA AGGREGATE")
    print("=" * 80)

    agg: Dict[str, Dict[str, Any]] = {}
    for v_name in sorted(all_results.keys()):
        all_q = []
        for dj in known_dramas:
            all_q.extend(all_results[v_name].get(dj, []))

        if not all_q:
            agg[v_name] = {"n_queries": 0}
            continue

        n = len(all_q)
        oracle_avg = sum(r["oracle_max"] for r in all_q) / n
        inter_avg = sum(
            r["intersection_mean_score"] for r in all_q
        ) / n
        # Weighted intersection mean (weight by intersection_count)
        inter_weighted_num = sum(
            r["intersection_mean_score"] * r["intersection_count"]
            for r in all_q
        )
        inter_weighted_den = sum(r["intersection_count"] for r in all_q)
        inter_weighted = (
            inter_weighted_num / inter_weighted_den
            if inter_weighted_den > 0
            else 0.0
        )
        ge40_total = sum(r["ge40_count"] for r in all_q)
        ge45_total = sum(r["ge45_count"] for r in all_q)
        # Also compute: queries with at least 1 >=4.0 / >=4.5 in top-20
        q_ge40 = sum(1 for r in all_q if r["ge40_count"] > 0)
        q_ge45 = sum(1 for r in all_q if r["ge45_count"] > 0)

        agg[v_name] = {
            "n_queries": n,
            "oracle_avg": oracle_avg,
            "inter_avg": inter_avg,
            "inter_weighted": inter_weighted,
            "ge40_total": ge40_total,
            "ge45_total": ge45_total,
            "q_ge40": q_ge40,
            "q_ge45": q_ge45,
        }

    # Print summary table
    print()
    header = f"{'Variant':20s} | {'Queries':>7s} | {'Oracle均分':>10s} | {'交集均分':>10s} | {'交集加权':>10s} | {'>=4.0(总数/覆盖)':>16s} | {'>=4.5(总数/覆盖)':>16s}"
    print(header)
    print("-" * len(header))
    for v_name in sorted(agg.keys()):
        a = agg[v_name]
        if a["n_queries"] == 0:
            continue
        print(
            f"{v_name:20s} | {a['n_queries']:7d} | {a['oracle_avg']:10.3f} | {a['inter_avg']:10.3f} | {a['inter_weighted']:10.3f} | "
            f"{a['ge40_total']:3d}/{a['q_ge40']:3d}   | {a['ge45_total']:3d}/{a['q_ge45']:3d}"
        )

    # Comparison: spread
    oracle_vals = [a["oracle_avg"] for a in agg.values() if a["n_queries"] > 0]
    inter_vals = [a["inter_avg"] for a in agg.values() if a["n_queries"] > 0]
    if oracle_vals:
        print(f"\nOracle 均分极差: {max(oracle_vals) - min(oracle_vals):.3f}")
    if inter_vals:
        print(f"交集均分极差: {max(inter_vals) - min(inter_vals):.3f}")


if __name__ == "__main__":
    main()
