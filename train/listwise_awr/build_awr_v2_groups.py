#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_awr_v2_groups.py — DPO2000-Aware Group Construction for AWR v2

复用 canonical judge rewards（不重跑 judge），交叉 DPO v1 retrieval 结果
识别 error negatives，构造 weighted groups。

候选组成：
  正样本：strong(2-3) + fallback(1) + ambiguous(1)
  DPO v1 易错负样本：v1 排名靠前的 hard/easy negative(3-5)
  校准负样本：hard(2-3) + easy(2-3)
  group size: 5-12

用法：
    python3 train/listwise_awr/build_awr_v2_groups.py
    python3 train/listwise_awr/build_awr_v2_groups.py --splits train,val,test
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# ── Buckets ────────────────────────────────────────────────────────────────────
POSITIVE_BUCKETS = {"strong_positive", "fallback_positive", "ambiguous_positive"}
NEGATIVE_BUCKETS = {"hard_negative", "easy_negative"}

# ── Group composition ─────────────────────────────────────────────────────────
STRONG_MAX = 3
FALLBACK_MAX = 1
AMBIGUOUS_MAX = 1
DPO_ERROR_NEG_MAX = 5      # DPO v1 排名靠前的负样本
CALIBRATION_HARD_MAX = 3    # 普通 hard negative（按 overall 高优先）
CALIBRATION_EASY_MAX = 3    # easy negative（锚定底部）
GROUP_MIN_SIZE = 5
GROUP_MAX_SIZE = 12

RNG_SEED = 42


# ── Data loading ──────────────────────────────────────────────────────────────

def load_samples_by_query(path: Path) -> Dict[str, List[dict]]:
    """Load samples.jsonl, group by query_id."""
    by_query: Dict[str, List[dict]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        s = json.loads(line)
        by_query[s["query_id"]].append(s)
    return dict(by_query)


def load_dpo_v1_retrieval(dj: str, cache_root: Path) -> Dict[int, List[dict]]:
    """
    Load DPO v1 retrieval results for a DJ.

    Returns:
        {segment_sequence: [{episode, segment_id, reranker_score, rank(1-based)}]}
    """
    pattern = str(cache_root / dj / "*_structured_qwen_q3_rerank_q1_dpo2000.json")
    files = glob.glob(pattern)
    if not files:
        return {}

    data = json.loads(Path(files[0]).read_text(encoding="utf-8"))
    result: Dict[int, List[dict]] = {}
    for seg in data:
        seg_seq = seg.get("segment_sequence")
        if seg_seq is None:
            continue
        mr = seg.get("match_result") or {}
        top_matches = mr.get("top_matches") or []
        entries = []
        for rank_0, tm in enumerate(top_matches):
            entries.append({
                "episode": tm.get("episode"),
                "segment_id": tm.get("segment_id"),
                "reranker_score": tm.get("reranker_score") or tm.get("similarity") or 0.0,
                "rank": rank_0 + 1,
            })
        result[seg_seq] = entries
    return result


# ── Helpers ───────────────────────────────────────────────────────────────────

def _dpo_v1_lookup(
    sample: dict,
    dpo_top_matches: List[dict],
) -> Tuple[Optional[int], Optional[float]]:
    """Return (rank, reranker_score) if sample's candidate is in DPO v1 top-K, else (None, None)."""
    ep = sample["candidate"]["episode"]
    seg_id = sample["candidate"]["segment_id"]
    for tm in dpo_top_matches:
        if tm["episode"] == ep and tm["segment_id"] == seg_id:
            return tm["rank"], tm["reranker_score"]
    return None, None


def compute_reward(bucket: str, overall: float) -> float:
    if bucket in POSITIVE_BUCKETS:
        return overall
    return overall - 4.0


# ── Group building ────────────────────────────────────────────────────────────

def build_group(
    query_id: str,
    samples: List[dict],
    dpo_top_matches: List[dict],
    rng: random.Random,
) -> Optional[Tuple[dict, dict]]:
    """
    Build one DPO2000-aware weighted group.

    Returns:
        (group_dict, meta_dict) or None if skipped.
    """
    # Annotate DPO v1 rank
    for s in samples:
        rank, score = _dpo_v1_lookup(s, dpo_top_matches)
        s["_dpo_rank"] = rank
        s["_dpo_score"] = score

    # Classify by bucket
    by_bucket: Dict[str, List[dict]] = defaultdict(list)
    for s in samples:
        b = s["labels"]["sample_bucket"]
        if b in POSITIVE_BUCKETS or b in NEGATIVE_BUCKETS:
            by_bucket[b].append(s)

    has_positive = any(b in POSITIVE_BUCKETS for b in by_bucket)
    if not has_positive:
        return None

    selected: List[dict] = []
    selected_ids: set = set()

    def _add(s: dict):
        if s["sample_id"] not in selected_ids:
            selected.append(s)
            selected_ids.add(s["sample_id"])

    # ── 1. Positives ──────────────────────────────────────────────────────
    for bucket, limit in [
        ("strong_positive", STRONG_MAX),
        ("fallback_positive", FALLBACK_MAX),
        ("ambiguous_positive", AMBIGUOUS_MAX),
    ]:
        cands = sorted(
            by_bucket.get(bucket, []),
            key=lambda s: s["judge"]["scores"]["overall"],
            reverse=True,
        )
        for s in cands[:limit]:
            _add(s)

    # ── 2. DPO v1 error negatives ─────────────────────────────────────────
    #    负样本中 DPO v1 排名最靠前的（即 v1 认为好但 judge 认为差）
    all_negs = by_bucket.get("hard_negative", []) + by_bucket.get("easy_negative", [])
    error_negs = [
        s for s in all_negs
        if s["_dpo_rank"] is not None and s["sample_id"] not in selected_ids
    ]
    error_negs.sort(key=lambda s: s["_dpo_rank"])  # rank 1 first
    for s in error_negs[:DPO_ERROR_NEG_MAX]:
        _add(s)

    # ── 3. Calibration negatives ──────────────────────────────────────────
    remaining_hard = [
        s for s in by_bucket.get("hard_negative", [])
        if s["sample_id"] not in selected_ids
    ]
    remaining_easy = [
        s for s in by_bucket.get("easy_negative", [])
        if s["sample_id"] not in selected_ids
    ]
    # hard: overall 高的优先（更接近正样本边界 → 更有区分度）
    remaining_hard.sort(key=lambda s: s["judge"]["scores"]["overall"], reverse=True)
    remaining_easy.sort(key=lambda s: s["judge"]["scores"]["overall"])

    slots = GROUP_MAX_SIZE - len(selected)
    for s in remaining_hard[: min(CALIBRATION_HARD_MAX, max(0, slots))]:
        _add(s)

    slots = GROUP_MAX_SIZE - len(selected)
    for s in remaining_easy[: min(CALIBRATION_EASY_MAX, max(0, slots))]:
        _add(s)

    # ── 4. Fill to minimum ────────────────────────────────────────────────
    if len(selected) < GROUP_MIN_SIZE:
        filler = [s for s in all_negs if s["sample_id"] not in selected_ids]
        filler.sort(key=lambda s: s["judge"]["scores"]["overall"], reverse=True)
        for s in filler:
            if len(selected) >= GROUP_MIN_SIZE:
                break
            _add(s)

    if len(selected) < GROUP_MIN_SIZE:
        return None

    # ── 5. Compute group weight ───────────────────────────────────────────
    rewards = {s["sample_id"]: compute_reward(s["labels"]["sample_bucket"],
                                               s["judge"]["scores"]["overall"])
               for s in selected}
    judge_best_id = max(rewards, key=rewards.get)

    dpo_ranked_in_group = [
        (s["sample_id"], s["_dpo_rank"])
        for s in selected if s["_dpo_rank"] is not None
    ]

    weight = 1.0
    dpo_top_id = None
    if dpo_ranked_in_group:
        dpo_top_id, _ = min(dpo_ranked_in_group, key=lambda x: x[1])
        if dpo_top_id == judge_best_id:
            weight = 1.0
        else:
            sorted_ids = sorted(rewards, key=rewards.get, reverse=True)
            judge_rank_of_dpo_top = sorted_ids.index(dpo_top_id) + 1
            weight = 2.0 if judge_rank_of_dpo_top <= 3 else 3.0

    # ── 6. Assemble ───────────────────────────────────────────────────────
    rng.shuffle(selected)

    src = samples[0]
    n_error = sum(1 for s in selected
                  if s["_dpo_rank"] is not None and s["labels"]["sample_bucket"] in NEGATIVE_BUCKETS)

    group = {
        "group_id": query_id,
        "query_id": query_id,
        "source": {
            "dj": src["source"]["dj"],
            "run_id": src["source"]["run_id"],
        },
        "query": {
            "script_text": src["query"]["script_text"],
            "query_text": src["query"].get("query_text", src["query"]["script_text"]),
        },
        "weight": weight,
        "candidates": [
            {
                "sample_id": s["sample_id"],
                "candidate_id": s["candidate_id"],
                "bucket": s["labels"]["sample_bucket"],
                "overall": s["judge"]["scores"]["overall"],
                "reward": rewards[s["sample_id"]],
                "video_segment_path": s["candidate"]["video_segment_path"],
                "dpo_v1_rank": s["_dpo_rank"],
            }
            for s in selected
        ],
    }

    meta = {
        "weight": weight,
        "group_size": len(selected),
        "n_error_neg": n_error,
        "dpo_top_id": dpo_top_id,
        "judge_best_id": judge_best_id,
    }
    return group, meta


# ── Main ──────────────────────────────────────────────────────────────────────

def build_split(
    samples_path: Path,
    cache_root: Path,
    rng: random.Random,
) -> Tuple[List[dict], dict]:
    samples_by_query = load_samples_by_query(samples_path)
    logger.info("Loaded %d queries from %s", len(samples_by_query), samples_path.name)

    # Collect DJs → load retrieval once per DJ
    djs = sorted({ss[0]["source"]["dj"] for ss in samples_by_query.values()})
    dpo_retrieval: Dict[str, Dict[int, List[dict]]] = {}
    for dj in djs:
        r = load_dpo_v1_retrieval(dj, cache_root)
        dpo_retrieval[dj] = r
    logger.info("Loaded DPO v1 retrieval for %d / %d DJs", sum(1 for v in dpo_retrieval.values() if v), len(djs))

    groups: List[dict] = []
    stats: Dict[str, Any] = {
        "total_queries": len(samples_by_query),
        "groups_built": 0,
        "skipped_no_positive": 0,
        "skipped_too_small": 0,
        "weight_dist": {1.0: 0, 2.0: 0, 3.0: 0},
        "total_candidates": 0,
        "bucket_counts": defaultdict(int),
        "avg_group_size": 0,
        "avg_error_negs": 0,
    }

    sizes, error_counts = [], []

    for qid in sorted(samples_by_query):
        samps = samples_by_query[qid]
        dj = samps[0]["source"]["dj"]
        seg_seq = samps[0]["query"].get("segment_sequence")

        dpo_top = dpo_retrieval.get(dj, {}).get(seg_seq, []) if seg_seq is not None else []

        result = build_group(qid, samps, dpo_top, rng)
        if result is None:
            has_pos = any(s["labels"]["sample_bucket"] in POSITIVE_BUCKETS for s in samps)
            if not has_pos:
                stats["skipped_no_positive"] += 1
            else:
                stats["skipped_too_small"] += 1
            continue

        group, meta = result
        groups.append(group)
        stats["groups_built"] += 1
        stats["weight_dist"][meta["weight"]] = stats["weight_dist"].get(meta["weight"], 0) + 1
        stats["total_candidates"] += meta["group_size"]
        sizes.append(meta["group_size"])
        error_counts.append(meta["n_error_neg"])
        for c in group["candidates"]:
            stats["bucket_counts"][c["bucket"]] += 1

    stats["bucket_counts"] = dict(stats["bucket_counts"])
    if sizes:
        stats["avg_group_size"] = round(sum(sizes) / len(sizes), 2)
        stats["min_group_size"] = min(sizes)
        stats["max_group_size"] = max(sizes)
    if error_counts:
        stats["avg_error_negs"] = round(sum(error_counts) / len(error_counts), 2)

    return groups, stats


def main():
    parser = argparse.ArgumentParser(
        description="Build DPO2000-aware weighted groups for AWR v2",
    )
    parser.add_argument(
        "--splits-dir",
        default="judge/training_data/v3/splits",
        help="Directory with {split}_samples.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default="train/listwise_awr",
        help="Output directory for weighted groups",
    )
    parser.add_argument(
        "--cache-root",
        default="text_video_match/cache",
        help="Retrieval cache root (contains per-DJ dirs)",
    )
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    parser.add_argument(
        "--splits",
        default="train,val,test",
        help="Comma-separated splits to build",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    splits_dir = Path(args.splits_dir)
    output_dir = Path(args.output_dir)
    cache_root = Path(args.cache_root)
    for p in (splits_dir, output_dir, cache_root):
        if not p.is_absolute():
            p_abs = REPO_ROOT / p
        else:
            p_abs = p
        if p is splits_dir:
            splits_dir = p_abs
        elif p is output_dir:
            output_dir = p_abs
        else:
            cache_root = p_abs

    rng = random.Random(args.seed)
    all_stats: Dict[str, Any] = {}

    for split in args.splits.split(","):
        split = split.strip()
        samples_path = splits_dir / f"{split}_samples.jsonl"
        if not samples_path.exists():
            logger.warning("Skip %s: %s not found", split, samples_path)
            continue

        logger.info("=== %s ===", split.upper())
        groups, stats = build_split(samples_path, cache_root, rng)

        out_path = output_dir / f"{split}_groups_dpo2000_weighted.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            "\n".join(json.dumps(g, ensure_ascii=False) for g in groups) + ("\n" if groups else ""),
            encoding="utf-8",
        )
        logger.info(
            "  %s → %d groups, avg_size=%.1f, weight: 1.0=%d / 2.0=%d / 3.0=%d, "
            "avg_error_negs=%.1f",
            out_path.name,
            stats["groups_built"],
            stats.get("avg_group_size", 0),
            stats["weight_dist"].get(1.0, 0),
            stats["weight_dist"].get(2.0, 0),
            stats["weight_dist"].get(3.0, 0),
            stats.get("avg_error_negs", 0),
        )
        all_stats[split] = stats

    stats_path = output_dir / "group_build_stats.json"
    stats_path.write_text(json.dumps(all_stats, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Stats → %s", stats_path)


if __name__ == "__main__":
    main()
