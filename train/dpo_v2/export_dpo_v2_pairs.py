#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_dpo_v2_pairs.py  —  从 train_samples.jsonl 构造 DPO v2 hard-mined pairs

设计文档：train/dpo_v2/findings/dpo_v2_hard_mining_design.md

核心规则：
  chosen:  strong_positive 优先，无 strong 时用 fallback_positive
           in_top20, system_rank 非空, overall >= 4.0
  rejected: hard_negative only
           in_top20, system_rank 非空, overall <= 3.5
  hard rule:
    inversion:  rank_gap <= 0
    semi-hard:  0 < rank_gap <= 3 且 rejected_rank <= 10
  限额：per-query 20, per-chosen 6, per-rejected 8

用法：
    python train/dpo_v2/export_dpo_v2_pairs.py
    python train/dpo_v2/export_dpo_v2_pairs.py --margin 3 --max-rejected-rank 10
"""

import argparse
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# ── 默认参数 ──────────────────────────────────────────────────────────────────
CHOSEN_OVERALL_MIN = 4.0
REJECTED_OVERALL_MAX = 3.5
MARGIN = 3                  # semi-hard 最大 gap
MAX_REJECTED_RANK = 10      # semi-hard 头部聚焦
MAX_PAIRS_PER_QUERY = 20
MAX_NEG_PER_CHOSEN = 6
MAX_POS_PER_REJECTED = 8

POSITIVE_BUCKETS = {"strong_positive", "fallback_positive"}
REJECTED_BUCKET = "hard_negative"


def load_samples(samples_path: Path):
    """读 train_samples.jsonl，按 query_id 分组。"""
    by_query = defaultdict(list)
    total = 0
    with open(samples_path) as f:
        for line in f:
            s = json.loads(line)
            by_query[s["query_id"]].append(s)
            total += 1
    logger.info("读取 %d samples, %d queries", total, len(by_query))
    return by_query


def extract_record(s):
    """从 sample 中提取配对所需字段。"""
    return {
        "sample_id": s["sample_id"],
        "candidate_id": s.get("candidate_id", s["sample_id"].split("_", 1)[-1] if "_" in s["sample_id"] else ""),
        "bucket": s["labels"]["sample_bucket"],
        "overall": s["judge"]["scores"]["overall"],
        "video_segment_path": s["candidate"]["video_segment_path"],
        "system_rank": s["retrieval"].get("system_rank"),
    }


def build_pairs(
    by_query: dict,
    margin: int = MARGIN,
    max_rejected_rank: int = MAX_REJECTED_RANK,
    max_pairs_per_query: int = MAX_PAIRS_PER_QUERY,
    max_neg_per_chosen: int = MAX_NEG_PER_CHOSEN,
    max_pos_per_rejected: int = MAX_POS_PER_REJECTED,
):
    """对每个 query 构造 hard-mined DPO pairs。"""
    all_pairs = []
    stats = Counter()

    for query_id, samples in by_query.items():
        # ── 1. 筛选 chosen ────────────────────────────────────────────────
        strongs = []
        fallbacks = []
        for s in samples:
            bucket = s["labels"]["sample_bucket"]
            rank = s["retrieval"].get("system_rank")
            in_top = s["retrieval"].get("in_system_top_matches", False)
            overall = s["judge"]["scores"]["overall"]

            if not in_top or rank is None:
                continue
            if overall < CHOSEN_OVERALL_MIN:
                continue

            if bucket == "strong_positive":
                strongs.append(s)
            elif bucket == "fallback_positive":
                fallbacks.append(s)

        chosen_pool = strongs if strongs else fallbacks
        if not chosen_pool:
            stats["skip_no_chosen"] += 1
            continue

        # ── 2. 筛选 rejected ──────────────────────────────────────────────
        rejected_pool = []
        for s in samples:
            bucket = s["labels"]["sample_bucket"]
            rank = s["retrieval"].get("system_rank")
            in_top = s["retrieval"].get("in_system_top_matches", False)
            overall = s["judge"]["scores"]["overall"]

            if (
                bucket == REJECTED_BUCKET
                and in_top
                and rank is not None
                and overall <= REJECTED_OVERALL_MAX
            ):
                rejected_pool.append(s)

        if not rejected_pool:
            stats["skip_no_rejected"] += 1
            continue

        # ── 3. 枚举 pairs + hard rule 过滤 ────────────────────────────────
        query_text = (
            samples[0]["query"].get("query_text")
            or samples[0]["query"].get("script_text")
            or ""
        )
        script_text = samples[0]["query"].get("script_text", "")
        source_dj = samples[0].get("source", {}).get("dj", "")

        candidates = []
        for pos in chosen_pool:
            pos_rank = pos["retrieval"]["system_rank"]
            for neg in rejected_pool:
                neg_rank = neg["retrieval"]["system_rank"]
                rank_gap = neg_rank - pos_rank

                if rank_gap <= 0:
                    pair_subtype = "inversion"
                elif rank_gap <= margin and neg_rank <= max_rejected_rank:
                    pair_subtype = "semi_hard"
                else:
                    continue

                pos_overall = pos["judge"]["scores"]["overall"]
                neg_overall = neg["judge"]["scores"]["overall"]
                judge_gap = pos_overall - neg_overall

                candidates.append({
                    "pos": pos,
                    "neg": neg,
                    "pair_subtype": pair_subtype,
                    "rank_gap": rank_gap,
                    "judge_gap": round(judge_gap, 2),
                    "pos_rank": pos_rank,
                    "neg_rank": neg_rank,
                })

        if not candidates:
            stats["skip_no_hard_pair"] += 1
            continue

        # ── 4. 排序优先级 ─────────────────────────────────────────────────
        candidates.sort(key=lambda p: (
            0 if p["pair_subtype"] == "inversion" else 1,  # inversion 优先
            p["neg_rank"],                                  # rejected_rank 小优先
            p["rank_gap"],                                  # gap 小优先
            p["judge_gap"],                                 # judge_gap 小优先
            -p["pos_rank"],                                 # chosen_rank 大优先
        ))

        # ── 5. 限额裁剪 ──────────────────────────────────────────────────
        chosen_neg_count = Counter()
        rejected_pos_count = Counter()
        query_pairs = []

        for c in candidates:
            if len(query_pairs) >= max_pairs_per_query:
                break
            pos_id = c["pos"]["sample_id"]
            neg_id = c["neg"]["sample_id"]
            if chosen_neg_count[pos_id] >= max_neg_per_chosen:
                continue
            if rejected_pos_count[neg_id] >= max_pos_per_rejected:
                continue

            chosen_neg_count[pos_id] += 1
            rejected_pos_count[neg_id] += 1

            pos_rec = extract_record(c["pos"])
            neg_rec = extract_record(c["neg"])

            # pair_type: strong_vs_hard / fallback_vs_hard
            pos_bucket_short = "strong" if pos_rec["bucket"] == "strong_positive" else "fallback"
            pair_type = f"{pos_bucket_short}_vs_hard"

            pair = {
                "pair_id": f"{query_id}_c{pos_rec['candidate_id']}_r{neg_rec['candidate_id']}",
                "query_id": query_id,
                "pair_type": pair_type,
                "pair_subtype": c["pair_subtype"],
                "rank_gap": c["rank_gap"],
                "judge_gap": c["judge_gap"],
                "query": {
                    "script_text": script_text,
                    "query_text": query_text,
                },
                "chosen": pos_rec,
                "rejected": neg_rec,
                "source": {"dj": source_dj},
            }
            query_pairs.append(pair)

        all_pairs.extend(query_pairs)
        stats["queries_with_pairs"] += 1
        stats[f"pairs_total"] += len(query_pairs)
        for qp in query_pairs:
            stats[f"subtype_{qp['pair_subtype']}"] += 1
            stats[f"pairtype_{qp['pair_type']}"] += 1

    return all_pairs, stats


def print_summary(pairs, stats):
    """输出统计摘要。"""
    total = len(pairs)
    logger.info("=" * 60)
    logger.info("DPO v2 Hard-Mining 数据导出摘要")
    logger.info("=" * 60)
    logger.info("总 pairs: %d", total)
    logger.info("覆盖 queries: %d", stats.get("queries_with_pairs", 0))
    logger.info("")

    # subtype 分布
    inv = stats.get("subtype_inversion", 0)
    semi = stats.get("subtype_semi_hard", 0)
    logger.info("按 subtype:")
    logger.info("  inversion:  %d (%.1f%%)", inv, 100 * inv / max(total, 1))
    logger.info("  semi_hard:  %d (%.1f%%)", semi, 100 * semi / max(total, 1))
    logger.info("")

    # pair_type 分布
    logger.info("按 pair_type:")
    for key in sorted(stats):
        if key.startswith("pairtype_"):
            name = key[len("pairtype_"):]
            cnt = stats[key]
            logger.info("  %s: %d (%.1f%%)", name, cnt, 100 * cnt / max(total, 1))
    logger.info("")

    # 跳过统计
    for key in ("skip_no_chosen", "skip_no_rejected", "skip_no_hard_pair"):
        if stats.get(key, 0) > 0:
            logger.info("跳过 (%s): %d queries", key, stats[key])

    # rank_gap 分布
    if pairs:
        gaps = [p["rank_gap"] for p in pairs]
        gap_counter = Counter(gaps)
        logger.info("")
        logger.info("rank_gap 分布:")
        for g in sorted(gap_counter):
            logger.info("  gap=%+d: %d pairs", g, gap_counter[g])

    # judge_gap 统计
    if pairs:
        jgaps = [p["judge_gap"] for p in pairs]
        logger.info("")
        logger.info("judge_gap: min=%.1f, max=%.1f, mean=%.2f",
                     min(jgaps), max(jgaps), sum(jgaps) / len(jgaps))

    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="DPO v2 Hard-Mining 数据导出")
    parser.add_argument(
        "--samples-path",
        default=str(REPO_ROOT / "judge/training_data/v3/splits/train_samples.jsonl"),
        help="train_samples.jsonl 路径",
    )
    parser.add_argument(
        "--output-path",
        default=str(REPO_ROOT / "train/dpo_v2/train_pairs.jsonl"),
        help="输出 pairs 路径",
    )
    parser.add_argument("--margin", type=int, default=MARGIN, help="semi-hard 最大 rank gap")
    parser.add_argument("--max-rejected-rank", type=int, default=MAX_REJECTED_RANK,
                        help="semi-hard 要求 rejected_rank <= N")
    parser.add_argument("--max-pairs-per-query", type=int, default=MAX_PAIRS_PER_QUERY)
    parser.add_argument("--max-neg-per-chosen", type=int, default=MAX_NEG_PER_CHOSEN)
    parser.add_argument("--max-pos-per-rejected", type=int, default=MAX_POS_PER_REJECTED)
    args = parser.parse_args()

    by_query = load_samples(Path(args.samples_path))

    pairs, stats = build_pairs(
        by_query,
        margin=args.margin,
        max_rejected_rank=args.max_rejected_rank,
        max_pairs_per_query=args.max_pairs_per_query,
        max_neg_per_chosen=args.max_neg_per_chosen,
        max_pos_per_rejected=args.max_pos_per_rejected,
    )

    print_summary(pairs, stats)

    # 写出
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    logger.info("写入 %s (%d pairs)", output_path, len(pairs))


if __name__ == "__main__":
    main()
