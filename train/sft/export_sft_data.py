#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_sft_data.py  —  从 train_samples.jsonl 筛选 SFT 训练数据

筛选规则（定稿方案，见 findings/three_stage_finetune_design.md）：
  正样本（target=1）：
    - in_system_top_matches=True
    - system_rank > 10
    - bucket ∈ {strong_positive, fallback_positive}
    - overall >= 4.0
    - per-query 上限 3 条（strong 优先，overall 降序）
  负样本（target=0）：
    - system_rank = 1
    - (hard_negative 且 overall < 3.5) 或 easy_negative
    - 天然 per-query ≤ 1

用法：
    python train/sft/export_sft_data.py
    python train/sft/export_sft_data.py --samples-path judge/training_data/v3/splits/train_samples.jsonl
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

POS_RANK_THRESHOLD = 10      # rank > 10
POS_OVERALL_MIN = 4.0        # overall >= 4.0
NEG_OVERALL_MAX = 3.5        # overall < 3.5 (for hard_negative)
POS_PER_QUERY_MAX = 3        # per-query 正样本上限


def export_sft_data(samples_path: Path, output_path: Path, frame_cache_dir: Path = None):
    # ── 读取并分类 ─────────────────────────────────────────────────────────
    pos_by_query = defaultdict(list)  # query_id -> list of positive candidates
    negatives = []

    with open(samples_path) as f:
        for line in f:
            s = json.loads(line)
            bucket = s["labels"]["sample_bucket"]
            rank = s["retrieval"].get("system_rank")
            in_top = s["retrieval"].get("in_system_top_matches", False)
            overall = s["judge"]["scores"]["overall"]
            query_text = s["query"].get("query_text") or s["query"].get("script_text") or ""
            video_path = s["candidate"]["video_segment_path"]

            record = {
                "sample_id": s["sample_id"],
                "query_id": s["query_id"],
                "query_text": query_text,
                "video_segment_path": video_path,
                "overall": overall,
                "bucket": bucket,
                "system_rank": rank,
            }

            # 正样本：in_top20 + rank>10 + positive bucket + overall>=4.0
            if (
                in_top
                and rank is not None
                and rank > POS_RANK_THRESHOLD
                and bucket in ("strong_positive", "fallback_positive")
                and overall >= POS_OVERALL_MIN
            ):
                pos_by_query[s["query_id"]].append(record)

            # 负样本：rank=1 + (hard且overall<3.5 或 easy)
            elif in_top and rank == 1:
                if bucket == "easy_negative":
                    negatives.append(record)
                elif bucket == "hard_negative" and overall < NEG_OVERALL_MAX:
                    negatives.append(record)

    # ── Per-query cap：每 query 最多 3 条正样本 ────────────────────────────
    positives = []
    for qid, candidates in pos_by_query.items():
        # strong 优先，overall 降序
        candidates.sort(
            key=lambda x: (x["bucket"] == "strong_positive", x["overall"]),
            reverse=True,
        )
        positives.extend(candidates[:POS_PER_QUERY_MAX])

    # ── 帧缓存校验（过滤视频源文件缺失的样本）─────────────────────────
    if frame_cache_dir:
        def has_frames(video_path: str) -> bool:
            stem = Path(video_path).stem
            fdir = frame_cache_dir / stem
            return fdir.exists() and len(list(fdir.glob("frame_*.jpg"))) > 0

        before_pos = len(positives)
        before_neg = len(negatives)
        positives = [r for r in positives if has_frames(r["video_segment_path"])]
        negatives = [r for r in negatives if has_frames(r["video_segment_path"])]
        dropped_pos = before_pos - len(positives)
        dropped_neg = before_neg - len(negatives)
        if dropped_pos or dropped_neg:
            logger.info("帧缓存过滤: 丢弃 %d 正样本 + %d 负样本（视频源缺失）",
                        dropped_pos, dropped_neg)

    # ── 统计 ──────────────────────────────────────────────────────────────
    pos_strong = sum(1 for r in positives if r["bucket"] == "strong_positive")
    pos_fallback = sum(1 for r in positives if r["bucket"] == "fallback_positive")
    neg_hard = sum(1 for r in negatives if r["bucket"] == "hard_negative")
    neg_easy = sum(1 for r in negatives if r["bucket"] == "easy_negative")

    logger.info("正样本: %d (strong=%d, fallback=%d)", len(positives), pos_strong, pos_fallback)
    logger.info("负样本: %d (hard=%d, easy=%d)", len(negatives), neg_hard, neg_easy)
    logger.info("总计: %d, 正负比 1:%.1f", len(positives) + len(negatives),
                len(negatives) / max(len(positives), 1))

    # ── 写出 ──────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for r in positives:
            r["target"] = 1
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        for r in negatives:
            r["target"] = 0
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info("写入 %s (%d 条)", output_path, len(positives) + len(negatives))


def main():
    parser = argparse.ArgumentParser(description="导出 SFT 训练数据")
    parser.add_argument(
        "--samples-path",
        default=str(REPO_ROOT / "judge/training_data/v3/splits/train_samples.jsonl"),
        help="train_samples.jsonl 路径",
    )
    parser.add_argument(
        "--output-path",
        default=str(REPO_ROOT / "train/sft/train_sft.jsonl"),
        help="输出路径",
    )
    parser.add_argument(
        "--frame-cache-dir",
        default="",
        help="帧缓存目录，用于过滤缺帧样本（留空则不过滤）",
    )
    args = parser.parse_args()
    fcd = Path(args.frame_cache_dir) if args.frame_cache_dir else None
    export_sft_data(Path(args.samples_path), Path(args.output_path), fcd)


if __name__ == "__main__":
    main()
