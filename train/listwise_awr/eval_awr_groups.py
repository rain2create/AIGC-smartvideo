#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_awr_groups.py — 离线 Group 级评估（快速选 Checkpoint）

对每个 checkpoint，在 groups.jsonl 上逐 group 打分，计算排序指标。
无需跑完整推理管线，几分钟出结果。

用法：
    # 单个 checkpoint
    python3 train/listwise_awr/eval_awr_groups.py \
        --checkpoint train/checkpoints/qwen_awr_v2/checkpoint-100 \
        --groups train/listwise_awr/val_groups_dpo2000_weighted.jsonl

    # 多个 checkpoint 对比
    python3 train/listwise_awr/eval_awr_groups.py \
        --checkpoints train/checkpoints/qwen_awr_v2/checkpoint-50 \
                      train/checkpoints/qwen_awr_v2/checkpoint-100 \
                      train/checkpoints/qwen_awr_v2/checkpoint-150 \
        --groups train/listwise_awr/val_groups_dpo2000_weighted.jsonl

    # 也评估 reference（DPO v1 merged，无 AWR LoRA）作为 baseline
    加 --eval-reference
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from train.dpo.dpo_model import compute_score, load_lora_weights_compat, load_model_with_merged_lora
from train.listwise_awr.awr_dataset import AWRDataCollator, AWRDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ── 指标计算 ──────────────────────────────────────────────────────────────────

def ndcg_at_k(scores: List[float], rewards: List[float], k: int) -> float:
    """按 scores 排序，用 rewards 作为 relevance，算 NDCG@k。"""
    paired = sorted(zip(scores, rewards), key=lambda x: x[0], reverse=True)
    dcg = sum(r / math.log2(i + 2) for i, (_, r) in enumerate(paired[:k]) if r > 0)
    ideal = sorted(rewards, reverse=True)
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal[:k]) if r > 0)
    return dcg / idcg if idcg > 0 else 0.0


def pairwise_accuracy(scores: List[float], rewards: List[float]) -> float:
    """所有 (pos, neg) pair 中 score(pos) > score(neg) 的比例。"""
    correct, total = 0, 0
    for i in range(len(scores)):
        for j in range(len(scores)):
            if rewards[i] > rewards[j]:
                total += 1
                if scores[i] > scores[j]:
                    correct += 1
    return correct / total if total > 0 else 0.0


def spearman_rank(scores: List[float], rewards: List[float]) -> float:
    """Spearman rank correlation."""
    n = len(scores)
    if n < 2:
        return 0.0

    def _rank(vals):
        order = sorted(range(n), key=lambda i: vals[i], reverse=True)
        ranks = [0.0] * n
        for r, idx in enumerate(order):
            ranks[idx] = r + 1.0
        return ranks

    rs = _rank(scores)
    rr = _rank(rewards)
    d2 = sum((a - b) ** 2 for a, b in zip(rs, rr))
    return 1.0 - 6.0 * d2 / (n * (n * n - 1))


def bucket_mean_scores(
    scores: List[float],
    buckets: List[str],
) -> Dict[str, float]:
    """每个 bucket 的平均 policy score。"""
    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for s, b in zip(scores, buckets):
        sums[b] = sums.get(b, 0.0) + s
        counts[b] = counts.get(b, 0) + 1
    return {b: sums[b] / counts[b] for b in sums}


# ── 模型打分 ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def score_groups(
    model: torch.nn.Module,
    yes_id: int,
    no_id: int,
    collator: AWRDataCollator,
    dataset: AWRDataset,
    device: torch.device,
    use_reference: bool = False,
) -> List[Dict[str, Any]]:
    """对每个 group 的候选打分，返回 [{group_id, scores, rewards, buckets}]。"""
    model.eval()
    results = []

    for idx in range(len(dataset)):
        item = dataset[idx]
        batch = collator([item])

        candidate_inputs = batch["candidate_inputs"]
        rewards = batch["rewards"].tolist()
        group_id = batch["group_id"]
        buckets = [c["bucket"] for c in item["candidates"]]

        scores = []
        for cand_inputs in candidate_inputs:
            cand_inputs = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in cand_inputs.items()
            }
            if use_reference:
                model.disable_adapter_layers()
                s = compute_score(model, cand_inputs, yes_id, no_id)
                model.enable_adapter_layers()
            else:
                s = compute_score(model, cand_inputs, yes_id, no_id)
            scores.append(s.squeeze(0).item())

        results.append({
            "group_id": group_id,
            "scores": scores,
            "rewards": rewards,
            "buckets": buckets,
        })

        if (idx + 1) % 5 == 0:
            logger.info("  scored %d / %d groups", idx + 1, len(dataset))

    return results


def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """从打分结果计算汇总指标。"""
    ndcg3s, ndcg5s, ndcg10s = [], [], []
    pw_accs = []
    spearmans = []
    top1_hits = 0
    pos_ranks = []
    all_bucket_scores: Dict[str, List[float]] = {}

    for r in results:
        scores = r["scores"]
        rewards = r["rewards"]
        buckets = r["buckets"]

        ndcg3s.append(ndcg_at_k(scores, rewards, 3))
        ndcg5s.append(ndcg_at_k(scores, rewards, 5))
        ndcg10s.append(ndcg_at_k(scores, rewards, 10))
        pw_accs.append(pairwise_accuracy(scores, rewards))
        spearmans.append(spearman_rank(scores, rewards))

        # top1 hit: policy top-1 == reward top-1?
        policy_top = max(range(len(scores)), key=lambda i: scores[i])
        reward_top = max(range(len(rewards)), key=lambda i: rewards[i])
        if policy_top == reward_top:
            top1_hits += 1

        # positive mean rank
        sorted_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        rank_map = {idx: rank + 1 for rank, idx in enumerate(sorted_idx)}
        for i, b in enumerate(buckets):
            if b in ("strong_positive", "fallback_positive", "ambiguous_positive"):
                pos_ranks.append(rank_map[i])

        # bucket scores
        for s, b in zip(scores, buckets):
            all_bucket_scores.setdefault(b, []).append(s)

    n = len(results)
    metrics = {
        "n_groups": n,
        "ndcg@3": round(sum(ndcg3s) / n, 4) if n else 0,
        "ndcg@5": round(sum(ndcg5s) / n, 4) if n else 0,
        "ndcg@10": round(sum(ndcg10s) / n, 4) if n else 0,
        "pairwise_acc": round(sum(pw_accs) / n, 4) if n else 0,
        "spearman": round(sum(spearmans) / n, 4) if n else 0,
        "top1_hit_rate": round(top1_hits / n, 4) if n else 0,
        "positive_mean_rank": round(sum(pos_ranks) / len(pos_ranks), 2) if pos_ranks else 0,
        "bucket_mean_score": {
            b: round(sum(vs) / len(vs), 4)
            for b, vs in sorted(all_bucket_scores.items())
        },
    }
    return metrics


# ── 主流程 ────────────────────────────────────────────────────────────────────

def load_model_for_eval(
    model_path: str,
    init_lora_path: str,
    checkpoint_path: Optional[str],
    device: torch.device,
):
    """加载模型用于评估。

    checkpoint_path=None 时只加载 reference（DPO v1 merged，无 AWR LoRA）。
    """
    from types import SimpleNamespace

    cfg = SimpleNamespace(
        model_path=model_path,
        init_lora_path=init_lora_path,
        lora_rank=64,
        lora_alpha=128,
        lora_dropout=0.0,
        lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj"],
        bf16=True,
        gradient_checkpointing=False,
    )

    model, processor, yes_id, no_id = load_model_with_merged_lora(cfg)

    if checkpoint_path:
        # 加载训练好的 AWR LoRA 权重
        logger.info("Loading AWR checkpoint: %s", checkpoint_path)
        load_lora_weights_compat(model, checkpoint_path, adapter_name="default")
    # else: 只用 merged base + 全零 LoRA = reference

    model = model.to(device)
    model.eval()
    return model, processor, yes_id, no_id


def main():
    parser = argparse.ArgumentParser(description="Offline group-level eval for AWR checkpoints")
    ckpt_group = parser.add_mutually_exclusive_group(required=True)
    ckpt_group.add_argument("--checkpoint", help="单个 checkpoint 路径")
    ckpt_group.add_argument("--checkpoints", nargs="+", help="多个 checkpoint 路径")
    parser.add_argument("--groups", required=True, help="groups jsonl 路径")
    parser.add_argument(
        "--model-path",
        default="/home/jovyan/data03/search_public/xby/models/Qwen3-VL-Reranker-8B",
        help="Base model 路径",
    )
    parser.add_argument(
        "--init-lora-path",
        default="train/checkpoints/qwen_dpo/checkpoint-2000",
        help="前置 LoRA（merge 进 base）",
    )
    parser.add_argument(
        "--frame-cache-dir",
        default="judge/training_data/frame_cache",
        help="Frame cache 目录",
    )
    parser.add_argument("--eval-reference", action="store_true",
                        help="同时评估 reference（DPO v1 merged，无 AWR LoRA）作为 baseline")
    parser.add_argument("--output", default="", help="输出 JSON 路径（默认打印到 stdout）")
    args = parser.parse_args()

    # 路径解析
    init_lora = args.init_lora_path
    if not Path(init_lora).is_absolute():
        init_lora = str(REPO_ROOT / init_lora)
    frame_cache = args.frame_cache_dir
    if not Path(frame_cache).is_absolute():
        frame_cache = str(REPO_ROOT / frame_cache)

    checkpoints = args.checkpoints or [args.checkpoint]
    checkpoints = [str(REPO_ROOT / c) if not Path(c).is_absolute() else c for c in checkpoints]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据（只需一次）
    dataset = AWRDataset(
        groups_jsonl_path=args.groups,
        frame_cache_dir=frame_cache,
        max_frames=8,
    )

    all_results: Dict[str, Any] = {"groups_file": args.groups, "n_groups": len(dataset), "evals": {}}

    # 加载模型一次，后续换 adapter
    logger.info("Loading model (base + merge DPO v1)...")
    model, processor, yes_id, no_id = load_model_for_eval(
        model_path=args.model_path,
        init_lora_path=init_lora,
        checkpoint_path=checkpoints[0],
        device=device,
    )

    collator = AWRDataCollator(
        processor=processor,
        instruction="Retrieve the most relevant video clip for the given query.",
        total_pixels=10485760,
        max_seq_len=4096,
    )

    # 评估 reference（可选）
    if args.eval_reference:
        logger.info("=== Evaluating reference (DPO v1, no AWR) ===")
        scored = score_groups(model, yes_id, no_id, collator, dataset, device, use_reference=True)
        metrics = compute_metrics(scored)
        all_results["evals"]["reference"] = metrics
        logger.info("  reference: NDCG@5=%.4f  pw_acc=%.4f  top1_hit=%.4f  spearman=%.4f",
                     metrics["ndcg@5"], metrics["pairwise_acc"],
                     metrics["top1_hit_rate"], metrics["spearman"])

    # 评估各 checkpoint
    for i, ckpt in enumerate(checkpoints):
        ckpt_name = Path(ckpt).name
        logger.info("=== Evaluating %s (%d/%d) ===", ckpt_name, i + 1, len(checkpoints))

        if i > 0:
            # 换 checkpoint 权重（不重新加载 base，不创建同名 adapter）
            load_lora_weights_compat(model, ckpt, adapter_name="default")
            model.eval()

        scored = score_groups(model, yes_id, no_id, collator, dataset, device)
        metrics = compute_metrics(scored)
        all_results["evals"][ckpt_name] = metrics

        logger.info("  %s: NDCG@5=%.4f  pw_acc=%.4f  top1_hit=%.4f  spearman=%.4f",
                     ckpt_name, metrics["ndcg@5"], metrics["pairwise_acc"],
                     metrics["top1_hit_rate"], metrics["spearman"])

    # 汇总表
    logger.info("\n=== Summary ===")
    header = f"{'checkpoint':<20} {'NDCG@3':>8} {'NDCG@5':>8} {'NDCG@10':>8} {'pw_acc':>8} {'top1':>8} {'spearman':>8} {'pos_rank':>8}"
    logger.info(header)
    for name, m in all_results["evals"].items():
        logger.info(
            f"{name:<20} {m['ndcg@3']:>8.4f} {m['ndcg@5']:>8.4f} {m['ndcg@10']:>8.4f} "
            f"{m['pairwise_acc']:>8.4f} {m['top1_hit_rate']:>8.4f} {m['spearman']:>8.4f} "
            f"{m['positive_mean_rank']:>8.2f}"
        )

    # 输出
    output_str = json.dumps(all_results, indent=2, ensure_ascii=False)
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output_str, encoding="utf-8")
        logger.info("Results written to %s", out_path)
    else:
        print(output_str)


if __name__ == "__main__":
    main()
