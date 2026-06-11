#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare retrieval variants with one canonical Judge matrix.

The canonical Judge run evaluates:
  script segment x every video segment in the target episode

Variant comparison then only reads each variant's best_match and looks it up
in the canonical matrix. This avoids re-judging the same candidate pool for
V4/Qwen Q1/Qwen Q2/Qwen Q3 and removes LLM re-run noise from comparisons.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent
JUDGE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = JUDGE_DIR / "outputs"
CACHE_ROOT = PROJECT_ROOT / "text_video_match" / "cache"

TOPK_THRESHOLDS = [1, 3, 5, 10, 20]
_DISABLE_20S_FILTER = False
SCORE_THRESHOLDS = [4.0, 4.5]
DIMENSIONS = [
    "entity_match",
    "event_match",
    "scene_match",
    "tone_match",
    "editing_suitability",
]
DIMENSION_LABELS = {
    "entity_match": "entity",
    "event_match": "event",
    "scene_match": "scene",
    "tone_match": "tone",
    "editing_suitability": "editing",
}


def _to_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _pct(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.1f}%"


def _count_pct(count: Any, total: Any, rate: Optional[float] = None) -> str:
    try:
        count_int = int(count)
        total_int = int(total)
    except (TypeError, ValueError):
        return "-"
    if rate is None:
        rate = count_int / total_int if total_int else 0.0
    return f"{count_int} / {total_int} ({_pct(rate)})"


def _score(value: Optional[float], ndigits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value:.{ndigits}f}"


def _score_range(min_value: Optional[float], max_value: Optional[float]) -> str:
    if min_value is None or max_value is None:
        return "-"
    return f"{_score(min_value)} ~ {_score(max_value)}"


def _pct_delta(value: float) -> str:
    return f"{value * 100:.1f}pp"


def _md_cell(value: Any, limit: int = 80) -> str:
    if value is None:
        return "-"
    text = " ".join(str(value).split())
    text = text.replace("|", "\\|")
    if limit > 0 and len(text) > limit:
        return text[: limit - 1] + "…"
    return text or "-"


def _ep_seg(episode: Any, segment_id: Any) -> str:
    ep = _to_int(episode)
    sid = _to_int(segment_id)
    if ep is None and sid is None:
        return "-"
    if ep is None:
        return f"seg{sid}"
    if sid is None:
        return f"ep{ep}"
    return f"ep{ep}/seg{sid}"


def _candidate_key(episode: Any, segment_id: Any) -> Tuple[Optional[int], Optional[int]]:
    return (_to_int(episode), _to_int(segment_id))


def _candidate_sort_key(item: Dict[str, Any]) -> Tuple[float, int, int]:
    score = _to_float(item.get("scores", {}).get("overall")) or 0.0
    episode = _to_int(item.get("candidate", {}).get("episode")) or 0
    segment_id = _to_int(item.get("candidate", {}).get("segment_id")) or 0
    return (-score, episode, segment_id)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_task_djs(task_file: str) -> List[str]:
    path = Path(task_file)
    djs: List[str] = []
    seen = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        dj = line.split(",")[0].strip()
        if dj and dj not in seen:
            djs.append(dj)
            seen.add(dj)
    return djs


def load_structured_segments(structured_json_path: Path) -> List[Dict[str, Any]]:
    data = json.loads(structured_json_path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    return data.get("segments", [])


def _select_segments_within_20_seconds(
    segments: List[Dict[str, Any]],
    target_total: float = 20.0,
) -> List[Dict[str, Any]]:
    if _DISABLE_20S_FILTER:
        return segments
    selected: List[Dict[str, Any]] = []
    accumulated_time = 0.0
    for seg in segments:
        audio_dur = float(seg.get("audio_duration") or 3.0)
        if accumulated_time >= target_total:
            break
        accumulated_time += audio_dur
        selected.append(seg)
        if accumulated_time >= target_total:
            break
    return selected


def _project_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)


def find_latest_completed_run(variant: str, dj: str, run_id: str = "") -> Path:
    base = OUTPUTS_DIR / variant / dj
    if run_id:
        run_dir = base / run_id
        _assert_completed_run(run_dir)
        return run_dir

    if not base.exists():
        raise FileNotFoundError(f"Judge output dir not found: {base}")

    runs = []
    for run_dir in base.iterdir():
        if not run_dir.is_dir():
            continue
        try:
            _assert_completed_run(run_dir)
        except Exception:
            continue
        runs.append(run_dir)

    if not runs:
        raise FileNotFoundError(f"No completed Judge run found under: {base}")
    return max(runs, key=lambda p: (p / "report.json").stat().st_mtime)


def _assert_completed_run(run_dir: Path) -> None:
    config_path = run_dir / "config.json"
    results_path = run_dir / "results.jsonl"
    report_path = run_dir / "report.json"
    summary_path = run_dir / "summary.md"
    missing = [p.name for p in [config_path, results_path, report_path, summary_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Incomplete run {run_dir}; missing: {', '.join(missing)}")
    config = _load_json(config_path)
    if not config.get("completed_at"):
        raise RuntimeError(f"Incomplete run {run_dir}; config has no completed_at")


def find_variant_structured_json(dj: str, variant: str) -> Path:
    cache_dir = CACHE_ROOT / dj
    if not cache_dir.exists():
        raise FileNotFoundError(f"cache dir not found: {cache_dir}")

    if variant in {"v4", "base", "origin", "original"}:
        pattern = "*_structured.json"
    else:
        pattern = f"*_structured_{variant}.json"

    candidates = sorted(cache_dir.glob(pattern), key=lambda p: p.stat().st_mtime)

    # Fallback: new naming includes query_version (qwen_b1_q3), old naming omits it (qwen_b1).
    # Try stripping the query_version suffix for non-V1 video index variants.
    if not candidates:
        m = re.match(r"(qwen_[a-z]+\d+)_(q\d+)(_rerank_.*)?$", variant)
        if m:
            old_variant = m.group(1) + (m.group(3) or "")
            old_pattern = f"*_structured_{old_variant}.json"
            candidates = sorted(cache_dir.glob(old_pattern), key=lambda p: p.stat().st_mtime)

    if not candidates:
        raise FileNotFoundError(f"No structured json for dj={dj}, variant={variant}, pattern={pattern}")
    return candidates[-1]


def load_canonical_matrix(run_dir: Path) -> Dict[int, List[Dict[str, Any]]]:
    results_path = run_dir / "results.jsonl"
    by_script: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    with results_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            seg_seq = item.get("query", {}).get("segment_sequence")
            if seg_seq is None:
                seg_seq = _parse_seg_seq_from_eval_id(item.get("evaluation_id", ""))
            seg_seq = _to_int(seg_seq)
            if seg_seq is None:
                continue
            by_script[seg_seq].append(item)

    for candidates in by_script.values():
        candidates.sort(key=_candidate_sort_key)
    return dict(by_script)


def _parse_seg_seq_from_eval_id(eval_id: str) -> Optional[int]:
    marker = "_seg"
    if marker not in eval_id:
        return None
    tail = eval_id.split(marker, 1)[1]
    value = tail.split("_", 1)[0]
    return _to_int(value)


def load_script_meta(structured_json: Path) -> Dict[int, Dict[str, Any]]:
    segments = _select_segments_within_20_seconds(load_structured_segments(structured_json))
    meta = {}
    for seg in segments:
        seg_seq = _to_int(seg.get("segment_sequence"))
        if seg_seq is None:
            continue
        meta[seg_seq] = {
            "segment_sequence": seg_seq,
            "script_text": seg.get("script_text", ""),
            "episode_target": _to_int(seg.get("episode_target")),
            "audio_duration": _to_float(seg.get("audio_duration")),
        }
    return meta


def extract_variant_best_matches(structured_json: Path) -> Dict[int, Dict[str, Any]]:
    segments = _select_segments_within_20_seconds(load_structured_segments(structured_json))
    matches = {}
    for seg in segments:
        seg_seq = _to_int(seg.get("segment_sequence"))
        if seg_seq is None:
            continue
        match_result = seg.get("match_result") or {}
        if isinstance(match_result, str):
            try:
                match_result = json.loads(match_result)
            except json.JSONDecodeError:
                match_result = {}

        best = {}
        if isinstance(match_result, dict):
            best = (
                match_result.get("best_match")
                or match_result.get("top1_match")
                or ((match_result.get("top_matches") or [{}])[0])
                or {}
            )

        top_matches_keys: List[Tuple[Optional[int], Optional[int]]] = []
        if isinstance(match_result, dict):
            for tm in (match_result.get("top_matches") or []):
                tm_seg = _to_int(tm.get("segment_id"))
                tm_ep = _to_int(tm.get("episode"))
                if tm_seg is not None:
                    top_matches_keys.append((tm_ep, tm_seg))

        segment_id = _to_int(best.get("segment_id"))
        episode = _to_int(best.get("episode"))
        if segment_id is None:
            matches[seg_seq] = {
                "status": match_result.get("status") if isinstance(match_result, dict) else "missing",
                "segment_id": None,
                "episode": episode,
                "source_score": None,
                "segment_file_path": "",
                "top_matches_keys": top_matches_keys,
                "top_matches_count": len(top_matches_keys),
            }
            continue

        matches[seg_seq] = {
            "status": match_result.get("status", "success") if isinstance(match_result, dict) else "success",
            "segment_id": segment_id,
            "episode": episode,
            "source_score": _to_float(best.get("score") or best.get("similarity")),
            "segment_file_path": best.get("segment_file_path", ""),
            "top_matches_keys": top_matches_keys,
            "top_matches_count": len(top_matches_keys),
        }
    return matches


def evaluate_variant_on_matrix(
    script_meta: Dict[int, Dict[str, Any]],
    matrix: Dict[int, List[Dict[str, Any]]],
    variant_matches: Dict[int, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows = []
    for seg_seq in sorted(script_meta):
        candidates = matrix.get(seg_seq, [])
        candidate_lookup = {
            _candidate_key(c.get("candidate", {}).get("episode"), c.get("candidate", {}).get("segment_id")): (rank, c)
            for rank, c in enumerate(candidates, 1)
        }
        best_judge = candidates[0] if candidates else None
        best_match = variant_matches.get(seg_seq, {})
        best_key = _candidate_key(best_match.get("episode"), best_match.get("segment_id"))
        rank_item = candidate_lookup.get(best_key)
        rank = rank_item[0] if rank_item else None
        judged = rank_item[1] if rank_item else None
        system_score = _to_float(judged.get("scores", {}).get("overall")) if judged else None
        system_dimension_scores = {
            dim: _to_float(judged.get("scores", {}).get(dim)) if judged else None
            for dim in DIMENSIONS
        }
        judge_best_score = _to_float(best_judge.get("scores", {}).get("overall")) if best_judge else None
        judge_best_dimension_scores = {
            dim: _to_float(best_judge.get("scores", {}).get(dim)) if best_judge else None
            for dim in DIMENSIONS
        }

        high_counts = {
            threshold: sum(
                1 for c in candidates
                if (_to_float(c.get("scores", {}).get("overall")) or 0.0) >= threshold
            )
            for threshold in SCORE_THRESHOLDS
        }

        # 召回率 (Recall@K): 系统 topK 与 Judge topK 的交集占比
        judge_keys = [
            _candidate_key(c.get("candidate", {}).get("episode"), c.get("candidate", {}).get("segment_id"))
            for c in candidates
        ]
        system_keys = best_match.get("top_matches_keys", [])
        recall: Dict[str, Optional[float]] = {}
        for k in TOPK_THRESHOLDS:
            if k > len(system_keys) and len(system_keys) > 0:
                recall[f"top{k}"] = None  # 候选数不足，不计算
            else:
                judge_set = set(judge_keys[:k]) if judge_keys else set()
                system_set = set(system_keys[:k])
                recall[f"top{k}"] = len(system_set & judge_set) / k if judge_set and k > 0 else None

        # NDCG@K: 系统排序 vs Judge 理想排序
        judge_score_lookup: Dict[Tuple[Optional[int], Optional[int]], float] = {}
        for c in candidates:
            jk = _candidate_key(c.get("candidate", {}).get("episode"), c.get("candidate", {}).get("segment_id"))
            js = _to_float(c.get("scores", {}).get("overall")) or 0.0
            judge_score_lookup[jk] = js
        all_judge_scores = sorted(judge_score_lookup.values(), reverse=True)

        def _dcg_at_k(k: int, ranked_keys: List[Tuple[Optional[int], Optional[int]]]) -> float:
            dcg = 0.0
            for i, rk in enumerate(ranked_keys[:k]):
                dcg += judge_score_lookup.get(rk, 0.0) / math.log2(i + 2)
            return dcg

        def _idcg_at_k(k: int) -> float:
            idcg = 0.0
            for i in range(min(k, len(all_judge_scores))):
                idcg += all_judge_scores[i] / math.log2(i + 2)
            return idcg

        ndcg: Dict[str, Optional[float]] = {}
        for k in TOPK_THRESHOLDS:
            if not system_keys:
                ndcg[f"top{k}"] = None
            else:
                idcg = _idcg_at_k(k)
                ndcg[f"top{k}"] = _dcg_at_k(k, system_keys) / idcg if idcg > 0 else None

        rows.append({
            "segment_sequence": seg_seq,
            "script_text": script_meta[seg_seq].get("script_text", ""),
            "episode_target": script_meta[seg_seq].get("episode_target"),
            "n_candidates": len(candidates),
            "system_segment_id": best_match.get("segment_id"),
            "system_episode": best_match.get("episode"),
            "system_source_score": best_match.get("source_score"),
            "system_status": best_match.get("status", "missing"),
            "system_top_matches_count": best_match.get("top_matches_count", 0),
            "system_in_candidate_pool": rank is not None,
            "system_judge_rank": rank,
            "system_judge_score": system_score,
            "system_dimension_scores": system_dimension_scores,
            "judge_best_segment_id": best_judge.get("candidate", {}).get("segment_id") if best_judge else None,
            "judge_best_episode": best_judge.get("candidate", {}).get("episode") if best_judge else None,
            "judge_best_score": judge_best_score,
            "judge_best_dimension_scores": judge_best_dimension_scores,
            "score_gap_to_judge_best": (
                round(judge_best_score - system_score, 4)
                if judge_best_score is not None and system_score is not None else None
            ),
            "high_score_counts": {
                f"ge_{str(threshold).replace('.', '_')}": high_counts[threshold]
                for threshold in SCORE_THRESHOLDS
            },
            "recall": recall,
            "ndcg": ndcg,
            "diagnosis": diagnose_row(rank, system_score, judge_best_score, high_counts),
        })
    return rows


def diagnose_row(
    rank: Optional[int],
    system_score: Optional[float],
    judge_best_score: Optional[float],
    high_counts: Dict[float, int],
) -> str:
    if rank is None or system_score is None:
        return "未命中候选池"
    if rank == 1 and system_score >= 4.0:
        return "已命中高质量"
    if rank == 1:
        return "命中但上限不足"
    if high_counts.get(4.0, 0) == 0:
        return "候选池不足"
    if rank <= 3:
        return "接近，可优化"
    if judge_best_score is not None and judge_best_score - system_score >= 1.0:
        return "有好候选但错排"
    return "排序可优化"


def aggregate_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    ranked = [row for row in rows if row.get("system_judge_rank") is not None]
    ranks = [int(row["system_judge_rank"]) for row in ranked]
    scores = [float(row["system_judge_score"]) for row in ranked if row.get("system_judge_score") is not None]
    gaps = [float(row["score_gap_to_judge_best"]) for row in ranked if row.get("score_gap_to_judge_best") is not None]
    dimension_means = {}
    for dim in DIMENSIONS:
        values = [
            float(row.get("system_dimension_scores", {}).get(dim))
            for row in ranked
            if row.get("system_dimension_scores", {}).get(dim) is not None
        ]
        dimension_means[dim] = statistics.mean(values) if values else None

    def topk_rate(k: int) -> float:
        return sum(1 for rank in ranks if rank <= k) / total if total else 0.0

    def score_rate(threshold: float) -> float:
        return sum(1 for score in scores if score >= threshold) / total if total else 0.0

    def recall_at_k(k: int) -> Optional[float]:
        values = [row.get("recall", {}).get(f"top{k}") for row in rows if row.get("recall", {}).get(f"top{k}") is not None]
        return float(statistics.mean(values)) if values else None

    score_threshold_by_script = {}
    for threshold in SCORE_THRESHOLDS:
        key = f"ge_{str(threshold).replace('.', '_')}"
        scripts_with_any_count = sum(
            1 for row in rows
            if (row.get("high_score_counts", {}).get(key) or 0) > 0
        )
        system_best_score_count = sum(
            1 for row in rows
            if row.get("system_judge_score") is not None
            and float(row["system_judge_score"]) >= threshold
        )
        score_threshold_by_script[key] = {
            "scripts_with_any_count": scripts_with_any_count,
            "scripts_with_any_rate": scripts_with_any_count / total if total else 0.0,
            "system_best_score_count": system_best_score_count,
            "system_best_score_rate": system_best_score_count / total if total else 0.0,
        }

    max_tm = max((row.get("system_top_matches_count", 0) for row in rows), default=0)

    return {
        "script_count": total,
        "ranked_script_count": len(ranked),
        "coverage": len(ranked) / total if total else 0.0,
        "topk_rate": {f"top{k}": topk_rate(k) for k in TOPK_THRESHOLDS},
        "mean_rank": statistics.mean(ranks) if ranks else None,
        "median_rank": statistics.median(ranks) if ranks else None,
        "mean_system_judge_score": statistics.mean(scores) if scores else None,
        "min_system_judge_score": min(scores) if scores else None,
        "max_system_judge_score": max(scores) if scores else None,
        "mean_system_dimension_scores": dimension_means,
        "mean_gap_to_judge_best": statistics.mean(gaps) if gaps else None,
        "system_score_rate": {
            f"ge_{str(threshold).replace('.', '_')}": score_rate(threshold)
            for threshold in SCORE_THRESHOLDS
        },
        "score_threshold_by_script": score_threshold_by_script,
        "recall": {f"top{k}": recall_at_k(k) for k in TOPK_THRESHOLDS},
        "ndcg": {
            f"top{k}": (
                float(statistics.mean(vals))
                if (vals := [row.get("ndcg", {}).get(f"top{k}") for row in rows if row.get("ndcg", {}).get(f"top{k}") is not None])
                else None
            )
            for k in TOPK_THRESHOLDS
        },
        "max_top_matches": max_tm,
    }


def candidate_pool_stats(matrix: Dict[int, List[Dict[str, Any]]]) -> Dict[str, Any]:
    all_scores = []
    all_dimension_scores: Dict[str, List[float]] = {dim: [] for dim in DIMENSIONS}
    per_script = []
    for seg_seq, candidates in sorted(matrix.items()):
        scores = [_to_float(c.get("scores", {}).get("overall")) or 0.0 for c in candidates]
        all_scores.extend(scores)
        for candidate in candidates:
            for dim in DIMENSIONS:
                value = _to_float(candidate.get("scores", {}).get(dim))
                if value is not None:
                    all_dimension_scores[dim].append(value)
        per_script.append({
            "segment_sequence": seg_seq,
            "n_candidates": len(candidates),
            "best_score": max(scores) if scores else None,
            "mean_score": statistics.mean(scores) if scores else None,
            "score_threshold_stats": {
                f"ge_{str(threshold).replace('.', '_')}": {
                    "count": sum(1 for score in scores if score >= threshold),
                    "total": len(scores),
                    "rate": (
                        sum(1 for score in scores if score >= threshold) / len(scores)
                        if scores else 0.0
                    ),
                }
                for threshold in SCORE_THRESHOLDS
            },
            "score_rate": {
                f"ge_{str(threshold).replace('.', '_')}": (
                    sum(1 for score in scores if score >= threshold) / len(scores)
                    if scores else 0.0
                )
                for threshold in SCORE_THRESHOLDS
            },
        })
    return {
        "total_evaluations": len(all_scores),
        "script_count": len(matrix),
        "mean": statistics.mean(all_scores) if all_scores else None,
        "min": min(all_scores) if all_scores else None,
        "max": max(all_scores) if all_scores else None,
        "score_sum": sum(all_scores),
        "dimension_sums": {
            dim: sum(values) for dim, values in all_dimension_scores.items()
        },
        "dimension_counts": {
            dim: len(values) for dim, values in all_dimension_scores.items()
        },
        "dimension_means": {
            dim: statistics.mean(values) if values else None
            for dim, values in all_dimension_scores.items()
        },
        "score_threshold_stats": {
            f"ge_{str(threshold).replace('.', '_')}": {
                "count": sum(1 for score in all_scores if score >= threshold),
                "total": len(all_scores),
                "rate": (
                    sum(1 for score in all_scores if score >= threshold) / len(all_scores)
                    if all_scores else 0.0
                ),
            }
            for threshold in SCORE_THRESHOLDS
        },
        "score_rate": {
            f"ge_{str(threshold).replace('.', '_')}": (
                sum(1 for score in all_scores if score >= threshold) / len(all_scores)
                if all_scores else 0.0
            )
            for threshold in SCORE_THRESHOLDS
        },
        "per_script": per_script,
    }


def compare_dj(
    dj: str,
    variants: List[str],
    canonical_variant: str,
    canonical_run_id: str = "",
) -> Dict[str, Any]:
    canonical_run = find_latest_completed_run(canonical_variant, dj, canonical_run_id)
    canonical_config = _load_json(canonical_run / "config.json")
    canonical_structured = CACHE_ROOT / dj / canonical_config["structured_json"]
    script_meta = load_script_meta(canonical_structured)
    matrix = load_canonical_matrix(canonical_run)
    pool = candidate_pool_stats(matrix)

    variant_results: Dict[str, Any] = {}
    for variant in variants:
        try:
            structured_json = find_variant_structured_json(dj, variant)
            matches = extract_variant_best_matches(structured_json)
            rows = evaluate_variant_on_matrix(script_meta, matrix, matches)
            variant_results[variant] = {
                "status": "ok",
                "structured_json": str(structured_json),
                "summary": aggregate_rows(rows),
                "scripts": rows,
            }
        except Exception as exc:
            rows = evaluate_variant_on_matrix(script_meta, matrix, {})
            variant_results[variant] = {
                "status": "error",
                "error": str(exc),
                "structured_json": "",
                "summary": aggregate_rows(rows),
                "scripts": rows,
            }

    return {
        "dj": dj,
        "canonical_variant": canonical_variant,
        "canonical_run": str(canonical_run),
        "canonical_structured_json": str(canonical_structured),
        "canonical_config": canonical_config,
        "candidate_pool": pool,
        "variants": variant_results,
    }


def aggregate_variant_across_djs(results: List[Dict[str, Any]], variant: str) -> Dict[str, Any]:
    rows = []
    for result in results:
        variant_result = result["variants"].get(variant, {})
        rows.extend(variant_result.get("scripts") or [])
    return aggregate_rows(rows)


def aggregate_candidate_pool_across_djs(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    pools = [result["candidate_pool"] for result in results]
    total_evaluations = sum(pool.get("total_evaluations", 0) for pool in pools)
    script_count = sum(pool.get("script_count", 0) for pool in pools)
    score_sum = sum(pool.get("score_sum", 0.0) for pool in pools)
    mins = [pool.get("min") for pool in pools if pool.get("min") is not None]
    maxs = [pool.get("max") for pool in pools if pool.get("max") is not None]

    dimension_sums = {
        dim: sum(pool.get("dimension_sums", {}).get(dim, 0.0) for pool in pools)
        for dim in DIMENSIONS
    }
    dimension_counts = {
        dim: sum(pool.get("dimension_counts", {}).get(dim, 0) for pool in pools)
        for dim in DIMENSIONS
    }

    return {
        "total_evaluations": total_evaluations,
        "script_count": script_count,
        "mean": score_sum / total_evaluations if total_evaluations else None,
        "min": min(mins) if mins else None,
        "max": max(maxs) if maxs else None,
        "score_sum": score_sum,
        "dimension_means": {
            dim: (
                dimension_sums[dim] / dimension_counts[dim]
                if dimension_counts[dim] else None
            )
            for dim in DIMENSIONS
        },
        "score_threshold_stats": {
            f"ge_{str(threshold).replace('.', '_')}": {
                "count": sum(
                    pool.get("score_threshold_stats", {})
                    .get(f"ge_{str(threshold).replace('.', '_')}", {})
                    .get("count", 0)
                    for pool in pools
                ),
                "total": total_evaluations,
                "rate": (
                    sum(
                        pool.get("score_threshold_stats", {})
                        .get(f"ge_{str(threshold).replace('.', '_')}", {})
                        .get("count", 0)
                        for pool in pools
                    ) / total_evaluations
                    if total_evaluations else 0.0
                ),
            }
            for threshold in SCORE_THRESHOLDS
        },
    }


def _variant_metric_row(variant: str, summary: Dict[str, Any]) -> Dict[str, Any]:
    topk = summary.get("topk_rate", {})
    score_rate = summary.get("system_score_rate", {})
    recall = summary.get("recall", {})
    ndcg = summary.get("ndcg", {})
    return {
        "variant": variant,
        "script_count": summary.get("script_count", 0),
        "coverage": summary.get("coverage"),
        "top1": topk.get("top1"),
        "top3": topk.get("top3"),
        "top5": topk.get("top5"),
        "top10": topk.get("top10"),
        "top20": topk.get("top20"),
        "recall_top1": recall.get("top1"),
        "recall_top3": recall.get("top3"),
        "recall_top5": recall.get("top5"),
        "recall_top10": recall.get("top10"),
        "recall_top20": recall.get("top20"),
        "ndcg_top1": ndcg.get("top1"),
        "ndcg_top3": ndcg.get("top3"),
        "ndcg_top5": ndcg.get("top5"),
        "ndcg_top10": ndcg.get("top10"),
        "ndcg_top20": ndcg.get("top20"),
        "max_top_matches": summary.get("max_top_matches", 0),
        "mean_rank": summary.get("mean_rank"),
        "median_rank": summary.get("median_rank"),
        "mean_system_judge_score": summary.get("mean_system_judge_score"),
        "mean_gap_to_judge_best": summary.get("mean_gap_to_judge_best"),
        "ge_4_0": score_rate.get("ge_4_0"),
        "ge_4_5": score_rate.get("ge_4_5"),
    }


# --- 三套排序指标：命中 / 召回 / NDCG ---
_METRICS_HIT_FIRST = [
    ("top1", "Hit@1", True, _pct),
    ("top3", "Hit@3", True, _pct),
    ("top5", "Hit@5", True, _pct),
    ("top10", "Hit@10", True, _pct),
    ("top20", "Hit@20", True, _pct),
]

_METRICS_RECALL_FIRST = [
    ("recall_top1", "R@1", True, _pct),
    ("recall_top3", "R@3", True, _pct),
    ("recall_top5", "R@5", True, _pct),
    ("recall_top10", "R@10", True, _pct),
    ("recall_top20", "R@20", True, _pct),
]

_METRICS_NDCG_FIRST = [
    ("ndcg_top1", "NDCG@1", True, _score),
    ("ndcg_top3", "NDCG@3", True, _score),
    ("ndcg_top5", "NDCG@5", True, _score),
    ("ndcg_top10", "NDCG@10", True, _score),
    ("ndcg_top20", "NDCG@20", True, _score),
]

# 通用辅助指标，三个导向都展示
_AUX_METRICS = [
    ("mean_system_judge_score", "系统均分", True, _score),
    ("ge_4_0", ">=4.0", True, _pct),
    ("ge_4_5", ">=4.5", True, _pct),
    ("mean_gap_to_judge_best", "平均分差", False, _score),
    ("mean_rank", "均rank", False, _score),
]

ALL_METRICS = _METRICS_HIT_FIRST + _METRICS_RECALL_FIRST + _METRICS_NDCG_FIRST + _AUX_METRICS

# 判断理由中用到的指标顺序，三导向通用
_JUDGE_KEY_ORDER = [
    ("top1", "Hit@1"),
    ("top3", "Hit@3"),
    ("top5", "Hit@5"),
    ("recall_top1", "R@1"),
    ("recall_top3", "R@3"),
    ("recall_top5", "R@5"),
    ("ndcg_top1", "NDCG@1"),
    ("ndcg_top3", "NDCG@3"),
    ("ndcg_top5", "NDCG@5"),
    ("mean_system_judge_score", "均分"),
    ("ge_4_0", ">=4.0"),
    ("ge_4_5", ">=4.5"),
]


def _make_sort_key(metric_specs: List[Tuple]) -> Tuple[Any, ...]:
    """Build a sort-key callable from a list of (key, label, higher_is_better, formatter) entries."""
    def sort_key(row: Dict[str, Any]) -> Tuple[float, ...]:
        return tuple(
            _to_float(row.get(key)) or 0.0 if higher_is_better
            else -(_to_float(row.get(key)) or 9999.0)
            for key, _, higher_is_better, _ in metric_specs
        )
    return sort_key


def _metric_leader_reasons(ranking: List[Dict[str, Any]], metric_specs: List[Tuple]) -> List[str]:
    if not ranking:
        return []

    best = ranking[0]
    reasons = []
    for key, label, higher_is_better, formatter in metric_specs:
        values = [_to_float(row.get(key)) for row in ranking if row.get(key) is not None]
        best_value = _to_float(best.get(key))
        if not values or best_value is None:
            continue
        target = max(values) if higher_is_better else min(values)
        if abs(best_value - target) < 1e-9:
            suffix = "最高" if higher_is_better else "最低"
            reasons.append(f"{label}{suffix}（{formatter(best_value)}）")
    return reasons


def _variant_judgement(row: Dict[str, Any], best: Dict[str, Any]) -> str:
    if row.get("rank_in_section") == 1:
        return "综合最佳"

    notes = []
    for key, label in _JUDGE_KEY_ORDER:
        best_val = _to_float(best.get(key))
        row_val = _to_float(row.get(key))
        if best_val is not None and row_val is not None and best_val > row_val:
            notes.append(f"{label} 低 {_pct_delta(best_val - row_val)}")

    best_gap = _to_float(best.get("mean_gap_to_judge_best"))
    row_gap = _to_float(row.get("mean_gap_to_judge_best"))
    if best_gap is not None and row_gap is not None and row_gap > best_gap:
        notes.append(f"分差高 {_score(row_gap - best_gap)}")

    return "；".join(notes[:3]) if notes else "接近最佳"


def build_variant_conclusion(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    """生成两套排序结论：命中导向 + 召回导向。"""
    variants_rows = [
        _variant_metric_row(variant, report["overall"][variant])
        for variant in report["variants"]
    ]

    conclusions = []
    for mode, metric_specs, method_desc in [
        ("hit", _METRICS_HIT_FIRST, "按 Hit@1 → Hit@3 → Hit@5 → Hit@10 → Hit@20 字典序排列。"),
        ("recall", _METRICS_RECALL_FIRST, "按 R@1 → R@3 → R@5 → R@10 → R@20 字典序排列。"),
        ("ndcg", _METRICS_NDCG_FIRST, "按 NDCG@1 → NDCG@3 → NDCG@5 → NDCG@10 → NDCG@20 字典序排列。"),
    ]:
        ranked = [dict(row) for row in variants_rows]
        ranked.sort(key=_make_sort_key(metric_specs), reverse=True)

        for index, row in enumerate(ranked, 1):
            row["rank_in_section"] = index

        best = ranked[0] if ranked else {}
        reasons = _metric_leader_reasons(ranked, metric_specs)
        for row in ranked:
            row["judgement"] = _variant_judgement(row, best) if best else "-"

        conclusions.append({
            "mode": mode,
            "method": method_desc,
            "best_variant": best.get("variant", ""),
            "best_reasons": reasons,
            "ranking": ranked,
        })

    return conclusions


def build_report(results: List[Dict[str, Any]], variants: List[str], canonical_variant: str) -> Dict[str, Any]:
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "canonical_variant": canonical_variant,
        "variants": variants,
        "djs": [result["dj"] for result in results],
        "candidate_pool_overall": aggregate_candidate_pool_across_djs(results),
        "overall": {
            variant: aggregate_variant_across_djs(results, variant)
            for variant in variants
        },
        "per_dj": results,
    }
    report["conclusions"] = build_variant_conclusion(report)
    return report


def write_json_report(report: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def write_markdown_report(report: Dict[str, Any], output_path: Path, include_scripts: bool = False) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    variants = report["variants"]
    variant_title = " vs ".join(variants)
    lines = [
        f"# {variant_title} 对比报告",
        "",
        f"> 生成时间：{report['generated_at']}  ",
        f"> Canonical Judge 来源：`judge/outputs/{report['canonical_variant']}`  ",
        f"> 对比版本：{', '.join(f'`{variant}`' for variant in variants)}",
        "",
        "## 1. 口径说明",
        "",
        "Canonical Judge 只评估一次：每条文案与目标集数内所有视频片段逐一打分。"
        "不同 variant 只提供自己的 `best_match` 和 `top_matches`，本报告把这些候选放回同一份 Judge 矩阵里查排名和分数。",
        "",
        "这样比较时，候选池和 Judge 分数固定，不会混入每次重跑 LLM-as-Judge 的随机波动。",
        "",
        "## 2. 指标说明",
        "",
        "### 命中率 (Hit@K)",
        "系统选出的 `best_match` 在 Judge 排序中位于前 K 名的文案占比。",
        "衡量「系统最看好的那一个，Judge 觉得好不好」。",
        "",
        "### 召回率 (Recall@K)",
        "系统返回的前 K 个候选 (`top_matches`) 与 Judge 排名前 K 个候选的交集占比：",
        "`Recall@K = |系统 topK ∩ Judge topK| / K`。",
        "衡量「系统推荐的范围，能覆盖多少 Judge 认为好的」。",
        "例如：系统 top5 里有 3 个也在 Judge top5 里 → R@5 = 3/5 = 60%。",
        "注意：当某版本某 query 候选数不足 K 时，Recall@K 标记为 -（不适用）。",
        "",
        "### 排序质量 (NDCG@K)",
        "衡量系统排序与 Judge 理想排序的接近程度，范围 0~1。",
        "公式：NDCG@K = DCG@K / IDCG@K，其中 DCG = Σ(score_i / log2(i+1))，位置越靠后权重越低。",
        "NDCG@K = 1 代表系统前 K 个排序与 Judge 理想排序完全一致。",
        "",
        "## 3. 推荐结论",
        "",
    ]

    conclusions = report.get("conclusions", [])
    section_labels = [
        ("3a. 命中导向排名", "hit", "选中最重要：只看系统 best_match 在 Judge 排序中的位置"),
        ("3b. 召回导向排名", "recall", "覆盖最重要：只看系统 topK 与 Judge topK 的交集"),
        ("3c. NDCG 导向排名", "ndcg", "排序质量最重要：衡量系统排序与 Judge 理想排序的接近程度"),
    ]

    for section_title, mode, description in section_labels:
        conclusion = next((c for c in conclusions if c.get("mode") == mode), {})
        ranking = conclusion.get("ranking", [])
        best_variant = conclusion.get("best_variant") or "-"
        best_reasons = conclusion.get("best_reasons") or []
        reason_text = "；".join(best_reasons) if best_reasons else "关键指标综合排序最高"

        if mode == "hit":
            table_header = "| 排名 | Variant | Hit@1 | Hit@3 | Hit@5 | Hit@10 | Hit@20 | 判断 |"
            table_sep = "|---:|---|---:|---:|---:|---:|---:|---|"
        elif mode == "recall":
            table_header = "| 排名 | Variant | R@1 | R@3 | R@5 | R@10 | R@20 | 判断 |"
            table_sep = "|---:|---|---:|---:|---:|---:|---:|---|"
        else:  # ndcg
            table_header = "| 排名 | Variant | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | NDCG@20 | 判断 |"
            table_sep = "|---:|---|---:|---:|---:|---:|---:|---|"

        lines.extend([
            f"### {section_title}",
            "",
            f"> {description}。",
            "",
            f"**综合最佳：`{best_variant}`。**",
            "",
            f"排序口径：{conclusion.get('method', '-')}",
            "",
            f"胜出原因：{reason_text}。",
            "",
            table_header,
            table_sep,
        ])
        for row in ranking:
            if mode == "hit":
                lines.append(
                    f"| {row.get('rank_in_section', '-')} | `{row['variant']}` | "
                    f"{_pct(row.get('top1'))} | {_pct(row.get('top3'))} | {_pct(row.get('top5'))} | "
                    f"{_pct(row.get('top10'))} | {_pct(row.get('top20'))} | "
                    f"{row.get('judgement', '-')} |"
                )
            elif mode == "recall":
                lines.append(
                    f"| {row.get('rank_in_section', '-')} | `{row['variant']}` | "
                    f"{_pct(row.get('recall_top1'))} | {_pct(row.get('recall_top3'))} | {_pct(row.get('recall_top5'))} | "
                    f"{_pct(row.get('recall_top10'))} | {_pct(row.get('recall_top20'))} | "
                    f"{row.get('judgement', '-')} |"
                )
            else:  # ndcg
                lines.append(
                    f"| {row.get('rank_in_section', '-')} | `{row['variant']}` | "
                    f"{_score(row.get('ndcg_top1'))} | {_score(row.get('ndcg_top3'))} | {_score(row.get('ndcg_top5'))} | "
                    f"{_score(row.get('ndcg_top10'))} | {_score(row.get('ndcg_top20'))} | "
                    f"{row.get('judgement', '-')} |"
                )
        lines.extend(["", ""])

    lines.extend([
        "",
        "## 4. 总体指标",
        "",
        "| Variant | 文案数 | 覆盖率 | Hit@1 | Hit@3 | Hit@5 | Hit@10 | Hit@20 | R@1 | R@3 | R@5 | R@10 | R@20 | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | NDCG@20 | 系统均分 | 平均分差 | >=4.0 | >=4.5 | 均rank |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])

    for variant in variants:
        summary = report["overall"][variant]
        topk = summary["topk_rate"]
        score_rate = summary["system_score_rate"]
        recall = summary.get("recall", {})
        ndcg = summary.get("ndcg", {})
        lines.append(
            f"| `{variant}` | {summary['script_count']} | {_pct(summary['coverage'])} | "
            f"{_pct(topk['top1'])} | {_pct(topk['top3'])} | {_pct(topk['top5'])} | "
            f"{_pct(topk['top10'])} | {_pct(topk['top20'])} | "
            f"{_pct(recall.get('top1'))} | {_pct(recall.get('top3'))} | {_pct(recall.get('top5'))} | "
            f"{_pct(recall.get('top10'))} | {_pct(recall.get('top20'))} | "
            f"{_score(ndcg.get('top1'))} | {_score(ndcg.get('top3'))} | {_score(ndcg.get('top5'))} | "
            f"{_score(ndcg.get('top10'))} | {_score(ndcg.get('top20'))} | "
            f"{_score(summary['mean_system_judge_score'])} | "
            f"{_score(summary['mean_gap_to_judge_best'])} | "
            f"{_pct(score_rate['ge_4_0'])} | {_pct(score_rate['ge_4_5'])} | "
            f"{_score(summary['mean_rank'])} |"
        )

    pool_overall = report.get("candidate_pool_overall", {})
    lines.extend([
        "",
        "## 5. Canonical 候选池阈值命中",
        "",
        f"- 候选池总评估数：{pool_overall.get('total_evaluations', 0)}",
        f"- 候选池 overall：{_score(pool_overall.get('mean'))}（{_score_range(pool_overall.get('min'), pool_overall.get('max'))}）",
        "",
        "| 阈值 | 命中数 | 命中率 |",
        "|---|---:|---:|",
    ])
    for threshold in SCORE_THRESHOLDS:
        key = f"ge_{str(threshold).replace('.', '_')}"
        stats = pool_overall.get("score_threshold_stats", {}).get(key, {})
        lines.append(
            f"| >={threshold:.1f} | {stats.get('count', 0)} / {stats.get('total', 0)} | "
            f"{_pct(stats.get('rate'))} |"
        )

    lines.extend([
        "",
        "## 6. 系统选中片段 Judge 均分",
        "",
        "| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for variant in variants:
        summary = report["overall"][variant]
        dims = summary.get("mean_system_dimension_scores", {})
        lines.append(
            f"| `{variant}` | {_score(summary.get('mean_system_judge_score'))} | "
            f"{_score_range(summary.get('min_system_judge_score'), summary.get('max_system_judge_score'))} | "
            f"{_score(dims.get('entity_match'))} | {_score(dims.get('event_match'))} | "
            f"{_score(dims.get('scene_match'))} | {_score(dims.get('tone_match'))} | "
            f"{_score(dims.get('editing_suitability'))} |"
        )

    lines.extend([
        "",
        "## 7. 文案级高分覆盖",
        "",
        "说明：候选覆盖是文案级覆盖率，计算方式为“至少有 1 个候选 overall 达到阈值的文案段数 / 总文案段数”；同一条文案下多个高分候选仍只算 1 条文案命中。",
        "",
        "| Variant | >=4.0 候选覆盖 | >=4.0 系统达标 | >=4.5 候选覆盖 | >=4.5 系统达标 |",
        "|---|---:|---:|---:|---:|",
    ])
    for variant in variants:
        summary = report["overall"][variant]
        by_script = summary.get("score_threshold_by_script", {})
        ge40 = by_script.get("ge_4_0", {})
        ge45 = by_script.get("ge_4_5", {})
        lines.append(
            f"| `{variant}` | "
            f"{_count_pct(ge40.get('scripts_with_any_count'), summary['script_count'], ge40.get('scripts_with_any_rate'))} | "
            f"{_count_pct(ge40.get('system_best_score_count'), summary['script_count'], ge40.get('system_best_score_rate'))} | "
            f"{_count_pct(ge45.get('scripts_with_any_count'), summary['script_count'], ge45.get('scripts_with_any_rate'))} | "
            f"{_count_pct(ge45.get('system_best_score_count'), summary['script_count'], ge45.get('system_best_score_rate'))} |"
        )

    lines.extend([
        "",
        "## 8. 分剧指标",
        "",
    ])

    for result in report["per_dj"]:
        lines.extend([
            f"### `{result['dj']}`",
            "",
            f"- canonical run: `{_project_rel(Path(result['canonical_run']))}`",
            f"- candidate pool: {result['candidate_pool']['script_count']} 条文案，"
            f"{result['candidate_pool']['total_evaluations']} 条评估，"
            f"overall mean={_score(result['candidate_pool']['mean'])}",
            "",
            "| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for variant in variants:
            item = result["variants"].get(variant, {})
            if item.get("status") != "ok":
                lines.append(f"| `{variant}` | {item.get('error', 'error')} | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |")
                continue
            summary = item["summary"]
            topk = summary["topk_rate"]
            score_rate = summary["system_score_rate"]
            recall = summary.get("recall", {})
            lines.append(
                f"| `{variant}` | {_pct(summary['coverage'])} | "
                f"{_pct(recall.get('top1'))} | {_pct(recall.get('top3'))} | {_pct(recall.get('top5'))} | "
                f"{_pct(recall.get('top10'))} | {_pct(recall.get('top20'))} | "
                f"{_pct(topk['top1'])} | {_pct(topk['top3'])} | "
                f"{_pct(topk['top5'])} | {_pct(topk['top20'])} | "
                f"{_score(summary['mean_rank'])} | "
                f"{_score(summary['median_rank'])} | {_score(summary['mean_system_judge_score'])} | "
                f"{_score(summary['mean_gap_to_judge_best'])} | "
                f"{_pct(score_rate['ge_4_0'])} | {_pct(score_rate['ge_4_5'])} |"
            )
        lines.extend([
            "",
            "| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for variant in variants:
            item = result["variants"].get(variant, {})
            if item.get("status") != "ok":
                lines.append(f"| `{variant}` | - | - | - | - | - | - | - |")
                continue
            summary = item["summary"]
            dims = summary.get("mean_system_dimension_scores", {})
            lines.append(
                f"| `{variant}` | {_score(summary.get('mean_system_judge_score'))} | "
                f"{_score_range(summary.get('min_system_judge_score'), summary.get('max_system_judge_score'))} | "
                f"{_score(dims.get('entity_match'))} | {_score(dims.get('event_match'))} | "
                f"{_score(dims.get('scene_match'))} | {_score(dims.get('tone_match'))} | "
                f"{_score(dims.get('editing_suitability'))} |"
            )
        lines.append("")

    lines.extend([
        "## 9. 明显错排样本",
        "",
        "| 剧名 | Variant | 文案段 | 系统片段 | Judge 排名 | 系统分 | Judge 最优片段 | Judge 最优分 | 分差 | 诊断 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ])
    issue_count = 0
    for result in report["per_dj"]:
        for variant in variants:
            item = result["variants"].get(variant, {})
            for row in item.get("scripts") or []:
                gap = row.get("score_gap_to_judge_best")
                rank = row.get("system_judge_rank")
                if gap is None or rank is None:
                    continue
                if gap < 1.0 and rank <= 5:
                    continue
                issue_count += 1
                lines.append(
                    f"| `{result['dj']}` | `{variant}` | {row['segment_sequence']} | "
                    f"{row.get('system_segment_id') or '-'} | {rank} | {_score(row.get('system_judge_score'))} | "
                    f"{row.get('judge_best_segment_id') or '-'} | {_score(row.get('judge_best_score'))} | "
                    f"{_score(gap)} | {row.get('diagnosis', '-')} |"
                )
    if issue_count == 0:
        lines.append("| - | - | - | - | - | - | - | - | - | 无明显错排 |")
    lines.append("")

    if include_scripts:
        lines.extend([
            "## 10. 全量逐文案明细",
            "",
        ])
        for result in report["per_dj"]:
            for variant in variants:
                item = result["variants"].get(variant, {})
                if item.get("status") != "ok":
                    continue
                lines.extend([
                    f"### `{result['dj']}` / `{variant}`",
                    "",
                    "| 文案段 | 系统片段 | Judge 排名 | 系统分 | Judge 最优片段 | Judge 最优分 | 分差 | R@1 | R@3 | R@5 | R@10 | R@20 | 诊断 |",
                    "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
                ])
                for row in item["scripts"]:
                    rec = row.get("recall", {})
                    lines.append(
                        f"| {row['segment_sequence']} | {row.get('system_segment_id') or '-'} | "
                        f"{row.get('system_judge_rank') or '-'} | {_score(row.get('system_judge_score'))} | "
                        f"{row.get('judge_best_segment_id') or '-'} | {_score(row.get('judge_best_score'))} | "
                        f"{_score(row.get('score_gap_to_judge_best'))} | "
                        f"{_pct(rec.get('top1'))} | {_pct(rec.get('top3'))} | {_pct(rec.get('top5'))} | "
                        f"{_pct(rec.get('top10'))} | {_pct(rec.get('top20'))} | "
                        f"{row.get('diagnosis', '-')} |"
                    )
                lines.append("")

        # 11. 分段召回对比 (R@5)
        if len(variants) > 1:
            lines.extend([
                "## 11. 分段召回对比 (R@5)",
                "",
                "各 variant 在每个文案段上的 Recall@5 对比。`-` 表示候选不足无法计算。",
                "",
            ])
            for result in report["per_dj"]:
                seg_rows: Dict[int, Dict[str, float]] = {}
                all_segs: List[int] = []
                for variant in variants:
                    item = result["variants"].get(variant, {})
                    for row in item.get("scripts") or []:
                        seg = row["segment_sequence"]
                        rec = row.get("recall", {})
                        if seg not in seg_rows:
                            seg_rows[seg] = {}
                            all_segs.append(seg)
                        seg_rows[seg][variant] = rec.get("top5")
                if not all_segs:
                    continue
                lines.append(f"### `{result['dj']}`")
                lines.append("")
                header = "| 文案段 |"
                sep = "|---:|"
                for variant in variants:
                    header += f" `{variant}` |"
                    sep += "---:|"
                lines.append(header)
                lines.append(sep)
                for seg_seq in sorted(all_segs):
                    cells = f"| {seg_seq} |"
                    for variant in variants:
                        val = seg_rows[seg_seq].get(variant)
                        cells += f" {_pct(val)} |"
                    lines.append(cells)
                lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def _variant_detail_markdown(
    report: Dict[str, Any],
    dj_result: Dict[str, Any],
    variant: str,
    variant_result: Dict[str, Any],
) -> str:
    lines = [
        f"# {dj_result['dj']} / {variant} 匹配效果报告",
        "",
        f"> 生成时间：{report['generated_at']}  ",
        f"> Canonical Judge：`{_project_rel(Path(dj_result['canonical_run']))}`  ",
        f"> Canonical 文案：`{_project_rel(Path(dj_result['canonical_structured_json']))}`",
        "",
    ]

    if variant_result.get("status") != "ok":
        lines.extend([
            "## 状态",
            "",
            f"- variant structured: `{variant_result.get('structured_json') or '-'}`",
            f"- error: `{variant_result.get('error', 'unknown error')}`",
            "",
            "这个 variant 没有可用匹配结果，因此下面的覆盖率和命中率会按 0 处理。",
            "",
        ])
    else:
        lines.extend([
            f"> Variant 匹配结果：`{_project_rel(Path(variant_result['structured_json']))}`",
            "",
        ])

    summary = variant_result["summary"]
    topk = summary["topk_rate"]
    score_rate = summary["system_score_rate"]
    recall = summary.get("recall", {})
    ndcg = summary.get("ndcg", {})
    score_by_script = summary.get("score_threshold_by_script", {})
    selected_dims = summary.get("mean_system_dimension_scores", {})
    pool = dj_result["candidate_pool"]
    pool_dims = pool.get("dimension_means", {})
    pool_thresholds = pool.get("score_threshold_stats", {})

    lines.extend([
        "## 指标说明",
        "",
        "- **命中率 (Hit@K)**：系统选出的 best_match 在 Judge 排序中位于前 K 名的文案占比。",
        "- **召回率 (Recall@K)**：系统返回的前 K 个候选与 Judge 排名前 K 个候选的交集占比。",
        "",
        "## 汇总",
        "",
        "### 命中率",
        "",
        "| 指标 | 值 |",
        "|---|---:|",
        f"| 文案数 | {summary['script_count']} |",
        f"| 覆盖率 | {_pct(summary['coverage'])} |",
        f"| Hit@1 | {_pct(topk['top1'])} |",
        f"| Hit@3 | {_pct(topk['top3'])} |",
        f"| Hit@5 | {_pct(topk['top5'])} |",
        f"| Hit@10 | {_pct(topk['top10'])} |",
        f"| Hit@20 | {_pct(topk['top20'])} |",
        f"| 平均 Judge 排名 | {_score(summary['mean_rank'])} |",
        f"| 中位 Judge 排名 | {_score(summary['median_rank'])} |",
        f"| 系统选中片段 Judge 均分 | {_score(summary['mean_system_judge_score'])} |",
        f"| 系统选中片段 overall 范围 | {_score_range(summary.get('min_system_judge_score'), summary.get('max_system_judge_score'))} |",
        f"| 与 Judge 最优平均分差 | {_score(summary['mean_gap_to_judge_best'])} |",
        f"| 系统选中片段 >=4.0 | {_pct(score_rate['ge_4_0'])} |",
        f"| 系统选中片段 >=4.5 | {_pct(score_rate['ge_4_5'])} |",
        "",
        "### 召回率",
        "",
        "| 指标 | 值 | 说明 |",
        "|---|---:|---|",
        f"| Recall@1 | {_pct(recall.get('top1'))} | 系统 top1 与 Judge top1 是否一致 |",
        f"| Recall@3 | {_pct(recall.get('top3'))} | 系统 top3 与 Judge top3 交集占比 |",
        f"| Recall@5 | {_pct(recall.get('top5'))} | 系统 top5 与 Judge top5 交集占比 |",
        f"| Recall@10 | {_pct(recall.get('top10'))} | 系统 top10 与 Judge top10 交集占比 |",
        f"| Recall@20 | {_pct(recall.get('top20'))} | 系统 top20 与 Judge top20 交集占比 |",
        "",
        f"### NDCG (排序质量)",
        "",
        "| 指标 | 值 | 说明 |",
        "|---|---:|---|",
        f"| NDCG@1 | {_score(ndcg.get('top1'))} | 排序首位是否最优 |",
        f"| NDCG@3 | {_score(ndcg.get('top3'))} | 前3排序与Judge理想排序接近度 |",
        f"| NDCG@5 | {_score(ndcg.get('top5'))} | 前5排序与Judge理想排序接近度 |",
        f"| NDCG@10 | {_score(ndcg.get('top10'))} | 前10排序与Judge理想排序接近度 |",
        f"| NDCG@20 | {_score(ndcg.get('top20'))} | 前20排序与Judge理想排序接近度 |",
        "",
        "",
        "| 指标 | 值 |",
        "|---|---:|",
        f"| entity | {_score(selected_dims.get('entity_match'))} |",
        f"| event | {_score(selected_dims.get('event_match'))} |",
        f"| scene | {_score(selected_dims.get('scene_match'))} |",
        f"| tone | {_score(selected_dims.get('tone_match'))} |",
        f"| editing | {_score(selected_dims.get('editing_suitability'))} |",
        "",
        "## Canonical 候选池",
        "",
        "| 指标 | 值 |",
        "|---|---:|",
        f"| 文案数 | {pool['script_count']} |",
        f"| 总评估数 | {pool['total_evaluations']} |",
        f"| 候选池 overall 均值 | {_score(pool['mean'])} |",
        f"| 候选池 overall 最小值 | {_score(pool['min'])} |",
        f"| 候选池 overall 最大值 | {_score(pool['max'])} |",
        f"| 候选池 entity 均值 | {_score(pool_dims.get('entity_match'))} |",
        f"| 候选池 event 均值 | {_score(pool_dims.get('event_match'))} |",
        f"| 候选池 scene 均值 | {_score(pool_dims.get('scene_match'))} |",
        f"| 候选池 tone 均值 | {_score(pool_dims.get('tone_match'))} |",
        f"| 候选池 editing 均值 | {_score(pool_dims.get('editing_suitability'))} |",
        f"| 候选池片段 >=4.0 | {_count_pct(pool_thresholds.get('ge_4_0', {}).get('count'), pool_thresholds.get('ge_4_0', {}).get('total'), pool_thresholds.get('ge_4_0', {}).get('rate'))} |",
        f"| 候选池片段 >=4.5 | {_count_pct(pool_thresholds.get('ge_4_5', {}).get('count'), pool_thresholds.get('ge_4_5', {}).get('total'), pool_thresholds.get('ge_4_5', {}).get('rate'))} |",
        "",
        "## 候选池阈值命中",
        "",
        "| 阈值 | 命中数 | 命中率 |",
        "|---|---:|---:|",
    ])
    for threshold in SCORE_THRESHOLDS:
        key = f"ge_{str(threshold).replace('.', '_')}"
        stats = pool_thresholds.get(key, {})
        lines.append(
            f"| >={threshold:.1f} | {stats.get('count', 0)} / {stats.get('total', 0)} | "
            f"{_pct(stats.get('rate'))} |"
        )

    lines.extend([
        "",
        "## 文案级高分覆盖",
        "",
        "说明：候选覆盖是文案级覆盖率，计算方式为“至少有 1 个候选 overall 达到阈值的文案段数 / 总文案段数”；同一条文案下多个高分候选仍只算 1 条文案命中。",
        "",
        "| 阈值 | 存在高分候选的文案占比 | 系统选中片段达标的文案占比 |",
        "|---|---:|---:|",
    ])
    for threshold in SCORE_THRESHOLDS:
        key = f"ge_{str(threshold).replace('.', '_')}"
        stats = score_by_script.get(key, {})
        lines.append(
            f"| >={threshold:.1f} | "
            f"{_count_pct(stats.get('scripts_with_any_count'), summary['script_count'], stats.get('scripts_with_any_rate'))} | "
            f"{_count_pct(stats.get('system_best_score_count'), summary['script_count'], stats.get('system_best_score_rate'))} |"
        )

    lines.extend([
        "",
        "## 逐文案效果",
        "",
        "| 文案段 | 文案 | 目标集 | 系统片段 | 源分 | Judge 排名 | 系统 Judge 分 | Judge 最优片段 | Judge 最优分 | 分差 | >=4候选数 | 诊断 |",
        "|---:|---|---:|---|---:|---:|---:|---|---:|---:|---:|---|",
    ])

    for row in variant_result.get("scripts") or []:
        high_counts = row.get("high_score_counts") or {}
        lines.append(
            f"| {row['segment_sequence']} | {_md_cell(row.get('script_text'), 72)} | "
            f"{row.get('episode_target') or '-'} | "
            f"{_ep_seg(row.get('system_episode'), row.get('system_segment_id'))} | "
            f"{_score(row.get('system_source_score'), 4)} | "
            f"{row.get('system_judge_rank') or '-'} | "
            f"{_score(row.get('system_judge_score'))} | "
            f"{_ep_seg(row.get('judge_best_episode'), row.get('judge_best_segment_id'))} | "
            f"{_score(row.get('judge_best_score'))} | "
            f"{_score(row.get('score_gap_to_judge_best'))} | "
            f"{high_counts.get('ge_4_0', 0)} | "
            f"{row.get('diagnosis', '-')} |"
        )

    lines.extend([
        "",
        "## 逐文案召回率",
        "",
        "| 文案段 | 文案 | R@1 | R@3 | R@5 | R@10 | R@20 |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ])
    for row in variant_result.get("scripts") or []:
        rec = row.get("recall", {})
        lines.append(
            f"| {row['segment_sequence']} | {_md_cell(row.get('script_text'), 72)} | "
            f"{_pct(rec.get('top1'))} | {_pct(rec.get('top3'))} | {_pct(rec.get('top5'))} | "
            f"{_pct(rec.get('top10'))} | {_pct(rec.get('top20'))} |"
        )

    issue_rows = []
    for row in variant_result.get("scripts") or []:
        rank = row.get("system_judge_rank")
        gap = row.get("score_gap_to_judge_best")
        if rank is None or gap is None or rank > 5 or gap >= 1.0:
            issue_rows.append(row)

    lines.extend([
        "",
        "## 重点问题",
        "",
        "| 文案段 | 系统片段 | Judge 排名 | 系统分 | Judge 最优片段 | Judge 最优分 | 分差 | 诊断 |",
        "|---:|---|---:|---:|---|---:|---:|---|",
    ])
    if not issue_rows:
        lines.append("| - | - | - | - | - | - | - | 无明显问题 |")
    else:
        for row in issue_rows:
            lines.append(
                f"| {row['segment_sequence']} | "
                f"{_ep_seg(row.get('system_episode'), row.get('system_segment_id'))} | "
                f"{row.get('system_judge_rank') or '-'} | "
                f"{_score(row.get('system_judge_score'))} | "
                f"{_ep_seg(row.get('judge_best_episode'), row.get('judge_best_segment_id'))} | "
                f"{_score(row.get('judge_best_score'))} | "
                f"{_score(row.get('score_gap_to_judge_best'))} | "
                f"{row.get('diagnosis', '-')} |"
            )

    lines.append("")
    return "\n".join(lines)


def write_variant_detail_reports(
    report: Dict[str, Any],
    output_root: Path,
) -> List[Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    for dj_result in report["per_dj"]:
        dj = dj_result["dj"]
        for variant in report["variants"]:
            variant_result = dj_result["variants"].get(variant)
            if not variant_result:
                continue

            run_dir = output_root / variant / dj
            run_dir.mkdir(parents=True, exist_ok=True)

            detail_payload = {
                "report_type": "canonical_variant_detail",
                "generated_at": report["generated_at"],
                "canonical_variant": report["canonical_variant"],
                "dj": dj,
                "variant": variant,
                "canonical_run": dj_result["canonical_run"],
                "canonical_structured_json": dj_result["canonical_structured_json"],
                "candidate_pool": dj_result["candidate_pool"],
                "variant_result": variant_result,
            }
            config = {
                "report_type": "canonical_variant_detail",
                "generated_at": report["generated_at"],
                "dj": dj,
                "variant": variant,
                "canonical_variant": report["canonical_variant"],
                "canonical_run": _project_rel(Path(dj_result["canonical_run"])),
                "canonical_structured_json": _project_rel(Path(dj_result["canonical_structured_json"])),
                "variant_structured_json": (
                    _project_rel(Path(variant_result["structured_json"]))
                    if variant_result.get("structured_json") else ""
                ),
                "status": variant_result.get("status", "unknown"),
            }

            md_content = _variant_detail_markdown(report, dj_result, variant, variant_result)
            (run_dir / "summary.md").write_text(md_content, encoding="utf-8")
            (run_dir / "summary.txt").write_text(md_content, encoding="utf-8")
            (run_dir / "report.json").write_text(
                json.dumps(detail_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (run_dir / "config.json").write_text(
                json.dumps(config, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            written.append(run_dir / "summary.md")

    return written


def parse_variants(raw: str) -> List[str]:
    variants = [part.strip() for part in raw.split(",") if part.strip()]
    if not variants:
        raise ValueError("--variants cannot be empty")
    return variants


def _slugify_filename_part(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    slug = slug.strip("._-")
    return slug or "variant"


def default_compare_stem(variants: List[str], timestamp: str) -> str:
    variant_part = "_vs_".join(_slugify_filename_part(variant) for variant in variants)
    return f"compare_{variant_part}_{timestamp}"


def resolve_djs(args: argparse.Namespace) -> List[str]:
    djs: List[str] = []
    if args.input_arg:
        if args.input_arg.endswith(".txt"):
            djs.extend(load_task_djs(args.input_arg))
        else:
            djs.append(args.input_arg)
    if args.task_file:
        djs.extend(load_task_djs(args.task_file))
    if args.dj:
        djs.append(args.dj)

    ordered: List[str] = []
    seen = set()
    for dj in djs:
        if dj and dj not in seen:
            ordered.append(dj)
            seen.add(dj)
    if not ordered:
        raise ValueError("Please provide a dj or task file")
    return ordered


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare variants using one canonical Judge matrix")
    parser.add_argument("input_arg", nargs="?", default="", help="剧名或 task txt 路径")
    parser.add_argument("--task-file", default="", help="任务文件")
    parser.add_argument("--dj", default="", help="单剧名")
    parser.add_argument("--variants", default="v4,qwen_q1,qwen_q2,qwen_q3", help="逗号分隔的 variant 列表")
    parser.add_argument("--canonical-variant", default="canonical", help="canonical Judge 所在 outputs variant")
    parser.add_argument("--canonical-run-id", default="", help="指定 canonical run_id；默认每部剧取最新完成 run")
    parser.add_argument("--out", default="", help="Markdown 输出路径")
    parser.add_argument("--json-out", default="", help="JSON 输出路径")
    parser.add_argument("--include-scripts", action="store_true", help="Markdown 中包含全量逐文案明细")
    parser.add_argument("--detail-output-root", default="", help="每个 variant/dj 明细报告输出根目录，默认 judge/outputs")
    parser.add_argument("--no-detail-reports", action="store_true", help="只生成总汇总，不写 outputs/{variant}/{dj} 明细")
    parser.add_argument("--no-20s-filter", action="store_true", help="禁用前20秒截断，对比全量文案段")
    args = parser.parse_args()

    global _DISABLE_20S_FILTER
    if args.no_20s_filter:
        _DISABLE_20S_FILTER = True

    djs = resolve_djs(args)
    variants = parse_variants(args.variants)

    results = [
        compare_dj(
            dj=dj,
            variants=variants,
            canonical_variant=args.canonical_variant,
            canonical_run_id=args.canonical_run_id,
        )
        for dj in djs
    ]
    report = build_report(results, variants, args.canonical_variant)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    default_stem = default_compare_stem(variants, timestamp)
    md_path = Path(args.out) if args.out else OUTPUTS_DIR / f"{default_stem}.md"
    json_path = Path(args.json_out) if args.json_out else md_path.with_suffix(".json")
    if not md_path.is_absolute():
        md_path = PROJECT_ROOT / md_path
    if not json_path.is_absolute():
        json_path = PROJECT_ROOT / json_path

    write_markdown_report(report, md_path, include_scripts=args.include_scripts)
    write_json_report(report, json_path)
    print(f"Markdown report: {md_path}")
    print(f"JSON report: {json_path}")

    if not args.no_detail_reports:
        detail_output_root = Path(args.detail_output_root) if args.detail_output_root else OUTPUTS_DIR
        if not detail_output_root.is_absolute():
            detail_output_root = PROJECT_ROOT / detail_output_root
        detail_paths = write_variant_detail_reports(report, detail_output_root)
        print(f"Variant detail reports: {len(detail_paths)} dirs under {detail_output_root}")


if __name__ == "__main__":
    main()
