#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RL / Reranker 训练数据构造器（v1）

当前阶段只做 samples 层：
1. 读取 Judge results.jsonl
2. 对齐 structured.json / structured_description.json
3. 导出 samples.jsonl / dropped.jsonl / stats.json
"""

import argparse
import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


LOGGER = logging.getLogger(__name__)
BUILDER_VERSION = "samples.v2"
EVAL_ID_RE = re.compile(
    r"^(?P<dj>.+?)_seg(?P<segment_sequence>\d+)(?:_ep(?P<episode>\d+))?_cand(?P<candidate>\d+)$"
)
SCORE_DIMENSIONS = (
    "entity_match",
    "event_match",
    "scene_match",
    "tone_match",
    "editing_suitability",
)
POSITIVE_BUCKETS = {"strong_positive", "fallback_positive"}
NEGATIVE_BUCKETS = {"hard_negative", "easy_negative", "dimension_negative"}


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def cache_base(dj: str, cache_dir: Optional[str] = None) -> Path:
    if cache_dir:
        return Path(cache_dir)
    return repo_root() / "text_video_match" / "cache" / dj


def judge_run_dir(dj: str, run_id: str, judge_output_dir: Optional[str] = None) -> Path:
    if judge_output_dir:
        return Path(judge_output_dir) / dj / run_id
    return repo_root() / "judge" / "outputs" / dj / run_id


def default_output_dir(dataset_version: str, dj: str, run_id: str) -> Path:
    return repo_root() / "judge" / "training_data" / dataset_version / dj / run_id


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> List[Tuple[int, Dict[str, Any]]]:
    rows: List[Tuple[int, Dict[str, Any]]] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            rows.append((lineno, json.loads(raw)))
        except json.JSONDecodeError as exc:
            rows.append((lineno, {"_parse_error": str(exc), "_raw_line": raw}))
    return rows


def latest_structured_json(base: Path) -> Optional[Path]:
    candidates = sorted(base.glob("*_structured.json"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def make_segment_key(segment_id: Any, episode: Any) -> Optional[str]:
    if segment_id is None:
        return None
    if episode is None:
        return f"seg{int(segment_id)}"
    return f"ep{int(episode)}_seg{int(segment_id)}"


def build_segment_indexes(sd_path: Path) -> Tuple[Dict[str, Dict[str, Any]], Dict[int, List[Dict[str, Any]]]]:
    data = load_json(sd_path)
    key_index: Dict[str, Dict[str, Any]] = {}
    id_index: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    for seg in data.get("segments", []):
        seg_id = seg.get("segment_id")
        if seg_id is None:
            continue
        key = make_segment_key(seg_id, seg.get("episode"))
        if key:
            key_index[key] = seg
        id_index[int(seg_id)].append(seg)

    return key_index, id_index


def build_query_index(structured_json_path: Path) -> Dict[int, Dict[str, Any]]:
    segments = load_json(structured_json_path)
    if isinstance(segments, dict):
        segments = segments.get("segments", [])

    query_index: Dict[int, Dict[str, Any]] = {}
    for seg in segments:
        seg_seq = seg.get("segment_sequence")
        if seg_seq is not None:
            query_index[int(seg_seq)] = seg
    return query_index


def parse_eval_id(eval_id: str) -> Dict[str, Optional[int]]:
    match = EVAL_ID_RE.match(eval_id or "")
    if not match:
        return {"segment_sequence": None, "episode": None, "candidate": None}
    groups = match.groupdict()
    return {
        "segment_sequence": int(groups["segment_sequence"]) if groups.get("segment_sequence") else None,
        "episode": int(groups["episode"]) if groups.get("episode") else None,
        "candidate": int(groups["candidate"]) if groups.get("candidate") else None,
    }


def find_candidate_segment(
    candidate: Dict[str, Any],
    key_index: Dict[str, Dict[str, Any]],
    id_index: Dict[int, List[Dict[str, Any]]],
) -> Optional[Dict[str, Any]]:
    seg_id = candidate.get("segment_id")
    episode = candidate.get("episode")

    key = make_segment_key(seg_id, episode)
    if key and key in key_index:
        return key_index[key]

    if seg_id is None:
        return None

    matches = id_index.get(int(seg_id), [])
    if not matches:
        return None
    if episode is None:
        return matches[0]
    for item in matches:
        if item.get("episode") == episode:
            return item
    return matches[0]


def find_system_rank(
    top_matches: List[Dict[str, Any]],
    candidate_segment_id: Optional[int],
    candidate_episode: Optional[int],
) -> Tuple[Optional[int], Optional[float]]:
    if candidate_segment_id is None:
        return None, None

    for idx, item in enumerate(top_matches, start=1):
        item_seg_id = item.get("segment_id")
        if item_seg_id is None or int(item_seg_id) != int(candidate_segment_id):
            continue
        item_episode = item.get("episode")
        if item_episode is not None and candidate_episode is not None and int(item_episode) != int(candidate_episode):
            continue
        similarity = item.get("similarity")
        return idx, float(similarity) if similarity is not None else None

    return None, None


def is_same_candidate(left: Dict[str, Any], right_segment_id: Optional[int], right_episode: Optional[int]) -> bool:
    left_segment_id = left.get("segment_id")
    if left_segment_id is None or right_segment_id is None:
        return False
    if int(left_segment_id) != int(right_segment_id):
        return False

    left_episode = left.get("episode")
    if left_episode is None or right_episode is None:
        return True
    return int(left_episode) == int(right_episode)


def build_sample(
    result: Dict[str, Any],
    query_seg: Dict[str, Any],
    candidate_seg: Dict[str, Any],
    config: Dict[str, Any],
    dataset_version: str,
) -> Dict[str, Any]:
    query = result.get("query", {}) or {}
    candidate = result.get("candidate", {}) or {}
    metadata = result.get("metadata", {}) or {}
    match_result = query_seg.get("match_result", {}) or {}
    top_matches = match_result.get("top_matches", []) or []
    system_best = match_result.get("best_match", {}) or {}

    segment_sequence = int(query.get("segment_sequence") or query_seg.get("segment_sequence") or 0)
    candidate_segment_id = candidate.get("segment_id")
    candidate_episode = candidate.get("episode")
    system_rank, system_similarity = find_system_rank(top_matches, candidate_segment_id, candidate_episode)

    query_id = f"{config['dj']}_seg{segment_sequence}"
    if candidate_segment_id is None:
        candidate_id = "unknown"
    else:
        candidate_id = make_segment_key(candidate_segment_id, candidate_episode) or f"seg{candidate_segment_id}"

    return {
        "sample_id": result.get("evaluation_id"),
        "query_id": query_id,
        "candidate_id": candidate_id,
        "source": {
            "dj": config.get("dj"),
            "run_id": config.get("run_id") or metadata.get("run_id"),
            "structured_json": config.get("structured_json"),
            "judge_model": metadata.get("judge_model"),
            "judge_version": metadata.get("judge_version"),
            "judge_temperature": metadata.get("temperature"),
            "judge_timestamp": metadata.get("timestamp"),
            "dataset_version": dataset_version,
            "builder_version": BUILDER_VERSION,
        },
        "query": {
            "segment_sequence": segment_sequence,
            "sentence_idx": query_seg.get("_sentence_idx"),
            "script_text": query.get("script_text") or query_seg.get("script_text"),
            "episode_target": query.get("episode_target") or query_seg.get("episode_target"),
            "episode_confidence": query_seg.get("episode_confidence"),
            "audio_path": query_seg.get("audio_path"),
            "audio_duration": query_seg.get("audio_duration"),
            "visual_query": query_seg.get("visual_query"),
        },
        "candidate": {
            "segment_id": candidate_segment_id,
            "episode": candidate_episode if candidate_episode is not None else candidate_seg.get("episode"),
            "duration": candidate_seg.get("duration"),
            "start_time": candidate_seg.get("start_time"),
            "end_time": candidate_seg.get("end_time"),
            "segment_file_path": candidate_seg.get("segment_file_path"),
            "video_segment_path": metadata.get("video_segment_path"),
            "structured_description": candidate_seg.get("structured_description"),
        },
        "judge": {
            "scores": result.get("scores"),
            "reasoning": result.get("reasoning"),
            "confidence": result.get("confidence"),
            "would_use": result.get("would_use"),
            "match_highlights": result.get("match_highlights", []),
            "match_issues": result.get("match_issues", []),
        },
        "retrieval": {
            "match_breakdown": match_result.get("match_breakdown"),
            "system_best_match": system_best,
            "system_top_matches": top_matches,
            "is_system_best_match": is_same_candidate(system_best, candidate_segment_id, candidate_episode),
            "in_system_top_matches": system_rank is not None,
            "system_rank": system_rank,
            "system_similarity": system_similarity,
        },
        "labels": {
            "sample_bucket": None,
            "tie_group": None,
            "is_positive_candidate": None,
            "is_negative_candidate": None,
        },
    }


def sample_scores(sample: Dict[str, Any]) -> Dict[str, float]:
    return sample.get("judge", {}).get("scores", {}) or {}


def sample_overall(sample: Dict[str, Any]) -> Optional[float]:
    overall = sample_scores(sample).get("overall")
    return float(overall) if overall is not None else None


def dimension_values(sample: Dict[str, Any]) -> Dict[str, float]:
    scores = sample_scores(sample)
    return {
        dim: float(scores[dim])
        for dim in SCORE_DIMENSIONS
        if scores.get(dim) is not None
    }


def is_strong_positive(sample: Dict[str, Any]) -> bool:
    overall = sample_overall(sample)
    confidence = sample.get("judge", {}).get("confidence")
    would_use = sample.get("judge", {}).get("would_use")
    return (
        overall is not None
        and overall >= 4.5
        and bool(would_use)
        and confidence is not None
        and float(confidence) >= 0.8
    )


def is_fallback_positive(sample: Dict[str, Any]) -> bool:
    overall = sample_overall(sample)
    would_use = sample.get("judge", {}).get("would_use")
    return overall is not None and overall >= 4.0 and bool(would_use)


def is_dimension_negative(sample: Dict[str, Any]) -> bool:
    overall = sample_overall(sample)
    if overall is None or overall < 2.0 or overall > 3.5:
        return False
    if sample.get("judge", {}).get("would_use"):
        return False

    dims = dimension_values(sample)
    low_dims = [name for name, value in dims.items() if value <= 2.0]
    high_dims = [name for name, value in dims.items() if value >= 4.0]
    return bool(low_dims) and len(high_dims) >= 2


def is_hard_negative(sample: Dict[str, Any]) -> bool:
    overall = sample_overall(sample)
    if overall is None or overall < 2.5 or overall > 3.5:
        return False

    dims = dimension_values(sample)
    has_high_dimension = any(value >= 4.0 for value in dims.values())
    retrieval = sample.get("retrieval", {}) or {}
    return has_high_dimension or bool(retrieval.get("in_system_top_matches"))


def is_easy_negative(sample: Dict[str, Any]) -> bool:
    overall = sample_overall(sample)
    return overall is not None and overall <= 2.0


def classify_sample_bucket(sample: Dict[str, Any], has_strong_positive: bool) -> str:
    if is_strong_positive(sample):
        return "strong_positive"
    if not has_strong_positive and is_fallback_positive(sample):
        return "fallback_positive"
    if is_dimension_negative(sample):
        return "dimension_negative"
    if is_hard_negative(sample):
        return "hard_negative"
    if is_easy_negative(sample):
        return "easy_negative"
    return "unusable"


def tie_group_name(query_id: str, overall: float) -> str:
    return f"{query_id}_overall_{overall:.2f}"


def annotate_samples(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        grouped[sample["query_id"]].append(sample)

    for query_id, query_samples in grouped.items():
        has_strong_positive = any(is_strong_positive(sample) for sample in query_samples)

        for sample in query_samples:
            bucket = classify_sample_bucket(sample, has_strong_positive)
            labels = sample.setdefault("labels", {})
            labels["sample_bucket"] = bucket
            labels["is_positive_candidate"] = bucket in POSITIVE_BUCKETS
            labels["is_negative_candidate"] = bucket in NEGATIVE_BUCKETS

        tie_groups: Dict[float, List[Dict[str, Any]]] = defaultdict(list)
        for sample in query_samples:
            overall = sample_overall(sample)
            if overall is not None:
                tie_groups[overall].append(sample)

        for overall, tied_samples in tie_groups.items():
            group_name = tie_group_name(query_id, overall) if len(tied_samples) > 1 else None
            for sample in tied_samples:
                sample.setdefault("labels", {})["tie_group"] = group_name

    return samples


def stats_from_values(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"count": 0, "mean": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": round(sum(values) / len(values), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


def build_stats(
    samples: List[Dict[str, Any]],
    dropped: List[Dict[str, Any]],
    config: Dict[str, Any],
    dataset_version: str,
) -> Dict[str, Any]:
    query_counter = Counter(sample["query_id"] for sample in samples)
    bucket_counter = Counter((sample.get("labels", {}) or {}).get("sample_bucket", "unlabeled") for sample in samples)
    overall_values = [
        float(sample["judge"]["scores"]["overall"])
        for sample in samples
        if sample.get("judge", {}).get("scores", {}).get("overall") is not None
    ]
    confidence_values = [
        float(sample["judge"]["confidence"])
        for sample in samples
        if sample.get("judge", {}).get("confidence") is not None
    ]
    would_use_count = sum(1 for sample in samples if sample.get("judge", {}).get("would_use"))
    in_top_matches_count = sum(1 for sample in samples if sample.get("retrieval", {}).get("in_system_top_matches"))
    best_match_count = sum(1 for sample in samples if sample.get("retrieval", {}).get("is_system_best_match"))
    rank_values = [
        sample["retrieval"]["system_rank"]
        for sample in samples
        if sample.get("retrieval", {}).get("system_rank") is not None
    ]
    dropped_reasons = Counter(item.get("reason", "unknown") for item in dropped)
    candidates_per_query = list(query_counter.values())
    positive_count = sum(1 for sample in samples if (sample.get("labels", {}) or {}).get("is_positive_candidate"))
    negative_count = sum(1 for sample in samples if (sample.get("labels", {}) or {}).get("is_negative_candidate"))
    tie_groups = Counter(
        (sample.get("labels", {}) or {}).get("tie_group")
        for sample in samples
        if (sample.get("labels", {}) or {}).get("tie_group")
    )

    query_label_stats: Dict[str, Dict[str, Any]] = {}
    queries_without_positive: List[str] = []
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        grouped[sample["query_id"]].append(sample)

    for query_id, query_samples in grouped.items():
        query_bucket_counts = Counter(
            (sample.get("labels", {}) or {}).get("sample_bucket", "unlabeled")
            for sample in query_samples
        )
        query_positive_count = sum(
            1 for sample in query_samples if (sample.get("labels", {}) or {}).get("is_positive_candidate")
        )
        query_negative_count = sum(
            1 for sample in query_samples if (sample.get("labels", {}) or {}).get("is_negative_candidate")
        )
        query_tie_groups = {
            (sample.get("labels", {}) or {}).get("tie_group")
            for sample in query_samples
            if (sample.get("labels", {}) or {}).get("tie_group")
        }
        query_tied_samples = sum(
            1 for sample in query_samples if (sample.get("labels", {}) or {}).get("tie_group")
        )
        if query_positive_count == 0:
            queries_without_positive.append(query_id)

        query_label_stats[query_id] = {
            "total_candidates": len(query_samples),
            "positive_count": query_positive_count,
            "negative_count": query_negative_count,
            "unusable_count": query_bucket_counts.get("unusable", 0),
            "has_strong_positive": query_bucket_counts.get("strong_positive", 0) > 0,
            "has_fallback_positive": query_bucket_counts.get("fallback_positive", 0) > 0,
            "bucket_counts": dict(sorted(query_bucket_counts.items())),
            "tie_group_count": len(query_tie_groups),
            "tied_sample_count": query_tied_samples,
            "best_overall": max(
                (sample_overall(sample) for sample in query_samples if sample_overall(sample) is not None),
                default=None,
            ),
        }

    return {
        "dj": config.get("dj"),
        "run_id": config.get("run_id"),
        "structured_json": config.get("structured_json"),
        "dataset_version": dataset_version,
        "builder_version": BUILDER_VERSION,
        "input_results": len(samples) + len(dropped),
        "kept_samples": len(samples),
        "dropped_samples": len(dropped),
        "dropped_by_reason": dict(sorted(dropped_reasons.items())),
        "unique_queries": len(query_counter),
        "candidates_per_query": stats_from_values([float(v) for v in candidates_per_query]),
        "judge_overall": stats_from_values(overall_values),
        "judge_confidence": stats_from_values(confidence_values),
        "would_use": {
            "count": would_use_count,
            "rate": round(would_use_count / len(samples), 4) if samples else 0.0,
        },
        "retrieval_coverage": {
            "in_system_top_matches_count": in_top_matches_count,
            "in_system_top_matches_rate": round(in_top_matches_count / len(samples), 4) if samples else 0.0,
            "is_system_best_match_count": best_match_count,
            "is_system_best_match_rate": round(best_match_count / len(samples), 4) if samples else 0.0,
            "system_rank": stats_from_values([float(v) for v in rank_values]),
        },
        "labels": {
            "bucket_counts": dict(sorted(bucket_counter.items())),
            "positive_count": positive_count,
            "positive_rate": round(positive_count / len(samples), 4) if samples else 0.0,
            "negative_count": negative_count,
            "negative_rate": round(negative_count / len(samples), 4) if samples else 0.0,
            "tie_group_count": len(tie_groups),
            "tied_sample_count": sum(tie_groups.values()),
            "queries_without_positive": sorted(queries_without_positive),
            "per_query": query_label_stats,
        },
    }


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    lines = [json.dumps(row, ensure_ascii=False) for row in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def ensure_run_inputs(dj: str, run_id: str, cache_dir: Optional[str], judge_output_dir: Optional[str]) -> Tuple[Path, Path, Path, Path]:
    run_dir = judge_run_dir(dj, run_id, judge_output_dir)
    config_path = run_dir / "config.json"
    results_path = run_dir / "results.jsonl"

    if not config_path.exists():
        raise FileNotFoundError(f"config.json 不存在: {config_path}")
    if not results_path.exists():
        raise FileNotFoundError(f"results.jsonl 不存在: {results_path}")

    config = load_json(config_path)
    base = cache_base(dj, cache_dir)
    structured_json_name = config.get("structured_json")
    structured_json_path = base / structured_json_name if structured_json_name else latest_structured_json(base)
    if not structured_json_path or not structured_json_path.exists():
        raise FileNotFoundError(f"structured_json 不存在: {structured_json_path}")

    structured_description_path = base / "structured_description.json"
    if not structured_description_path.exists():
        raise FileNotFoundError(f"structured_description.json 不存在: {structured_description_path}")

    return config_path, results_path, structured_json_path, structured_description_path


def build_samples(
    dj: str,
    run_id: str,
    dataset_version: str,
    cache_dir: Optional[str],
    judge_output_dir: Optional[str],
    output_dir: Optional[str],
) -> Path:
    config_path, results_path, structured_json_path, structured_description_path = ensure_run_inputs(
        dj=dj,
        run_id=run_id,
        cache_dir=cache_dir,
        judge_output_dir=judge_output_dir,
    )
    config = load_json(config_path)
    query_index = build_query_index(structured_json_path)
    key_index, id_index = build_segment_indexes(structured_description_path)
    results = load_jsonl(results_path)

    samples: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []

    for lineno, result in results:
        if result.get("_parse_error"):
            dropped.append({
                "line_number": lineno,
                "evaluation_id": None,
                "reason": "invalid_jsonl",
                "parse_error": result.get("_parse_error"),
            })
            continue

        eval_id = result.get("evaluation_id")
        parsed = parse_eval_id(eval_id or "")

        query = result.get("query", {}) or {}
        candidate = result.get("candidate", {}) or {}

        segment_sequence = query.get("segment_sequence") or parsed.get("segment_sequence")
        if segment_sequence is None:
            dropped.append({
                "line_number": lineno,
                "evaluation_id": eval_id,
                "reason": "missing_segment_sequence",
            })
            continue

        query_seg = query_index.get(int(segment_sequence))
        if not query_seg:
            dropped.append({
                "line_number": lineno,
                "evaluation_id": eval_id,
                "reason": "missing_query_segment",
                "segment_sequence": int(segment_sequence),
            })
            continue

        if candidate.get("episode") is None and parsed.get("episode") is not None:
            candidate["episode"] = parsed["episode"]
        if candidate.get("segment_id") is None and parsed.get("candidate") is not None:
            candidate["segment_id"] = parsed["candidate"]

        candidate_seg = find_candidate_segment(candidate, key_index, id_index)
        if not candidate_seg:
            dropped.append({
                "line_number": lineno,
                "evaluation_id": eval_id,
                "reason": "missing_candidate_segment",
                "candidate": candidate,
            })
            continue

        sample = build_sample(
            result=result,
            query_seg=query_seg,
            candidate_seg=candidate_seg,
            config=config,
            dataset_version=dataset_version,
        )
        samples.append(sample)

    samples = annotate_samples(samples)

    target_dir = Path(output_dir) if output_dir else default_output_dir(dataset_version, dj, run_id)
    target_dir.mkdir(parents=True, exist_ok=True)

    write_jsonl(target_dir / "samples.jsonl", samples)
    write_jsonl(target_dir / "dropped.jsonl", dropped)
    stats = build_stats(samples, dropped, config, dataset_version)
    (target_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    LOGGER.info("samples=%s dropped=%s output=%s", len(samples), len(dropped), target_dir)
    return target_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 reranker 训练样本层（samples.jsonl）")
    parser.add_argument("--dj", required=True, help="剧名")
    parser.add_argument("--run-id", required=True, help="Judge run_id")
    parser.add_argument("--dataset-version", default="v1", help="数据集版本")
    parser.add_argument("--cache-dir", default=None, help="text_video_match/cache/{dj} 路径")
    parser.add_argument("--judge-output-dir", default=None, help="judge/outputs 根目录")
    parser.add_argument("--output-dir", default=None, help="输出目录，默认 judge/training_data/{dataset_version}/{dj}/{run_id}")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    output_dir = build_samples(
        dj=args.dj,
        run_id=args.run_id,
        dataset_version=args.dataset_version,
        cache_dir=args.cache_dir,
        judge_output_dir=args.judge_output_dir,
        output_dir=args.output_dir,
    )
    print(f"samples 已生成: {output_dir}")


if __name__ == "__main__":
    main()
