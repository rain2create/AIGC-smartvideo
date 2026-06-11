#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RL / Reranker 训练数据构造器（v3）

输出：
- samples.jsonl / dropped.jsonl / stats.json  (v3 schema)
- pairs.jsonl   (DPO 偏好对，从 samples 派生)
- groups.jsonl  (Listwise AWR 候选组，从 samples 派生)

用法示例：
    # 单剧
    python judge/rl_data_builder.py --dj 剧名 --run-id run_001 --export-all
    # 批量
    python judge/rl_data_builder.py --all --export-all --dataset-version v3
"""

import argparse
import json
import logging
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


LOGGER = logging.getLogger(__name__)
BUILDER_VERSION = "samples.v3"
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

# Bucket sets（Phase C 定稿）
POSITIVE_BUCKETS = {"strong_positive", "fallback_positive", "ambiguous_positive"}
NEGATIVE_BUCKETS = {"hard_negative", "easy_negative"}

# DPO 采样参数
DPO_HARD_PER_STRONG = 4
DPO_EASY_PER_STRONG = 3
DPO_HARD_PER_FALLBACK = 4

# Listwise AWR 采样参数
AWR_STRONG_MAX = 3
AWR_AMBIGUOUS_MAX = 1
AWR_FALLBACK_MAX = 1
AWR_HARD_MIN = 3
AWR_HARD_MAX = 4
AWR_EASY_MIN = 2
AWR_EASY_MAX = 3
AWR_GROUP_MIN_SIZE = 5
AWR_GROUP_MAX_SIZE = 25


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def cache_base(dj: str, cache_dir: Optional[str] = None) -> Path:
    if cache_dir:
        return Path(cache_dir)
    return repo_root() / "text_video_match" / "cache" / dj


def judge_run_dir(dj: str, run_id: str, judge_output_dir: Optional[str] = None) -> Path:
    if judge_output_dir:
        return Path(judge_output_dir) / dj / run_id
    return repo_root() / "judge" / "outputs" / "canonical" / dj / run_id


def default_output_dir(dataset_version: str, dj: str, run_id: str) -> Path:
    return repo_root() / "judge" / "training_data" / dataset_version / dj / run_id


# ---------------------------------------------------------------------------
# I/O utilities
# ---------------------------------------------------------------------------

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


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    lines = [json.dumps(row, ensure_ascii=False) for row in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


# ---------------------------------------------------------------------------
# Structured data index helpers
# ---------------------------------------------------------------------------

def latest_structured_json(base: Path) -> Optional[Path]:
    candidates = sorted(base.glob("*_structured.json"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def has_query_text(path: Path) -> bool:
    """Return True if the structured JSON already carries reranker query_text."""
    try:
        data = load_json(path)
    except Exception:
        return False
    segments = data if isinstance(data, list) else data.get("segments", [])
    if not isinstance(segments, list):
        return False
    for seg in segments[:5]:
        match_result = seg.get("match_result", {}) or {}
        match_breakdown = match_result.get("match_breakdown") or {}
        query_text = match_breakdown.get("query_text")
        if isinstance(query_text, str) and query_text.strip():
            return True
    return False


def resolve_training_structured_json(path: Path) -> Path:
    """Prefer the Q3+reranker structured file used by reranker training."""
    if has_query_text(path):
        return path

    name = path.name
    if name.endswith("_structured.json"):
        prefix = name[: -len("_structured.json")]
        exact = path.with_name(f"{prefix}_structured_qwen_q3_rerank_q1.json")
        if exact.exists():
            return exact
        candidates = sorted(
            path.parent.glob(f"{prefix}_structured*qwen*q3*rerank*q1*.json"),
            key=lambda p: p.stat().st_mtime,
        )
        if candidates:
            return candidates[-1]

    return path


def make_segment_key(segment_id: Any, episode: Any) -> Optional[str]:
    if segment_id is None:
        return None
    if episode is None:
        return f"seg{int(segment_id)}"
    return f"ep{int(episode)}_seg{int(segment_id)}"


def build_segment_indexes(
    sd_path: Path,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[int, List[Dict[str, Any]]]]:
    data = load_json(sd_path)
    segments = data if isinstance(data, list) else data.get("segments", [])
    key_index: Dict[str, Dict[str, Any]] = {}
    id_index: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for seg in segments:
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


def build_q3_query_text(script_text: str, visual_query: Optional[Dict[str, Any]]) -> str:
    """Build the Q3 query string used by qwen.embedder.build_query_text."""
    visual_query = visual_query or {}
    script_text = (script_text or "").strip()
    char_types = visual_query.get("character_types") or []
    action_keywords = visual_query.get("action_keywords") or []
    environment = visual_query.get("environment")
    expanded_tags = visual_query.get("expanded_tags")
    interaction_type = visual_query.get("interaction_type") or []

    parts: List[str] = [script_text] if script_text else []
    if char_types:
        parts.append(f"角色类型: {'、'.join(map(str, char_types))}")
    if action_keywords:
        parts.append(f"动作: {'、'.join(map(str, action_keywords))}")
    if environment:
        parts.append(f"环境: {environment}")
    if expanded_tags:
        parts.append(f"扩展标签: {expanded_tags}")
    if interaction_type:
        parts.append(f"互动关系: {'、'.join(map(str, interaction_type))}")
    return "\n".join(part for part in parts if part)


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


def is_same_candidate(
    left: Dict[str, Any], right_segment_id: Optional[int], right_episode: Optional[int]
) -> bool:
    left_segment_id = left.get("segment_id")
    if left_segment_id is None or right_segment_id is None:
        return False
    if int(left_segment_id) != int(right_segment_id):
        return False
    left_episode = left.get("episode")
    if left_episode is None or right_episode is None:
        return True
    return int(left_episode) == int(right_episode)


# ---------------------------------------------------------------------------
# Sample building (v3 schema)
# ---------------------------------------------------------------------------

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
    match_breakdown = match_result.get("match_breakdown") or {}

    segment_sequence = int(query.get("segment_sequence") or query_seg.get("segment_sequence") or 0)
    candidate_segment_id = candidate.get("segment_id")
    candidate_episode = candidate.get("episode")
    system_rank, system_similarity = find_system_rank(top_matches, candidate_segment_id, candidate_episode)

    query_id = f"{config['dj']}_seg{segment_sequence}"
    if candidate_segment_id is None:
        candidate_id = "unknown"
    else:
        candidate_id = make_segment_key(candidate_segment_id, candidate_episode) or f"seg{candidate_segment_id}"

    script_text = query.get("script_text") or query_seg.get("script_text")

    # query_text: Q3 格式化查询字符串，优先来自 match_breakdown.query_text（reranker 的实际输入）；
    # 老数据缺失时按同一 Q3 规则从 visual_query 重建，避免训练静默退回裸 script_text。
    query_text = match_breakdown.get("query_text") if isinstance(match_breakdown, dict) else None
    if not query_text:
        query_text = build_q3_query_text(script_text or "", query_seg.get("visual_query"))

    # system_best_match 精简为两个标量字段
    sys_best_seg_id_raw = system_best.get("segment_id")
    sys_best_seg_id = int(sys_best_seg_id_raw) if sys_best_seg_id_raw is not None else None

    # system_top_matches 精简：每条只保留 {segment_id, episode, similarity}
    slim_top_matches = [
        {
            "segment_id": m.get("segment_id"),
            "episode": m.get("episode"),
            "similarity": m.get("similarity"),
        }
        for m in top_matches
    ]

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
            "script_text": script_text,
            "query_text": query_text,
            "episode_target": query.get("episode_target") or query_seg.get("episode_target"),
            "visual_query": query_seg.get("visual_query"),
        },
        "candidate": {
            "segment_id": candidate_segment_id,
            "episode": candidate_episode if candidate_episode is not None else candidate_seg.get("episode"),
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
            "in_system_top_matches": system_rank is not None,
            "system_rank": system_rank,
            "system_similarity": system_similarity,
            "is_system_best_match": is_same_candidate(system_best, candidate_segment_id, candidate_episode),
            "system_best_match_segment_id": sys_best_seg_id,
            "system_best_match_overall": None,  # 在 annotate_samples 中填充
            "system_top_matches": slim_top_matches,
        },
        "labels": {
            "sample_bucket": None,
            "is_positive": None,
            "is_negative": None,
            "tie_group": None,
        },
    }


# ---------------------------------------------------------------------------
# Bucket classification（Phase C 定稿，context-independent）
# ---------------------------------------------------------------------------

def sample_scores(sample: Dict[str, Any]) -> Dict[str, float]:
    return sample.get("judge", {}).get("scores", {}) or {}


def sample_overall(sample: Dict[str, Any]) -> Optional[float]:
    overall = sample_scores(sample).get("overall")
    return float(overall) if overall is not None else None


def is_strong_positive(sample: Dict[str, Any]) -> bool:
    """overall >= 4.5 且 would_use=True 且 confidence >= 0.8"""
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


def is_ambiguous_positive(sample: Dict[str, Any]) -> bool:
    """overall >= 4.0 且 would_use=False（Judge 高分但不可实操）"""
    overall = sample_overall(sample)
    would_use = sample.get("judge", {}).get("would_use")
    return overall is not None and overall >= 4.0 and not bool(would_use)


def is_fallback_positive(sample: Dict[str, Any]) -> bool:
    """overall >= 3.5 且 would_use=True 且 非 strong（context-independent 回退正样本）"""
    overall = sample_overall(sample)
    would_use = sample.get("judge", {}).get("would_use")
    return overall is not None and overall >= 3.5 and bool(would_use)


def is_hard_negative(sample: Dict[str, Any]) -> bool:
    """2.0 <= overall < 4.0（吸收原 dimension_negative + unusable 2.0-4.0 区间；
    3.5-4.0 且 would_use=False 不被 ambiguous/fallback 覆盖，归入此桶）"""
    overall = sample_overall(sample)
    return overall is not None and 2.0 <= overall < 4.0


def is_easy_negative(sample: Dict[str, Any]) -> bool:
    """overall < 2.0"""
    overall = sample_overall(sample)
    return overall is not None and overall < 2.0


def classify_sample_bucket(sample: Dict[str, Any]) -> str:
    """Phase C 定稿：5 bucket，context-independent，优先级从高到低。"""
    if is_strong_positive(sample):
        return "strong_positive"
    if is_ambiguous_positive(sample):
        return "ambiguous_positive"
    if is_fallback_positive(sample):
        return "fallback_positive"
    if is_hard_negative(sample):
        return "hard_negative"
    if is_easy_negative(sample):
        return "easy_negative"
    return "unusable"  # overall=None 或其他边缘情况


def tie_group_name(query_id: str, overall: float) -> str:
    return f"{query_id}_overall_{overall:.2f}"


def annotate_samples(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        grouped[sample["query_id"]].append(sample)

    for query_id, query_samples in grouped.items():
        # 1. 分桶标注（context-independent）
        for sample in query_samples:
            bucket = classify_sample_bucket(sample)
            labels = sample.setdefault("labels", {})
            labels["sample_bucket"] = bucket
            labels["is_positive"] = bucket in POSITIVE_BUCKETS
            labels["is_negative"] = bucket in NEGATIVE_BUCKETS

        # 2. Tie group 标注
        tie_groups_by_overall: Dict[float, List[Dict[str, Any]]] = defaultdict(list)
        for sample in query_samples:
            overall = sample_overall(sample)
            if overall is not None:
                tie_groups_by_overall[overall].append(sample)
        for overall, tied in tie_groups_by_overall.items():
            group_name = tie_group_name(query_id, overall) if len(tied) > 1 else None
            for sample in tied:
                sample.setdefault("labels", {})["tie_group"] = group_name

        # 3. 填充 system_best_match_overall（找 is_system_best_match=True 的样本的 judge 分）
        best_match_overall: Optional[float] = None
        for sample in query_samples:
            if sample.get("retrieval", {}).get("is_system_best_match"):
                best_match_overall = sample_overall(sample)
                break
        for sample in query_samples:
            sample.setdefault("retrieval", {})["system_best_match_overall"] = best_match_overall

    return samples


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def stats_from_values(values: List[float]) -> Dict[str, Any]:
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
    bucket_counter = Counter(
        (sample.get("labels", {}) or {}).get("sample_bucket", "unlabeled") for sample in samples
    )
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
    in_top_matches_count = sum(
        1 for sample in samples if sample.get("retrieval", {}).get("in_system_top_matches")
    )
    query_text_count = sum(
        1
        for sample in samples
        if isinstance(sample.get("query", {}).get("query_text"), str)
        and sample["query"]["query_text"].strip()
    )
    best_match_count = sum(
        1 for sample in samples if sample.get("retrieval", {}).get("is_system_best_match")
    )
    rank_values = [
        sample["retrieval"]["system_rank"]
        for sample in samples
        if sample.get("retrieval", {}).get("system_rank") is not None
    ]
    dropped_reasons = Counter(item.get("reason", "unknown") for item in dropped)
    candidates_per_query = list(query_counter.values())
    positive_count = sum(1 for sample in samples if (sample.get("labels", {}) or {}).get("is_positive"))
    negative_count = sum(1 for sample in samples if (sample.get("labels", {}) or {}).get("is_negative"))
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
            1 for sample in query_samples if (sample.get("labels", {}) or {}).get("is_positive")
        )
        query_negative_count = sum(
            1 for sample in query_samples if (sample.get("labels", {}) or {}).get("is_negative")
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
            "has_strong_positive": query_bucket_counts.get("strong_positive", 0) > 0,
            "has_fallback_positive": query_bucket_counts.get("fallback_positive", 0) > 0,
            "has_ambiguous_positive": query_bucket_counts.get("ambiguous_positive", 0) > 0,
            "bucket_counts": dict(sorted(query_bucket_counts.items())),
            "tie_group_count": len(query_tie_groups),
            "tied_sample_count": query_tied_samples,
            "best_overall": max(
                (sample_overall(s) for s in query_samples if sample_overall(s) is not None),
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
        "query_text_coverage": {
            "count": query_text_count,
            "rate": round(query_text_count / len(samples), 4) if samples else 0.0,
        },
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


# ---------------------------------------------------------------------------
# DPO pairs building
# ---------------------------------------------------------------------------

def _get_bucket(sample: Dict[str, Any]) -> str:
    return (sample.get("labels", {}) or {}).get("sample_bucket", "")


def _make_pair(chosen: Dict[str, Any], rejected: Dict[str, Any], pair_type: str) -> Dict[str, Any]:
    query_id = chosen["query_id"]
    chosen_cand_id = chosen["candidate_id"]
    rejected_cand_id = rejected["candidate_id"]
    return {
        "pair_id": f"{query_id}_c{chosen_cand_id}_r{rejected_cand_id}",
        "query_id": query_id,
        "pair_type": pair_type,
        "source": {
            "dj": chosen["source"]["dj"],
            "run_id": chosen["source"]["run_id"],
        },
        "query": {
            "script_text": chosen["query"]["script_text"],
            "query_text": chosen["query"].get("query_text"),
        },
        "chosen": {
            "sample_id": chosen["sample_id"],
            "candidate_id": chosen_cand_id,
            "bucket": _get_bucket(chosen),
            "overall": sample_overall(chosen),
            "video_segment_path": chosen["candidate"]["video_segment_path"],
        },
        "rejected": {
            "sample_id": rejected["sample_id"],
            "candidate_id": rejected_cand_id,
            "bucket": _get_bucket(rejected),
            "overall": sample_overall(rejected),
            "video_segment_path": rejected["candidate"]["video_segment_path"],
        },
    }


def build_pairs(
    samples: List[Dict[str, Any]],
    dpo_hard_per_strong: int = DPO_HARD_PER_STRONG,
    dpo_easy_per_strong: int = DPO_EASY_PER_STRONG,
    dpo_hard_per_fallback: int = DPO_HARD_PER_FALLBACK,
) -> List[Dict[str, Any]]:
    """从 annotated samples 构建 DPO 偏好对。

    三类 pair：
    - strong_vs_hard:    所有有 strong 的 query，每 strong 配 ≤N 个 hard（overall 降序）
    - strong_vs_easy:    所有有 strong 的 query，每 strong 配 ≤N 个 easy（overall 降序）
    - fallback_vs_hard:  仅无 strong 的 query，每 fallback 配 ≤N 个 hard（overall 降序）

    ambiguous_positive 不参与 DPO（would_use=False 作为 chosen 信号矛盾）。
    """
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for s in samples:
        grouped[s["query_id"]].append(s)

    pairs: List[Dict[str, Any]] = []
    for query_id, query_samples in sorted(grouped.items()):
        strongs = [s for s in query_samples if _get_bucket(s) == "strong_positive"]
        fallbacks = [s for s in query_samples if _get_bucket(s) == "fallback_positive"]
        # 负样本按 overall 降序（最难区分的在前，信号最强）
        hards = sorted(
            [s for s in query_samples if _get_bucket(s) == "hard_negative"],
            key=lambda s: sample_overall(s) or 0,
            reverse=True,
        )
        easys = sorted(
            [s for s in query_samples if _get_bucket(s) == "easy_negative"],
            key=lambda s: sample_overall(s) or 0,
            reverse=True,
        )

        if strongs:
            hard_pool = hards[:dpo_hard_per_strong]
            easy_pool = easys[:dpo_easy_per_strong]
            for chosen in strongs:
                for rejected in hard_pool:
                    pairs.append(_make_pair(chosen, rejected, "strong_vs_hard"))
                for rejected in easy_pool:
                    pairs.append(_make_pair(chosen, rejected, "strong_vs_easy"))
        else:
            # fallback_vs_hard 仅用于无 strong 的 query
            hard_pool = hards[:dpo_hard_per_fallback]
            for chosen in fallbacks:
                for rejected in hard_pool:
                    pairs.append(_make_pair(chosen, rejected, "fallback_vs_hard"))

    return pairs


# ---------------------------------------------------------------------------
# Listwise AWR groups building
# ---------------------------------------------------------------------------

def compute_awr_reward(sample: Dict[str, Any]) -> float:
    """Bucket-aware reward：正样本=overall，负样本=overall-4.0。

    设计动机：正负两区间之间制造 ~3.5 分断层，确保负样本 reward 为负。
    归一化由训练框架在 forward 阶段完成，groups.jsonl 只存原始 reward。
    """
    bucket = _get_bucket(sample)
    overall = sample_overall(sample) or 0.0
    if bucket in POSITIVE_BUCKETS:
        return overall
    return overall - 4.0


def build_groups(
    samples: List[Dict[str, Any]],
    strong_max: int = AWR_STRONG_MAX,
    ambiguous_max: int = AWR_AMBIGUOUS_MAX,
    fallback_max: int = AWR_FALLBACK_MAX,
    hard_max: int = AWR_HARD_MAX,
    easy_max: int = AWR_EASY_MAX,
    group_min_size: int = AWR_GROUP_MIN_SIZE,
    group_max_size: int = AWR_GROUP_MAX_SIZE,
    rng_seed: int = 42,
) -> List[Dict[str, Any]]:
    """从 annotated samples 构建 Listwise AWR 候选组。

    每 query 一组，候选构成：
    - strong_positive:    1-3 个
    - ambiguous_positive: 0-1 个
    - fallback_positive:  0-1 个
    - hard_negative:      最多 8 个（数量多，弥补小梯度）
    - easy_negative:      最多 6 个（压缩，避免拉低组内均值）
    总计：~15-19，上限 25

    边界处理：
    - 无任何正样本 → 跳过
    - 总候选 < group_min_size → 跳过（对比度不足）
    - 总候选 > group_max_size → 截断（easy 优先截，其次 hard）
    """
    rng = random.Random(rng_seed)
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for s in samples:
        grouped[s["query_id"]].append(s)

    groups: List[Dict[str, Any]] = []
    for query_id, query_samples in sorted(grouped.items()):
        strongs = [s for s in query_samples if _get_bucket(s) == "strong_positive"]
        ambiguous = [s for s in query_samples if _get_bucket(s) == "ambiguous_positive"]
        fallbacks = [s for s in query_samples if _get_bucket(s) == "fallback_positive"]
        hards = [s for s in query_samples if _get_bucket(s) == "hard_negative"]
        easys = [s for s in query_samples if _get_bucket(s) == "easy_negative"]

        # 无正样本 → 跳过
        if not strongs and not fallbacks and not ambiguous:
            continue

        # 构成候选列表
        selected: List[Dict[str, Any]] = []
        selected += strongs[:strong_max]
        selected += ambiguous[:ambiguous_max]
        selected += fallbacks[:fallback_max]
        selected += hards[:hard_max]
        selected += easys[:easy_max]

        # 总候选 < group_min_size → 跳过（对比度不足）
        if len(selected) < group_min_size:
            continue

        # 超过上限则截断：easy 优先截，其次 hard
        while len(selected) > group_max_size:
            easy_idxs = [i for i, s in enumerate(selected) if _get_bucket(s) == "easy_negative"]
            if easy_idxs:
                selected.pop(easy_idxs[-1])
            else:
                hard_idxs = [i for i, s in enumerate(selected) if _get_bucket(s) == "hard_negative"]
                if hard_idxs:
                    selected.pop(hard_idxs[-1])
                else:
                    selected.pop()

        # 打乱，避免候选顺序带来位置偏差
        rng.shuffle(selected)

        source_sample = query_samples[0]
        groups.append({
            "group_id": query_id,
            "query_id": query_id,
            "source": {
                "dj": source_sample["source"]["dj"],
                "run_id": source_sample["source"]["run_id"],
            },
            "query": {
                "script_text": source_sample["query"]["script_text"],
                "query_text": source_sample["query"].get("query_text"),
            },
            "candidates": [
                {
                    "sample_id": s["sample_id"],
                    "candidate_id": s["candidate_id"],
                    "bucket": _get_bucket(s),
                    "overall": sample_overall(s),
                    "reward": compute_awr_reward(s),
                    "video_segment_path": s["candidate"]["video_segment_path"],
                }
                for s in selected
            ],
        })

    return groups


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def ensure_run_inputs(
    dj: str,
    run_id: str,
    cache_dir: Optional[str],
    judge_output_dir: Optional[str],
) -> Tuple[Path, Path, Path, Path]:
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
    structured_json_path = (
        base / structured_json_name if structured_json_name else latest_structured_json(base)
    )
    if not structured_json_path or not structured_json_path.exists():
        raise FileNotFoundError(f"structured_json 不存在: {structured_json_path}")
    training_structured_json_path = resolve_training_structured_json(structured_json_path)
    if training_structured_json_path != structured_json_path:
        LOGGER.info(
            "使用训练结构化文件: %s -> %s",
            structured_json_path.name,
            training_structured_json_path.name,
        )
    structured_json_path = training_structured_json_path

    structured_description_path = base / "structured_description.json"
    if not structured_description_path.exists():
        raise FileNotFoundError(f"structured_description.json 不存在: {structured_description_path}")

    return config_path, results_path, structured_json_path, structured_description_path


# ---------------------------------------------------------------------------
# Single-drama builder
# ---------------------------------------------------------------------------

def build_for_drama(
    dj: str,
    run_id: str,
    dataset_version: str,
    cache_dir: Optional[str],
    judge_output_dir: Optional[str],
    output_dir: Optional[str],
    export_pairs: bool = False,
    export_groups: bool = False,
) -> Path:
    config_path, results_path, structured_json_path, structured_description_path = ensure_run_inputs(
        dj=dj,
        run_id=run_id,
        cache_dir=cache_dir,
        judge_output_dir=judge_output_dir,
    )
    config = load_json(config_path)
    config = dict(config)
    config["structured_json"] = structured_json_path.name
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

    if export_pairs:
        pairs = build_pairs(samples)
        write_jsonl(target_dir / "pairs.jsonl", pairs)
        LOGGER.info("pairs=%s output=%s/pairs.jsonl", len(pairs), target_dir)

    if export_groups:
        groups = build_groups(samples)
        write_jsonl(target_dir / "groups.jsonl", groups)
        LOGGER.info("groups=%s output=%s/groups.jsonl", len(groups), target_dir)

    LOGGER.info(
        "dj=%s run_id=%s samples=%s dropped=%s output=%s",
        dj, run_id, len(samples), len(dropped), target_dir,
    )
    return target_dir


# ---------------------------------------------------------------------------
# Batch builder
# ---------------------------------------------------------------------------

def find_all_judge_runs(judge_output_dir: Optional[str] = None) -> List[Tuple[str, str]]:
    """枚举 judge/outputs/canonical 下所有有效 (dj, run_id) 对。"""
    base = Path(judge_output_dir) if judge_output_dir else repo_root() / "judge" / "outputs" / "canonical"
    runs: List[Tuple[str, str]] = []
    if not base.exists():
        return runs
    for dj_dir in sorted(base.iterdir()):
        if not dj_dir.is_dir():
            continue
        for run_dir in sorted(dj_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            if (run_dir / "results.jsonl").exists() and (run_dir / "config.json").exists():
                runs.append((dj_dir.name, run_dir.name))
    return runs


def build_all_dramas(
    dataset_version: str,
    cache_dir: Optional[str],
    judge_output_dir: Optional[str],
    output_base: Optional[str],
    export_pairs: bool = False,
    export_groups: bool = False,
) -> Dict[str, Any]:
    """批量构建所有剧的训练数据。"""
    runs = find_all_judge_runs(judge_output_dir)
    LOGGER.info("发现 %d 个 judge runs", len(runs))

    succeeded: Dict[str, str] = {}
    failed: Dict[str, str] = {}
    for dj, run_id in runs:
        key = f"{dj}/{run_id}"
        out_dir_str = str(Path(output_base) / dj / run_id) if output_base else None
        try:
            out_dir = build_for_drama(
                dj=dj,
                run_id=run_id,
                dataset_version=dataset_version,
                cache_dir=cache_dir,
                judge_output_dir=judge_output_dir,
                output_dir=out_dir_str,
                export_pairs=export_pairs,
                export_groups=export_groups,
            )
            succeeded[key] = str(out_dir)
        except Exception as exc:
            failed[key] = str(exc)
            LOGGER.error("构建失败 %s: %s", key, exc)

    return {"total": len(runs), "succeeded": succeeded, "failed": failed}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="构建 reranker RL 训练数据（v3 schema）")

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--dj", help="单剧模式：剧名")
    mode_group.add_argument("--all", action="store_true", help="批量模式：处理所有剧")

    parser.add_argument("--run-id", help="Judge run_id（单剧模式必填）")
    parser.add_argument("--dataset-version", default="v3", help="数据集版本（默认 v3）")
    parser.add_argument("--cache-dir", default=None, help="text_video_match/cache/{dj} 路径")
    parser.add_argument("--judge-output-dir", default=None, help="judge/outputs 根目录")
    parser.add_argument("--output-dir", default=None, help="输出目录（单剧模式）")
    parser.add_argument("--output-base", default=None, help="输出根目录（批量模式）")
    parser.add_argument("--export-pairs", action="store_true", help="同时导出 pairs.jsonl（DPO）")
    parser.add_argument("--export-groups", action="store_true", help="同时导出 groups.jsonl（Listwise AWR）")
    parser.add_argument("--export-all", action="store_true", help="同时导出 pairs.jsonl 和 groups.jsonl")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    export_pairs = args.export_pairs or args.export_all
    export_groups = args.export_groups or args.export_all

    if args.all:
        result = build_all_dramas(
            dataset_version=args.dataset_version,
            cache_dir=args.cache_dir,
            judge_output_dir=args.judge_output_dir,
            output_base=args.output_base,
            export_pairs=export_pairs,
            export_groups=export_groups,
        )
        print(f"批量构建完成: 成功 {len(result['succeeded'])}/{result['total']} 个剧")
        if result["failed"]:
            print(f"失败 {len(result['failed'])} 个:")
            for key, err in sorted(result["failed"].items()):
                print(f"  {key}: {err}")
    else:
        if not args.run_id:
            parser.error("单剧模式需要 --run-id")
        output_dir = build_for_drama(
            dj=args.dj,
            run_id=args.run_id,
            dataset_version=args.dataset_version,
            cache_dir=args.cache_dir,
            judge_output_dir=args.judge_output_dir,
            output_dir=args.output_dir,
            export_pairs=export_pairs,
            export_groups=export_groups,
        )
        print(f"构建完成: {output_dir}")


if __name__ == "__main__":
    main()
