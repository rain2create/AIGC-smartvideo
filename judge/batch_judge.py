#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM-as-Judge 批量评估脚本

核心思路：
  对每条文案，找到其目标集数（episode_target），然后评估该文案与
  目标集数内【所有】视频片段的匹配质量。

  评估完成后自动生成系统质量报告（best_match 排名、分数分布等）。

  不再区分 eval / annotation 模式：
  - 全量标注是唯一的评估动作
  - eval 报告从标注结果中直接派生

用法:
    # 基本用法（全量标注 + 自动生成报告）
    python3 judge/batch_judge.py \
        --dj weinishiguit4

    # 指定模型（默认 flash，标注量大用 flash 即可）
    python3 judge/batch_judge.py \
        --dj weinishiguit4 \
        --model google/gemini-3-flash-preview

    # 断点续评（已有结果自动跳过）
    python3 judge/batch_judge.py \
        --dj weinishiguit4 \
        --run-id annotation_20260417_001

    # 只生成报告（不跑评估，从已有 results.jsonl 生成）
    python3 judge/batch_judge.py \
        --dj weinishiguit4 \
        --run-id annotation_20260417_001 \
        --report-only

    # 只处理前 N 条文案（调试用）
    python3 judge/batch_judge.py \
        --dj weinishiguit4 \
        --max-segments 3
"""

import os
import json
import logging
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
from dataclasses import asdict

from match_judge import (
    MatchJudge,
    JudgeInput,
    resolve_video_path,
    VALID_DIMENSIONS,
    DIMENSION_WEIGHTS,
)

logger = logging.getLogger(__name__)


def load_task_djs(task_file: str) -> List[str]:
    """从 task txt/csv 中提取剧名列表，默认取每行第一列。"""
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


# ─── 数据加载 ───

def find_input_json(dj: str, cache_dir: str = None, variant: str = None) -> Optional[Path]:
    """优先找到结构化原始口径的 *_structured.json；找不到再回退到 final.json。"""
    base = _cache_base(dj, cache_dir)
    if not base.exists():
        return None

    patterns = []
    if variant:
        patterns.append(f"*_structured_{variant}.json")
    patterns.append("*_structured.json")

    for pattern in patterns:
        candidates = sorted(base.glob(pattern), key=lambda p: p.stat().st_mtime)
        if candidates:
            return candidates[-1]

    final_candidates: List[Path] = []
    if variant:
        final_candidates.append(base / f"{dj}_final_{variant}.json")
    final_candidates.append(base / f"{dj}_final.json")

    for candidate in final_candidates:
        if candidate.exists():
            return candidate

    return None


def load_structured_segments(structured_json_path: Path) -> List[Dict[str, Any]]:
    """加载 Judge 输入 JSON 中的文案段列表。支持 list 或 {segments:[...]}。"""
    data = json.loads(structured_json_path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    return data.get("segments", [])


def _make_segment_key(segment_id: Any, episode: Any) -> str:
    if episode is None:
        return f"seg{segment_id}"
    return f"ep{int(episode)}_seg{int(segment_id)}"


def build_segment_index(dj: str, cache_dir: str = None) -> Dict[str, Dict[str, Any]]:
    """构建 (episode, segment_id) → segment_info 的索引"""
    base = _cache_base(dj, cache_dir)
    sd_path = base / "structured_description.json"

    if not sd_path.exists():
        logger.error(f"structured_description.json 不存在: {sd_path}")
        return {}

    data = json.loads(sd_path.read_text(encoding="utf-8"))
    index: Dict[str, Dict[str, Any]] = {}
    for seg in data.get("segments", []):
        sid = seg.get("segment_id")
        if sid is not None:
            seg_key = _make_segment_key(sid, seg.get("episode"))
            index[seg_key] = seg
    logger.info(f"视频片段索引: {len(index)} 个")
    return index


def build_episode_index(seg_index: Dict[str, Dict[str, Any]]) -> Dict[int, List[str]]:
    """
    构建 episode → [segment_ids] 的索引

    优先从 segment 的 episode 字段分组；
    如果没有 episode 字段（旧数据），返回空 dict（后续走兜底逻辑）
    """
    ep_index: Dict[int, List[str]] = defaultdict(list)
    has_episode = False

    for seg_key, seg in seg_index.items():
        ep = seg.get("episode")
        if ep is not None:
            has_episode = True
            ep_index[int(ep)].append(seg_key)

    if has_episode:
        for ep in ep_index:
            ep_index[ep].sort(key=lambda key: int(seg_index[key].get("segment_id", 0)))
        logger.info(f"集数索引: {len(ep_index)} 集, " +
                     ", ".join(f"ep{ep}={len(ids)}段" for ep, ids in sorted(ep_index.items())[:5]) + "...")
        return dict(ep_index)

    logger.warning("视频片段无 episode 字段（旧数据），将评估全部片段")
    return {}


def resolve_video_path_cached(
    segment_key: str,
    seg_index: Dict[str, Dict[str, Any]],
    dj: str,
    cache_dir: str = None,
) -> Optional[str]:
    """从索引中解析视频路径，优先使用当前 cache 下可稳定访问的片段文件。"""
    base = _cache_base(dj, cache_dir)

    seg_info = seg_index.get(segment_key)
    if seg_info:
        path = seg_info.get("segment_file_path", "")
        filename = Path(path).name if path else ""
        episode = seg_info.get("episode")
        seg_dir = base / "segment"

        # 1) 优先在当前 cache 下按 episode + filename 精确找，避免误用旧绝对路径。
        if filename and seg_dir.exists():
            if episode is not None:
                local_candidate = seg_dir / f"ep_{int(episode)}" / filename
                if local_candidate.exists():
                    return str(local_candidate.resolve())

            candidate = seg_dir / filename
            if candidate.exists():
                return str(candidate.resolve())

            for ep_dir in seg_dir.glob("ep_*"):
                candidate = ep_dir / filename
                if candidate.exists():
                    return str(candidate.resolve())

        # 2) 如果当前 cache 找不到，再退回旧路径。
        if path and os.path.exists(path):
            return path

        # 3) 最后按当前 segment_id 在当前 cache 中做一次 episode-aware 兜底解析。
        sid = seg_info.get("segment_id")
        if sid is not None:
            resolved = resolve_video_path(int(sid), dj, str(base))
            if resolved:
                return resolved

    return None


def get_episode_target(seg: Dict[str, Any], seg_index: Dict[str, Dict[str, Any]]) -> Optional[int]:
    """
    获取文案段的目标集数

    优先级：
    1. _structured.json 中的 episode_target 字段（S3 suoying 模块写入）
    2. 从 top_matches 推断（大多数候选来自同一集）
    3. None（兜底：评估全部片段）
    """
    # 方式 1: 直接读取
    ep = seg.get("episode_target")
    if ep is not None:
        return int(ep)

    # 方式 2: 从 top_matches 推断
    mr = seg.get("match_result", {})
    matches = mr.get("top_matches", [])
    if matches:
        ep_counts: Dict[int, int] = defaultdict(int)
        for m in matches[:10]:
            seg_ep = m.get("episode")
            if seg_ep is not None:
                ep_counts[int(seg_ep)] += 1
        if ep_counts:
            return max(ep_counts, key=ep_counts.get)

    return None


def get_episode_segment_ids(
    episode: Optional[int],
    ep_index: Dict[int, List[str]],
    seg_index: Dict[str, Dict[str, Any]],
) -> List[str]:
    """
    获取目标集数内的所有 segment_id

    如果有 ep_index（新数据），按集数过滤
    如果没有（旧数据），返回全部 segment_id
    """
    if episode is not None and ep_index:
        return ep_index.get(episode, [])

    # 兜底：全部片段
    return sorted(seg_index.keys(), key=lambda key: (
        int(seg_index[key].get("episode") or 0),
        int(seg_index[key].get("segment_id") or 0),
    ))


def _cache_base(dj: str, cache_dir: str = None) -> Path:
    return Path(cache_dir) if cache_dir else (
        Path(__file__).resolve().parent.parent / "text_video_match" / "cache" / dj
    )


def _fmt_score(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}"


def _fmt_pct(value: float) -> str:
    return f"{value:.1%}"


def _diagnose_script(script_report: Dict[str, Any]) -> str:
    if script_report.get("n_candidates", 0) == 0:
        return "无成功评估结果"

    best_rank = script_report.get("system_best_match_rank")
    judge_best_score = script_report.get("best_judge_score")
    ge_4_count = script_report.get("score_ge_stats", {}).get("ge_4_0", {}).get("count", 0)

    if best_rank == 1:
        if judge_best_score is not None and judge_best_score >= 4.0:
            return "已命中高质量"
        return "命中但上限不足"
    if ge_4_count == 0:
        return "候选池不足"
    if best_rank is not None and best_rank <= 3:
        return "接近，可优化"
    return "有好候选但错排"


def _select_segments_within_20_seconds(segments: List[Dict[str, Any]], target_total: float = 20.0) -> List[Dict[str, Any]]:
    """按 audio_duration 累加选择前 20 秒文案段，行为与主流程保持一致。"""
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


# ─── 断点续评 ───

def load_existing_results(results_path: Path) -> Dict[str, Dict[str, Any]]:
    """加载已有 results.jsonl，返回 {evaluation_id: result}"""
    if not results_path.exists():
        return {}

    existing = {}
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                eid = item.get("evaluation_id", "")
                if eid:
                    existing[eid] = item
            except json.JSONDecodeError:
                continue

    logger.info(f"已有评估结果: {len(existing)} 条")
    return existing


# ─── 系统质量报告 ───

def generate_report(
    run_dir: Path,
    structured_json: Path,
    seg_index: Dict,
    dj: str,
    hide_system_quality: bool = False,
):
    """
    从全量标注结果生成系统质量报告

    核心指标：
    - 每条文案的 best_match 在 Judge 排名中排第几
    - overall 分数分布
    - 各维度平均分
    """
    results_path = run_dir / "results.jsonl"
    if not results_path.exists():
        logger.warning("无评估结果，跳过报告生成")
        return

    # 加载所有结果；不能按 evaluation_id 去重。
    # 旧 run 中 evaluation_id 曾错误使用 episode_target，导致不同文案碰撞。
    all_results: List[Dict[str, Any]] = []
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                all_results.append(item)
            except json.JSONDecodeError:
                continue

    if not all_results:
        return

    # segments = _select_segments_within_20_seconds(load_structured_segments(structured_json))
    segments = load_structured_segments(structured_json)
    segment_meta: Dict[int, Dict[str, Any]] = {}
    expected_seg_sequences = []
    script_key_to_seq: Dict[Tuple[str, int], int] = {}
    best_match_map = {}  # seg_seq -> best_match_info
    for seg in segments:
        seg_seq = int(seg.get("segment_sequence", 0))
        expected_seg_sequences.append(seg_seq)
        segment_meta[seg_seq] = seg

        script_text = seg.get("script_text", "")
        ep_target = int(seg.get("episode_target") or 0)
        script_key_to_seq[(script_text, ep_target)] = seg_seq

        mr = seg.get("match_result", {})
        best = mr.get("best_match") or {}
        top1 = mr.get("top1_match") or {}
        tops = mr.get("top_matches", []) or []
        chosen = best or top1 or (tops[0] if tops else {})
        if chosen:
            best_match_map[seg_seq] = {
                "segment_id": chosen.get("segment_id"),
                "episode": chosen.get("episode"),
            }

    # 按 script segment 分组
    from collections import defaultdict
    by_script: Dict[int, List[Dict]] = defaultdict(list)
    for seg_seq in expected_seg_sequences:
        by_script[int(seg_seq)] = []
    for result in all_results:
        eid = result.get("evaluation_id", "")
        query = result.get("query", {})
        seg_seq = query.get("segment_sequence")
        if seg_seq is None:
            seg_seq = script_key_to_seq.get((
                query.get("script_text", ""),
                int(query.get("episode_target") or 0)
            ))
        if seg_seq is None:
            # 兼容旧结果：如果 evaluation_id 曾错误写成 seg{episode_target}，这里不再信任它。
            parts = eid.split("_seg")
            if len(parts) == 2:
                try:
                    seg_seq = int(parts[1].split("_cand")[0])
                except ValueError:
                    seg_seq = None
        if seg_seq is None:
            continue
        by_script[int(seg_seq)].append(result)

    # ── 计算指标 ──
    report = {
        "dj": dj,
        "total_evaluations": len(all_results),
        "total_scripts": len(expected_seg_sequences),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "scripts": [],
        "overall_stats": {},
    }

    all_overalls = []
    dimension_sums = {d: [] for d in VALID_DIMENSIONS}
    best_match_ranks = []
    topk_thresholds = [1, 3, 5, 10, 20]
    best_match_topk_hits = {k: 0 for k in topk_thresholds}
    score_thresholds = [4.0, 4.5]
    total_ge_hits = {threshold: 0 for threshold in score_thresholds}
    scripts_with_ge_hits = {threshold: 0 for threshold in score_thresholds}
    system_best_ge_hits = {threshold: 0 for threshold in score_thresholds}

    total_scripts = len(expected_seg_sequences)
    for seg_seq in sorted(expected_seg_sequences):
        candidates = by_script[seg_seq]
        # 按 overall 降序排列
        candidates.sort(key=lambda x: x.get("scores", {}).get("overall", 0), reverse=True)

        n_candidates = len(candidates)
        overalls = [c["scores"]["overall"] for c in candidates]
        all_overalls.extend(overalls)
        episode_target = candidates[0].get("query", {}).get("episode_target") if candidates else None
        if episode_target is None:
            seg_meta = segment_meta.get(int(seg_seq), {})
            episode_target = seg_meta.get("episode_target")

        for c in candidates:
            for d in VALID_DIMENSIONS:
                dimension_sums[d].append(c["scores"].get(d, 0))

        # best_match 排名
        best_info = best_match_map.get(seg_seq, {})
        best_sid = best_info.get("segment_id")
        best_episode = best_info.get("episode")
        best_rank = None
        best_judge_score = None
        for rank, c in enumerate(candidates, 1):
            cand_sid = c["candidate"].get("segment_id")
            cand_episode = c["candidate"].get("episode")
            if cand_sid == best_sid and (best_episode is None or cand_episode == best_episode):
                best_rank = rank
                best_judge_score = c["scores"]["overall"]
                break

        if best_rank is not None:
            best_match_ranks.append(best_rank)
            for k in topk_thresholds:
                if best_rank <= k:
                    best_match_topk_hits[k] += 1

        top1 = candidates[0] if candidates else None
        score_ge_counts = {
            threshold: sum(1 for score in overalls if score >= threshold)
            for threshold in score_thresholds
        }
        for threshold in score_thresholds:
            total_ge_hits[threshold] += score_ge_counts[threshold]
            if score_ge_counts[threshold] > 0:
                scripts_with_ge_hits[threshold] += 1
            if best_judge_score is not None and best_judge_score >= threshold:
                system_best_ge_hits[threshold] += 1

        script_report = {
            "segment_sequence": seg_seq,
            "episode_target": episode_target,
            "n_candidates": n_candidates,
            "best_judge_segment_id": top1["candidate"]["segment_id"] if top1 else None,
            "best_judge_score": top1["scores"]["overall"] if top1 else None,
            "system_best_match_id": best_sid,
            "system_best_match_episode": best_episode,
            "system_best_match_in_candidate_pool": best_rank is not None,
            "system_best_match_rank": best_rank,
            "system_best_match_score": best_judge_score,
            "score_range": [min(overalls), max(overalls)] if overalls else None,
            "would_use_count": sum(1 for c in candidates if c.get("would_use")),
            "system_match_status": (segment_meta.get(int(seg_seq), {}).get("match_result", {}) or {}).get("status"),
            "score_ge_stats": {
                f"ge_{str(threshold).replace('.', '_')}": {
                    "count": score_ge_counts[threshold],
                    "rate": round(score_ge_counts[threshold] / n_candidates, 3) if n_candidates else 0.0,
                }
                for threshold in score_thresholds
            },
        }
        script_report["score_gap_to_judge_best"] = (
            round(script_report["best_judge_score"] - best_judge_score, 2)
            if script_report["best_judge_score"] is not None and best_judge_score is not None
            else None
        )
        script_report["diagnosis"] = _diagnose_script(script_report)
        report["scripts"].append(script_report)

    # 汇总统计
    if all_overalls:
        report["overall_stats"] = {
            "mean": round(sum(all_overalls) / len(all_overalls), 3),
            "min": round(min(all_overalls), 2),
            "max": round(max(all_overalls), 2),
            "dimension_means": {
                d: round(sum(v) / len(v), 3) for d, v in dimension_sums.items() if v
            },
            "score_threshold_stats": {
                f"ge_{str(threshold).replace('.', '_')}": {
                    "count": total_ge_hits[threshold],
                    "rate": round(total_ge_hits[threshold] / len(all_overalls), 3),
                }
                for threshold in score_thresholds
            },
        }

    if best_match_ranks:
        total_ranked = len(best_match_ranks)
        report["system_quality"] = {
            "ranked_script_count": total_ranked,
            "best_match_mean_rank": round(sum(best_match_ranks) / len(best_match_ranks), 2),
            "best_match_median_rank": sorted(best_match_ranks)[len(best_match_ranks) // 2],
            "best_match_topk_rate": {
                f"top{k}": round(best_match_topk_hits[k] / total_scripts, 3) if total_scripts else 0.0
                for k in topk_thresholds
            },
            "score_threshold_by_script": {
                f"ge_{str(threshold).replace('.', '_')}": {
                    "scripts_with_any_rate": round(scripts_with_ge_hits[threshold] / total_scripts, 3) if total_scripts else 0.0,
                    "system_best_score_rate": round(system_best_ge_hits[threshold] / total_scripts, 3) if total_scripts else 0.0,
                }
                for threshold in score_thresholds
            },
        }

    # 写入报告
    report_path = run_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # 人类可读摘要（Markdown 表格）
    summary_lines = [
        f"# Judge 评估报告 - {dj}",
        "",
    ]

    if report.get("overall_stats"):
        stats = report["overall_stats"]
        summary_lines.extend([
            "## 总览",
            "",
            "| 指标 | 值 |",
            "|---|---|",
            f"| 总评估数 | {len(all_overalls)} |",
            f"| 文案段数 | {total_scripts} |",
            f"| overall 均值 | {_fmt_score(stats['mean'])} |",
            f"| overall 范围 | {_fmt_score(stats['min'])} ~ {_fmt_score(stats['max'])} |",
            f"| entity 均值 | {_fmt_score(stats.get('dimension_means', {}).get('entity_match'))} |",
            f"| event 均值 | {_fmt_score(stats.get('dimension_means', {}).get('event_match'))} |",
            f"| scene 均值 | {_fmt_score(stats.get('dimension_means', {}).get('scene_match'))} |",
            f"| tone 均值 | {_fmt_score(stats.get('dimension_means', {}).get('tone_match'))} |",
            f"| editing 均值 | {_fmt_score(stats.get('dimension_means', {}).get('editing_suitability'))} |",
            "",
            (
                "核心要点：这张表保留原来的整体质量指标，先看候选总体质量分布，"
                "再决定问题更偏向候选池还是排序。"
            ),
            "",
            "## 候选池阈值命中",
            "",
            "| 阈值 | 命中数 | 命中率 |",
            "|---|---:|---:|",
        ])
        for threshold in score_thresholds:
            key = f"ge_{str(threshold).replace('.', '_')}"
            threshold_stats = stats["score_threshold_stats"][key]
            summary_lines.append(
                f"| >={threshold:.1f} | {threshold_stats['count']} / {len(all_overalls)} | "
                f"{_fmt_pct(threshold_stats['rate'])} |"
            )
        summary_lines.extend([
            "",
            "核心要点：这里看的不是系统排得准不准，而是整个候选池里高分片段密度高不高。",
            "",
        ])

    if report.get("system_quality") and not hide_system_quality:
        sq = report["system_quality"]
        summary_lines.extend([
            "## 系统排序表现",
            "",
            "| 指标 | Top1 | Top3 | Top5 | Top10 | Top20 |",
            "|---|---:|---:|---:|---:|---:|",
            "| 系统 best_match 在 Judge 排名中的命中率 | " +
            " | ".join(_fmt_pct(sq["best_match_topk_rate"][f"top{k}"]) for k in topk_thresholds) + " |",
            "",
            "| 指标 | 值 |",
            "|---|---|",
            f"| 系统 best_match 平均排名 | {_fmt_score(sq['best_match_mean_rank'])} |",
            f"| 系统 best_match 中位排名 | {sq['best_match_median_rank']} |",
            "",
            (
                "核心要点：这里专门看排序能力，判断系统自己选中的片段在 Judge 看来排得靠不靠前。"
            ),
            "",
            "## 文案级高分覆盖",
            "",
            "说明：这是文案级覆盖率，计算方式为“至少有 1 个候选 overall 达到阈值的文案段数 / 总文案段数”；同一条文案下多个高分候选仍只算 1 条文案命中。",
            "",
            "| 阈值 | 存在高分候选的文案占比 | 系统选中片段达标的文案占比 |",
            "|---|---:|---:|",
        ])
        for threshold in score_thresholds:
            key = f"ge_{str(threshold).replace('.', '_')}"
            threshold_stats = sq["score_threshold_by_script"][key]
            summary_lines.append(
                f"| >={threshold:.1f} | {_fmt_pct(threshold_stats['scripts_with_any_rate'])} | "
                f"{_fmt_pct(threshold_stats['system_best_score_rate'])} |"
            )
        summary_lines.extend([
            "",
            (
                "核心要点：左列看候选池里有没有高分片段，右列看系统自己选中的片段是否达到同一条线。"
            ),
            "",
        ])
    elif hide_system_quality and report.get("system_quality"):
        sq = report["system_quality"]
        summary_lines.extend([
            "## 文案级高分覆盖",
            "",
            "说明：这是文案级覆盖率，计算方式为“至少有 1 个候选 overall 达到阈值的文案段数 / 总文案段数”；同一条文案下多个高分候选仍只算 1 条文案命中。",
            "",
            "| 阈值 | 存在高分候选的文案占比 |",
            "|---|---:|",
        ])
        for threshold in score_thresholds:
            key = f"ge_{str(threshold).replace('.', '_')}"
            threshold_stats = sq["score_threshold_by_script"][key]
            summary_lines.append(
                f"| >={threshold:.1f} | {_fmt_pct(threshold_stats['scripts_with_any_rate'])} |"
            )
        summary_lines.extend([
            "",
            "核心要点：这里只看每条文案的候选池里是否存在高分片段，不展示匹配版本的排序指标。",
            "",
        ])

    if hide_system_quality:
        summary_lines.extend([
            "## 逐条文案候选质量",
            "",
            "| 文案段 | 候选数 | Judge 最优片段 | Judge 最优分 | overall 范围 | >=4.0 候选数 | >=4.5 候选数 | would_use 数 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for script_report in report["scripts"]:
            ge_4_stats = script_report.get("score_ge_stats", {}).get("ge_4_0", {})
            ge_45_stats = script_report.get("score_ge_stats", {}).get("ge_4_5", {})
            score_range = script_report.get("score_range") or [None, None]
            summary_lines.append(
                "| seg{seg} | {n_candidates} | {judge_id} | {judge_score} | {score_min} ~ {score_max} | "
                "{ge4_count} | {ge45_count} | {would_use_count} |".format(
                    seg=script_report["segment_sequence"],
                    n_candidates=script_report["n_candidates"],
                    judge_id=script_report.get("best_judge_segment_id") or "-",
                    judge_score=_fmt_score(script_report.get("best_judge_score")),
                    score_min=_fmt_score(score_range[0]),
                    score_max=_fmt_score(score_range[1]),
                    ge4_count=ge_4_stats.get("count", 0),
                    ge45_count=ge_45_stats.get("count", 0),
                    would_use_count=script_report.get("would_use_count", 0),
                )
            )
        summary_lines.extend([
            "",
            "核心要点：这张表只看每条文案的候选池质量和 Judge 最优候选，不展示系统选中片段。",
            "",
            f"详细报告: {report_path}",
        ])
    else:
        summary_lines.extend([
            "## 逐条文案对比",
            "",
            "| 文案段 | 候选数 | 系统片段 | 系统 Judge 排名 | 系统分 | Judge 最优片段 | Judge 最优分 | 分差 | >=4.0 候选数 | 诊断 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ])
        for script_report in report["scripts"]:
            ge_4_stats = script_report.get("score_ge_stats", {}).get("ge_4_0", {})
            summary_lines.append(
                "| seg{seg} | {n_candidates} | {system_id} | {rank} | {system_score} | "
                "{judge_id} | {judge_score} | {gap} | {ge4_count} | {diagnosis} |".format(
                    seg=script_report["segment_sequence"],
                    n_candidates=script_report["n_candidates"],
                    system_id=script_report.get("system_best_match_id") or "-",
                    rank=script_report.get("system_best_match_rank") or "-",
                    system_score=_fmt_score(script_report.get("system_best_match_score")),
                    judge_id=script_report.get("best_judge_segment_id") or "-",
                    judge_score=_fmt_score(script_report.get("best_judge_score")),
                    gap=_fmt_score(script_report.get("score_gap_to_judge_best")),
                    ge4_count=ge_4_stats.get("count", 0),
                    diagnosis=script_report.get("diagnosis", "-"),
                )
            )
        summary_lines.extend([
            "",
            (
                "核心要点：这张表直接对比“系统选中了谁”和“Judge 认为谁最好”。"
                "如果某条有高分候选但系统排名靠后，就是明显的排序问题。"
            ),
            "",
            f"详细报告: {report_path}",
        ])

    summary_content = "\n".join(summary_lines)
    summary_path = run_dir / "summary.txt"
    summary_md_path = run_dir / "summary.md"
    summary_path.write_text(summary_content, encoding="utf-8")
    summary_md_path.write_text(summary_content, encoding="utf-8")
    print("\n" + summary_content)


# ─── 主逻辑 ───

def run_batch(
    dj: str,
    model: str = "google/gemini-3-pro-preview",
    run_id: str = "",
    cache_dir: str = None,
    output_dir: str = None,
    variant: str = None,
    structured_json_path: str = None,
    max_segments: int = 0,
    report_only: bool = False,
):
    """
    批量评估：每条文案 vs 目标集数内所有视频片段
    """
    # ── 路径 ──
    cache_base = _cache_base(dj, cache_dir)
    if output_dir:
        out_base = Path(output_dir)
    elif variant:
        out_base = Path(__file__).resolve().parent / "outputs" / variant / dj
    else:
        out_base = Path(__file__).resolve().parent / "outputs" / dj

    if not run_id:
        run_id = f"annotation_{time.strftime('%Y%m%d_%H%M%S')}"

    run_dir = out_base / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "results.jsonl"
    config_path = run_dir / "config.json"

    # ── 加载数据 ──
    if structured_json_path:
        structured_json = Path(structured_json_path)
        if not structured_json.is_absolute():
            structured_json = cache_base / structured_json
        structured_json = structured_json.resolve()
        if not structured_json.exists():
            logger.error(f"指定 structured_json 不存在: {structured_json}")
            return
    else:
        structured_json = find_input_json(dj, str(cache_base), variant)
        if not structured_json:
            logger.error(f"未找到可用于 Judge 的 final/structured json: {cache_base}")
            return

    logger.info(f"文案文件: {structured_json.name}")
    # segments = _select_segments_within_20_seconds(load_structured_segments(structured_json))
    segments = load_structured_segments(structured_json)
    seg_index = build_segment_index(dj, str(cache_base))

    if not seg_index:
        logger.error("视频片段索引为空，无法评估")
        return

    ep_index = build_episode_index(seg_index)

    # ── 报告模式：只生成报告不跑评估 ──
    if report_only:
        logger.info("report-only 模式，从已有结果生成报告")
        generate_report(run_dir, structured_json, seg_index, dj, hide_system_quality=(variant == "canonical"))
        return

    # ── 断点续评 ──
    existing = load_existing_results(results_path)

    # ── 保存配置 ──
    run_config = {
        "dj": dj,
        "variant": variant or "",
        "model": model,
        "run_id": run_id,
        "structured_json": structured_json.name,
        "total_scripts": len(segments),
        "total_video_segments": len(seg_index),
        "episodes_with_index": bool(ep_index),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    config_path.write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── 构建任务列表 ──
    tasks = []
    seg_count = 0
    for seg in segments:
        seg_count += 1
        if max_segments > 0 and seg_count > max_segments:
            break

        seg_seq = seg.get("segment_sequence", 0)
        episode_target = get_episode_target(seg, seg_index)
        candidate_keys = get_episode_segment_ids(episode_target, ep_index, seg_index)

        script_text = seg["script_text"]
        visual_query = seg.get("visual_query", {})
        audio_duration = seg.get("audio_duration")

        logger.info(
            f"文案 seg{seg_seq}: episode_target={episode_target}, "
            f"候选片段数={len(candidate_keys)}"
        )

        for cand_key in candidate_keys:
            cand_info = seg_index.get(cand_key, {})
            cand_id = cand_info.get("segment_id")
            cand_episode = cand_info.get("episode", 0)
            eval_id = f"{dj}_seg{seg_seq}_ep{cand_episode}_cand{cand_id}"
            if eval_id in existing:
                continue

            tasks.append({
                "eval_id": eval_id,
                "seg_seq": seg_seq,
                "cand_key": cand_key,
                "cand_id": cand_id,
                "cand_episode": cand_episode,
                "script_text": script_text,
                "visual_query": visual_query,
                "audio_duration": audio_duration,
                "episode_target": episode_target or 0,
            })

    if not tasks:
        logger.info("无新任务，全部已评估")
        print(f"全部已评估，生成报告...")
        generate_report(run_dir, structured_json, seg_index, dj, hide_system_quality=(variant == "canonical"))
        return

    logger.info(f"待评估: {len(tasks)} 条 (跳过已有 {len(existing)} 条)")
    print(f"开始评估: {len(tasks)} 条 (模型: {model})")
    print(f"输出目录: {run_dir}")

    # ── 初始化 Judge ──
    judge = MatchJudge(model=model)

    # ── 逐条评估 ──
    success_count = 0
    fail_count = 0
    start_time = time.time()

    with open(results_path, "a", encoding="utf-8") as f_out:
        for i, task in enumerate(tasks):
            eval_id = task["eval_id"]
            cand_key = task["cand_key"]
            cand_id = task["cand_id"]

            logger.info(f"[{i + 1}/{len(tasks)}] {eval_id}")

            seg_info = seg_index.get(cand_key)
            if not seg_info:
                logger.warning(f"candidate_key={cand_key} 不在索引中，跳过")
                fail_count += 1
                continue

            video_path = resolve_video_path_cached(cand_key, seg_index, dj, str(cache_base))
            if not video_path:
                logger.warning(f"candidate_key={cand_key} 视频文件不存在，跳过")
                fail_count += 1
                continue

            sd = seg_info.get("structured_description", {})
            ep = seg_info.get("episode", 0)
            duration = seg_info.get("duration", 0)

            judge_input = JudgeInput(
                script_text=task["script_text"],
                visual_query=task["visual_query"],
                episode_target=task["episode_target"],
                segment_id=cand_id,
                episode=task["cand_episode"] or ep,
                video_segment_path=video_path,
                structured_description=sd,
                duration=duration,
                audio_duration=task["audio_duration"],
            )

            try:
                output = judge.evaluate_single(judge_input, dj=dj, run_id=run_id)
                result_dict = asdict(output)
                result_dict["evaluation_id"] = eval_id
                result_dict.setdefault("query", {})
                result_dict["query"]["segment_sequence"] = task["seg_seq"]
                f_out.write(json.dumps(result_dict, ensure_ascii=False) + "\n")
                f_out.flush()
                success_count += 1

                overall = output.scores.get("overall", 0)
                logger.info(f"  -> overall={overall:.2f}, would_use={output.would_use}")

            except Exception as e:
                logger.error(f"  -> 评估失败: {e}")
                fail_count += 1

            # 进度
            if (i + 1) % 10 == 0 or (i + 1) == len(tasks):
                elapsed = time.time() - start_time
                avg = elapsed / (i + 1)
                eta = avg * (len(tasks) - i - 1)
                print(
                    f"进度: {i + 1}/{len(tasks)} | "
                    f"成功 {success_count} | 失败 {fail_count} | "
                    f"ETA {eta / 60:.1f}min"
                )

    # ── 完成 ──
    elapsed = time.time() - start_time
    run_config.update({
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "success_count": success_count,
        "fail_count": fail_count,
        "elapsed_seconds": round(elapsed, 1),
    })
    config_path.write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n评估完成: {success_count} 成功 / {fail_count} 失败 / {elapsed / 60:.1f}min")

    # ── 自动生成报告 ──
    generate_report(run_dir, structured_json, seg_index, dj, hide_system_quality=(variant == "canonical"))


# ─── CLI ───

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s"
    )

    parser = argparse.ArgumentParser(description="LLM-as-Judge 批量评估")
    parser.add_argument("input_arg", nargs="?", default="", help="剧名或 task txt 路径")
    parser.add_argument("--dj", default="", help="剧名")
    parser.add_argument("--task-file", default="", help="任务文件（每行一条，默认取第一列剧名）")
    parser.add_argument("--model", default="google/gemini-3-pro-preview", help="模型")
    parser.add_argument("--run-id", default="", help="批次ID（断点续评时指定）")
    parser.add_argument("--cache-dir", default=None, help="缓存目录")
    parser.add_argument("--output-dir", default=None, help="输出目录")
    parser.add_argument("--variant", default=None, help="输出版本标识，例如 v4 / qwen_q1")
    parser.add_argument("--structured-json", default=None, help="显式指定待评估的输入 json 路径（可传 final.json 或 _structured.json）")
    parser.add_argument("--max-segments", type=int, default=0, help="最多处理几条文案（0=全部，调试用）")
    parser.add_argument("--report-only", action="store_true", help="只生成报告，不跑评估")
    args = parser.parse_args()

    target_djs: List[str] = []
    input_arg = args.input_arg.strip()

    if input_arg:
        if input_arg.endswith(".txt"):
            target_djs.extend(load_task_djs(input_arg))
        else:
            target_djs.append(input_arg)

    if args.task_file:
        target_djs.extend(load_task_djs(args.task_file))
    if args.dj:
        target_djs.append(args.dj)

    # 保持顺序同时去重
    ordered_djs: List[str] = []
    seen = set()
    for dj in target_djs:
        if dj not in seen:
            ordered_djs.append(dj)
            seen.add(dj)

    if not ordered_djs:
        parser.error("必须提供 --dj 或 --task-file")
    if args.structured_json and len(ordered_djs) > 1:
        parser.error("--structured-json 仅支持单剧模式，请只传一个 --dj")

    for dj in ordered_djs:
        print(f"\n{'=' * 60}")
        print(f"开始处理剧: {dj}")
        print(f"{'=' * 60}")
        run_batch(
            dj=dj,
            model=args.model,
            run_id=args.run_id,
            cache_dir=args.cache_dir,
            output_dir=args.output_dir,
            variant=args.variant,
            structured_json_path=args.structured_json,
            max_segments=args.max_segments,
            report_only=args.report_only,
        )
