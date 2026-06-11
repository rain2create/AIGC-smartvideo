#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B3 双路检索器：视觉路 + 文本路 → 加权分数融合。

视觉路: visual_query (Q3) → visual_emb (V1 embeddings_vl.pkl)
文本路: structured fields query → text_emb (B2 text_embeddings_b2.pkl)
融合:   分数级 min-max normalize 后加权求和, α=0.8 (视觉主导)

产物：{txt_stem}_structured_qwen_b3.json（match_result 写回）
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from text_video_match.qwen.embedder import QwenEmbedderConfig, QwenVLMultimodalEmbedder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _to_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_pkl(path: Path) -> List[Dict[str, Any]]:
    with open(path, "rb") as fh:
        data = pickle.load(fh)
    if isinstance(data, dict):
        return data.get("embeddings", [])
    return data


def _cosine_similarity(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """query_vec: (D,), matrix: (N, D)"""
    q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    m_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)
    return np.dot(m_norm, q_norm)


def _min_max_normalize(scores: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]. Returns all-0.5 if single value."""
    mn = float(scores.min())
    mx = float(scores.max())
    if mx - mn < 1e-12:
        return np.full_like(scores, 0.5, dtype=np.float64)
    return (scores - mn) / (mx - mn)


class B3Retriever:
    def __init__(
        self,
        dj: str,
        script_json_path: str,
        top_k: int = 20,
        alpha: float = 0.8,
        query_version: str = "Q3",
        model_path: Optional[str] = None,
        output_path: Optional[str] = None,
        overwrite: bool = False,
        max_segments: Optional[int] = None,
    ):
        self.dj = dj
        self.script_json_path = Path(script_json_path)
        self.top_k = top_k
        self.alpha = alpha
        self.query_version = query_version
        self.overwrite = overwrite
        self.max_segments = max_segments

        self.cache_dir = PROJECT_ROOT / "text_video_match" / "cache" / dj
        self.vis_emb_path = self.cache_dir / "embeddings_vl.pkl"
        self.txt_emb_path = self.cache_dir / "text_embeddings_b2.pkl"
        self.output_path = Path(output_path) if output_path else self.script_json_path

        self.embedder = QwenVLMultimodalEmbedder(
            QwenEmbedderConfig(
                model_path=Path(model_path).expanduser().resolve()
                if model_path
                else QwenEmbedderConfig().model_path,
                query_version=query_version,
            )
        )

    def _load_script_items(self) -> List[Dict[str, Any]]:
        data = _load_json(self.script_json_path)
        if isinstance(data, dict):
            segments = data.get("segments") or data.get("data") or []
        elif isinstance(data, list):
            segments = data
        else:
            raise ValueError(f"无法解析 script JSON: {self.script_json_path}")
        return [s for s in segments if s.get("segment_sequence") is not None]

    def _build_candidate_index(
        self, vis_records: List[Dict[str, Any]], txt_records: List[Dict[str, Any]]
    ) -> Tuple[
        List[Tuple[Dict[str, Any], Dict[str, Any]]],
        np.ndarray, np.ndarray,
        Dict[int, List[int]],
    ]:
        vis_map: Dict[int, Dict[str, Any]] = {}
        for c in vis_records:
            seg_id = _to_int(c.get("segment_id"))
            if seg_id is not None:
                vis_map[seg_id] = c

        aligned: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
        vis_embs: List[np.ndarray] = []
        txt_embs: List[np.ndarray] = []
        episode_to_indices: Dict[int, List[int]] = {}

        for c in txt_records:
            seg_id = _to_int(c.get("segment_id"))
            if seg_id is None or seg_id not in vis_map:
                continue

            vis_rec = vis_map[seg_id]
            vis_emb = vis_rec.get("embedding")
            txt_emb = c.get("embedding")
            if vis_emb is None or txt_emb is None:
                continue

            align_idx = len(aligned)
            aligned.append((vis_rec, c))
            vis_embs.append(np.asarray(vis_emb, dtype=np.float32).flatten())
            txt_embs.append(np.asarray(txt_emb, dtype=np.float32).flatten())

            ep = _to_int(c.get("episode"))
            if ep is not None:
                episode_to_indices.setdefault(ep, []).append(align_idx)

        if not aligned:
            raise ValueError("无有效候选（visual 和 text embedding 交集为空）")

        vis_matrix = np.stack(vis_embs, axis=0).astype(np.float32, copy=False)
        txt_matrix = np.stack(txt_embs, axis=0).astype(np.float32, copy=False)

        return aligned, vis_matrix, txt_matrix, episode_to_indices

    def _filter_indices(
        self,
        episode_target: Optional[int],
        episode_to_indices: Dict[int, List[int]],
        total: int,
    ) -> List[int]:
        if episode_target is None:
            return list(range(total))
        return episode_to_indices.get(int(episode_target), [])

    def _build_top_matches(
        self, ranked: List[Tuple[Dict[str, Any], float]]
    ) -> List[Dict[str, Any]]:
        return [
            {
                "segment_id": c["segment_id"],
                "episode": c.get("episode"),
                "similarity": round(float(score), 6),
            }
            for c, score in ranked
        ]

    def _build_best_match(self, candidate: Dict[str, Any], score: float) -> Dict[str, Any]:
        return {
            "segment_id": candidate["segment_id"],
            "score": round(float(score), 6),
            "similarity": round(float(score), 6),
            "episode": candidate.get("episode"),
            "segment_file_path": candidate.get("segment_file_path", ""),
            "video_info": {
                "start_time": candidate.get("start_time"),
                "end_time": candidate.get("end_time"),
                "episode": candidate.get("episode"),
                "segment_file_path": candidate.get("segment_file_path", ""),
                "video_path": candidate.get("segment_file_path", ""),
            },
        }

    def process(self) -> Dict[str, Any]:
        if self.output_path.exists() and self.output_path != self.script_json_path and not self.overwrite:
            raise FileExistsError(f"输出文件已存在: {self.output_path}。如需覆盖请传 --overwrite")

        logger.info("B3 加权融合检索: dj=%s top_k=%s alpha=%s query_version=%s",
                     self.dj, self.top_k, self.alpha, self.query_version)

        if not self.vis_emb_path.exists():
            raise FileNotFoundError(f"视觉 embedding 不存在: {self.vis_emb_path}（请先跑 V1 index_builder）")
        if not self.txt_emb_path.exists():
            raise FileNotFoundError(f"文本 embedding 不存在: {self.txt_emb_path}（请先跑 B2 index_builder）")

        script_items = self._load_script_items()
        vis_records = _load_pkl(self.vis_emb_path)
        txt_records = _load_pkl(self.txt_emb_path)

        aligned, vis_matrix, txt_matrix, episode_to_indices = self._build_candidate_index(
            vis_records, txt_records
        )
        logger.info("候选索引构建完成: visual=%s text=%s aligned=%s",
                     len(vis_records), len(txt_records), len(aligned))

        target_items: List[Dict[str, Any]] = []
        accumulated = 0.0
        for item in script_items:
            duration = item.get("audio_duration") or 0.0
            accumulated += duration
            if accumulated <= 20.0:
                target_items.append(item)
        if self.max_segments is not None:
            target_items = target_items[:self.max_segments]

        logger.info("B3 加权融合检索: %s 条文案, 候选池 %s", len(target_items), vis_matrix.shape[0])

        self.embedder.load()
        logger.info("Qwen3-VL embedder 加载完成")

        results: List[Dict[str, Any]] = []
        success_count = 0

        for idx, item in enumerate(target_items, start=1):
            seq = item.get("segment_sequence")
            episode_target = item.get("episode_target")
            script_text = item.get("script_text", "")
            visual_query = item.get("visual_query") or {}

            candidate_indices = self._filter_indices(episode_target, episode_to_indices, vis_matrix.shape[0])

            if not candidate_indices:
                results.append({
                    "segment_sequence": seq,
                    "status": "success",
                    "episode_target": episode_target,
                    "top_matches": [],
                    "match_breakdown": {"mode": "b3_weighted_fusion", "error": "no_candidates"},
                    "best_match": None,
                })
                logger.warning("  Segment %s: 无候选片段", seq)
                continue

            # --- 视觉路 ---
            vis_query_text = self.embedder.build_query_text(script_text, visual_query, query_version=self.query_version)
            vis_query_emb = self.embedder.encode_texts([vis_query_text], batch_size=1)[0]
            if vis_query_emb is None:
                results.append({
                    "segment_sequence": seq,
                    "status": "success",
                    "episode_target": episode_target,
                    "top_matches": [],
                    "match_breakdown": {"mode": "b3_weighted_fusion", "error": "visual_query_encode_failed"},
                    "best_match": None,
                })
                continue

            vis_raw = _cosine_similarity(vis_query_emb, vis_matrix[candidate_indices])

            # --- 文本路 ---
            txt_query_text = self.embedder.build_text_query(visual_query)
            if not txt_query_text:
                vis_scores = vis_raw
                ranked_indices = sorted(
                    range(len(candidate_indices)),
                    key=lambda i: float(vis_scores[i]), reverse=True,
                )[:self.top_k]
                ranked = [(aligned[candidate_indices[i]][0], float(vis_scores[i])) for i in ranked_indices]
                breakdown = {
                    "mode": "b3_visual_only",
                    "reason": "text_query_empty",
                    "alpha": self.alpha,
                }
            else:
                txt_query_emb = self.embedder.encode_texts([txt_query_text], batch_size=1)[0]
                if txt_query_emb is None:
                    vis_scores = vis_raw
                    ranked_indices = sorted(
                        range(len(candidate_indices)),
                        key=lambda i: float(vis_scores[i]), reverse=True,
                    )[:self.top_k]
                    ranked = [(aligned[candidate_indices[i]][0], float(vis_scores[i])) for i in ranked_indices]
                    breakdown = {
                        "mode": "b3_visual_only",
                        "reason": "text_query_encode_failed",
                        "alpha": self.alpha,
                    }
                else:
                    txt_raw = _cosine_similarity(txt_query_emb, txt_matrix[candidate_indices])

                    vis_norm = _min_max_normalize(vis_raw)
                    txt_norm = _min_max_normalize(txt_raw)

                    combined = self.alpha * vis_norm + (1.0 - self.alpha) * txt_norm

                    ranked_indices = sorted(
                        range(len(candidate_indices)),
                        key=lambda i: float(combined[i]), reverse=True,
                    )[:self.top_k]
                    ranked = [(aligned[candidate_indices[i]][0], float(combined[i])) for i in ranked_indices]
                    breakdown = {
                        "mode": "b3_weighted_fusion",
                        "alpha": self.alpha,
                        "vis_query_text": vis_query_text[:200],
                        "txt_query_text": txt_query_text[:200],
                        "vis_mean_cosine": round(float(vis_raw.mean()), 4),
                        "txt_mean_cosine": round(float(txt_raw.mean()), 4),
                    }

            if ranked:
                best_candidate, best_score = ranked[0]
                match_result = {
                    "segment_sequence": seq,
                    "status": "success",
                    "episode_target": episode_target,
                    "top_matches": self._build_top_matches(ranked),
                    "match_breakdown": {
                        **breakdown,
                        "candidate_count": len(ranked),
                        "top_k": self.top_k,
                        "query_version": self.query_version,
                    },
                    "best_match": self._build_best_match(best_candidate, best_score),
                }
                success_count += 1
            else:
                match_result = {
                    "segment_sequence": seq,
                    "status": "success",
                    "episode_target": episode_target,
                    "top_matches": [],
                    "match_breakdown": {**breakdown, "error": "fusion_empty"},
                    "best_match": None,
                }

            results.append(match_result)
            logger.info("  Segment %s/%s (seq=%s): mode=%s top_matches=%s",
                         idx, len(target_items), seq,
                         breakdown.get("mode", "unknown"),
                         len(match_result.get("top_matches", [])))

        logger.info("B3 加权融合检索完成: success=%s/%s", success_count, len(target_items))
        return {"results": results, "success_count": success_count}

    def write_results(self, process_output: Dict[str, Any]) -> None:
        source = _load_json(self.script_json_path)
        segments = source if isinstance(source, list) else source.get("segments") or source.get("data") or []
        results_map = {r["segment_sequence"]: r for r in process_output["results"]}

        for seg in segments:
            seq = seg.get("segment_sequence")
            if seq in results_map:
                seg["match_result"] = {
                    k: v for k, v in results_map[seq].items()
                    if k != "segment_sequence"
                }

        os.makedirs(self.output_path.parent, exist_ok=True)
        if self.output_path == self.script_json_path:
            alt_path = self.script_json_path.parent / f"{self.script_json_path.stem}_qwen_b3.json"
            logger.warning("output_path == script_json_path, 写入独立文件: %s", alt_path)
            with open(alt_path, "w", encoding="utf-8") as fh:
                json.dump(segments if isinstance(source, list) else {**source, "segments": segments},
                          fh, ensure_ascii=False, indent=2)
        else:
            with open(self.output_path, "w", encoding="utf-8") as fh:
                json.dump(segments if isinstance(source, list) else {**source, "segments": segments},
                          fh, ensure_ascii=False, indent=2)
            logger.info("B3 检索结果已写入: %s", self.output_path)


def main():
    parser = argparse.ArgumentParser(description="B3 双路检索器（加权分数融合）")
    parser.add_argument("dj", help="剧名")
    parser.add_argument("script_json", help="S3 解说文案 JSON 路径")
    parser.add_argument("--output-path", help="输出 JSON 路径")
    parser.add_argument("--top-k", type=int, default=20, help="最终返回 top_k 候选（默认 20）")
    parser.add_argument("--alpha", type=float, default=0.8, help="视觉路融合权重（默认 0.8）")
    parser.add_argument("--query-version", choices=["Q1", "Q2", "Q3", "Q4"], default="Q3", help="视觉路 query 版本")
    parser.add_argument("--model-path", help="Qwen3-VL 模型路径")
    parser.add_argument("--overwrite", action="store_true", help="强制覆盖已有输出")
    parser.add_argument("--max-segments", type=int, help="限制处理片段数（测试用）")
    args = parser.parse_args()

    retriever = B3Retriever(
        dj=args.dj,
        script_json_path=args.script_json,
        top_k=args.top_k,
        alpha=args.alpha,
        query_version=args.query_version,
        model_path=args.model_path,
        output_path=args.output_path,
        overwrite=args.overwrite,
        max_segments=args.max_segments,
    )
    output = retriever.process()
    retriever.write_results(output)
    logger.info("完成: %s 段匹配成功", output["success_count"])


if __name__ == "__main__":
    main()
