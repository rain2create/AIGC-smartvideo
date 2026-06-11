#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B2 双路检索器：视觉路 + 文本路 → RRF 合并。

视觉路: visual_query (Q3) → visual_emb (V1 embeddings_vl.pkl)
文本路: structured fields query → text_emb (B2 text_embeddings_b2.pkl)
合并:   RRF (Reciprocal Rank Fusion), k=60

产物：{txt_stem}_structured_qwen_b2.json（match_result 写回）
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


def _rrf_merge(
    vis_ranked: List[Tuple[Dict[str, Any], float]],
    txt_ranked: List[Tuple[Dict[str, Any], float]],
    k: int = 60,
    top_k: int = 20,
) -> List[Tuple[Dict[str, Any], float]]:
    """Reciprocal Rank Fusion 合并两路候选。"""
    scores: Dict[int, float] = {}
    candidates: Dict[int, Dict[str, Any]] = {}

    for rank, (cand, _score) in enumerate(vis_ranked, start=1):
        seg_id = _to_int(cand["segment_id"])
        if seg_id is None:
            continue
        scores[seg_id] = scores.get(seg_id, 0.0) + 1.0 / (k + rank)
        candidates[seg_id] = cand

    for rank, (cand, _score) in enumerate(txt_ranked, start=1):
        seg_id = _to_int(cand["segment_id"])
        if seg_id is None:
            continue
        scores[seg_id] = scores.get(seg_id, 0.0) + 1.0 / (k + rank)
        candidates[seg_id] = cand

    merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(candidates[seg_id], rrf_score) for seg_id, rrf_score in merged[:top_k]]


class B2Retriever:
    def __init__(
        self,
        dj: str,
        script_json_path: str,
        top_k: int = 20,
        per_path_top_k: int = 30,
        query_version: str = "Q3",
        model_path: Optional[str] = None,
        output_path: Optional[str] = None,
        overwrite: bool = False,
        max_segments: Optional[int] = None,
    ):
        self.dj = dj
        self.script_json_path = Path(script_json_path)
        self.top_k = top_k
        self.per_path_top_k = per_path_top_k
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
        """构建两路候选索引，按 segment_id 对齐。

        Returns:
            aligned: [(vis_record, txt_record), ...] — 矩阵索引 → record
            vis_matrix: (N, D) 视觉 embedding 矩阵
            txt_matrix: (N, D) 文本 embedding 矩阵
            episode_to_indices: {episode: [matrix_index, ...]}
        """
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

        logger.info("B2 双路检索: dj=%s top_k=%s query_version=%s", self.dj, self.top_k, self.query_version)

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

        # 只处理 20s 以内的文案
        target_items: List[Dict[str, Any]] = []
        accumulated = 0.0
        for item in script_items:
            duration = item.get("audio_duration") or 0.0
            accumulated += duration
            if accumulated <= 20.0:
                target_items.append(item)
        if self.max_segments is not None:
            target_items = target_items[:self.max_segments]

        logger.info("B2 双路检索: %s 条文案, 候选池 %s", len(target_items), vis_matrix.shape[0])

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
                    "match_breakdown": {"mode": "b2_dual_path_rrf", "error": "no_candidates"},
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
                    "match_breakdown": {"mode": "b2_dual_path_rrf", "error": "visual_query_encode_failed"},
                    "best_match": None,
                })
                continue

            vis_scores = _cosine_similarity(vis_query_emb, vis_matrix[candidate_indices])
            vis_ranked = sorted(
                [(i, float(vis_scores[i])) for i in range(len(candidate_indices))],
                key=lambda x: x[1], reverse=True,
            )
            vis_top_n = vis_ranked[:self.per_path_top_k]

            # --- 文本路 ---
            txt_query_text = self.embedder.build_text_query(visual_query)
            if not txt_query_text:
                # 无结构化字段，仅用视觉路结果
                ranked = [(aligned[candidate_indices[i]][0], s) for i, s in vis_top_n]
                breakdown = {"mode": "b2_visual_only", "reason": "text_query_empty"}
            else:
                txt_query_emb = self.embedder.encode_texts([txt_query_text], batch_size=1)[0]
                if txt_query_emb is None:
                    ranked = [(aligned[candidate_indices[i]][0], s) for i, s in vis_top_n]
                    breakdown = {"mode": "b2_visual_only", "reason": "text_query_encode_failed"}
                else:
                    txt_scores = _cosine_similarity(txt_query_emb, txt_matrix[candidate_indices])
                    txt_ranked = sorted(
                        [(i, float(txt_scores[i])) for i in range(len(candidate_indices))],
                        key=lambda x: x[1], reverse=True,
                    )
                    txt_top_n = txt_ranked[:self.per_path_top_k]

                    vis_candidates = [(aligned[candidate_indices[i]][0], s) for i, s in vis_top_n]
                    txt_candidates = [(aligned[candidate_indices[i]][1], s) for i, s in txt_top_n]
                    ranked = _rrf_merge(vis_candidates, txt_candidates, k=60, top_k=self.top_k)
                    breakdown = {
                        "mode": "b2_dual_path_rrf",
                        "vis_query_text": vis_query_text[:200],
                        "txt_query_text": txt_query_text[:200],
                        "vis_candidates": len(vis_top_n),
                        "txt_candidates": len(txt_top_n),
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
                    "match_breakdown": {**breakdown, "error": "rrf_merge_empty"},
                    "best_match": None,
                }

            results.append(match_result)
            logger.info("  Segment %s/%s (seq=%s): vis_candidates=%s txt_candidates=%s top_matches=%s",
                         idx, len(target_items), seq,
                         breakdown.get("vis_candidates", 0),
                         breakdown.get("txt_candidates", 0),
                         len(match_result.get("top_matches", [])))

        logger.info("B2 双路检索完成: success=%s/%s", success_count, len(target_items))
        return {"results": results, "success_count": success_count}

    def write_results(self, process_output: Dict[str, Any]) -> None:
        """将 B2 检索结果写回 structured JSON。"""
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
            # 不覆盖源文件，写独立文件
            alt_path = self.script_json_path.parent / f"{self.script_json_path.stem}_qwen_b2.json"
            logger.warning("output_path == script_json_path, 写入独立文件: %s", alt_path)
            with open(alt_path, "w", encoding="utf-8") as fh:
                json.dump(segments if isinstance(source, list) else {**source, "segments": segments},
                          fh, ensure_ascii=False, indent=2)
        else:
            with open(self.output_path, "w", encoding="utf-8") as fh:
                json.dump(segments if isinstance(source, list) else {**source, "segments": segments},
                          fh, ensure_ascii=False, indent=2)
            logger.info("B2 检索结果已写入: %s", self.output_path)


def main():
    parser = argparse.ArgumentParser(description="B2 双路检索器（Late Fusion RRF）")
    parser.add_argument("dj", help="剧名")
    parser.add_argument("script_json", help="S3 解说文案 JSON 路径")
    parser.add_argument("--output-path", help="输出 JSON 路径（默认写回 S3 JSON，但会改为独立文件）")
    parser.add_argument("--top-k", type=int, default=20, help="最终返回 top_k 候选（默认 20）")
    parser.add_argument("--per-path-top-k", type=int, default=30, help="每路进入 RRF 的候选数（默认 30）")
    parser.add_argument("--query-version", choices=["Q1", "Q2", "Q3", "Q4"], default="Q3", help="视觉路 query 版本")
    parser.add_argument("--model-path", help="Qwen3-VL 模型路径")
    parser.add_argument("--overwrite", action="store_true", help="强制覆盖已有输出")
    parser.add_argument("--max-segments", type=int, help="限制处理片段数（测试用）")
    args = parser.parse_args()

    retriever = B2Retriever(
        dj=args.dj,
        script_json_path=args.script_json,
        top_k=args.top_k,
        per_path_top_k=args.per_path_top_k,
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
