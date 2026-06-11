#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-VL 多模态检索脚本。

职责：
1. 读取文案结构化结果（*_structured.json）
2. 读取视频侧 visual embedding 索引（embeddings_vl.pkl）
3. 对每条文案构造 query embedding
4. 在目标 episode 内做 cosine TopK 检索
5. 以兼容 Judge / S5 / S6 的格式回写 match_result

当前阶段只做：
- 文案文本 -> 视频 visual embedding 裸检索
- 不叠旧 Stage2 / Stage3 文本精排
- 输出 best_match + top_matches
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from text_video_match.qwen.embedder import QwenEmbedderConfig, QwenVLMultimodalEmbedder


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


VIDEO_INDEX_EMBEDDING_MAP = {
    "V1": "embeddings_vl.pkl",
    "B1": "embeddings_vl_b1.pkl",
}


@dataclass
class RetrieverConfig:
    dj: str
    script_json_path: str
    model_path: Optional[str] = None
    embedding_file: str = ""
    output_path: Optional[str] = None
    top_k: int = 10
    max_segments: Optional[int] = None
    use_episode_target: bool = True
    query_version: str = "Q1"
    overwrite: bool = False
    video_index_version: str = "V1"

    def __post_init__(self):
        if not self.embedding_file:
            self.embedding_file = VIDEO_INDEX_EMBEDDING_MAP.get(
                self.video_index_version, f"embeddings_vl_{self.video_index_version.lower()}.pkl"
            )


class QwenMatchRetriever:
    def __init__(self, config: RetrieverConfig):
        self.config = config
        self.cache_dir = PROJECT_ROOT / "text_video_match" / "cache" / config.dj
        self.script_json_path = Path(config.script_json_path)
        self.output_path = Path(config.output_path) if config.output_path else self.script_json_path
        self.embedding_path = self.cache_dir / config.embedding_file
        self.structured_path = self.cache_dir / "structured_description.json"
        self.embedder = QwenVLMultimodalEmbedder(
            QwenEmbedderConfig(
                model_path=Path(config.model_path).expanduser().resolve()
                if config.model_path
                else QwenEmbedderConfig().model_path,
                query_version=config.query_version,
            )
        )

    def _select_items_within_20_seconds(self, script_items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        selected: List[Dict[str, Any]] = []
        accumulated_time = 0.0

        for idx, item in enumerate(script_items, start=1):
            audio_dur = item.get("audio_duration") or 3

            if accumulated_time >= 20:
                logger.info("超过20秒，Qwen 检索停止处理后续文案。共处理 %s 句", len(selected))
                break

            accumulated_time += float(audio_dur)
            selected.append(item)

            if accumulated_time >= 20:
                logger.info(
                    "达到20秒目标，Qwen 检索停止处理后续文案。当前段=%s，累计时长=%.2fs，共处理 %s 句",
                    item.get("segment_sequence", idx),
                    accumulated_time,
                    len(selected),
                )
                break

        return selected

    def _load_script_items(self) -> List[Dict[str, Any]]:
        if not self.script_json_path.exists():
            raise FileNotFoundError(f"文案结构化文件不存在: {self.script_json_path}")
        data = json.loads(self.script_json_path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("文案结构化文件格式错误，必须为 list")
        if not data:
            raise ValueError("文案结构化文件为空")
        return data

    def _load_structured_segments(self) -> List[Dict[str, Any]]:
        if not self.structured_path.exists():
            raise FileNotFoundError(f"structured_description.json 不存在: {self.structured_path}")
        data = json.loads(self.structured_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            segments = data.get("segments") or data.get("segment_infos") or []
        elif isinstance(data, list):
            segments = data
        else:
            raise ValueError("structured_description.json 格式错误")
        if not isinstance(segments, list) or not segments:
            raise ValueError("structured_description.json 中未找到有效 segments")
        return segments

    def _load_embeddings(self) -> List[Dict[str, Any]]:
        if not self.embedding_path.exists():
            raise FileNotFoundError(f"视频 embedding 文件不存在: {self.embedding_path}")
        with self.embedding_path.open("rb") as f:
            data = pickle.load(f)

        if isinstance(data, dict) and "embeddings" in data:
            embeddings = data["embeddings"]
        elif isinstance(data, list):
            embeddings = data
        else:
            raise ValueError("embeddings_vl.pkl 格式错误，必须包含 embeddings")

        if not isinstance(embeddings, list) or not embeddings:
            raise ValueError("embeddings_vl.pkl 中未找到有效 embedding")
        return embeddings

    def _build_candidate_index(
        self,
        segments: Sequence[Dict[str, Any]],
        embedding_records: Sequence[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], Dict[int, List[int]]]:
        seg_by_episode_and_id: Dict[Tuple[int, int], Dict[str, Any]] = {}
        seg_by_id: Dict[int, List[Dict[str, Any]]] = {}
        seg_by_path: Dict[str, Dict[str, Any]] = {}

        for seg in segments:
            segment_id = seg.get("segment_id")
            if segment_id is None:
                continue

            segment_id_int = int(segment_id)
            seg_by_id.setdefault(segment_id_int, []).append(seg)

            episode = seg.get("episode")
            if episode is not None:
                seg_by_episode_and_id[(int(episode), segment_id_int)] = seg

            segment_file_path = seg.get("segment_file_path")
            if segment_file_path:
                seg_by_path[str(segment_file_path)] = seg

        candidates: List[Dict[str, Any]] = []
        episode_to_indices: Dict[int, List[int]] = {}

        for record in embedding_records:
            segment_id = record.get("segment_id")
            record_episode = record.get("episode")
            if segment_id is None:
                continue

            segment_id_int = int(segment_id)
            segment = None

            if record_episode is not None:
                segment = seg_by_episode_and_id.get((int(record_episode), segment_id_int))

            if segment is None:
                record_path = record.get("segment_file_path")
                if record_path:
                    segment = seg_by_path.get(str(record_path))

            if segment is None:
                id_matches = seg_by_id.get(segment_id_int, [])
                if len(id_matches) == 1:
                    segment = id_matches[0]

            if not segment:
                logger.warning(
                    "跳过无法唯一定位的 embedding 记录: segment_id=%s episode=%s path=%s",
                    segment_id,
                    record_episode,
                    record.get("segment_file_path"),
                )
                continue

            emb = np.asarray(record.get("embedding"), dtype=np.float32)
            episode = record.get("episode") or segment.get("episode") or 1
            candidate = {
                "segment_id": segment_id_int,
                "episode": episode,
                "segment_file_path": record.get("segment_file_path") or segment.get("segment_file_path"),
                "start_time": record.get("start_time", segment.get("start_time")),
                "end_time": record.get("end_time", segment.get("end_time")),
                "duration": record.get("duration", segment.get("duration")),
                "structured_description": segment.get("structured_description", {}),
                "embedding": emb,
            }
            idx = len(candidates)
            candidates.append(candidate)
            episode_to_indices.setdefault(int(episode), []).append(idx)

        if not candidates:
            raise ValueError("没有可用的视频候选索引")
        return candidates, episode_to_indices

    def _cosine_scores(self, query_embedding: np.ndarray, candidate_embeddings: np.ndarray) -> np.ndarray:
        return candidate_embeddings @ query_embedding

    def _select_candidate_indices(
        self,
        episode_target: Optional[int],
        episode_to_indices: Dict[int, List[int]],
        total_candidates: int,
    ) -> Tuple[List[int], Dict[str, Any]]:
        breakdown = {
            "use_episode_target": bool(self.config.use_episode_target),
            "requested_episode_target": episode_target,
            "effective_episode_target": None,
            "fallback_to_all_candidates": False,
        }

        if not self.config.use_episode_target:
            breakdown["fallback_to_all_candidates"] = True
            return list(range(total_candidates)), breakdown

        if episode_target is None:
            raise ValueError("严格集内检索要求每条文案都提供 episode_target")

        episode_indices = episode_to_indices.get(int(episode_target), [])
        if not episode_indices:
            available_episodes = sorted(episode_to_indices.keys())
            raise ValueError(
                "严格集内检索失败: "
                f"episode_target={int(episode_target)} 在当前视频索引中无候选。"
                f"可用集数={available_episodes[:20]}"
            )

        breakdown["effective_episode_target"] = int(episode_target)
        return episode_indices, breakdown

    def _build_best_match(self, candidate: Dict[str, Any], score: float) -> Dict[str, Any]:
        return {
            "segment_id": candidate["segment_id"],
            "score": round(float(score), 6),
            "similarity": round(float(score), 6),
            "episode": candidate["episode"],
            "segment_file_path": candidate["segment_file_path"],
            "start_time": candidate.get("start_time"),
            "end_time": candidate.get("end_time"),
            "duration": candidate.get("duration"),
            "video_info": {
                "start_time": candidate.get("start_time"),
                "end_time": candidate.get("end_time"),
                "episode": candidate["episode"],
                "segment_file_path": candidate["segment_file_path"],
                "video_path": candidate["segment_file_path"],
            },
        }

    def _build_top_matches(self, ranked_candidates: Sequence[Tuple[Dict[str, Any], float]]) -> List[Dict[str, Any]]:
        results = []
        for candidate, score in ranked_candidates:
            results.append(
                {
                    "segment_id": candidate["segment_id"],
                    "episode": candidate["episode"],
                    "similarity": round(float(score), 6),
                    "segment_file_path": candidate["segment_file_path"],
                }
            )
        return results

    def _build_match_result(
        self,
        item: Dict[str, Any],
        ranked_candidates: Sequence[Tuple[Dict[str, Any], float]],
        breakdown: Dict[str, Any],
    ) -> Dict[str, Any]:
        seq = item.get("segment_sequence")
        episode_target = item.get("episode_target")

        if not ranked_candidates:
            return {
                "segment_sequence": seq,
                "status": "success",
                "episode_target": episode_target,
                "top_matches": [],
                "match_breakdown": {
                    **breakdown,
                    "mode": "qwen_vl_retrieval",
                    "candidate_count": 0,
                    "top_k": self.config.top_k,
                },
                "best_match": None,
            }

        best_candidate, best_score = ranked_candidates[0]
        return {
            "segment_sequence": seq,
            "status": "success",
            "episode_target": episode_target,
            "top_matches": self._build_top_matches(ranked_candidates),
            "match_breakdown": {
                **breakdown,
                "mode": "qwen_vl_retrieval",
                "candidate_count": len(ranked_candidates),
                "top_k": self.config.top_k,
                "query_version": self.config.query_version,
                "query_text": self.embedder.build_query_text(
                    item.get("script_text", ""),
                    item.get("visual_query", {}),
                    query_version=self.config.query_version,
                ),
                "score_type": "cosine_on_normalized_embeddings",
            },
            "best_match": self._build_best_match(best_candidate, best_score),
        }

    def process(self) -> Dict[str, Any]:
        if self.output_path.exists() and self.output_path != self.script_json_path and not self.config.overwrite:
            raise FileExistsError(f"输出文件已存在: {self.output_path}。如需覆盖请传 --overwrite")

        script_items = self._load_script_items()
        structured_segments = self._load_structured_segments()
        embedding_records = self._load_embeddings()

        # target_items = self._select_items_within_20_seconds(script_items)
        target_items = list(script_items)
        if self.config.max_segments is not None:
            target_items = target_items[: self.config.max_segments]

        candidates, episode_to_indices = self._build_candidate_index(structured_segments, embedding_records)
        candidate_matrix = np.stack([c["embedding"] for c in candidates], axis=0).astype(np.float32, copy=False)

        logger.info("开始加载 Qwen3-VL embedder")
        self.embedder.load()
        logger.info("Qwen3-VL embedder 加载完成")

        logger.info("开始编码 %s 条文案 query", len(target_items))
        query_embeddings = self.embedder.encode_queries(
            target_items,
            batch_size=min(8, len(target_items)),
            query_version=self.config.query_version,
        )

        success_count = 0
        for idx, (item, query_embedding) in enumerate(zip(target_items, query_embeddings), start=1):
            episode_target = item.get("episode_target")
            candidate_indices, breakdown = self._select_candidate_indices(
                episode_target=episode_target,
                episode_to_indices=episode_to_indices,
                total_candidates=len(candidates),
            )

            subset_matrix = candidate_matrix[candidate_indices]
            subset_scores = self._cosine_scores(query_embedding.astype(np.float32, copy=False), subset_matrix)
            ranked_local = np.argsort(subset_scores)[::-1][: self.config.top_k]

            ranked_candidates = [
                (candidates[candidate_indices[local_idx]], float(subset_scores[local_idx]))
                for local_idx in ranked_local
            ]

            match_result = self._build_match_result(item, ranked_candidates, breakdown)
            item["match_result"] = match_result
            if match_result.get("best_match"):
                success_count += 1

            logger.info(
                "完成检索: %s/%s segment_sequence=%s episode_target=%s top1_segment=%s score=%.4f",
                idx,
                len(target_items),
                item.get("segment_sequence"),
                episode_target,
                match_result.get("best_match", {}).get("segment_id"),
                match_result.get("best_match", {}).get("score", 0.0) or 0.0,
            )

        if target_items is not script_items:
            updated = target_items + script_items[len(target_items):]
        else:
            updated = script_items

        self.output_path.write_text(
            json.dumps(updated, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return {
            "status": "success",
            "dj": self.config.dj,
            "script_json_path": str(self.script_json_path),
            "output_path": str(self.output_path),
            "processed_segments": len(target_items),
            "success_count": success_count,
            "failed_count": len(target_items) - success_count,
            "embedding_file": str(self.embedding_path),
            "runtime": self.embedder.get_runtime_info(),
        }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qwen3-VL 多模态 TopK 检索")
    parser.add_argument("dj", help="剧名 / 数据集名")
    parser.add_argument("script_json_path", help="文案结构化 JSON 路径")
    parser.add_argument("--model-path", help="Qwen3-VL-Embedding 本地模型路径")
    parser.add_argument("--embedding-file", default="", help="视频侧 embedding 文件名，默认按 video_index_version 自动命名")
    parser.add_argument("--output-path", help="输出 JSON 路径；默认覆盖输入文件")
    parser.add_argument("--top-k", type=int, default=10, help="TopK 候选数")
    parser.add_argument("--max-segments", type=int, help="仅处理前 N 条文案")
    parser.add_argument(
        "--ignore-episode-target",
        action="store_true",
        help="显式忽略 episode_target，直接全量检索；默认严格集内检索",
    )
    parser.add_argument("--query-version", choices=["Q1", "Q2", "Q3", "Q4"], default="Q1", help="文案 query 组织版本")
    parser.add_argument("--video-index-version", choices=["V1", "B1"], default="V1", help="视频索引版本：V1=纯视觉，B1=视觉+文本sidecar")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖输出文件")
    return parser


def main():
    args = build_arg_parser().parse_args()
    retriever = QwenMatchRetriever(
        RetrieverConfig(
            dj=args.dj,
            script_json_path=args.script_json_path,
            model_path=args.model_path,
            embedding_file=args.embedding_file,
            output_path=args.output_path,
            top_k=args.top_k,
            max_segments=args.max_segments,
            use_episode_target=not args.ignore_episode_target,
            query_version=args.query_version,
            overwrite=args.overwrite,
            video_index_version=args.video_index_version,
        )
    )
    result = retriever.process()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
