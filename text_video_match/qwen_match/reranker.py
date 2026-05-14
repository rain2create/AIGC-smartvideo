#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-VL-Reranker 第二阶段精排 (S4.5)。

读取 S4 B1 检索结果，对每个文案的 top-K 候选逐个 (query, video) 打分，
按 reranker_score 重排后写回新 JSON。

doc_version:
  Q1 - 纯视频 document（第一版）
  Q2 - 视频 + 文本 sidecar document（未来扩展）

产物：{txt_stem}_structured_qwen_b1_rerank_{doc_version}.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
WORKSPACE_ROOT = CURRENT_FILE.parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _default_reranker_model_path() -> Path:
    env_path = os.getenv("QWEN3_VL_RERANKER_MODEL_PATH")
    if env_path:
        return Path(env_path).expanduser().resolve()

    candidates = [
        WORKSPACE_ROOT / "models" / "Qwen3-VL-Reranker-8B",
        PROJECT_ROOT / "models" / "Qwen3-VL-Reranker-8B",
        Path("/home/jovyan/data03/search_public/xby/models/Qwen3-VL-Reranker-8B"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return candidates[0]


@dataclass
class RerankerConfig:
    dj: str
    input_json_path: str
    model_path: Optional[str] = None
    output_path: Optional[str] = None
    fps: float = 1.0
    max_frames: int = 8
    doc_version: str = "Q1"
    instruction: str = "Retrieve the most relevant video clip for the given query."
    overwrite: bool = False


class QwenVLRerankerScorer:
    def __init__(self, config: RerankerConfig):
        self.config = config
        self.input_json_path = Path(config.input_json_path)
        self.output_path = (
            Path(config.output_path) if config.output_path else self.input_json_path
        )
        self.model_path = Path(config.model_path) if config.model_path else _default_reranker_model_path()
        self._model: Any = None
        self._torch: Any = None

    def _load_module(self):
        script_path = self.model_path / "scripts" / "qwen3_vl_reranker.py"
        if not script_path.exists():
            raise FileNotFoundError(
                f"Reranker 模型脚本不存在: {script_path}\n"
                f"模型路径: {self.model_path}\n"
                f"可通过 --model-path 或环境变量 QWEN3_VL_RERANKER_MODEL_PATH 指定。"
            )

        spec = importlib.util.spec_from_file_location(
            "local_qwen3_vl_reranker",
            script_path,
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"无法加载模型脚本: {script_path}")

        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except ModuleNotFoundError as exc:
            raise ImportError(
                f"Reranker 模型依赖缺失。缺失模块: {exc.name or 'unknown'}"
            ) from exc
        return module

    def load_model(self, force_reload: bool = False):
        if self._model is not None and not force_reload:
            return self._model

        module = self._load_module()
        self._torch = module.torch
        reranker_cls = module.Qwen3VLReranker

        self._model = reranker_cls(
            model_name_or_path=str(self.model_path),
            max_frames=self.config.max_frames,
            fps=self.config.fps,
            default_instruction=self.config.instruction,
            torch_dtype=self._torch.bfloat16,
        )
        logger.info("Reranker 模型加载完成: %s", self.model_path)
        return self._model

    def _ensure_loaded(self):
        if self._model is None:
            self.load_model()

    def _load_input_json(self) -> List[Dict[str, Any]]:
        if not self.input_json_path.exists():
            raise FileNotFoundError(f"输入 JSON 不存在: {self.input_json_path}")
        data = json.loads(self.input_json_path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"输入 JSON 格式错误，期望 list，实际 {type(data)}")
        return data

    def _load_structured_descriptions(self) -> Dict[int, Dict[str, Any]]:
        sd_path = self.input_json_path.parent / "structured_description.json"
        if not sd_path.exists():
            logger.warning("structured_description.json 不存在: %s，Q2 将降级为纯视频", sd_path)
            return {}
        raw = json.loads(sd_path.read_text(encoding="utf-8"))
        segments = raw.get("segments", raw) if isinstance(raw, dict) else raw
        if not isinstance(segments, list):
            logger.warning("structured_description.json 格式异常")
            return {}
        lookup: Dict[int, Dict[str, Any]] = {}
        for seg in segments:
            sid = seg.get("segment_id")
            sd = seg.get("structured_description")
            if sid is not None and isinstance(sd, dict):
                lookup[sid] = sd
        return lookup

    def _build_documents(
        self,
        top_matches: List[Dict[str, Any]],
        doc_version: str,
        sd_lookup: Dict[int, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        documents: List[Dict[str, Any]] = []
        for candidate in top_matches:
            video_path = candidate.get("segment_file_path", "")
            if not video_path:
                continue
            if not Path(video_path).exists():
                logger.warning("候选视频不存在，跳过: %s", video_path)
                continue

            doc: Dict[str, Any] = {"video": video_path}

            if doc_version.upper() == "Q2":
                segment_id = candidate.get("segment_id")
                sd = sd_lookup.get(segment_id) if segment_id is not None else None
                sidecar = self._build_sidecar_text(sd) if sd else ""
                if sidecar:
                    doc["text"] = sidecar

            documents.append(doc)
        return documents

    @staticmethod
    def _build_sidecar_text(sd: Dict[str, Any]) -> str:
        """构建视频侧结构化文本，字段与 index_builder B1 完全对齐。"""
        if not isinstance(sd, dict):
            return ""

        lines: List[str] = []

        visual_summary = sd.get("visual_summary")
        if visual_summary:
            lines.append(f"画面: {visual_summary}")

        characters = sd.get("characters") or []
        if characters:
            lines.append(f"角色: {', '.join(str(c) for c in characters)}")

        actions = sd.get("actions") or []
        if actions:
            lines.append(f"动作: {', '.join(str(a) for a in actions)}")

        environment = sd.get("environment")
        if environment:
            lines.append(f"环境: {environment}")

        emotion_intensity = sd.get("emotion_intensity")
        if emotion_intensity:
            if isinstance(emotion_intensity, list):
                emotion_intensity = ', '.join(str(e) for e in emotion_intensity)
            lines.append(f"情绪强度: {emotion_intensity}")

        camera_movement = sd.get("camera_movement")
        if camera_movement:
            lines.append(f"镜头: {camera_movement}")

        dialogue_or_text = sd.get("dialogue_or_text")
        if dialogue_or_text and str(dialogue_or_text).lower() != "null":
            lines.append(f"对白: {dialogue_or_text}")

        return "\n".join(lines)

    def score_candidates(
        self,
        query_text: str,
        documents: List[Dict[str, Any]],
    ) -> List[float]:
        self._ensure_loaded()
        if not documents:
            return []

        inputs = {
            "instruction": self.config.instruction,
            "query": {"text": query_text},
            "documents": documents,
            "fps": self.config.fps,
            "max_frames": self.config.max_frames,
        }

        try:
            scores = self._model.process(inputs)
            return [float(s) for s in scores]
        except Exception as exc:
            logger.error("Reranker 批量打分失败: %s", exc)
            return [0.0] * len(documents)

    def process(self) -> Dict[str, Any]:
        if (
            self.output_path.exists()
            and self.output_path != self.input_json_path
            and not self.config.overwrite
        ):
            raise FileExistsError(
                f"输出文件已存在: {self.output_path}。如需覆盖请传 --overwrite"
            )

        logger.info(
            "S4.5 Reranker: dj=%s doc_version=%s fps=%.1f max_frames=%s",
            self.config.dj,
            self.config.doc_version,
            self.config.fps,
            self.config.max_frames,
        )

        items = self._load_input_json()
        logger.info("加载 %s 条文案", len(items))

        sd_lookup = self._load_structured_descriptions() if self.config.doc_version.upper() == "Q2" else {}

        self._ensure_loaded()

        total_segments = 0
        total_candidates = 0

        for item in items:
            mr = item.get("match_result") or {}
            if mr.get("status") != "success":
                continue
            top_matches = mr.get("top_matches") or []
            if not top_matches:
                continue
            total_segments += 1
            total_candidates += len(top_matches)

            query_text = mr.get("match_breakdown", {}).get("query_text", "")
            if not query_text:
                query_text = item.get("script_text", "")

            documents = self._build_documents(top_matches, self.config.doc_version, sd_lookup)
            if not documents:
                logger.warning("Segment %s: 无有效候选视频", item.get("segment_sequence"))
                continue

            scores = self.score_candidates(query_text, documents)

            valid_doc_indices = [
                i for i, d in enumerate(documents) if d.get("video")
            ]
            if not valid_doc_indices or len(scores) != len(documents):
                logger.warning("Segment %s: 打分数量与候选不匹配", item.get("segment_sequence"))
                continue

            scored_candidates = list(zip(valid_doc_indices, scores, strict=False))
            scored_candidates.sort(key=lambda x: x[1], reverse=True)

            reranked_matches = []
            for doc_idx, rerank_score in scored_candidates:
                candidate = top_matches[doc_idx]
                entry: Dict[str, Any] = {
                    "segment_id": candidate.get("segment_id"),
                    "episode": candidate.get("episode"),
                    "similarity": candidate.get("similarity"),
                    "segment_file_path": candidate.get("segment_file_path"),
                    "reranker_score": round(float(rerank_score), 6),
                }
                reranked_matches.append(entry)

            if reranked_matches:
                best = reranked_matches[0]
                original_b1_best = mr.get("best_match") or {}

                mr["best_match"] = {
                    "segment_id": best["segment_id"],
                    "score": best["reranker_score"],
                    "similarity": best["similarity"],
                    "b1_score": original_b1_best.get("score"),
                    "episode": best["episode"],
                    "segment_file_path": best["segment_file_path"],
                    "start_time": original_b1_best.get("start_time"),
                    "end_time": original_b1_best.get("end_time"),
                    "duration": original_b1_best.get("duration"),
                    "video_info": original_b1_best.get("video_info", {}),
                }

            mr["top_matches"] = reranked_matches

            breakdown = mr.get("match_breakdown") or {}
            breakdown["reranker"] = {
                "mode": "qwen3_vl_reranker",
                "doc_version": self.config.doc_version,
                "model_path": str(self.model_path),
                "fps": self.config.fps,
                "max_frames": self.config.max_frames,
                "instruction": self.config.instruction,
                "candidates_reranked": len(reranked_matches),
            }
            mr["match_breakdown"] = breakdown

        logger.info(
            "S4.5 Reranker 完成: segments=%s candidates=%s",
            total_segments,
            total_candidates,
        )

        self._write_output(items)
        return {
            "status": "success",
            "dj": self.config.dj,
            "input_json_path": str(self.input_json_path),
            "output_path": str(self.output_path),
            "segments_processed": total_segments,
            "candidates_total": total_candidates,
        }

    def _write_output(self, data: List[Dict[str, Any]]) -> None:
        os.makedirs(self.output_path.parent, exist_ok=True)
        self.output_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Reranker 结果已写入: %s", self.output_path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qwen3-VL-Reranker S4.5 精排")
    parser.add_argument("dj", help="剧名")
    parser.add_argument("input_json_path", help="S4 B1 检索结果 JSON 路径")
    parser.add_argument("--output-path", help="输出 JSON 路径")
    parser.add_argument("--model-path", help="Reranker 模型路径")
    parser.add_argument("--fps", type=float, default=1.0, help="视频抽帧帧率（默认 1.0）")
    parser.add_argument("--max-frames", type=int, default=8, help="每个候选最大帧数（默认 8）")
    parser.add_argument(
        "--doc-version",
        choices=["Q1", "Q2"],
        default="Q1",
        help="Reranker document 版本：Q1=纯视频, Q2=视频+文本（默认 Q1）",
    )
    parser.add_argument(
        "--instruction",
        default="Retrieve the most relevant video clip for the given query.",
        help="系统指令",
    )
    parser.add_argument("--overwrite", action="store_true", help="强制覆盖已有输出")
    return parser


def main():
    args = build_arg_parser().parse_args()

    config = RerankerConfig(
        dj=args.dj,
        input_json_path=args.input_json_path,
        model_path=args.model_path,
        output_path=args.output_path,
        fps=args.fps,
        max_frames=args.max_frames,
        doc_version=args.doc_version,
        instruction=args.instruction,
        overwrite=args.overwrite,
    )

    scorer = QwenVLRerankerScorer(config)
    result = scorer.process()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
