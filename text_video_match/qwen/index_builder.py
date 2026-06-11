#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-VL 视频侧 embedding 索引构建脚本。

职责：
1. 读取 text_video_match/cache/{dj}/structured_description.json
2. 解析每个 segment 的视频路径
3. 调用 embedder 对视频片段做 embedding
4. 输出 embeddings_vl.pkl / embeddings_vl_manifest.json / embeddings_vl_failures.json

当前阶段定位：
- 只做 V6-A：视频片段 -> 视觉 embedding 索引
- 不接主流程，不做检索，不写回旧 match_result
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


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


VIDEO_INDEX_OUTPUT_MAP = {
    "V1": "embeddings_vl.pkl",
    "B1": "embeddings_vl_b1.pkl",
    "B2": "text_embeddings_b2.pkl",
}


@dataclass
class IndexBuilderConfig:
    dj: str
    model_path: Optional[str] = None
    fps: float = 3.0
    max_frames: int = 30
    max_length: int = 32768
    normalize: bool = True
    output_dtype: str = "float16"
    output_name: str = ""
    overwrite: bool = False
    max_segments: Optional[int] = None
    batch_size: int = 1
    video_index_version: str = "V1"

    def __post_init__(self):
        if not self.output_name:
            self.output_name = VIDEO_INDEX_OUTPUT_MAP.get(
                self.video_index_version, f"embeddings_vl_{self.video_index_version.lower()}.pkl"
            )


class QwenVideoIndexBuilder:
    def __init__(self, config: IndexBuilderConfig):
        self.config = config
        self.cache_dir = PROJECT_ROOT / "text_video_match" / "cache" / config.dj
        self.structured_path = self.cache_dir / "structured_description.json"
        version_suffix = "" if config.video_index_version == "V1" else f"_{config.video_index_version.lower()}"
        self.output_path = self.cache_dir / config.output_name
        self.manifest_path = self.cache_dir / f"embeddings_vl{version_suffix}_manifest.json"
        self.failures_path = self.cache_dir / f"embeddings_vl{version_suffix}_failures.json"
        self.embedder = QwenVLMultimodalEmbedder(
            QwenEmbedderConfig(
                model_path=Path(config.model_path).expanduser().resolve()
                if config.model_path
                else QwenEmbedderConfig().model_path,
                fps=config.fps,
                num_frames=config.max_frames,
                max_frames=config.max_frames,
                max_length=config.max_length,
                normalize=config.normalize,
                output_dtype=config.output_dtype,
            )
        )

    def _load_segments(self) -> List[Dict[str, Any]]:
        if not self.structured_path.exists():
            raise FileNotFoundError(f"structured_description.json 不存在: {self.structured_path}")

        data = json.loads(self.structured_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            segments = data.get("segments") or data.get("segment_infos") or []
        elif isinstance(data, list):
            segments = data
        else:
            raise ValueError("structured_description.json 格式错误，必须是 list 或 dict")

        if not isinstance(segments, list) or not segments:
            raise ValueError("structured_description.json 中未找到有效 segments")
        return segments

    def _candidate_video_paths(self, segment: Dict[str, Any]) -> List[Path]:
        candidates: List[Path] = []
        raw_paths = [
            segment.get("segment_file_path"),
            segment.get("video_path"),
            segment.get("video_info", {}).get("segment_file_path") if isinstance(segment.get("video_info"), dict) else None,
            segment.get("video_info", {}).get("video_path") if isinstance(segment.get("video_info"), dict) else None,
        ]

        for raw in raw_paths:
            if not raw:
                continue
            candidates.append(Path(str(raw)))

        segment_file_path = segment.get("segment_file_path")
        if segment_file_path:
            path_str = str(segment_file_path)
            if "smart_videoV4" in path_str:
                candidates.append(Path(path_str.replace("smart_videoV4", PROJECT_ROOT.name)))
            if "smart_videoV6" in path_str and PROJECT_ROOT.name != "smart_videoV6":
                candidates.append(Path(path_str.replace("smart_videoV6", PROJECT_ROOT.name)))

        episode = segment.get("episode")
        filename = None
        if segment.get("segment_file_path"):
            filename = Path(str(segment["segment_file_path"])).name
        elif segment.get("video_path"):
            filename = Path(str(segment["video_path"])).name

        if episode is not None and filename:
            candidates.append(self.cache_dir / "segment" / f"ep_{episode}" / filename)

        unique_candidates: List[Path] = []
        seen = set()
        for candidate in candidates:
            candidate = candidate.expanduser()
            key = str(candidate)
            if key not in seen:
                seen.add(key)
                unique_candidates.append(candidate)
        return unique_candidates

    def _resolve_video_path(self, segment: Dict[str, Any]) -> Path:
        candidates = self._candidate_video_paths(segment)
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        raise FileNotFoundError(
            f"找不到视频片段文件，segment_id={segment.get('segment_id')}，已检查: "
            + ", ".join(str(path) for path in candidates)
        )

    def _build_sidecar_text(self, segment: Dict[str, Any]) -> str:
        sd = segment.get("structured_description") or {}
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

    def _build_manifest(
        self,
        total_segments: int,
        success_records: Sequence[Dict[str, Any]],
        failures: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        runtime = self.embedder.get_runtime_info()
        return {
            "builder": "qwen.index_builder.v2",
            "dj": self.config.dj,
            "video_index_version": self.config.video_index_version,
            "structured_path": str(self.structured_path),
            "output_path": str(self.output_path),
            "total_segments": total_segments,
            "success_count": len(success_records),
            "failed_count": len(failures),
            "fps": self.config.fps,
            "max_frames": self.config.max_frames,
            "max_length": self.config.max_length,
            "normalize": self.config.normalize,
            "output_dtype": self.config.output_dtype,
            "batch_size": self.config.batch_size,
            "model_path": str(self.embedder.model_path),
            "runtime": runtime,
        }

    def _process_text_only(self, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """B2 文本 embedding 索引构建：从 structured_description 字段编码文本。

        生成产物：text_embeddings_b2.pkl + manifest + failures。
        不碰 embeddings_vl.pkl（复用 V1 的视觉 embedding）。
        """
        logger.info("开始加载 Qwen3-VL embedder (text-only mode for B2)")
        self.embedder.load()
        logger.info("Qwen3-VL embedder 加载完成")

        total = len(segments)
        logger.info("开始编码 %s 个 segment 的文本描述", total)

        text_embeddings = self.embedder.encode_video_texts(segments, batch_size=min(8, total))

        success_records: List[Dict[str, Any]] = []
        failures: List[Dict[str, Any]] = []
        for idx, (segment, emb) in enumerate(zip(segments, text_embeddings), start=1):
            segment_id = segment.get("segment_id")
            episode = segment.get("episode")
            if emb is not None:
                success_records.append({
                    "segment_id": segment_id,
                    "episode": episode,
                    "segment_file_path": segment.get("segment_file_path", ""),
                    "embedding": emb,
                    "embedding_dim": int(emb.shape[0]),
                    "embedding_dtype": str(emb.dtype),
                })
                logger.info("完成 text embedding: %s/%s segment_id=%s episode=%s dim=%s",
                            idx, total, segment_id, episode, int(emb.shape[0]))
            else:
                failures.append({
                    "segment_id": segment_id,
                    "episode": episode,
                    "segment_file_path": segment.get("segment_file_path", ""),
                    "error": "embedding is None",
                })
                logger.warning("text embedding 为空: segment_id=%s", segment_id)

        with open(self.output_path, "wb") as fh:
            pickle.dump(success_records, fh)

        manifest = self._build_manifest(total, success_records, failures)
        manifest["mode"] = "text_only_b2"
        with open(self.manifest_path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, ensure_ascii=False, indent=2)

        with open(self.failures_path, "w", encoding="utf-8") as fh:
            json.dump(failures, fh, ensure_ascii=False, indent=2)

        logger.info("B2 text embedding 构建完成: success=%s failed=%s output=%s",
                     len(success_records), len(failures), self.output_path)
        return manifest

    def process(self) -> Dict[str, Any]:
        if self.output_path.exists() and not self.config.overwrite:
            raise FileExistsError(
                f"输出文件已存在: {self.output_path}。如需覆盖，请传 --overwrite"
            )

        segments = self._load_segments()
        if self.config.max_segments is not None:
            segments = segments[: self.config.max_segments]

        # B2: 纯文本 embedding，不需要加载视频模型
        if self.config.video_index_version == "B2":
            return self._process_text_only(segments)

        logger.info("开始加载 Qwen3-VL embedder")
        self.embedder.load()
        logger.info("Qwen3-VL embedder 加载完成")

        success_records: List[Dict[str, Any]] = []
        failures: List[Dict[str, Any]] = []

        total = len(segments)
        for index, segment in enumerate(segments, start=1):
            segment_id = segment.get("segment_id")
            episode = segment.get("episode")
            try:
                resolved_video_path = self._resolve_video_path(segment)
                use_sidecar = self.config.video_index_version != "V1"
                if use_sidecar:
                    sidecar_text = self._build_sidecar_text(segment)
                    if sidecar_text:
                        embedding = self.embedder.encode_video_with_sidecar(
                            resolved_video_path,
                            sidecar_text,
                            fps=self.config.fps,
                            max_frames=self.config.max_frames,
                        )
                    else:
                        embedding = self.embedder.encode_video(
                            resolved_video_path,
                            fps=self.config.fps,
                            max_frames=self.config.max_frames,
                        )
                else:
                    embedding = self.embedder.encode_video(
                        resolved_video_path,
                        fps=self.config.fps,
                        max_frames=self.config.max_frames,
                    )
                record = {
                    "segment_id": segment_id,
                    "episode": episode,
                    "segment_file_path": str(resolved_video_path),
                    "embedding": embedding,
                    "embedding_dim": int(embedding.shape[0]),
                    "embedding_dtype": str(embedding.dtype),
                    "duration": segment.get("duration"),
                    "start_time": segment.get("start_time"),
                    "end_time": segment.get("end_time"),
                }
                success_records.append(record)
                logger.info(
                    "完成视频 embedding: %s/%s segment_id=%s episode=%s dim=%s",
                    index,
                    total,
                    segment_id,
                    episode,
                    record["embedding_dim"],
                )
            except Exception as exc:
                failure = {
                    "segment_id": segment_id,
                    "episode": episode,
                    "segment_file_path": segment.get("segment_file_path"),
                    "error": str(exc),
                }
                failures.append(failure)
                logger.exception("视频 embedding 失败: segment_id=%s", segment_id)

        payload = {
            "manifest": self._build_manifest(total, success_records, failures),
            "embeddings": success_records,
        }

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("wb") as f:
            pickle.dump(payload, f)

        self.manifest_path.write_text(
            json.dumps(payload["manifest"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.failures_path.write_text(
            json.dumps(failures, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return {
            "status": "success" if not failures else "partial_success",
            "dj": self.config.dj,
            "output_path": str(self.output_path),
            "manifest_path": str(self.manifest_path),
            "failures_path": str(self.failures_path),
            "total_segments": total,
            "success_count": len(success_records),
            "failed_count": len(failures),
            "runtime": self.embedder.get_runtime_info(),
        }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="构建 Qwen3-VL 视频 embedding 索引")
    parser.add_argument("dj", help="剧名 / 数据集名，例如 chongfan80")
    parser.add_argument("--model-path", help="Qwen3-VL-Embedding 本地模型路径")
    parser.add_argument("--fps", type=float, default=3.0, help="视频编码 fps")
    parser.add_argument("--max-frames", type=int, default=30, help="每个视频最多送入模型的帧数")
    parser.add_argument("--max-length", type=int, default=32768, help="Qwen3-VL processor 最大长度")
    parser.add_argument("--batch-size", type=int, default=1, help="当前默认建议 1，避免视频侧显存尖峰")
    parser.add_argument("--max-segments", type=int, help="仅处理前 N 个 segment，用于小样本验证")
    parser.add_argument("--output-name", default="", help="输出 pkl 文件名，默认按 video_index_version 自动命名")
    parser.add_argument("--output-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--video-index-version", choices=["V1", "B1", "B2"], default="V1", help="视频索引版本：V1=纯视觉，B1=视觉+文本sidecar，B2=独立文本embedding")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有输出")
    return parser


def main():
    args = build_arg_parser().parse_args()
    builder = QwenVideoIndexBuilder(
        IndexBuilderConfig(
            dj=args.dj,
            model_path=args.model_path,
            fps=args.fps,
            max_frames=args.max_frames,
            max_length=args.max_length,
            output_dtype=args.output_dtype,
            output_name=args.output_name,
            overwrite=args.overwrite,
            max_segments=args.max_segments,
            batch_size=args.batch_size,
            video_index_version=args.video_index_version,
        )
    )
    result = builder.process()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
