#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-VL-Embedding 本地封装。

职责：
1. 懒加载本地 Qwen3-VL-Embedding-8B 模型
2. 提供 encode_text / encode_video / batch 接口
3. 统一输出 numpy embedding，便于后续建索引和检索

说明：
- 这里不做服务化，只做 V6 第一阶段的本地离线推理封装
- 真正的大规模索引构建和检索逻辑会分别放到 index_builder.py / retriever.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


class _EmbedderTimeout(Exception):
    """Embedder 批次超时异常。"""


def _embedder_timeout_handler(signum, frame):
    raise _EmbedderTimeout("Embedding 批次超时")

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
WORKSPACE_ROOT = CURRENT_FILE.parents[3]


def _default_model_path() -> Path:
    env_path = os.getenv("QWEN3_VL_EMBED_MODEL_PATH")
    if env_path:
        return Path(env_path).expanduser().resolve()

    candidates = [
        WORKSPACE_ROOT / "models" / "Qwen3-VL-Embedding-8B",
        PROJECT_ROOT / "models" / "Qwen3-VL-Embedding-8B",
        Path.cwd() / "models" / "Qwen3-VL-Embedding-8B",
        Path.cwd().parent / "models" / "Qwen3-VL-Embedding-8B",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return candidates[0]


DEFAULT_MODEL_PATH = _default_model_path()
DEFAULT_QUERY_INSTRUCTION = "Retrieve the most relevant video clip for the query."
DEFAULT_VIDEO_INSTRUCTION = "Represent the video clip for retrieval."


@dataclass
class QwenEmbedderConfig:
    model_path: Path = DEFAULT_MODEL_PATH
    fps: float = 3.0
    num_frames: int = 30
    max_frames: int = 30
    max_length: int = 32768
    normalize: bool = True
    output_dtype: str = "float16"
    torch_dtype: str = "float16"
    attn_implementation: Optional[str] = None
    query_instruction: str = DEFAULT_QUERY_INSTRUCTION
    video_instruction: str = DEFAULT_VIDEO_INSTRUCTION
    query_version: str = "Q1"


class QwenVLMultimodalEmbedder:
    def __init__(self, config: Optional[QwenEmbedderConfig] = None):
        self.config = config or QwenEmbedderConfig()
        self._embedder: Any = None
        self._torch: Any = None

    @property
    def model_path(self) -> Path:
        return Path(self.config.model_path)

    @property
    def model_script_path(self) -> Path:
        return self.model_path / "scripts" / "qwen3_vl_embedding.py"

    def _load_module(self):
        if not self.model_script_path.exists():
            checked = [
                str(self.model_script_path),
                str(WORKSPACE_ROOT / "models" / "Qwen3-VL-Embedding-8B" / "scripts" / "qwen3_vl_embedding.py"),
                str(PROJECT_ROOT / "models" / "Qwen3-VL-Embedding-8B" / "scripts" / "qwen3_vl_embedding.py"),
            ]
            raise FileNotFoundError(
                "Qwen3-VL embedding script 不存在。"
                f"\n当前 model_path: {self.model_path}"
                f"\n已检查路径:\n- " + "\n- ".join(checked) +
                "\n可通过 --model-path 或环境变量 QWEN3_VL_EMBED_MODEL_PATH 显式指定。"
            )

        spec = importlib.util.spec_from_file_location(
            "local_qwen3_vl_embedding",
            self.model_script_path,
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"无法加载模型脚本: {self.model_script_path}")

        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except ModuleNotFoundError as exc:
            missing_name = exc.name or "unknown"
            raise ImportError(
                "Qwen3-VL embedding 依赖缺失。"
                f"\n缺失模块: {missing_name}"
                "\n通常是当前环境的 transformers / qwen-vl-utils / torch 版本不满足要求。"
                "\n模型 README 要求:"
                "\n- transformers>=4.57.0"
                "\n- qwen-vl-utils>=0.0.14"
                "\n- torch==2.8.0"
                "\n建议按顺序检查:"
                "\n1. python -c \"import torch; print(torch.__version__)\""
                "\n2. python -c \"import transformers; print(transformers.__version__)\""
                "\n3. python -c \"import torchvision; print(torchvision.__version__)\""
                "\n4. python -c \"import qwen_vl_utils; print(getattr(qwen_vl_utils, '__version__', 'unknown'))\""
                "\n如果第 3 步失败，先安装 torchvision；如果第 2 步版本过低，再升级 transformers。"
            ) from exc
        return module

    def _resolve_torch_dtype(self, torch_module):
        dtype_name = self.config.torch_dtype.lower()
        mapping = {
            "float16": torch_module.float16,
            "fp16": torch_module.float16,
            "bfloat16": torch_module.bfloat16,
            "bf16": torch_module.bfloat16,
            "float32": torch_module.float32,
            "fp32": torch_module.float32,
        }
        if dtype_name not in mapping:
            raise ValueError(f"不支持的 torch_dtype: {self.config.torch_dtype}")
        return mapping[dtype_name]

    def load(self, force_reload: bool = False):
        if self._embedder is not None and not force_reload:
            return self._embedder

        module = self._load_module()
        torch_module = module.torch
        qwen_cls = module.Qwen3VLEmbedder

        kwargs: Dict[str, Any] = {
            "dtype": self._resolve_torch_dtype(torch_module),
            "max_length": self.config.max_length,
            "fps": self.config.fps,
            "num_frames": self.config.num_frames,
            "max_frames": self.config.max_frames,
            "default_instruction": self.config.query_instruction,
        }
        if self.config.attn_implementation:
            kwargs["attn_implementation"] = self.config.attn_implementation

        self._embedder = qwen_cls(
            model_name_or_path=str(self.model_path),
            **kwargs,
        )
        self._torch = torch_module
        return self._embedder

    def _ensure_loaded(self):
        if self._embedder is None:
            self.load()

    def _to_numpy(self, tensor) -> np.ndarray:
        array = tensor.detach().cpu().numpy()
        if self.config.output_dtype == "float16":
            return array.astype(np.float16, copy=False)
        if self.config.output_dtype == "float32":
            return array.astype(np.float32, copy=False)
        raise ValueError(f"不支持的 output_dtype: {self.config.output_dtype}")

    def _batched(self, items: Sequence[Dict[str, Any]], batch_size: int) -> Iterable[Sequence[Dict[str, Any]]]:
        for start in range(0, len(items), batch_size):
            yield items[start:start + batch_size]

    def _process(self, items: Sequence[Dict[str, Any]], batch_size: int) -> np.ndarray:
        self._ensure_loaded()
        outputs: List[np.ndarray] = []
        for batch in self._batched(items, batch_size=batch_size):
            old_handler = signal.signal(signal.SIGALRM, _embedder_timeout_handler)
            signal.alarm(600)
            try:
                tensor = self._embedder.process(
                    list(batch),
                    normalize=self.config.normalize,
                )
            except _EmbedderTimeout:
                logger.error("Embedding 批次超时 (>600s)，NFS I/O 可能 hang，返回零向量")
                dim = getattr(self._embedder, 'hidden_size', 4096)
                n = len(batch)
                tensor = self._torch.zeros(n, dim, dtype=self._torch.float16 if self.config.output_dtype == "float16" else self._torch.float32)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
            outputs.append(self._to_numpy(tensor))
            if self._torch is not None and self._torch.cuda.is_available():
                self._torch.cuda.empty_cache()
        if not outputs:
            return np.empty((0, 0), dtype=np.float16 if self.config.output_dtype == "float16" else np.float32)
        return np.concatenate(outputs, axis=0)

    def get_runtime_info(self) -> Dict[str, Any]:
        self._ensure_loaded()
        model = getattr(self._embedder, "model", None)
        device = "unknown"
        model_dtype = "unknown"
        if model is not None:
            try:
                first_param = next(model.parameters())
                device = str(first_param.device)
                model_dtype = str(first_param.dtype)
            except StopIteration:
                pass
        cuda_available = bool(self._torch and self._torch.cuda.is_available())
        return {
            "device": device,
            "cuda_available": cuda_available,
            "model_dtype": model_dtype,
        }

    def build_query_text(
        self,
        script_text: str,
        visual_query: Optional[Dict[str, Any]] = None,
        query_version: Optional[str] = None,
    ) -> str:
        visual_query = visual_query or {}
        version = (query_version or self.config.query_version or "Q1").upper()
        script_text = (script_text or "").strip()
        char_types = visual_query.get("character_types") or []
        action_keywords = visual_query.get("action_keywords") or []
        environment = visual_query.get("environment")
        expanded_tags = visual_query.get("expanded_tags")
        interaction_type = visual_query.get("interaction_type") or []
        fuzzy_tags = visual_query.get("fuzzy_tags") or {}

        parts: List[str] = [script_text] if script_text else []

        if version == "Q1":
            return script_text

        if version == "Q2":
            if expanded_tags:
                parts.append(f"扩展标签: {expanded_tags}")
            return "\n".join(part for part in parts if part)

        if version in {"Q3", "Q4"}:
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

            if version == "Q4" and fuzzy_tags:
                parts.append(f"模糊意图: {json.dumps(fuzzy_tags, ensure_ascii=False, sort_keys=True)}")

            return "\n".join(part for part in parts if part)

        raise ValueError(f"不支持的 query_version: {version}")

    def build_text_query(self, visual_query: Optional[Dict[str, Any]] = None) -> str:
        """构造文本路 query（B2 双路检索用）。

        从 visual_query 的结构化字段提取：角色、动作、环境、互动、情绪、意图。
        不包含 script_text —— 文本路只做结构化语义匹配。
        """
        visual_query = visual_query or {}
        char_types = visual_query.get("character_types") or visual_query.get("characters") or []
        action_keywords = visual_query.get("action_keywords") or []
        environment = visual_query.get("environment")
        interaction_type = visual_query.get("interaction_type") or []
        fuzzy_tags = visual_query.get("fuzzy_tags") or {}

        parts: List[str] = []
        if char_types:
            parts.append(f"角色: {'、'.join(map(str, char_types))}")
        if action_keywords:
            parts.append(f"动作: {'、'.join(map(str, action_keywords))}")
        if environment:
            parts.append(f"环境: {environment}")
        if interaction_type:
            parts.append(f"互动: {'、'.join(map(str, interaction_type))}")
        emotion = fuzzy_tags.get("emotion_intensity")
        if emotion:
            parts.append(f"情绪: {emotion}")
        director_intent = fuzzy_tags.get("director_intent") or []
        if director_intent:
            parts.append(f"意图: {'、'.join(map(str, director_intent))}")

        return "\n".join(parts) if parts else ""

    def encode_video_texts(
        self,
        segments: Sequence[Dict[str, Any]],
        batch_size: int = 8,
    ) -> List[Optional[np.ndarray]]:
        """编码视频侧 structured_description 文本为 text embedding（B2 用）。

        每个 segment 抽取：characters, actions, environment, interaction_type,
        functional_intent, emotion_intensity。
        """
        texts = []
        for seg in segments:
            sd = seg.get("structured_description") or {}
            if not isinstance(sd, dict):
                texts.append("")
                continue

            chars = sd.get("characters") or []
            actions = sd.get("actions") or []
            env = sd.get("environment")
            interactions = sd.get("interaction_type") or []
            intent = sd.get("functional_intent") or []
            emotion = sd.get("emotion_intensity") or ""

            parts: List[str] = []
            if chars:
                parts.append(f"角色: {'、'.join(map(str, chars))}")
            if actions:
                parts.append(f"动作: {'、'.join(map(str, actions))}")
            if env:
                parts.append(f"环境: {env}")
            if interactions:
                parts.append(f"互动: {'、'.join(map(str, interactions))}")
            if intent:
                parts.append(f"意图: {'、'.join(map(str, intent))}")
            if emotion:
                parts.append(f"情绪: {emotion}")

            texts.append("\n".join(parts) if parts else "")

        embeddings = self.encode_texts(texts, batch_size=batch_size)
        return [emb for emb in embeddings]

    def encode_texts(
        self,
        texts: Sequence[str],
        instruction: Optional[str] = None,
        batch_size: int = 8,
    ) -> np.ndarray:
        items = [
            {
                "text": text,
                "instruction": instruction or self.config.query_instruction,
            }
            for text in texts
        ]
        return self._process(items, batch_size=batch_size)

    def encode_text(self, text: str, instruction: Optional[str] = None) -> np.ndarray:
        embeddings = self.encode_texts([text], instruction=instruction, batch_size=1)
        return embeddings[0]

    def encode_videos(
        self,
        video_paths: Sequence[str | Path],
        instruction: Optional[str] = None,
        batch_size: int = 1,
        fps: Optional[float] = None,
        max_frames: Optional[int] = None,
    ) -> np.ndarray:
        items = [
            {
                "video": str(Path(video_path)),
                "instruction": instruction or self.config.video_instruction,
                "fps": fps or self.config.fps,
                "max_frames": max_frames or self.config.max_frames,
            }
            for video_path in video_paths
        ]
        return self._process(items, batch_size=batch_size)

    def encode_video(
        self,
        video_path: str | Path,
        instruction: Optional[str] = None,
        fps: Optional[float] = None,
        max_frames: Optional[int] = None,
    ) -> np.ndarray:
        embeddings = self.encode_videos(
            [video_path],
            instruction=instruction,
            batch_size=1,
            fps=fps,
            max_frames=max_frames,
        )
        return embeddings[0]

    def encode_videos_with_sidecar(
        self,
        video_paths: Sequence[str | Path],
        sidecar_texts: Sequence[str],
        instruction: Optional[str] = None,
        batch_size: int = 1,
        fps: Optional[float] = None,
        max_frames: Optional[int] = None,
    ) -> np.ndarray:
        items = [
            {
                "video": str(Path(video_path)),
                "text": sidecar_text,
                "instruction": instruction or self.config.video_instruction,
                "fps": fps or self.config.fps,
                "max_frames": max_frames or self.config.max_frames,
            }
            for video_path, sidecar_text in zip(video_paths, sidecar_texts)
        ]
        return self._process(items, batch_size=batch_size)

    def encode_video_with_sidecar(
        self,
        video_path: str | Path,
        sidecar_text: str,
        instruction: Optional[str] = None,
        fps: Optional[float] = None,
        max_frames: Optional[int] = None,
    ) -> np.ndarray:
        embeddings = self.encode_videos_with_sidecar(
            [video_path],
            [sidecar_text],
            instruction=instruction,
            batch_size=1,
            fps=fps,
            max_frames=max_frames,
        )
        return embeddings[0]

    def encode_queries(
        self,
        script_items: Sequence[Dict[str, Any]],
        batch_size: int = 8,
        query_version: Optional[str] = None,
    ) -> np.ndarray:
        query_texts = [
            self.build_query_text(
                item.get("script_text", ""),
                item.get("visual_query", {}),
                query_version=query_version,
            )
            for item in script_items
        ]
        return self.encode_texts(query_texts, batch_size=batch_size)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qwen3-VL-Embedding 本地自检入口")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH), help="本地模型目录")
    parser.add_argument("--text", action="append", default=[], help="待编码文本，可传多次")
    parser.add_argument("--video", action="append", default=[], help="待编码视频路径，可传多次")
    parser.add_argument("--fps", type=float, default=3.0, help="视频编码 fps")
    parser.add_argument("--max-frames", type=int, default=30, help="视频最大帧数")
    parser.add_argument("--max-length", type=int, default=32768, help="Qwen3-VL processor 最大长度")
    parser.add_argument("--query-version", choices=["Q1", "Q2", "Q3", "Q4"], default="Q1", help="文案 query 组织版本")
    parser.add_argument(
        "--output-dtype",
        choices=["float16", "float32"],
        default="float16",
        help="输出 embedding dtype",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()
    embedder = QwenVLMultimodalEmbedder(
        QwenEmbedderConfig(
            model_path=Path(args.model_path),
            fps=args.fps,
            num_frames=args.max_frames,
            max_frames=args.max_frames,
            max_length=args.max_length,
            output_dtype=args.output_dtype,
            query_version=args.query_version,
        )
    )
    embedder.load()

    results: Dict[str, Any] = {
        "model_path": str(embedder.model_path),
        "text_count": len(args.text),
        "video_count": len(args.video),
    }
    results.update(embedder.get_runtime_info())

    if args.text:
        text_embeddings = embedder.encode_texts(args.text, batch_size=min(8, len(args.text)))
        results["text_embedding_shape"] = list(text_embeddings.shape)
        results["text_embedding_dtype"] = str(text_embeddings.dtype)

    if args.video:
        video_embeddings = embedder.encode_videos(
            args.video,
            batch_size=1,
            fps=args.fps,
            max_frames=args.max_frames,
        )
        results["video_embedding_shape"] = list(video_embeddings.shape)
        results["video_embedding_dtype"] = str(video_embeddings.dtype)

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
