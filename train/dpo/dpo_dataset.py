"""
dpo_dataset.py  —  DPO Dataset + DataCollator

DPODataset:
    - 读 train_pairs.jsonl，返回 (query_text, chosen_frames, rejected_frames)
    - chosen/rejected_frames 为预抽帧 JPEG 路径列表

DPODataCollator:
    - 接收一组样本，对 chosen/rejected 各自独立 tokenize
    - 返回 { chosen_input_ids, chosen_attention_mask, chosen_pixel_values_videos, ...,
              rejected_input_ids, ... }
    - batch_size=1 时不需要 padding（单样本即为 batch）

对话格式（与推理侧 format_mm_instruction 完全一致）：
    system: "Judge whether the Document meets the requirements..."
    user:
      - text: "<Instruct>: ..."
      - text: "<Query>:"
      - text: <query_text>
      - text: "\n<Document>:"
      - video: [file://frame_0001.jpg, ...]  total_pixels=...
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

SYSTEM_MSG = (
    'Judge whether the Document meets the requirements based on the Query '
    'and the Instruct provided. Note that the answer can only be "yes" or "no".'
)


# ── 对话格式构建 ──────────────────────────────────────────────────────────────

def build_conversation(
    query_text: str,
    frame_paths: List[str],
    instruction: str,
    total_pixels: int,
) -> List[Dict[str, Any]]:
    """
    构建 processor.apply_chat_template 可用的对话格式。
    与 Qwen3VLReranker.format_mm_instruction 输出完全对齐。
    """
    video_content = [
        (f"file://{fp}" if not fp.startswith("file://") else fp)
        for fp in frame_paths
    ]
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_MSG}],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"<Instruct>: {instruction}"},
                {"type": "text", "text": "<Query>:"},
                {"type": "text", "text": query_text},
                {"type": "text", "text": "\n<Document>:"},
                {
                    "type": "video",
                    "video": video_content,
                    "total_pixels": total_pixels,
                },
            ],
        },
    ]


# ── Dataset ──────────────────────────────────────────────────────────────────

class DPODataset(Dataset):
    def __init__(
        self,
        pairs_jsonl_path: str | Path,
        frame_cache_dir: str | Path,
        max_frames: int = 8,
    ):
        self.frame_cache_dir = Path(frame_cache_dir)
        self.max_frames = max_frames

        self.pairs: List[Dict] = []
        with open(pairs_jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.pairs.append(json.loads(line))
        logger.info("DPODataset loaded %d pairs from %s", len(self.pairs), pairs_jsonl_path)

    def __len__(self) -> int:
        return len(self.pairs)

    def _get_frame_paths(self, video_segment_path: str) -> List[str]:
        stem = Path(video_segment_path).stem
        frame_dir = self.frame_cache_dir / stem
        frames = sorted(frame_dir.glob("frame_*.jpg"))[: self.max_frames]
        if not frames:
            raise FileNotFoundError(
                f"No frames found in {frame_dir}. "
                "Run train/extract_frames.py first."
            )
        return [str(f) for f in frames]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pair = self.pairs[idx]
        query = pair["query"]
        query_text = query.get("query_text") or query.get("script_text") or ""
        return {
            "query_text": query_text,
            "chosen_frames": self._get_frame_paths(pair["chosen"]["video_segment_path"]),
            "rejected_frames": self._get_frame_paths(pair["rejected"]["video_segment_path"]),
        }


# ── DataCollator ─────────────────────────────────────────────────────────────

class DPODataCollator:
    def __init__(
        self,
        processor,           # AutoProcessor
        instruction: str,
        total_pixels: int,
        max_seq_len: int,
    ):
        self.processor = processor
        self.instruction = instruction
        self.total_pixels = total_pixels
        self.max_seq_len = max_seq_len

        # 延迟导入 qwen_vl_utils（依赖 transformers 安装）
        try:
            from qwen_vl_utils import process_vision_info
            self._process_vision_info = process_vision_info
        except ImportError as e:
            raise ImportError(
                "qwen_vl_utils not found. Install with: pip install qwen-vl-utils"
            ) from e

    def _tokenize(self, conversation: List[Dict]) -> Dict[str, torch.Tensor]:
        """对单条对话做 tokenization，返回 tensor dict。"""
        text = self.processor.apply_chat_template(
            [conversation],
            tokenize=False,
            add_generation_prompt=True,
        )

        try:
            images, videos, video_kwargs = self._process_vision_info(
                [conversation],
                image_patch_size=16,
                return_video_kwargs=True,
                return_video_metadata=True,
            )
        except Exception as e:
            logger.error("process_vision_info failed: %s", e)
            raise

        if videos is not None:
            videos, video_metadatas = zip(*videos)
            videos, video_metadatas = list(videos), list(video_metadatas)
        else:
            video_metadatas = None

        inputs = self.processor(
            text=text,
            images=images,
            videos=videos,
            video_metadata=video_metadatas,
            truncation=True,
            padding=False,
            max_length=self.max_seq_len,
            do_resize=False,
            return_tensors="pt",
            **video_kwargs,
        )
        return inputs

    def __call__(self, samples: List[Dict]) -> Dict[str, Any]:
        """
        samples: list of __getitem__ 输出。
        返回 dict，key 格式：{chosen|rejected}_{tensor_name}
        """
        batch: Dict[str, Any] = {}

        for role in ("chosen", "rejected"):
            role_inputs_list = []
            for sample in samples:
                conv = build_conversation(
                    query_text=sample["query_text"],
                    frame_paths=sample[f"{role}_frames"],
                    instruction=self.instruction,
                    total_pixels=self.total_pixels,
                )
                role_inputs_list.append(self._tokenize(conv))

            # batch_size=1：直接取第一个（无需 pad across examples）
            # 若 batch_size>1 需在此处 pad input_ids / attention_mask
            merged = role_inputs_list[0]
            if len(role_inputs_list) > 1:
                merged = _pad_and_stack(role_inputs_list, self.processor.tokenizer.pad_token_id)

            for k, v in merged.items():
                batch[f"{role}_{k}"] = v

        return batch


def _pad_and_stack(
    inputs_list: List[Dict[str, torch.Tensor]],
    pad_token_id: int,
) -> Dict[str, torch.Tensor]:
    """
    将多个 tokenized 输入 pad 到相同长度后 stack。
    仅处理 1D token 序列（input_ids / attention_mask）；
    pixel_values_videos 保持 list（各样本帧数可能不同）。
    """
    # 找最长序列
    max_len = max(inp["input_ids"].shape[-1] for inp in inputs_list)
    out: Dict[str, Any] = {}

    for key in inputs_list[0]:
        tensors = [inp[key] for inp in inputs_list]
        if key == "input_ids":
            padded = [
                torch.nn.functional.pad(t, (0, max_len - t.shape[-1]), value=pad_token_id)
                for t in tensors
            ]
            out[key] = torch.cat(padded, dim=0)
        elif key == "attention_mask":
            padded = [
                torch.nn.functional.pad(t, (0, max_len - t.shape[-1]), value=0)
                for t in tensors
            ]
            out[key] = torch.cat(padded, dim=0)
        else:
            # pixel_values_videos / image_grid_thw 等：直接 concat 或保持 list
            try:
                out[key] = torch.cat(tensors, dim=0)
            except Exception:
                out[key] = tensors  # fallback: keep as list

    return out
