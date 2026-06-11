"""
sft_dataset.py  —  SFT Dataset + DataCollator

SFTDataset:
    - 读 train_sft.jsonl（export_sft_data.py 导出）
    - __getitem__ 返回 {query_text, frames, target}

SFTDataCollator:
    - 对每条样本构建对话并 tokenize
    - 返回 {input_ids, attention_mask, pixel_values_videos, ..., targets}
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# 复用 DPO 的对话格式构建
from train.dpo.dpo_dataset import SYSTEM_MSG, build_conversation, _pad_and_stack


class SFTDataset(Dataset):
    def __init__(
        self,
        sft_jsonl_path: str | Path,
        frame_cache_dir: str | Path,
        max_frames: int = 8,
    ):
        self.frame_cache_dir = Path(frame_cache_dir)
        self.max_frames = max_frames

        self.samples: List[Dict] = []
        with open(sft_jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))
        logger.info(
            "SFTDataset loaded %d samples from %s", len(self.samples), sft_jsonl_path
        )

        pos = sum(1 for s in self.samples if s["target"] == 1)
        neg = len(self.samples) - pos
        logger.info("  positive=%d, negative=%d", pos, neg)

    def __len__(self) -> int:
        return len(self.samples)

    def _get_frame_paths(self, video_segment_path: str) -> List[str]:
        """从 frame_cache 获取预抽帧 JPEG。"""
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
        sample = self.samples[idx]
        return {
            "query_text": sample["query_text"],
            "frames": self._get_frame_paths(sample["video_segment_path"]),
            "target": sample["target"],
        }


class SFTDataCollator:
    """
    对每条样本构建对话并 tokenize。
    支持 batch_size >= 1。
    """

    def __init__(
        self,
        processor,
        instruction: str,
        total_pixels: int,
        max_seq_len: int,
    ):
        self.processor = processor
        self.instruction = instruction
        self.total_pixels = total_pixels
        self.max_seq_len = max_seq_len

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
        返回:
            {input_ids, attention_mask, pixel_values_videos, ..., targets}
        """
        inputs_list = []
        targets = []

        for sample in samples:
            conv = build_conversation(
                query_text=sample["query_text"],
                frame_paths=sample["frames"],
                instruction=self.instruction,
                total_pixels=self.total_pixels,
            )
            inputs_list.append(self._tokenize(conv))
            targets.append(sample["target"])

        # batch_size=1：直接取第一个；>1 需要 padding
        if len(inputs_list) == 1:
            batch = inputs_list[0]
        else:
            batch = _pad_and_stack(inputs_list, self.processor.tokenizer.pad_token_id)

        batch["targets"] = torch.tensor(targets, dtype=torch.float32)
        return batch
