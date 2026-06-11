"""
awr_dataset.py  —  Listwise AWR Dataset + DataCollator

AWRDataset:
    - 读 train_groups.jsonl，每条为一个 group（同 query 的多候选）
    - __getitem__ 返回 group 内所有候选的 frames + rewards

AWRDataCollator:
    - 接收单个 group，对 G 个候选逐一 tokenize
    - 返回 { input_ids, attention_mask, pixel_values_videos, ...,
              rewards: [G], group_size: int }
    - batch_size 固定为 1（一个 group = 一个 training unit）
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# 复用 DPO 的对话格式构建
from train.dpo.dpo_dataset import SYSTEM_MSG, build_conversation


class AWRDataset(Dataset):
    def __init__(
        self,
        groups_jsonl_path: str | Path,
        frame_cache_dir: str | Path,
        max_frames: int = 8,
        video_root: str = "",
    ):
        self.frame_cache_dir = Path(frame_cache_dir)
        self.max_frames = max_frames
        self.video_root = video_root

        self.groups: List[Dict] = []
        with open(groups_jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.groups.append(json.loads(line))
        logger.info(
            "AWRDataset loaded %d groups from %s", len(self.groups), groups_jsonl_path
        )

    def __len__(self) -> int:
        return len(self.groups)

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
        group = self.groups[idx]
        query = group["query"]
        query_text = query.get("query_text") or query.get("script_text") or ""

        candidates = []
        for cand in group["candidates"]:
            candidates.append({
                "frames": self._get_frame_paths(cand["video_segment_path"]),
                "reward": cand["reward"],
                "bucket": cand["bucket"],
            })

        return {
            "query_text": query_text,
            "group_id": group["group_id"],
            "candidates": candidates,
            "weight": group.get("weight", 1.0),
        }


class AWRDataCollator:
    """
    一个 group = 一个 training unit。
    对 group 内 G 个候选逐一构建对话并 tokenize。
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
        samples: list of __getitem__ 输出。batch_size=1 时只有一个 group。
        返回:
            candidate_inputs: list of tokenized dicts (每个候选一个)
            rewards: tensor [G]
            group_size: int
        """
        # batch_size=1: 只取第一个 group
        sample = samples[0]
        candidates = sample["candidates"]
        group_size = len(candidates)

        candidate_inputs = []
        rewards = []

        for cand in candidates:
            conv = build_conversation(
                query_text=sample["query_text"],
                frame_paths=cand["frames"],
                instruction=self.instruction,
                total_pixels=self.total_pixels,
            )
            candidate_inputs.append(self._tokenize(conv))
            rewards.append(cand["reward"])

        return {
            "candidate_inputs": candidate_inputs,
            "rewards": torch.tensor(rewards, dtype=torch.float32),
            "group_size": group_size,
            "group_id": sample["group_id"],
            "weight": sample.get("weight", 1.0),
        }
