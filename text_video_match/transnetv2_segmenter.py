#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TransNetV2 镜头切分器（修复版：分批推理 + 进度显示）
"""
import os
import json
import logging
import subprocess
from pathlib import Path
from typing import List, Dict, Any
import sys, os, os.path as osp
sys.path.insert(0, osp.join(osp.dirname(__file__), "..", "TransNetV2", "inference-pytorch"))

import numpy as np
import torch
from transnetv2_pytorch import TransNetV2

logger = logging.getLogger(__name__)


class TransNetV2Segmenter:
    def __init__(self,
                 threshold: float = 0.52,
                 min_duration: float = 3.0,
                 fps_sampling: int = 8,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 batch_size: int = 50):      # 🔥 新增：每批处理的帧数
        self.threshold = threshold
        self.min_duration = min_duration
        self.fps_sampling = fps_sampling
        self.device = device
        self.batch_size = batch_size  # 防止爆显存

        # 加载模型
        self.model = TransNetV2().to(self.device)
        weights_path = Path("./transnetv2-pytorch-weights.pth")
        if not weights_path.exists():
            raise FileNotFoundError(f"请先下载/转换权重到 {weights_path}")
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()

    def cut_video_segments(self, video_path: str, output_dir: str) -> List[Dict[str, Any]]:
        os.makedirs(output_dir, exist_ok=True)
        boundaries = self._detect_boundaries(video_path)
        logger.info(f"TransNetV2 检测到 {len(boundaries)} 个镜头")

        video_name = Path(video_path).stem
        results = []
        for idx, (start_sec, end_sec) in enumerate(boundaries, 1):
            duration = end_sec - start_sec
            out_name = f"{idx:03d}_{video_name}.mp4"
            out_path = Path(output_dir) / out_name

            logger.info(f"[{idx:>3d}/{len(boundaries):>3d}] 切割 {start_sec:6.2f}s ~ {end_sec:6.2f}s -> {out_name}")

            cmd = [
                "ffmpeg", "-ss", str(start_sec), "-i", video_path,
                "-t", str(duration), "-c", "copy", "-y", str(out_path)
            ]
            subprocess.run(cmd, check=True, capture_output=True)

            results.append({
                "video_path": str(out_path),
                "segment_index": idx,
                "start_time": start_sec,
                "end_time": end_sec,
                "duration": duration,
                "start_timecode": self._sec_to_timecode(start_sec),
                "end_timecode": self._sec_to_timecode(end_sec),
                "filename": out_name
            })
        return results

    def _detect_boundaries(self, video_path: str) -> List[tuple]:
        """返回 [(start_sec, end_sec), ...] """
        import cv2

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        hop = max(1, int(fps // self.fps_sampling))
        
        total_segments = total_frames // hop
        logger.info(f"视频信息: {total_frames}帧, {fps}fps, 预计抽帧{total_segments}张, batch_size={self.batch_size}")

        frames = []
        idxs = []
        
        # 🔥 抽帧（带进度）
        logger.info("开始抽帧...")
        for i in range(0, total_frames, hop):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (48, 27))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            idxs.append(i)
            
            # 每100帧打印一次进度
            if len(frames) % 100 == 0:
                logger.info(f"  已抽帧 {len(frames)}/{total_segments} ({len(frames)*100//total_segments}%)")
        cap.release()
        
        if not frames:
            return []

        # 🔥 分批推理（关键修复！防止爆显存）
        logger.info(f"开始推理，共{len(frames)}帧，分{(len(frames) + self.batch_size - 1) // self.batch_size}批...")
        probs = []
        
        with torch.no_grad():
            for i in range(0, len(frames), self.batch_size):
                batch_frames = frames[i:i + self.batch_size]
                batch_tensor = torch.from_numpy(np.stack(batch_frames)).unsqueeze(0).to(self.device)
                
                single, _ = self.model(batch_tensor)
                batch_prob = torch.sigmoid(single).cpu().numpy().squeeze()
                probs.extend(batch_prob.tolist() if batch_prob.ndim > 0 else [batch_prob])
                
                if (i // self.batch_size + 1) % 10 == 0:
                    logger.info(f"  已推理 {min(i + self.batch_size, len(frames))}/{len(frames)}帧")
                
                # 🔥 及时清理显存
                del batch_tensor
                if self.device == "cuda":
                    torch.cuda.empty_cache()
        
        prob = np.array(probs)

        # 后处理逻辑不变
        cut_idx = np.where(prob >= self.threshold)[0]
        if len(cut_idx) == 0:
            return [(0., total_frames / fps)]

        cut_sec = np.array(idxs)[cut_idx] / fps
        cut_sec = np.unique(np.clip(cut_sec, 0, total_frames / fps))

        boundaries = []
        start = 0.
        for sec in cut_sec:
            if sec - start >= self.min_duration:
                boundaries.append((start, sec))
                start = sec
        if total_frames / fps - start >= self.min_duration:
            boundaries.append((start, total_frames / fps))
        else:
            if boundaries:
                boundaries[-1] = (boundaries[-1][0], total_frames / fps)
            else:
                boundaries.append((0., total_frames / fps))

        logger.info(f"检测到 {len(boundaries)} 个镜头边界")
        return boundaries

    @staticmethod
    def _sec_to_timecode(sec: float) -> str:
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        s = sec % 60
        return f"{h:02d}:{m:02d}:{s:06.3f}"