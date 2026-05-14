#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频切割模块
"""

import os
import subprocess
import logging
from typing import List, Tuple, Dict, Any
from scenedetect import VideoManager, SceneManager
from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector
from scenedetect.video_manager import FrameTimecode

logger = logging.getLogger(__name__)

class VideoSegmenter:
    """视频切割器"""
    
    def __init__(self, threshold: float = 30.0, min_duration: float = 5.0):  ## 3
        self.threshold = threshold
        self.min_duration = min_duration
    
    def detect_scenes(self, video_path: str) -> List[Tuple[FrameTimecode, FrameTimecode]]:
        """
        检测视频场景
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            场景列表
        """
        video_manager = VideoManager([video_path])
        #video_manager = open_video(video_path,backend ='pyav')
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=self.threshold))
        
        video_manager.start()
        scene_manager.detect_scenes(video_manager)
        scenes = scene_manager.get_scene_list()
        video_manager.release()
        
        return scenes
    
    def merge_short_scenes(self, scenes: List[Tuple[FrameTimecode, FrameTimecode]]) -> List[Tuple[FrameTimecode, FrameTimecode]]:
        """
        合并短场景
        
        Args:
            scenes: 原始场景列表
            
        Returns:
            合并后的场景列表
        """
        if not scenes:
            return []
        
        merged_scenes = []
        current_start = scenes[0][0]
        current_end = scenes[0][1]
        
        for i in range(1, len(scenes)):
            start, end = scenes[i]
            current_duration = current_end.get_seconds() - current_start.get_seconds()
            
            if current_duration < self.min_duration:
                current_end = end
            else:
                merged_scenes.append((current_start, current_end))
                current_start = start
                current_end = end
        
        # 处理最后一个场景
        current_duration = current_end.get_seconds() - current_start.get_seconds()
        if current_duration < self.min_duration and merged_scenes:
            last_start, last_end = merged_scenes[-1]
            merged_scenes[-1] = (last_start, current_end)
        else:
            merged_scenes.append((current_start, current_end))
        
        return merged_scenes
    
    def cut_video_segments(self, video_path: str, output_dir: str) -> List[Dict[str, Any]]:
        """
        切割视频为片段
        
        Args:
            video_path: 输入视频路径
            output_dir: 输出目录
            
        Returns:
            切割后的视频片段信息列表，每个元素包含文件路径、起始时间、结束时间等信息
        """
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 检测场景
        scenes = self.detect_scenes(video_path)
        logger.info(f"检测到 {len(scenes)} 个原始场景")
        
        # 合并短场景
        merged_scenes = self.merge_short_scenes(scenes)
        logger.info(f"合并后得到 {len(merged_scenes)} 个场景")
        
        # 获取视频文件名（不含扩展名）
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        logger.info(f"视频文件名: {video_name}")
        logger.info(f"切割输出目录: {output_dir}")
        
        cut_videos = []
        
        for idx, (start, end) in enumerate(merged_scenes, start=1):
            start_sec = start.get_seconds()
            end_sec = end.get_seconds()
            duration = end_sec - start_sec
            
            # 输出文件名
            output_filename = f"{idx:03d}_{video_name}.mp4"
            output_path = os.path.join(output_dir, output_filename)
            
            logger.info(f"切割场景 {idx}: {start.get_timecode()} - {end.get_timecode()} (持续时间: {duration:.2f}s)")
            
            try:
                # 使用ffmpeg切割视频
                cmd = [
                    'ffmpeg',
                    '-ss', str(start_sec),
                    '-i', video_path,
                    '-t', str(duration),
                    '-c', 'copy',
                    '-avoid_negative_ts', 'make_zero',
                    '-y',
                    output_path
                ]
                

                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    logger.info(f"✓ 成功切割: {output_filename}")
                    # 创建包含时间信息的字典
                    segment_info = {
                        "video_path": output_path,
                        "segment_index": idx,
                        "start_time": start_sec,
                        "end_time": end_sec,
                        "duration": duration,
                        "start_timecode": start.get_timecode(),
                        "end_timecode": end.get_timecode(),
                        "filename": output_filename
                    }
                    cut_videos.append(segment_info)
                else:
                    logger.error(f"✗ 切割失败: {output_filename}")
                    logger.error(f"错误信息: {result.stderr}")
                    
            except Exception as e:
                logger.error(f"✗ 切割场景 {idx} 时发生错误: {str(e)}")
        
        logger.info(f"视频切割完成！共切割 {len(cut_videos)} 个场景")
        return cut_videos
