#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S1-NEW: 视频集数索引化模块
- 输入: /mnt_liuhuan/search_public/aigc/smart_video_v2/audio/videos/{dj}/{dj}_{episode}.mp4
- 处理: 每集用原有的VideoSegmenter切割，只增加episode标记
- 输出: cache/{dj}/structured_description.json（含episode字段）
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from tools.video_segmenter import VideoSegmenter
from tools.video_evaluator import VideoEvaluator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VideoEpisodeIndexer:
    def __init__(self, 
                 video_segmenter: Optional[VideoSegmenter] = None, 
                 video_evaluator: Optional[VideoEvaluator] = None):
        # 完全复用你原有的切割和评估工具
        self.video_segmenter = video_segmenter or VideoSegmenter()
        self.video_evaluator = video_evaluator or VideoEvaluator()

    def _get_video_base_path(self, dj: str) -> Path:
        """视频目录: /mnt_liuhuan/search_public/aigc/smart_video_v2/audio/videos/{dj}/"""
        return Path(f"/mnt_liuhuan/search_public/aigc/smart_video_v2/audio/videos/{dj}")

    def _base_cache(self, dj: str) -> Path:
        """缓存目录: ./text_video_match/cache/{dj}/"""
        return Path(__file__).parent / "cache" / dj

    def _scan_episodes(self, dj: str) -> List[int]:
        """扫描目录下所有集数，格式: {dj}_{episode}.mp4"""
        video_dir = self._get_video_base_path(dj)
        if not video_dir.exists():
            raise FileNotFoundError(f"视频目录不存在: {video_dir}")
        
        pattern = re.compile(rf"^{re.escape(dj)}_(\d+)\.mp4$")
        episodes = set()
        
        for f in video_dir.iterdir():
            if f.is_file() and f.suffix == ".mp4":
                match = pattern.match(f.name)
                if match:
                    episodes.add(int(match.group(1)))
        
        if not episodes:
            raise ValueError(f"未找到匹配格式 {dj}_{{episode}}.mp4 的视频文件")
        
        sorted_eps = sorted(episodes)
        logger.info(f"扫描到 {len(sorted_eps)} 集: {sorted_eps}")
        return sorted_eps

    def _get_episode_video_path(self, dj: str, episode: int) -> Path:
        """获取单集视频路径"""
        return self._get_video_base_path(dj) / f"{dj}_{episode}.mp4"

    def process(self, dj: str, start_ep: Optional[int] = None, end_ep: Optional[int] = None) -> Dict[str, Any]:
        """
        主流程: 按集数处理，每集用原有逻辑切割，只增加episode标记
        """
        logger.info(f"开始视频集数索引化: {dj}")
        
        # 扫描集数
        all_episodes = self._scan_episodes(dj)
        
        # 确定范围
        if start_ep is None:
            start_ep = min(all_episodes)
        if end_ep is None:
            end_ep = max(all_episodes)
        
        episodes_to_process = [ep for ep in all_episodes if start_ep <= ep <= end_ep]
        logger.info(f"处理集数: {episodes_to_process}")
        
        # 创建缓存目录
        cache_dir = self._base_cache(dj)
        seg_base_dir = cache_dir / "segment"
        seg_base_dir.mkdir(parents=True, exist_ok=True)
        
        all_results = []
        
        # 逐集处理
        for episode in sorted(episodes_to_process):
            video_path = self._get_episode_video_path(dj, episode)
            
            if not video_path.exists():
                logger.warning(f"第 {episode} 集不存在，跳过")
                continue
            
            logger.info(f"处理第 {episode} 集: {video_path.name}")
            
            # 该集的片段目录: segment/ep_{episode}/
            ep_seg_dir = seg_base_dir / f"ep_{episode}"
            ep_seg_dir.mkdir(parents=True, exist_ok=True)
            
            # 片段信息文件
            seg_info_file = ep_seg_dir / "segment_infos.json"
            
            # 1. 切割（复用你原有的VideoSegmenter）
            if seg_info_file.exists():
                # 检查是否已完成
                try:
                    data = json.loads(seg_info_file.read_text(encoding="utf8"))
                    if data.get("status") == "done":
                        segment_infos = data.get("segment_infos", [])
                        logger.info(f"第 {episode} 集已切割，共 {len(segment_infos)} 片段")
                    else:
                        segment_infos = self._cut_episode(video_path, ep_seg_dir, seg_info_file)
                except Exception:
                    segment_infos = self._cut_episode(video_path, ep_seg_dir, seg_info_file)
            else:
                segment_infos = self._cut_episode(video_path, ep_seg_dir, seg_info_file)
            
            if not segment_infos:
                logger.warning(f"第 {episode} 集切割失败")
                continue
            
            # 2. 生成描述（复用你原有的VideoEvaluator）
            ep_results = self._describe_episode_segments(segment_infos, episode)
            all_results.extend(ep_results)
            
            logger.info(f"第 {episode} 集处理完成: {len(ep_results)} 片段")
        
        # 3. 保存统一结果（只增加episode字段，其他结构完全不变）
        output_file = self._save_results(all_results, cache_dir)
        
        return {
            "status": "success",
            "dj": dj,
            "episode_range": [start_ep, end_ep],
            "total_episodes": len(episodes_to_process),
            "total_segments": len(all_results),
            "output_file": str(output_file),
        }

    def _cut_episode(self, video_path: Path, ep_seg_dir: Path, seg_info_file: Path) -> List[Dict]:
        """切割单集（完全调用你原有的VideoSegmenter）"""
        logger.info(f"开始切割: {video_path.name}")
        
        # 直接调用你原有的切割方法
        segment_infos = self.video_segmenter.cut_video_segments(str(video_path), str(ep_seg_dir))
        
        if not segment_infos:
            raise ValueError(f"切割失败: {video_path}")
        
        # 保存片段信息（复用你原有的统计逻辑）
        statistics = self._calculate_statistics(segment_infos)
        seg_info_file.write_text(
            json.dumps({
                "status": "done", 
                "statistics": statistics, 
                "segment_infos": segment_infos
            }, ensure_ascii=False, indent=2),
            encoding="utf8"
        )
        
        logger.info(f"切割完成: {len(segment_infos)} 片段")
        return segment_infos

    def _calculate_statistics(self, segment_infos: List[Dict]) -> Dict:
        """复用你原有的统计逻辑"""
        durations = [s.get("duration", 0) for s in segment_infos if s.get("duration", 0) > 0]
        if not durations:
            return {
                "total_segments": len(segment_infos),
                "min_duration": 0, "max_duration": 0, "avg_duration": 0,
                "duration_distribution": {"0-2秒": 0, "2-4秒": 0, "4-6秒": 0, "6-8秒": 0, "8-10秒及以上": 0}
            }
        
        return {
            "total_segments": len(segment_infos),
            "min_duration": round(min(durations), 2),
            "max_duration": round(max(durations), 2),
            "avg_duration": round(sum(durations) / len(durations), 2),
            "duration_distribution": {
                "0-2秒": sum(1 for d in durations if d < 2),
                "2-4秒": sum(1 for d in durations if 2 <= d < 4),
                "4-6秒": sum(1 for d in durations if 4 <= d < 6),
                "6-8秒": sum(1 for d in durations if 6 <= d < 8),
                "8-10秒及以上": sum(1 for d in durations if d >= 8),
            },
        }

    def _describe_episode_segments(self, segment_infos: List[Dict], episode: int) -> List[Dict]:
        """为单集片段生成描述，只增加episode字段"""
        results = []
        
        for idx, seg_info in enumerate(segment_infos, 1):
            seg_path = seg_info["video_path"]
            seg_id = seg_info.get("segment_index", idx)
            
            try:
                # 调用你原有的VideoEvaluator
                desc_result = self.video_evaluator.description_video(seg_path)
                
                result = {
                    "segment_id": seg_id,
                    "episode": episode,  # ← 唯一新增：集数标记
                    "duration": seg_info.get("duration", 0),
                    "start_time": seg_info.get("start_time", 0),
                    "end_time": seg_info.get("end_time", 0),
                    "structured_description": desc_result.get("parsed_result", {}),
                    "segment_file_path": seg_path,
                }
            except Exception as e:
                logger.error(f"描述片段失败 {seg_path}: {e}")
                result = {
                    "segment_id": seg_id,
                    "episode": episode,  # ← 唯一新增
                    "duration": seg_info.get("duration", 0),
                    "start_time": seg_info.get("start_time", 0),
                    "end_time": seg_info.get("end_time", 0),
                    "structured_description": {"error": "处理失败", "error_message": str(e)},
                    "segment_file_path": seg_path,
                }
            
            results.append(result)
        
        return results

    def _save_results(self, results: List[Dict], cache_dir: Path) -> Path:
        """保存结果（结构与你原有的一致，只多了episode字段）"""
        out_json = cache_dir / "structured_description.json"
        out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf8")
        logger.info(f"结构化描述已保存: {out_json} 共 {len(results)} 片段")
        return out_json


# -------------------- CLI --------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法: python skill_video_episode_indexer.py <dj> [start_ep] [end_ep]")
        print("示例:")
        print("  python skill_video_episode_indexer.py buxi5       # 处理所有集")
        print("  python skill_video_episode_indexer.py buxi5 1 15  # 处理1-15集")
        sys.exit(1)
    
    dj = sys.argv[1]
    start_ep = int(sys.argv[2]) if len(sys.argv) > 2 else None
    end_ep = int(sys.argv[3]) if len(sys.argv) > 3 else None
    
    indexer = VideoEpisodeIndexer()
    result = indexer.process(dj, start_ep, end_ep)
    print(json.dumps(result, ensure_ascii=False, indent=2))