#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S1：视频结构化理解模块  →  输出统一落到 ./text_video_match/cache/{dj}/
"""
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from tools.video_segmenter import VideoSegmenter
from tools.video_evaluator import VideoEvaluator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VideoStructuredDescription:
    def __init__(self, video_segmenter: Optional[VideoSegmenter] = None, video_evaluator: Optional[VideoEvaluator] = None):
        self.video_segmenter = video_segmenter or VideoSegmenter()
        self.video_evaluator = video_evaluator or VideoEvaluator()

    # ---------- 路径统一 ----------
    def _base_cache(self, dj: str) -> Path:
        """所有产物都落在 ./text_video_match/cache/{dj}/ """
        return Path(__file__).parent / "cache" / dj

    def _get_video_path(self, dj: str) -> str:
        video_path = Path(f"/mnt_liuhuan/search_public/xby/smart_video/clip/merge/{dj}_merged.mp4")
        if not video_path.exists():
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        return str(video_path)

    def _setup_directories(self, dj: str) -> Tuple[Path, Path]:
        base = self._base_cache(dj)
        seg_dir = base / "segment"
        seg_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"创建缓存目录: {base}")
        return base, seg_dir

    # ---------- 其余逻辑与你原文件完全一致 ----------
    def process(self, dj: str) -> Dict[str, Any]:
        try:
            logger.info(f"开始处理视频结构化理解，dj: {dj}")
            video_path = self._get_video_path(dj)
            cache_dir, seg_dir = self._setup_directories(dj)

            # 片段信息文件
            seg_info_file = seg_dir / "segment_infos.json"
            loaded = self._load_segment_infos(seg_info_file)
            if loaded is None:
                segment_infos = self._cut_and_save_segments(video_path, seg_dir, seg_info_file)
            else:
                segment_infos, statistics = loaded
                if statistics is None:
                    self._update_segment_info_statistics(seg_info_file, segment_infos)

            results = self._process_all_segments(segment_infos)
            output_file = self._save_results(results, cache_dir)
            return {"status": "success", "dj": dj, "total_segments": len(results), "output_file": str(output_file)}
        except Exception as e:
            logger.error(f"处理失败: {e}", exc_info=True)
            return {"status": "error", "dj": dj, "error_message": str(e)}

    # ---------- 以下全是原实现，仅把路径换成上面 base_cache ----------
    def _load_segment_infos(self, seg_info_file: Path):
        if not seg_info_file.exists():
            return None
        try:
            data = json.loads(seg_info_file.read_text(encoding="utf8"))
            if data.get("status") != "done":
                return None
            seg_infos = data.get("segment_infos", [])
            if not seg_infos:
                return None
            if self._check_segments_exist(seg_infos):
                return seg_infos, data.get("statistics")
            return None
        except Exception as e:
            logger.warning(f"加载 segment_infos 失败: {e}，将重新切割")
            return None

    def _check_segments_exist(self, seg_infos: List[Dict[str, Any]]) -> bool:
        for info in seg_infos:
            if not Path(info["video_path"]).exists():
                return False
        return True

    def _cut_and_save_segments(self, video_path: str, seg_dir: Path, seg_info_file: Path):
        logger.info("开始视频切割...")
        segment_infos = self.video_segmenter.cut_video_segments(video_path, str(seg_dir))
        if not segment_infos:
            raise ValueError("视频切割失败，未生成任何片段")
        statistics = self._calculate_statistics(segment_infos)
        seg_info_file.write_text(
            json.dumps({"status": "done", "statistics": statistics, "segment_infos": segment_infos},
                       ensure_ascii=False, indent=2), encoding="utf8")
        logger.info(f"片段信息已保存: {seg_info_file}")
        return segment_infos

    def _calculate_statistics(self, segment_infos: List[Dict[str, Any]]) -> Dict[str, Any]:
        durations = [s.get("duration", 0) for s in segment_infos if s.get("duration", 0) > 0]
        if not durations:
            return {"total_segments": len(segment_infos), "min_duration": 0, "max_duration": 0,
                    "avg_duration": 0, "duration_distribution": {"0-2秒": 0, "2-4秒": 0, "4-6秒": 0, "6-8秒": 0, "8-10秒及以上": 0}}
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

    def _update_segment_info_statistics(self, seg_info_file: Path, seg_infos: List[Dict[str, Any]]):
        statistics = self._calculate_statistics(seg_infos)
        data = json.loads(seg_info_file.read_text(encoding="utf8"))
        data["statistics"] = statistics
        seg_info_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf8")

    def _process_all_segments(self, seg_infos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results = []
        for idx, seg in enumerate(seg_infos, 1):
            results.append(self._process_single_segment(seg, idx, len(seg_infos)))
        return results

    def _process_single_segment(self, seg_info: Dict[str, Any], idx: int, total: int):
        seg_path = seg_info["video_path"]
        seg_id = seg_info["segment_index"]
        logger.info(f"处理片段 {idx}/{total}: {seg_path}")
        try:
            desc_result = self.video_evaluator.description_video(seg_path)
            return {
                "segment_id": seg_id,
                "duration": seg_info["duration"],
                "start_time": seg_info["start_time"],
                "end_time": seg_info["end_time"],
                "structured_description": desc_result.get("parsed_result", {}),
                "segment_file_path": seg_path,
            }
        except Exception as e:
            logger.error(f"处理片段 {idx} 错误: {e}")
            return {
                "segment_id": seg_id,
                "duration": seg_info["duration"],
                "start_time": seg_info["start_time"],
                "end_time": seg_info["end_time"],
                "structured_description": {"error": "处理失败", "error_message": str(e)},
                "segment_file_path": seg_path,
            }

    def _save_results(self, results: List[Dict[str, Any]], cache_dir: Path) -> Path:
        out_json = cache_dir / "structured_description.json"
        out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf8")
        logger.info(f"结构化描述已保存: {out_json}  共 {len(results)} 片段")
        return out_json


# -------------------- CLI --------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法: python skill_video_structured_description.py <dj>")
        sys.exit(1)
    dj = sys.argv[1]
    print(json.dumps(VideoStructuredDescription().process(dj), ensure_ascii=False, indent=2))