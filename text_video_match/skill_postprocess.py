#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频最后合成模块（索引化版本）
- 只处理前20秒的文案
- 逻辑：视频从start_time开始，时长=音频时长
"""
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VideoPostProcessor:
    def __init__(self):
        pass

    def _cache_dir(self, dj: str) -> Path:
        return Path(__file__).parent / "cache" / dj

    def _check_necessary_elements(self, dj: str, structured_text_file: str):
        """检查必要文件，只返回前20秒的文案"""
        cache = self._cache_dir(dj)

        text_path = Path(structured_text_file)
        if not text_path.exists():
            raise FileNotFoundError(f"文案JSON不存在: {structured_text_file}")
        
        with text_path.open(encoding="utf8") as f:
            text_data = json.load(f)

        filtered_data = []
        accumulated_time = 0.0
        
        for seg in text_data:
            audio_dur = seg.get("audio_duration", 3)
            
            if accumulated_time >= 20:
                logger.info(f"超过20秒，停止处理后续文案。共处理 {len(filtered_data)} 句")
                break
            
            accumulated_time += audio_dur
            
            seq = seg.get("segment_sequence", "unknown")
            match_result = seg.get("match_result", {})
            
            if isinstance(match_result, str):
                try:
                    match_result = json.loads(match_result)
                    seg["match_result"] = match_result
                except Exception as e:
                    raise ValueError(f"Segment {seq}: match_result字符串解析失败: {e}")
            
            if not isinstance(match_result, dict):
                raise ValueError(f"Segment {seq}: match_result不是字典")
                
            if match_result.get("status") != "success":
                raise ValueError(f"Segment {seq}: match_result状态不是success")
            
            if not match_result.get("best_match"):
                raise ValueError(f"Segment {seq}: 没有best_match")
            
            best_match = match_result["best_match"]
            if not best_match.get("segment_file_path"):
                raise ValueError(f"Segment {seq}: best_match没有segment_file_path")
            
            filtered_data.append(seg)

        return filtered_data

    def _get_segment_video_path(self, seg_info: Dict, dj: str) -> Optional[Path]:
        """获取片段对应的视频文件路径"""
        video_path = seg_info.get("segment_file_path") or seg_info.get("video_path")
        
        if video_path and Path(video_path).exists():
            return Path(video_path)
        
        # 尝试构造路径
        episode = seg_info.get("episode")
        seg_id = seg_info.get("segment_id")
        
        if episode is not None and seg_id is not None:
            cache = self._cache_dir(dj)
            constructed_path = cache / "episode_index" / f"ep_{episode}" / "segments" / f"{seg_id}.mp4"
            if constructed_path.exists():
                return constructed_path
        
        return None

    def _clip_video_segment(self, 
                           video_path: Path, 
                           start_time: float, 
                           duration: float, 
                           output_path: Path) -> bool:
        """
        从视频片段中裁剪指定时间段
        关键：从start_time开始，裁剪duration时长（等于音频时长）
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if duration <= 0:
            logger.warning(f"无效duration: {duration}，使用默认3秒")
            duration = 3.0
        
        cmd = [
            "ffmpeg", 
            "-ss", str(start_time),
            "-i", str(video_path),
            "-t", str(duration),
            "-c:v", "libx264",
            "-c:a", "aac",
            "-avoid_negative_ts", "make_zero", 
            "-y", 
            str(output_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"ffmpeg裁剪失败: {result.stderr}")
            return False
        
        return True

    def _combine_video_audio(self, video_path: Path, audio_path: Path, output_path: Path) -> bool:
        """合并视频和音频，以音频长度为准"""
        if not audio_path.exists():
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")
        
        cmd = [
            "ffmpeg", 
            "-i", str(video_path), 
            "-i", str(audio_path),
            "-c:v", "copy", 
            "-c:a", "aac", 
            "-map", "0:v:0", 
            "-map", "1:a:0",
            "-shortest",  # 以较短的为准（这里应该是音频）
            "-y", 
            str(output_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0

    def _merge_video_segments(self, segments: List[Path], output_path: Path) -> bool:
        """合并多个视频片段"""
        if not segments:
            raise RuntimeError("没有视频片段需要合并")
        
        # 按序号排序
        segments.sort(key=lambda x: int(x.stem.split("_")[1]))
        
        list_file = output_path.with_suffix(".txt")
        with list_file.open("w", encoding="utf8") as f:
            for seg in segments:
                f.write(f"file '{seg.absolute()}'\n")
        
        cmd = [
            "ffmpeg", 
            "-f", "concat", 
            "-safe", "0", 
            "-i", str(list_file),
            "-c", "copy", 
            "-avoid_negative_ts", "make_zero", 
            "-y", 
            str(output_path)
        ]
        
        ok = subprocess.run(cmd, stdout=None, stderr=None).returncode == 0
        list_file.unlink(missing_ok=True)
        return ok

    def process(self, dj: str, structured_text_file: str, output_filename: Optional[str] = None) -> Dict[str, Any]:
        try:
            logger.info(f"开始视频合成，dj: {dj}，只处理前20秒")
            cache = self._cache_dir(dj)
            finish_mp4 = Path(__file__).parent.parent / "finish" / (output_filename or f"{dj}_final.mp4")
            finish_mp4.parent.mkdir(exist_ok=True)

            text_data = self._check_necessary_elements(dj, structured_text_file)
            total_segments = len(text_data)
            logger.info(f"共 {total_segments} 个文案片段需要处理（前20秒）")

            # 2. 逐段处理
            seg_out_dir = cache / "processed_segments"
            final_segments = []
            
            for idx, seg in enumerate(text_data, 1):
                seq = seg["segment_sequence"]
                match_result = seg.get("match_result", {})
                
                # 获取best_match信息
                best_match = match_result.get("best_match", {})
                seg_id = best_match.get("segment_id")
                
                # 获取视频文件路径
                video_path = self._get_segment_video_path(best_match, dj)
                if not video_path:
                    raise RuntimeError(f"Segment {seq}: 找不到视频文件")
                
                # 关键：获取时间信息
                start_time = best_match.get("start_time", 0)
                # 音频信息
                audio_path = cache / seg.get("audio_path", "")
                audio_dur = seg.get("audio_duration", 3.0)  # 默认3秒
                
                if not audio_path.exists():
                    raise RuntimeError(f"Segment {seq}: 音频文件不存在 {audio_path}")
                
                # 关键逻辑：视频从start_time开始，时长=音频时长
                # 这样能保证说完一整句文案才换画面
                duration = audio_dur
                
                logger.info(f"segment {seq}/{total_segments}: 从EP{best_match.get('episode')}片段{seg_id}裁剪 [{start_time:.1f}, {start_time+duration:.1f}], 音频时长={duration:.1f}s")

                # 裁剪视频（无声）
                seg_v_na = seg_out_dir / f"segment_{seq:04d}_no_audio.mp4"
                if not self._clip_video_segment(video_path, start_time, duration, seg_v_na):
                    raise RuntimeError(f"Segment {seq}: 视频裁剪失败")

                # 合并音频
                seg_v_final = seg_out_dir / f"segment_{seq:04d}_final.mp4"
                if not self._combine_video_audio(seg_v_na, audio_path, seg_v_final):
                    raise RuntimeError(f"Segment {seq}: 合并音频失败")
                
                final_segments.append(seg_v_final)
                logger.info(f"segment {seq} 处理成功")

            if not final_segments:
                raise RuntimeError("没有成功处理任何片段")
            
            if len(final_segments) != total_segments:
                raise RuntimeError(f"处理不完整: 期望{total_segments}个，实际{len(final_segments)}个")

            # 3. 按顺序合并
            if not self._merge_video_segments(final_segments, finish_mp4):
                raise RuntimeError("最终合并失败")

            logger.info(f"视频合成完成: {finish_mp4}, 共{len(final_segments)}个片段")
            return {
                "status": "success", 
                "dj": dj, 
                "output_path": str(finish_mp4), 
                "total_segments": len(final_segments)
            }

        except Exception as e:
            logger.error(f"合成失败: {e}", exc_info=True)
            return {"status": "error", "error_message": str(e)}


# -------------------- CLI --------------------
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python skill_postprocess.py <dj> <structured_text_json> [output_filename]")
        sys.exit(1)
    dj, json_file = sys.argv[1], sys.argv[2]
    out_file = sys.argv[3] if len(sys.argv) > 3 else None
    print(json.dumps(VideoPostProcessor().process(dj, json_file, out_file), ensure_ascii=False, indent=2))