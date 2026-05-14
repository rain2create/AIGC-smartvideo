#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频最后合成模块  →  去 ./text_video_match/cache/{dj}/ 拿 JSON 和 TTS
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

    # ---------- 统一缓存根 ----------
    def _cache_dir(self, dj: str) -> Path:
        return Path(__file__).parent / "cache" / dj

    # ---------- 视频路径 ----------
    def _get_video_path(self, dj: str) -> str:
        video_path = Path(f"/mnt_liuhuan/search_public/xby/smart_video/clip/merge/{dj}_merged.mp4")
        if not video_path.exists():
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        return str(video_path)

    # ---------- 检查必要文件 ----------
    def _check_necessary_elements(self, dj: str, structured_text_file: str):
        cache = self._cache_dir(dj)

        # structured_description.json
        structured_json = cache / "structured_description.json"
        if not structured_json.exists():
            raise FileNotFoundError(f"结构化描述文件不存在: {structured_json}")
        
        with structured_json.open(encoding="utf8") as f:
            desc_data = json.load(f)
        
        # 调试：打印数据结构
        logger.info(f"结构化描述数据类型: {type(desc_data)}")
        if isinstance(desc_data, list) and len(desc_data) > 0:
            logger.info(f"第一个元素的类型: {type(desc_data[0])}")
            logger.info(f"第一个元素的内容: {desc_data[0]}")

        # 文案 JSON
        text_path = Path(structured_text_file)
        if not text_path.exists():
            raise FileNotFoundError(f"文案JSON不存在: {structured_text_file}")
        with text_path.open(encoding="utf8") as f:
            text_data = json.load(f)

        # 检查每段状态
        failed_segments = []
        for seg in text_data:
            match_result = seg.get("match_result")
            
            # 如果是字符串，尝试解析
            if isinstance(match_result, str):
                try:
                    match_result = json.loads(match_result)
                    # 更新回原始数据
                    seg["match_result"] = match_result
                except Exception as e:
                    logger.warning(f"segment {seg.get('segment_sequence', 'unknown')} match_result 字符串解析失败: {e}")
                    failed_segments.append(seg)
                    continue
            
            # 检查是否为字典且状态为 success
            if not isinstance(match_result, dict):
                logger.warning(f"segment {seg.get('segment_sequence', 'unknown')} match_result 不是字典类型")
                failed_segments.append(seg)
                continue
                
            status = match_result.get("status")
            if status != "success":
                failed_segments.append(seg)

        if failed_segments:
            logger.warning(f"有 {len(failed_segments)} 个文案片段匹配状态不是 success")

        return self._get_video_path(dj), desc_data, text_data

    # ---------- ffmpeg 操作 ----------
    def _remove_video_audio(self, video_path: str, output_path: Path) -> bool:
        cmd = ["ffmpeg", "-i", video_path, "-c:v", "copy", "-an", "-y", str(output_path)]
        return subprocess.run(cmd, stdout=None, stderr=None).returncode == 0

    def _clip_video_segment(self, video_path: str, start_time: float, duration: float, output_path: Path) -> bool:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = ["ffmpeg", "-ss", str(start_time), "-i", video_path, "-t", str(duration),
               "-c:v", "copy", "-an", "-avoid_negative_ts", "make_zero", "-y", str(output_path)]
        return subprocess.run(cmd, stdout=None, stderr=None).returncode == 0

    def _combine_video_audio(self, video_path: Path, audio_path: Path, output_path: Path) -> bool:
        if not audio_path.exists():
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")
        cmd = ["ffmpeg", "-i", str(video_path), "-i", str(audio_path),
               "-c:v", "copy", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0",
               "-shortest", "-y", str(output_path)]
        return subprocess.run(cmd, stdout=None, stderr=None).returncode == 0

    def _merge_video_segments(self, segments: List[Path], output_path: Path) -> bool:
        if not segments:
            raise RuntimeError("没有视频片段需要合并")
        list_file = output_path.with_suffix(".txt")
        with list_file.open("w", encoding="utf8") as f:
            for seg in segments:
                f.write(f"file '{seg.absolute()}'\n")
        cmd = ["ffmpeg", "-f", "concat", "-safe", "0", "-i", str(list_file),
               "-c", "copy", "-avoid_negative_ts", "make_zero", "-y", str(output_path)]
        ok = subprocess.run(cmd, stdout=None, stderr=None).returncode == 0
        list_file.unlink(missing_ok=True)
        return ok

    # ---------- 查找视频片段信息 ----------
    def _find_segment_info(self, desc_data, seg_id):
        """查找视频片段信息，处理不同的数据结构"""
        # 如果 desc_data 是字典，可能包含 segments 字段
        if isinstance(desc_data, dict):
            segments = desc_data.get("segments", [])
            return next((s for s in segments if isinstance(s, dict) and s.get("segment_id") == seg_id), None)
        
        # 如果 desc_data 是列表
        elif isinstance(desc_data, list):
            return next((s for s in desc_data if isinstance(s, dict) and s.get("segment_id") == seg_id), None)
        
        # 其他情况
        return None

    # ---------- 主业务 ----------
    def process(self, dj: str, structured_text_file: str, output_filename: Optional[str] = None) -> Dict[str, Any]:
        try:
            logger.info(f"开始视频合成，dj: {dj}")
            cache = self._cache_dir(dj)
            finish_mp4 = Path(__file__).parent.parent / "finish" / (output_filename or f"{dj}_final.mp4")
            finish_mp4.parent.mkdir(exist_ok=True)

            # 1. 检查必要文件
            video_path, desc_data, text_data = self._check_necessary_elements(dj, structured_text_file)

            # 2. 去音频
            no_audio = cache / "video_no_audio.mp4"
            if not self._remove_video_audio(video_path, no_audio):
                raise RuntimeError("移除原音频失败")

            # 3. 逐段裁剪+合并音频
            seg_out_dir = cache / "processed_segments"
            final_segments = []
            
            for seg in text_data:
                seq = seg["segment_sequence"]
                audio_path = cache / seg["audio_path"]
                dur = seg["audio_duration"]

                # 正确处理 match_result
                match_result = seg.get("match_result")
                
                # 如果 match_result 是字符串，尝试解析
                if isinstance(match_result, str):
                    try:
                        match_result = json.loads(match_result)
                    except Exception:
                        logger.warning(f"segment {seq} match_result 字符串解析失败，跳过该片段")
                        continue
                
                # 如果 match_result 是 None 或者不是字典，跳过
                if not match_result or not isinstance(match_result, dict):
                    logger.warning(f"segment {seq} match_result 无效，跳过该片段")
                    continue
                    
                if match_result.get("status") != "success" or not match_result.get("top_matches"):
                    logger.warning(f"segment {seq} 状态不是 success 或无匹配，跳过")
                    continue

                seg_id = match_result["top_matches"][0]["segment_id"]
                
                # 使用新的查找方法
                seg_info = self._find_segment_info(desc_data, seg_id)
                
                if not seg_info:
                    logger.warning(f"未找到片段 {seq} 对应的视频信息（segment_id: {seg_id}），跳过")
                    continue
                    
                start_time = seg_info.get("start_time")
                if start_time is None:
                    logger.warning(f"片段 {seq} 没有 start_time，跳过")
                    continue

                # 裁剪
                seg_v_na = seg_out_dir / f"segment_{seq:04d}_no_audio.mp4"
                if not self._clip_video_segment(str(no_audio), start_time, dur, seg_v_na):
                    logger.error(f"片段 {seq} 裁剪失败")
                    continue

                # 合并音频
                seg_v_final = seg_out_dir / f"segment_{seq:04d}_final.mp4"
                if not self._combine_video_audio(seg_v_na, audio_path, seg_v_final):
                    logger.error(f"片段 {seq} 合并音频失败")
                    continue

                final_segments.append(seg_v_final)
                logger.info(f"片段 {seq} 处理成功")

            if not final_segments:
                raise RuntimeError("没有成功处理任何片段")

            # 4. 合并成片
            final_segments.sort(key=lambda x: int(x.stem.split("_")[1]))
            if not self._merge_video_segments(final_segments, finish_mp4):
                raise RuntimeError("最终合并失败")

            logger.info(f"视频合成完成: {finish_mp4}")
            return {"status": "success", "dj": dj, "output_path": str(finish_mp4), "total_segments": len(final_segments)}

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