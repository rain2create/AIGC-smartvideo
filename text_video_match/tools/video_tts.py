#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频重新合成脚本
读取 mp4 和 srt 文件，对解说类型的文本重新生成 TTS，并替换原音频片段 ！！！  srt为字幕文件
"""

import os
import re
import asyncio
import subprocess
import edge_tts
import pysrt
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import logging
import httpx

logger = logging.getLogger(__name__)

# 默认配置
DEFAULT_VOICE = "zh-CN-YunxiNeural"
DEFAULT_RATE = "+15%"  # 语速
MAX_RETRIES = 3  # 最大重试次数
RETRY_DELAY = 2  # 重试延迟（秒）


class VideoTTSProcessor:
    """视频TTS处理器，用于重新生成配音并替换原音频"""
    
    def __init__(self, voice: str = DEFAULT_VOICE, rate: str = DEFAULT_RATE):
        """
        初始化TTS处理器
        
        Args:
            voice: 语音名称，默认 zh-CN-YunxiNeural
            rate: 语速，默认 +15%
        """
        self.voice = voice
        self.rate = rate
        logger.info(f"VideoTTSProcessor 初始化完成: voice={voice}, rate={rate}")
    
    def parse_srt_file(self, srt_path: Path) -> List[Tuple[int, str, str, float, float, bool]]:
        """
        使用 pysrt 解析 srt 文件，返回所有字幕信息
        
        Args:
            srt_path: srt 文件路径
            
        Returns:
            list: 包含 (index, text, subtitle_type, start_time, end_time, is_narration) 的列表
            subtitle_type: "解说" 或 "原声"
            is_narration: True 表示需要 TTS，False 表示使用原音频
        """
        subtitles = []
        
        try:
            subs = pysrt.open(str(srt_path), encoding='utf-8')
            
            for sub in subs:
                text_content = sub.text.strip()
                
                # 判断类型
                is_narration = "类型：解说" in text_content
                subtitle_type = "解说" if is_narration else "原声"
                
                # 提取实际文本内容
                lines = text_content.split('\n')
                actual_text_lines = []
                
                for line in lines:
                    line = line.strip()
                    if line.startswith("类型：") or not line:
                        continue
                    actual_text_lines.append(line)
                
                actual_text = ' '.join(actual_text_lines) if actual_text_lines else ""
                
                # 获取时间（转换为秒）
                start_seconds = sub.start.ordinal / 1000.0
                end_seconds = sub.end.ordinal / 1000.0
                
                subtitles.append((
                    sub.index,
                    actual_text,
                    subtitle_type,
                    start_seconds,
                    end_seconds,
                    is_narration
                ))
            
            # 按索引排序
            subtitles.sort(key=lambda x: x[0])
            
        except Exception as e:
            logger.error(f"解析 srt 文件时出错: {e}")
            raise
        
        return subtitles

    def extract_audio_from_video(self, video_path: Path, output_audio_path: Path) -> bool:
        """
        从视频中提取音频（转换为统一格式以便后续处理）
        
        Args:
            video_path: 视频文件路径
            output_audio_path: 输出音频文件路径
            
        Returns:
            bool: 是否成功
        """
        try:
            # 统一转换为 wav 格式，便于后续切割和拼接
            cmd = [
                'ffmpeg', '-i', str(video_path),
                '-vn',
                '-acodec', 'pcm_s16le',  # PCM 16-bit little-endian
                '-ar', '44100',  # 采样率 44.1kHz
                '-ac', '2',  # 立体声
                '-y',  # 覆盖输出文件
                str(output_audio_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info(f"已提取音频: {output_audio_path.name}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"提取音频失败: {e.stderr}")
            return False
        except FileNotFoundError:
            logger.error("错误: 未找到 ffmpeg，请先安装 ffmpeg")
            return False


    def generate_silent_audio(self, output_audio: Path, duration: float) -> bool:
        """
        生成空白音频（静音）
        
        Args:
            output_audio: 输出音频文件路径
            duration: 时长（秒）
            
        Returns:
            bool: 是否成功
        """
        try:
            cmd = [
                'ffmpeg', '-f', 'lavfi',
                '-i', f'anullsrc=channel_layout=stereo:sample_rate=44100',
                '-t', str(duration),
                '-acodec', 'pcm_s16le',
                '-ar', '44100',
                '-ac', '2',
                '-y',
                str(output_audio)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"生成空白音频失败: {e.stderr[:200]}")
            return False


    def cut_audio_segment(self, input_audio: Path, output_segment: Path, start_time: float, duration: float) -> bool:
        """
        切割音频片段（保持格式一致）
        
        Args:
            input_audio: 输入音频文件
            output_segment: 输出片段文件
            start_time: 开始时间（秒）
            duration: 持续时间（秒）
            
        Returns:
            bool: 是否成功
        """
        try:
            # 使用相同的编码格式，确保格式一致
            cmd = [
                'ffmpeg', '-i', str(input_audio),
                '-ss', str(start_time),
                '-t', str(duration),
                '-acodec', 'pcm_s16le',
                '-ar', '44100',
                '-ac', '2',
                '-y',
                str(output_segment)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"切割音频失败 ({start_time:.3f}s-{start_time + duration:.3f}s): {e.stderr[:200]}")
            return False

    async def text_to_speech(self, text: str, output_path: Path) -> bool:
        """
        使用 edge-tts 将文本转换为语音（带重试机制）
        
        Args:
            text: 要转换的文本
            output_path: 输出音频文件路径
            
        Returns:
            bool: 是否成功
        """
        last_error = None
        
        for attempt in range(MAX_RETRIES + 1):
            try:
                communicate = edge_tts.Communicate(text, self.voice, rate=self.rate)
                await communicate.save(str(output_path))
                
                if attempt > 0:
                    logger.info(f"已生成 TTS: {output_path.name} (重试 {attempt} 次后成功)")
                else:
                    logger.info(f"已生成 TTS: {output_path.name}")
                return True
                
            except Exception as e:
                last_error = e
                error_msg = str(e)
                
                is_network_error = (
                    "Connection timeout" in error_msg or
                    "timeout" in error_msg.lower() or
                    "Connection" in error_msg or
                    "network" in error_msg.lower() or
                    "connect" in error_msg.lower()
                )
                
                if attempt < MAX_RETRIES and is_network_error:
                    logger.warning(f"第 {attempt + 1} 次尝试失败 ({error_msg})，{RETRY_DELAY} 秒后重试...")
                    await asyncio.sleep(RETRY_DELAY)
                    continue
                else:
                    if attempt < MAX_RETRIES and not is_network_error:
                        logger.error(f"生成失败 {output_path.name}: {error_msg} (非网络错误，不重试)")
                    else:
                        logger.error(f"生成失败 {output_path.name}: {error_msg} (已重试 {MAX_RETRIES} 次)")
                    return False
        
        return False



    async def text_to_speech_transfer(
        self, text: str, output_path: Path, s3_object_name: Optional[str] = None
    ) -> Optional[float]:
        """
        本机直连 edge-tts，不再走任何中转。
        返回音频时长（秒）；失败返回 None。
        """
        try:
            # 1. 本地 edge-tts 直接生成
            communicate = edge_tts.Communicate(text, self.voice, rate=self.rate)
            await communicate.save(str(output_path))

            # 2. 取时长
            duration = self.get_audio_duration(output_path)
            logger.info(
                f"edge-tts 本机生成成功: {output_path.name}  "
                f"时长={duration:.3f}s" if duration else "时长未知"
            )
            return duration if duration else 0.0   # 0.0 表示文件已落盘但没拿到时长
        except Exception as e:
            logger.error(f"edge-tts 本机生成失败: {e}")
            return None
    

    def get_audio_duration(self, audio_path: Path) -> Optional[float]:
        """
        获取音频文件的时长（秒）
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            float: 音频时长（秒），失败返回 None
        """
        try:
            cmd = [
                'ffprobe', '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                str(audio_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            duration = float(result.stdout.strip())
            return duration
        except (subprocess.CalledProcessError, ValueError) as e:
            logger.error(f"获取音频时长失败: {e}")
            return None


    def adjust_audio_speed(self, input_audio: Path, output_audio: Path, target_duration: float) -> bool:
        """
        调整音频速度以匹配目标时长
        
        Args:
            input_audio: 输入音频文件
            output_audio: 输出音频文件
            target_duration: 目标时长（秒）
            
        Returns:
            bool: 是否成功
        """
        # 获取当前音频时长
        current_duration = self.get_audio_duration(input_audio)
        if current_duration is None:
            return False
        
        # 如果时长已经很接近（误差小于 0.1 秒），直接复制
        if abs(current_duration - target_duration) < 0.1:
            try:
                import shutil
                shutil.copy2(input_audio, output_audio)
                return True
            except Exception as e:
                logger.error(f"复制音频失败: {e}")
                return False
        
        # 计算速度比例
        speed_ratio = current_duration / target_duration
        
        logger.info(f"当前音频时长: {current_duration:.3f}s, 目标时长: {target_duration:.3f}s, 需要调整速度: {speed_ratio:.3f}x")
        
        try:
            # 使用 atempo 滤镜调整速度（atempo 范围是 0.5-2.0，如果需要更大范围需要链式使用）
            atempo_filters = []
            remaining = speed_ratio
            
            if remaining < 0.5:
                # 需要减速（多次链式应用 atempo，每次最多 0.5x）
                while remaining < 0.5:
                    atempo_filters.append('atempo=0.5')
                    remaining *= 2
                # 如果还有剩余部分，继续处理
                if remaining != 1.0:
                    atempo_filters.append(f'atempo={remaining:.4f}')
            elif remaining > 2.0:
                # 需要加速（多次链式应用 atempo，每次最多 2.0x）
                while remaining > 2.0:
                    atempo_filters.append('atempo=2.0')
                    remaining /= 2
                # 如果还有剩余部分，继续处理
                if remaining != 1.0:
                    atempo_filters.append(f'atempo={remaining:.4f}')
            else:
                # 在正常范围内（0.5-2.0），直接使用
                atempo_filters.append(f'atempo={remaining:.4f}')
            
            atempo_filter = ','.join(atempo_filters)
            
            cmd = [
                'ffmpeg', '-i', str(input_audio),
                '-filter:a', atempo_filter,
                '-acodec', 'pcm_s16le',
                '-ar', '44100',
                '-ac', '2',
                '-y',
                str(output_audio)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # 验证调整后的时长
            adjusted_duration = self.get_audio_duration(output_audio)
            if adjusted_duration:
                diff = abs(adjusted_duration - target_duration)
                logger.info(f"音频时长调整: {current_duration:.3f}s -> {adjusted_duration:.3f}s (目标: {target_duration:.3f}s, 误差: {diff:.3f}s)")
            
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"调整音频速度失败: {e.stderr[:200]}")
            return False


    def convert_audio_format(self, input_audio: Path, output_audio: Path, target_format: str = "wav") -> bool:
        """
        转换音频格式（用于统一格式以便拼接）
        
        Args:
            input_audio: 输入音频文件
            output_audio: 输出音频文件
            target_format: 目标格式
            
        Returns:
            bool: 是否成功
        """
        try:
            cmd = [
                'ffmpeg', '-i', str(input_audio),
                '-acodec', 'pcm_s16le',
                '-ar', '44100',
                '-ac', '2',
                '-y',
                str(output_audio)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"转换音频格式失败: {e.stderr}")
            return False


    def replace_video_audio(self, video_path: Path, audio_path: Path, output_video: Path) -> bool:
        """
        将新音频替换到视频中，生成新视频
        
        Args:
            video_path: 原始视频文件路径
            audio_path: 新音频文件路径
            output_video: 输出视频文件路径
            
        Returns:
            bool: 是否成功
        """
        try:
            # 使用 -c:v copy 保持视频编码不变，只替换音频
            # 使用 -c:a aac 或 copy 来编码音频
            cmd = [
                'ffmpeg', '-i', str(video_path),
                '-i', str(audio_path),
                '-c:v', 'copy',  # 视频流直接复制，不重新编码
                '-c:a', 'aac',   # 音频编码为 AAC
                '-map', '0:v:0', # 使用第一个输入（视频）的视频流
                '-map', '1:a:0', # 使用第二个输入（音频）的音频流
                '-shortest',      # 以最短的流为准
                '-y',
                str(output_video)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info(f"已生成新视频: {output_video.name}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"替换视频音频失败: {e.stderr[:500]}")
            return False


    def concatenate_audio_segments(self, segment_files: List[Path], output_audio: Path) -> bool:
        """
        拼接多个音频片段
        
        Args:
            segment_files: 音频片段文件列表（按顺序）
            output_audio: 输出音频文件
            
        Returns:
            bool: 是否成功
        """
        if not segment_files:
            logger.error("没有音频片段可拼接")
            return False
        
        try:
            # 创建文件列表
            file_list_path = output_audio.parent / "concat_list.txt"
            with open(file_list_path, 'w', encoding='utf-8') as f:
                for segment in segment_files:
                    if segment.exists():
                        # 使用绝对路径，并转义特殊字符
                        abs_path = segment.resolve()
                        f.write(f"file '{abs_path}'\n")
            
            # 使用 concat demuxer 拼接
            cmd = [
                'ffmpeg', '-f', 'concat', '-safe', '0',
                '-i', str(file_list_path),
                '-acodec', 'copy',
                '-y',
                str(output_audio)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # 清理临时文件
            if file_list_path.exists():
                file_list_path.unlink()
            
            logger.info(f"已拼接音频: {output_audio}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"拼接音频失败: {e.stderr}")
            return False


    def split_long_subtitles_by_sentences(
        self, 
        input_srt_path: Path, 
        output_srt_path: Path,
        sentence_delimiters: Optional[List[str]] = None
    ) -> bool:
        """
        将长字幕按句子拆分，根据字符数比例分配时间
        
        示例：
            原始SRT条目：
            1
            00:00:00,180 --> 00:00:09,660
            类型：解说
            新娘脱下嫁衣和长裤时，宾客们才觉察到她已怀孕9个多月。但她为了取消婚事，竟做了出人意料的举动。网购十元的耳饰。
            
            拆分后（按字符数比例分配时间）：
            1
            00:00:00,180 --> 00:00:04,920
            类型：解说
            新娘脱下嫁衣和长裤时，宾客们才觉察到她已怀孕9个多月。
            
            2
            00:00:04,920 --> 00:00:08,200
            类型：解说
            但她为了取消婚事，竟做了出人意料的举动。
            
            3
            00:00:08,200 --> 00:00:09,660
            类型：解说
            网购十元的耳饰。
        
        Args:
            input_srt_path: 输入SRT文件路径
            output_srt_path: 输出SRT文件路径
            sentence_delimiters: 句子分隔符列表，默认 ['。', '！', '？', '.', '!', '?']
        
        Returns:
            bool: 是否成功
        """
        if sentence_delimiters is None:
            sentence_delimiters = ['。', '！', '？', '.', '!', '?']
        
        try:
            # 解析原始SRT
            entries = self._parse_srt_entries(input_srt_path)
            
            # 拆分长字幕
            new_entries = []
            new_index = 1
            
            for entry in entries:
                text = entry['text']
                start_time = entry['start_time']
                end_time = entry['end_time']
                total_duration = end_time - start_time
                
                # 移除"类型："行，提取实际文本
                text_lines = text.split('\n')
                actual_text = ''
                subtitle_type = None
                
                for line in text_lines:
                    if line.strip().startswith('类型：'):
                        subtitle_type = line.strip().replace('类型：', '').strip()
                    elif line.strip() and not line.strip().startswith('类型：'):
                        if actual_text:
                            actual_text += ' ' + line.strip()
                        else:
                            actual_text = line.strip()
                
                # 按句子分隔符拆分文本
                sentences = self._split_text_into_sentences(actual_text, sentence_delimiters)
                
                if len(sentences) <= 1:
                    # 只有一个句子或无法拆分，保持原样
                    new_entries.append({
                        'index': new_index,
                        'start_time': start_time,
                        'end_time': end_time,
                        'text': text,
                        'type': subtitle_type
                    })
                    new_index += 1
                else:
                    # 多个句子，按字符数比例分配时间
                    # 计算每个句子的字符数
                    sentence_lengths = [len(s.strip()) for s in sentences if s.strip()]
                    total_chars = sum(sentence_lengths)
                    
                    if total_chars == 0:
                        # 如果没有有效字符，平均分配
                        sentence_duration = total_duration / len(sentences)
                        current_start = start_time
                        
                        for i, sentence in enumerate(sentences):
                            sentence = sentence.strip()
                            if not sentence:
                                continue
                            
                            if i == len(sentences) - 1:
                                current_end = end_time
                            else:
                                current_end = current_start + sentence_duration
                            
                            if subtitle_type:
                                entry_text = f"类型：{subtitle_type}\n{sentence}"
                            else:
                                entry_text = sentence
                            
                            new_entries.append({
                                'index': new_index,
                                'start_time': current_start,
                                'end_time': current_end,
                                'text': entry_text,
                                'type': subtitle_type
                            })
                            
                            current_start = current_end
                            new_index += 1
                    else:
                        # 按字符数比例分配时间
                        current_start = start_time
                        sentence_idx = 0
                        
                        for i, sentence in enumerate(sentences):
                            sentence = sentence.strip()
                            if not sentence:
                                continue
                            
                            # 计算当前句子的时长（按字符数比例）
                            char_count = sentence_lengths[sentence_idx]
                            sentence_duration = total_duration * (char_count / total_chars)
                            
                            if i == len(sentences) - 1:
                                # 最后一个句子，使用原始结束时间（避免时间误差）
                                current_end = end_time
                            else:
                                current_end = current_start + sentence_duration
                            
                            # 构建文本（包含类型信息）
                            if subtitle_type:
                                entry_text = f"类型：{subtitle_type}\n{sentence}"
                            else:
                                entry_text = sentence
                            
                            new_entries.append({
                                'index': new_index,
                                'start_time': current_start,
                                'end_time': current_end,
                                'text': entry_text,
                                'type': subtitle_type
                            })
                            
                            current_start = current_end
                            sentence_idx += 1
                            new_index += 1
            
            # 生成新的SRT文件
            self._write_srt_entries(new_entries, output_srt_path)
            
            logger.info(f"字幕拆分完成: {len(entries)} 个条目 -> {len(new_entries)} 个条目")
            return True
            
        except Exception as e:
            logger.error(f"拆分字幕失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def _split_text_into_sentences(self, text: str, delimiters: List[str]) -> List[str]:
        """
        按分隔符将文本拆分成句子
        
        Args:
            text: 要拆分的文本
            delimiters: 分隔符列表
        
        Returns:
            句子列表
        """
        import re
        
        if not text.strip():
            return []
        
        # 构建正则表达式：匹配分隔符（保留分隔符）
        pattern = '|'.join(re.escape(d) for d in delimiters)
        # 使用正向前瞻，保留分隔符
        pattern = f'({pattern})'
        
        # 拆分文本
        parts = re.split(pattern, text)
        
        sentences = []
        current_sentence = ''
        
        for part in parts:
            if not part:
                continue
            
            current_sentence += part
            
            # 如果当前部分包含分隔符，说明句子结束
            if any(delimiter in part for delimiter in delimiters):
                if current_sentence.strip():
                    sentences.append(current_sentence.strip())
                current_sentence = ''
        
        # 处理最后剩余的文本
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
        
        return sentences
    
    def _parse_srt_entries(self, srt_path: Path) -> List[Dict]:
        """
        解析SRT文件为条目列表
        
        Args:
            srt_path: SRT文件路径
        
        Returns:
            条目列表，每个条目包含 index, start_time, end_time, text
        """
        entries = []
        
        with open(srt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.strip().split('\n')
        i = 0
        
        while i < len(lines):
            # 跳过空行
            if not lines[i].strip():
                i += 1
                continue
            
            try:
                # 提取索引
                index = int(lines[i].strip())
                i += 1
                
                # 提取时间行
                if i >= len(lines):
                    break
                time_line = lines[i].strip()
                i += 1
                
                # 解析时间
                start_time, end_time = self._parse_srt_time(time_line)
                
                # 提取文本（可能多行）
                text_lines = []
                while i < len(lines) and lines[i].strip():
                    text_lines.append(lines[i].strip())
                    i += 1
                
                text = '\n'.join(text_lines)
                
                entries.append({
                    'index': index,
                    'start_time': start_time,
                    'end_time': end_time,
                    'text': text
                })
                
            except (ValueError, IndexError) as e:
                logger.warning(f"解析SRT条目失败: {e}")
                i += 1
                continue
        
        return entries
    
    def _parse_srt_time(self, time_line: str) -> Tuple[float, float]:
        """
        解析SRT时间行为秒数
        
        Args:
            time_line: 时间行，格式 "00:00:00,180 --> 00:00:09,660"
        
        Returns:
            (start_time, end_time) 元组，单位为秒
        """
        parts = time_line.split('-->')
        if len(parts) != 2:
            raise ValueError(f"无效的时间格式: {time_line}")
        
        start_str = parts[0].strip()
        end_str = parts[1].strip()
        
        start_time = self._srt_time_to_seconds_helper(start_str)
        end_time = self._srt_time_to_seconds_helper(end_str)
        
        return start_time, end_time
    
    def _srt_time_to_seconds_helper(self, time_str: str) -> float:
        """
        将SRT时间格式转换为秒数
        
        Args:
            time_str: 时间字符串，格式 "00:00:00,180"
        
        Returns:
            秒数
        """
        # 格式：HH:MM:SS,mmm
        parts = time_str.split(':')
        hours = int(parts[0])
        minutes = int(parts[1])
        secs_parts = parts[2].split(',')
        secs = int(secs_parts[0])
        milliseconds = int(secs_parts[1])
        
        return hours * 3600 + minutes * 60 + secs + milliseconds / 1000.0
    
    def _seconds_to_srt_time_helper(self, seconds: float) -> str:
        """
        将秒数转换为SRT时间格式
        
        Args:
            seconds: 秒数
        
        Returns:
            时间字符串，格式 "00:00:00,180"
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"
    
    def _write_srt_entries(self, entries: List[Dict], output_path: Path) -> None:
        """
        将条目列表写入SRT文件
        
        Args:
            entries: 条目列表
            output_path: 输出文件路径
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in entries:
                f.write(f"{entry['index']}\n")
                start_str = self._seconds_to_srt_time_helper(entry['start_time'])
                end_str = self._seconds_to_srt_time_helper(entry['end_time'])
                f.write(f"{start_str} --> {end_str}\n")
                f.write(f"{entry['text']}\n")
                f.write("\n")
    
    def _clean_srt_for_subtitle(self, input_srt: Path, output_srt: Path) -> None:
        """
        清理SRT文件，移除"类型："行，只保留实际文本内容（用于字幕渲染）
        
        Args:
            input_srt: 输入SRT文件路径
            output_srt: 输出SRT文件路径
        """
        with open(input_srt, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        cleaned_lines = []
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # 如果是序号行或时间行，直接保留
            if line.strip().isdigit() or '-->' in line:
                cleaned_lines.append(line)
                i += 1
                continue
            
            # 如果是文本行
            if line.strip():
                # 跳过"类型："行
                if not line.strip().startswith('类型：'):
                    cleaned_lines.append(line)
            else:
                # 空行保留
                cleaned_lines.append(line)
            
            i += 1
        
        with open(output_srt, 'w', encoding='utf-8') as f:
            f.writelines(cleaned_lines)
    
    def render_subtitles_to_video(
        self, 
        video_path: Path, 
        srt_path: Path, 
        output_video_path: Path,
        subtitle_style: Optional[Dict] = None,
        split_long_subtitles: bool = True,
        font_path: Optional[Path] = None
    ) -> bool:
        """
        将字幕渲染到视频上
        
        Args:
            video_path: 输入视频路径
            srt_path: SRT字幕文件路径
            output_video_path: 输出视频路径
            subtitle_style: 字幕样式配置（可选）
            split_long_subtitles: 是否拆分长字幕为多个短句，默认True
            font_path: 字体文件路径（可选）
        
        Returns:
            bool: 是否成功
        """
        try:
            # 如果需要拆分长字幕
            if split_long_subtitles:
                split_srt_path = srt_path.parent / f"{srt_path.stem}_split.srt"
                if self.split_long_subtitles_by_sentences(srt_path, split_srt_path):
                    srt_path = split_srt_path
                    logger.info(f"已拆分长字幕: {split_srt_path}")
                else:
                    logger.warning("字幕拆分失败，使用原始SRT文件")
            
            # 清理SRT文件：移除"类型："行
            cleaned_srt_path = srt_path.parent / f"{srt_path.stem}_cleaned.srt"
            self._clean_srt_for_subtitle(srt_path, cleaned_srt_path)
            
            # 处理字体文件
            fontsdir = None
            font_name_in_style = None
            if font_path:
                font_path = Path(font_path)
                if font_path.exists():
                    # 获取字体文件所在目录
                    fontsdir = font_path.parent.resolve()
                    # 尝试获取字体的内部名称
                    font_name_in_style = self._get_font_internal_name(font_path)
                    if not font_name_in_style:
                        # 如果获取失败，使用文件名（去掉扩展名）
                        font_name_in_style = font_path.stem
                    logger.info(f"使用自定义字体: {font_path}, 字体名称: {font_name_in_style}")
                else:
                    logger.warning(f"字体文件不存在: {font_path}，使用默认字体")
            
            # 构建字幕样式（如果指定了字体，更新字体名称）
            if font_name_in_style and subtitle_style:
                subtitle_style = subtitle_style.copy()
                subtitle_style['FontName'] = font_name_in_style
            elif font_name_in_style:
                # 如果没有传入样式，创建默认样式并使用自定义字体
                subtitle_style = {'FontName': font_name_in_style}
            
            style_str = self._build_subtitle_style(subtitle_style)
            
            # 获取SRT文件的绝对路径（Windows下需要转义）
            srt_abs_path = cleaned_srt_path.resolve()
            # Windows路径转义：将反斜杠替换为正斜杠或转义
            srt_path_str = str(srt_abs_path).replace('\\', '/')
            
            # 构建ffmpeg命令
            # 使用 subtitles 滤镜渲染字幕
            # 注意：subtitles滤镜需要字体文件的内部名称，而不是文件路径
            # 我们通过fontsdir指定字体目录，然后在样式中使用字体名称
            if fontsdir and font_name_in_style:
                # 如果指定了字体目录和字体名称，使用fontsdir参数
                fontsdir_str = str(fontsdir).replace('\\', '/')
                # 转义单引号
                fontsdir_str_escaped = fontsdir_str.replace("'", "\\'")
                srt_path_str_escaped = srt_path_str.replace("'", "\\'")
                style_str_escaped = style_str.replace("'", "\\'")
                
                cmd = [
                    'ffmpeg', '-i', str(video_path),
                    '-vf', f"subtitles='{srt_path_str_escaped}':fontsdir='{fontsdir_str_escaped}':force_style='{style_str_escaped}'",
                    '-c:a', 'copy',  # 音频流直接复制
                    '-y',
                    str(output_video_path)
                ]
                logger.info(f"使用字体目录: {fontsdir_str}, 字体名称: {font_name_in_style}")
            else:
                # 不使用自定义字体
                srt_path_str_escaped = srt_path_str.replace("'", "\\'")
                style_str_escaped = style_str.replace("'", "\\'")
                cmd = [
                    'ffmpeg', '-i', str(video_path),
                    '-vf', f"subtitles='{srt_path_str_escaped}':force_style='{style_str_escaped}'",
                    '-c:a', 'copy',  # 音频流直接复制
                    '-y',
                    str(output_video_path)
                ]
                logger.info("使用默认字体")
            
            logger.info(f"开始渲染字幕到视频: {video_path.name}")
            logger.info(f"字幕文件: {srt_path_str}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # 清理临时文件
            if cleaned_srt_path.exists() and cleaned_srt_path != srt_path:
                cleaned_srt_path.unlink()
            if split_long_subtitles and srt_path != video_path.parent / f"{srt_path.stem}_split.srt":
                # 如果使用了拆分后的SRT，清理它
                pass  # 保留拆分后的SRT以便调试
            
            logger.info(f"字幕渲染完成: {output_video_path}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"字幕渲染失败: {e.stderr[:500]}")
            return False
        except Exception as e:
            logger.error(f"字幕渲染异常: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def _get_font_internal_name(self, font_path: Path) -> Optional[str]:
        """
        获取字体文件的内部名称（PostScript名称或字体族名称）
        
        Args:
            font_path: 字体文件路径
        
        Returns:
            字体内部名称，如果获取失败返回None
        """
        try:
            # 方法1: 使用 fc-list 命令（如果系统有 fontconfig）
            cmd = ['fc-list', '--format=%{postscriptname}\n%{family}\n', str(font_path)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and result.stdout.strip():
                lines = result.stdout.strip().split('\n')
                # 优先使用 PostScript 名称，如果没有则使用字体族名称
                for line in lines:
                    if line.strip():
                        logger.info(f"从fc-list获取字体名称: {line.strip()}")
                        return line.strip()
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            pass
        
        try:
            # 方法2: 使用 Python 的 fonttools 库（如果可用）
            try:
                from fontTools.ttLib import TTFont
                font = TTFont(str(font_path))
                # 获取 PostScript 名称
                if 'name' in font:
                    for record in font['name'].names:
                        if record.nameID == 6:  # PostScript name
                            ps_name = record.toUnicode()
                            if ps_name:
                                logger.info(f"从TTFont获取PostScript名称: {ps_name}")
                                return ps_name
                        elif record.nameID == 1:  # Font Family name
                            family_name = record.toUnicode()
                            if family_name:
                                logger.info(f"从TTFont获取字体族名称: {family_name}")
                                return family_name
            except ImportError:
                pass
        except Exception as e:
            logger.debug(f"使用fonttools获取字体名称失败: {e}")
        
        # 方法3: 对于常见的字体文件名，尝试已知的内部名称
        font_stem = font_path.stem
        known_font_names = {
            'WenYue-XinQingNianTi-W8-J-2': 'WenYue-XinQingNianTi-W8-J-2',  # 尝试使用文件名
        }
        if font_stem in known_font_names:
            return known_font_names[font_stem]
        
        # 如果都失败了，返回None，让调用者使用文件名
        return None
    
    def _build_subtitle_style(self, style: Optional[Dict]) -> str:
        """
        构建字幕样式字符串（ASS格式的Style行）
        
        Args:
            style: 样式配置字典
        
        Returns:
            样式字符串
        """
        if not style:
            # 默认样式：底部以上20%位置，白色文字，黑色描边
            style = {
                'FontName': 'Microsoft YaHei',
                'FontSize': 28,
                'PrimaryColour': '&H00FFFFFF',  # 白色
                'OutlineColour': '&H00000000',  # 黑色描边
                'BackColour': '&H80000000',  # 半透明黑色背景
                'Bold': 0,
                'Italic': 0,
                'Underline': 0,
                'StrikeOut': 0,
                'ScaleX': 100,
                'ScaleY': 100,
                'Spacing': 0,
                'Angle': 0,
                'BorderStyle': 1,
                'Outline': 3,  # 描边宽度
                'Shadow': 2,  # 阴影宽度
                'Alignment': 2,  # 2=底部居中
                'MarginL': 10,
                'MarginR': 10,
                'MarginV': 0,  # 底部边距，0表示使用Alignment控制位置
            }
        
        # 转换为ASS格式的Style字符串
        # MarginV在ASS中表示距离底部的像素数
        # Alignment=2 表示底部居中
        # 对于底部以上20%的位置，设置MarginV为合适的像素值（如180像素适用于1080p视频）
        
        style_parts = [
            f"FontName={style.get('FontName', 'Microsoft YaHei')}",
            f"FontSize={style.get('FontSize', 28)}",
            f"PrimaryColour={style.get('PrimaryColour', '&H00FFFFFF')}",
            f"OutlineColour={style.get('OutlineColour', '&H00000000')}",
            f"BackColour={style.get('BackColour', '&H80000000')}",
            f"Bold={style.get('Bold', 0)}",
            f"Italic={style.get('Italic', 0)}",
            f"BorderStyle={style.get('BorderStyle', 1)}",
            f"Outline={style.get('Outline', 3)}",
            f"Shadow={style.get('Shadow', 2)}",
            f"Alignment={style.get('Alignment', 2)}",
            f"MarginL={style.get('MarginL', 10)}",
            f"MarginR={style.get('MarginR', 10)}",
            f"MarginV={style.get('MarginV', 180)}",  # 底部边距，默认180像素（约20%屏幕高度，适用于1080p视频）
        ]
        
        return ','.join(style_parts)

    async def process_video_audio(self, video_path: Path, srt_path: Path, output_dir: Optional[Path] = None) -> Optional[Path]:
            """
            处理视频音频重新合成
            
            Args:
                video_path: 视频文件路径
                srt_path: srt 文件路径  字幕文件
                output_dir: 输出目录
                voice: 语音名称
            """
            video_path = Path(video_path)
            srt_path = Path(srt_path)
            
            if not video_path.exists():
                error_msg = f"视频文件不存在: {video_path}"
                logger.error(error_msg)
                return None
            
            if not srt_path.exists():
                error_msg = f"SRT文件不存在: {srt_path}"  ## 字幕文件
                logger.error(error_msg)
                return None
            
            # 确定输出目录
            if output_dir is None:
                output_dir = video_path.parent
            else:
                output_dir = Path(output_dir)
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 临时文件目录
            temp_dir = output_dir / "temp_tts"
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"正在解析SRT文件: {srt_path}")
            subtitles = self.parse_srt_file(srt_path) ## 处理 SRT 文件，返回字幕列表
            logger.info(f"共找到 {len(subtitles)} 条字幕")
            
            # 统计需要 TTS 的数量
            narration_count = sum(1 for _, _, _, _, _, is_narration in subtitles if is_narration)
            logger.info(f"其中 {narration_count} 条需要重新生成 TTS（解说类型）")  ## 是解说的需要重新生成音频
            
            if len(subtitles) == 0:
                logger.warning("没有找到字幕，退出")
                return None
            
            # 1. 提取视频音频
            logger.info("步骤 1: 提取视频音频...")
            extracted_audio = temp_dir / f"{video_path.stem}_extracted.wav"
            if not self.extract_audio_from_video(video_path, extracted_audio): 
                logger.error("提取音频失败，退出")
                return None
            
            # 2. 切割音频片段并生成 TTS
            logger.info("步骤 2: 切割音频片段并生成 TTS...")
            audio_segments = []  # 存储最终要拼接的音频片段路径
            
            for i, (index, text, subtitle_type, start_time, end_time, is_narration) in enumerate(subtitles):
                duration = end_time - start_time
                
                if is_narration:
                    # 需要 TTS
                    logger.info(f"[{index}] 类型: {subtitle_type}, 生成 TTS (目标时长: {duration:.3f}s)...")
                    tts_audio = temp_dir / f"tts_{index:04d}.mp3"
                    success = await self.text_to_speech(text, tts_audio)  ## 将文本转换为语音，语速和音调经过调整
                    
                    if success and tts_audio.exists():
                        # 先转换为 wav 格式
                        tts_wav_temp = temp_dir / f"tts_{index:04d}_temp.wav"
                        if self.convert_audio_format(tts_audio, tts_wav_temp):
                            # 检查时长并调整速度以匹配目标时长
                            tts_wav = temp_dir / f"tts_{index:04d}.wav"
                            if self.adjust_audio_speed(tts_wav_temp, tts_wav, duration):
                                audio_segments.append(tts_wav)
                                # 清理临时文件
                                if tts_wav_temp.exists():
                                    tts_wav_temp.unlink()
                            else:
                                logger.warning(f"调整音频速度失败，使用原格式")
                                # 如果调整失败，使用未调整的版本
                                if tts_wav_temp.exists():
                                    tts_wav_temp.rename(tts_wav)
                                    audio_segments.append(tts_wav)
                                else:
                                    # 如果转换也失败，使用原音频片段
                                    logger.warning(f"转换 TTS 音频格式失败，使用原音频片段")
                                    original_segment = temp_dir / f"original_{index:04d}.wav"
                                    if self.cut_audio_segment(extracted_audio, original_segment, start_time, duration):
                                        audio_segments.append(original_segment)
                        else:
                            logger.warning(f"转换 TTS 音频格式失败，使用原音频片段")
                            # 如果转换失败，使用原音频片段
                            original_segment = temp_dir / f"original_{index:04d}.wav"
                            if self.cut_audio_segment(extracted_audio, original_segment, start_time, duration):
                                audio_segments.append(original_segment)
                    else:
                        logger.warning(f"TTS 生成失败，使用原音频片段")
                        # TTS 失败，使用原音频片段
                        original_segment = temp_dir / f"original_{index:04d}.wav"
                        if self.cut_audio_segment(extracted_audio, original_segment, start_time, duration):
                            audio_segments.append(original_segment)
                else:
                    # 使用原音频片段
                    logger.info(f"[{index}] 类型: {subtitle_type}, 使用原音频片段...")
                    original_segment = temp_dir / f"original_{index:04d}.wav"
                    if self.cut_audio_segment(extracted_audio, original_segment, start_time, duration):  ## 根据每段字幕 拿对应的音频片段
                        audio_segments.append(original_segment)
                
                # 处理空白区间（相邻两个片段之间的间隔）
                if i < len(subtitles) - 1:
                    next_start = subtitles[i + 1][3]  # 下一个片段的开始时间
                    gap_duration = next_start - end_time
                    
                    if gap_duration > 0.01:  # 如果间隔大于 10ms，处理空白区间
                        gap_segment = temp_dir / f"gap_{index:04d}_to_{subtitles[i+1][0]:04d}.wav"
                        
                        if gap_duration < 0.5:
                            # 如果空白区间小于 0.5 秒，使用空白音频代替
                            logger.info(f"空白区间: {end_time:.3f}s - {next_start:.3f}s (时长: {gap_duration:.3f}s) < 0.5s，使用空白音频")
                            if self.generate_silent_audio(gap_segment, gap_duration):
                                audio_segments.append(gap_segment)
                        else:
                            # 如果空白区间 >= 0.5 秒，提取原始音频
                            logger.info(f"提取空白区间: {end_time:.3f}s - {next_start:.3f}s (时长: {gap_duration:.3f}s)")
                            if self.cut_audio_segment(extracted_audio, gap_segment, end_time, gap_duration):
                                audio_segments.append(gap_segment)
            
            logger.info(f"共准备 {len(audio_segments)} 个音频片段用于拼接")
            
            # 3. 拼接所有音频片段
            logger.info("步骤 3: 拼接音频片段...")
            final_audio = temp_dir / f"{video_path.stem}_rebuild.wav"
            if not self.concatenate_audio_segments(audio_segments, final_audio):
                logger.error("拼接音频失败")
                return None
            
            logger.info(f"重新合成的音频已保存到: {final_audio}")
            
            # 4. 将新音频替换到视频中
            logger.info("步骤 4: 替换视频音频...")
            output_video = output_dir / f"{video_path.stem}_rewritten.mp4"
            if self.replace_video_audio(video_path, final_audio, output_video):
                logger.info(f"完成! 新视频已保存到: {output_video}")
                return output_video
            else:
                logger.error("替换视频音频失败")
                return None


# 为了向后兼容，保留原有的函数接口
def parse_srt_file(srt_path: Path) -> List[Tuple[int, str, str, float, float, bool]]:
    """向后兼容的函数接口"""
    processor = VideoTTSProcessor()
    return processor.parse_srt_file(srt_path)


async def process_video_audio(video_path: Path, srt_path: Path, output_dir: Optional[Path] = None, voice: str = DEFAULT_VOICE):
    """向后兼容的函数接口"""
    processor = VideoTTSProcessor(voice=voice)
    return await processor.process_video_audio(video_path, srt_path, output_dir)


async def main():
    """主函数"""
    import sys
    
    if len(sys.argv) < 3:
        print("用法: python video_tts.py <video_file_path> <srt_file_path> [output_dir] [voice]")
        print("\n参数说明:")
        print("  video_file_path: 视频文件路径（必需）")
        print("  srt_file_path: srt 文件路径（必需）")
        print("  output_dir: 输出目录（可选，默认在视频文件目录下）")
        print("  voice: 语音名称（可选，默认 zh-CN-YunxiNeural）")
        print("\n示例:")
        print("  python video_tts.py videos/video.mp4 videos/video.srt")
        print("  python video_tts.py videos/video.mp4 videos/video.srt output/audio")
        print("  python video_tts.py videos/video.mp4 videos/video.srt output/audio zh-CN-YunxiNeural")
        sys.exit(1)
    
    video_path = sys.argv[1]
    srt_path = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else None
    voice = sys.argv[4] if len(sys.argv) > 4 else DEFAULT_VOICE
    
    processor = VideoTTSProcessor(voice=voice)
    result = await processor.process_video_audio(video_path, srt_path, output_dir)
    
    if result:
        print(f"\n✓ 处理成功! 输出视频: {result}")
    else:
        print("\n✗ 处理失败")


if __name__ == "__main__":
    asyncio.run(main())
