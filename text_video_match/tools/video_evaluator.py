#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频评估模块
"""

import re
import os
import cv2
import base64
import json
import logging
import requests
import time
from typing import Dict, Any, Optional, List, Tuple
from io import BytesIO
from PIL import Image

logger = logging.getLogger(__name__)

# 配置
DEFAULT_FPS = 1

class VideoEvaluator:
    """视频评估器"""
    
    def __init__(self, 
                 api_url: str = "http://10.176.196.25/v1/chat/completions",
                 api_key: str = "fk1379989049.aEmK59LGEcoiwxat0pGE18JPvtSBkrybaf481994",
                 api_model: str = "google/gemini-3-flash-preview"):
        """
        初始化视频评估器
        
        Args:
            api_url: API 地址
            api_key: API 密钥
            api_model: API 使用的模型名称
        """
        self.api_url = api_url
        self.api_key = api_key
        self.api_model = api_model
        
        logger.info("使用 API 模式（云端模型）")
    
    def create_evaluation_prompt(self) -> str:
        """创建评估prompt"""
        return """
               【重要】你是一个专业的视频内容分析AI，专门评估视频片段作为短视频素材开头的适用性。都要加分析，要给出你思考问题的过程；\
                                你的任务是综合画面和字幕信息，基于给定的标准进行快速、准确的判断。\
                                请根据以下五个方面评估视频片段：\n\

                                1，冲突场景：【条件1：包含吵架词/侮辱/嘲讽（住手/闭嘴/滚/报警/打/杀/不许/放开…）等；条件2：包含对着吼、推搡/指人、镜头快切/抖动的画面；条件3：包含反转词（但是/可/却/没想到/居然/竟然/反而/谁知/却是/原来/不料…），上一句肯定，下一句出现反转词】\n\
                                2，激情/擦边画面：【条件1：男女之间存在拥吻/贴靠/抚摸；条件2：存在男性或者女性的唇/腿/腰/胸/臀等特写；条件3：画面存在暧昧词（亲一下/今晚/酒店/撩…）。】\n\
                                3，夸张表情或动作：【条件1：画面包含人物夸张表情（大笑/愤怒/惊讶/翻白眼/怒吼/张口/痛哭…的镜头特写）；条件2：画面包含人物夸张动作（攻击性动作/剧烈手势/夸张姿势/大幅度的肢体动作/甩门/摔物/飞奔…）；条件3：画面节奏变化快（近景特写/快切/抖动/强对比…）】\n\
                                4，精彩打斗：【条件1：画面中包含多个人之间存在肢体碰撞/打击/格斗/追逐/枪/刀/棍…；条件2：画面中人物有连续快速动作；条件3：画面场景多次快速切换…）。】\n\
                                5，根据字幕判断对白是否具有吸引力，尤其是能煽动情绪（如兴奋、愤怒、毁三观等）、勾起想象或好奇心（如悬念、提问、情感表达）。夸张表达等\n\
                                吸引力对白：【条件1：字幕含强烈情绪或极端词/感叹/重复/反问（受够了/你敢…）；条件2：字幕内容是否毁三观，不符合正常道德逻辑；条件3：字幕内容是否有设置悬念（提问或者出现疑惑词但没给结果）】\n\

                                输出要求
                                详细分析：针对每个标准，提供以下内容：
                                评分标准（每项0-10分）：
                                1. 冲突场景：9-10同时满足3个条件或任意条件高等强烈，5-8满足任意2个条件或任意条件程度中等强烈，1-4满足任意1个条件或任意条件程度低等强烈，0不满足任意条件\n\
                                2. 激情/擦边画面：9-10同时满足3个条件或任意条件高等强烈，5-8满足任意2个条件或任意条件程度中等强烈，1-4满足任意1个条件或任意条件程度低等强烈，0不满足任意条件\n\
                                3. 夸张表情或动作：9-10同时满足3个条件或任意条件高等强烈，5-8满足任意2个条件或任意条件程度中等强烈，1-4满足任意1个条件或任意条件程度低等强烈，0不满足任意条件\n\
                                4. 精彩打斗场景：9-10同时满足3个条件或任意条件高等强烈，5-8满足任意2个条件或任意条件程度中等强烈，1-4满足任意1个条件或任意条件程度低等强烈，0不满足任意条件\n\
                                5. 吸引力对白：9-10同时满足3个条件或任意条件高等强烈，5-8满足任意2个条件或任意条件程度中等强烈，1-4满足任意1个条件或任意条件程度低等强烈，0不满足任意条件\n\

                                简要理由 
                                请作为专业的视频内容分析AI，评估这段视频作为短视频开头的适用性。
             			输出格式示例
             			   {{
            			    "evaluations": [
             			       {{"criterion": "冲突场景", "score": 0, "reason": "理由"}},
               			     {{"criterion": "激情/擦边画面", "score": 0, "reason": "理由"}},
               			     {{"criterion": "夸张表情或动作", "score": 0, "reason": "理由"}},
               			     {{"criterion": "精彩打斗场景", "score": 0, "reason": "理由"}},
                			    {{"criterion": "吸引力对白", "score": 0, "reason": "理由"}}
              			  ],
              			  "is_highlight": false,
              			  "summary": "综合评价"
               			   }}

                """
    
    def create_description_prompt(self) -> str:
        """创建视频描述prompt"""
        return """# 角色设定

你是一位专业的影视内容分析专家，擅长精准捕捉视频中的视觉细节、叙事逻辑、角色情感和转场信息。

# 任务描述

请深度分析上传的视频片段，并严格按照 JSON 格式输出多维度的理解结果。这些数据将用于后续的影视解说文案匹配。

# 输出维度要求

## visual_summary
用一句话高度概括画面主要内容（包含主体、动作、核心意图）。用于向量检索。

## characters
识别画面中出现的角色。如果无法识别姓名，请描述特征（如：穿黑西装的男人、长发女孩）。

## actions
描述画面中发生的关键动态动作（如：奔跑、争吵、流泪、推门而入）。

## environment
描述场景和光影（如：深夜的办公室、大雨中的街道、昏暗的地下室）。

## emotion_tone
识别画面传达的情绪基调（如：压抑、热血、凄美、反讽）。

## camera_movement
描述镜头语言（如：特写、俯拍、快速推镜头、主观视角）。

## dialogue_or_text
记录画面中出现的台词、旁白或关键文字（OCR）。

# 约束条件

- 输出必须是合法的 JSON 格式
- 描述文字请使用客观、具象的词汇，避免使用模糊的形容词
- 如果某个维度没有信息，请填入 `"null"`

# JSON 模板示例

```json
{
  "visual_summary": "男子在雨夜的巷子中绝望地跪倒，仰天长啸。",
  "characters": ["男子"],
  "actions": ["跪下", "仰头", "大喊"],
  "environment": "夜晚，暴雨，老旧小巷，冷色调",
  "emotion_tone": "绝望，崩溃",
  "camera_movement": "从全景逐渐拉近至面部特写",
  "dialogue_or_text": "为什么总是这样！"
}
```
"""

    def extract_frames(self, video_path: str, fps: float = 1.0) -> List[Tuple[int, float, Any]]:
        """
        从视频中抽取帧
        
        Args:
            video_path: 视频文件路径
            fps: 抽取帧率，默认每秒1帧
            
        Returns:
            List[tuple]: [(frame_number, timestamp, image_data), ...]
            其中 image_data 是 numpy array 格式的图片数据（RGB格式）
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
        
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps if video_fps > 0 else 0
        
        logger.info(f"视频信息: FPS={video_fps:.2f}, 总帧数={total_frames}, 时长={duration:.2f}秒")
        
        # 计算需要抽取的帧
        frame_interval = int(video_fps / fps) if video_fps > fps else 1
        frames = []
        
        frame_number = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 按照间隔抽取帧
            if frame_number % frame_interval == 0:
                timestamp = frame_number / video_fps if video_fps > 0 else 0
                # 将BGR转换为RGB（PIL使用RGB）
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append((frame_number, timestamp, frame_rgb))
            
            frame_number += 1
        
        cap.release()
        logger.info(f"成功抽取 {len(frames)} 帧")
        return frames
    
    def image_to_base64(self, image_array) -> str:
        """
        将图片数组转换为base64编码
        
        Args:
            image_array: numpy array格式的图片（RGB格式）
            
        Returns:
            str: base64编码的图片字符串，格式为 "data:image/jpeg;base64,{base64_string}"
        """
        # 将numpy array转换为PIL Image
        pil_image = Image.fromarray(image_array)
        
        # 转换为JPEG格式的字节流
        buffer = BytesIO()
        pil_image.save(buffer, format='JPEG')
        image_bytes = buffer.getvalue()
        
        # 转换为base64
        base64_string = base64.b64encode(image_bytes).decode('utf-8')
        
        return f"data:image/jpeg;base64,{base64_string}"
    
    def create_api_payload(self, prompt: str, frames: List[Tuple[int, float, Any]]) -> Dict[str, Any]:
        """
        创建 API 请求的 payload
        
        Args:
            prompt: 文本提示词
            frames: 抽取的帧列表，格式为 [(frame_number, timestamp, image_data), ...]
            
        Returns:
            Dict: API 请求 payload
        """
        content = [
            {
                "type": "text",
                "text": prompt
            }
        ]
        
        # 添加每个帧和时间戳
        for frame_number, timestamp, image_data in frames:
            # 添加时间戳文本
            content.append({
                "type": "text",
                "text": f"[{timestamp:.1f} second]"
            })
            
            # 添加图片（base64编码）
            base64_image = self.image_to_base64(image_data)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": base64_image
                }
            })
        
        payload = {
            "model": self.api_model,
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ]
        }
        
        return payload
    
    def call_api(self, payload: Dict[str, Any], timeout: int = 300, max_retries: int = 3) -> str:
        """
        调用 API，带重试机制
        
        Args:
            payload: 请求 payload
            timeout: 超时时间（秒）
            max_retries: 最大重试次数，默认3次
            
        Returns:
            str: API 响应文本内容。如果所有重试都失败，返回包含错误信息的 JSON 字符串
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "host": "api.360.cn"
        }
        
        last_error = None
        
        for attempt in range(max_retries):
            try:
                logger.info(f"API调用尝试 {attempt + 1}/{max_retries}")
                
                response = requests.post(
                    self.api_url,
                    json=payload,
                    headers=headers,
                    timeout=timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                    if content:
                        logger.info(f"API调用成功（尝试 {attempt + 1}）")
                        logger.info(f"API响应内容: {content}")
                        return content
                    else:
                        error_msg = "API响应中未找到有效内容"
                        logger.warning(f"尝试 {attempt + 1} 失败: {error_msg}")
                        last_error = error_msg
                else:
                    error_msg = f"API请求失败: HTTP {response.status_code}, {response.text[:200]}"
                    logger.warning(f"尝试 {attempt + 1} 失败: {error_msg}")
                    last_error = error_msg
                    
                    # 如果是客户端错误（4xx），不重试
                    if 400 <= response.status_code < 500:
                        logger.error(f"客户端错误，不重试: {error_msg}")
                        break
                        
            except requests.exceptions.Timeout as e:
                error_msg = f"API请求超时: {str(e)}"
                logger.warning(f"尝试 {attempt + 1} 失败: {error_msg}")
                last_error = error_msg
            except requests.exceptions.ConnectionError as e:
                error_msg = f"API连接错误: {str(e)}"
                logger.warning(f"尝试 {attempt + 1} 失败: {error_msg}")
                last_error = error_msg
            except requests.exceptions.RequestException as e:
                error_msg = f"API请求异常: {str(e)}"
                logger.warning(f"尝试 {attempt + 1} 失败: {error_msg}")
                last_error = error_msg
            except Exception as e:
                error_msg = f"未知错误: {str(e)}"
                logger.warning(f"尝试 {attempt + 1} 失败: {error_msg}")
                last_error = error_msg
            
            # 如果不是最后一次尝试，等待后重试（指数退避）
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # 1秒, 2秒, 4秒...
                logger.info(f"等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
        
        # 所有重试都失败，返回错误信息（格式化为JSON，以便后续解析）
        logger.error(f"API调用失败，已重试 {max_retries} 次。最后错误: {last_error}")
        error_result = {
            "error": "API调用失败",
            "error_message": last_error,
            "retries": max_retries,
            "summary": f"API调用失败，无法进行评估。错误信息: {last_error}"
        }
        return json.dumps(error_result, ensure_ascii=False)
    
    def evaluate_video(self, video_path: str, fps: float = None) -> Dict[str, Any]:
        """
        评估单个视频片段
        
        Args:
            video_path: 视频文件路径
            fps: 抽取帧率（默认使用 DEFAULT_FPS）
            
        Returns:
            评估结果字典。即使出现错误，也会返回包含错误信息的结果，不会抛出异常
        """
        try:
            prompt_text = self.create_evaluation_prompt()
            
            # API 模式
            if fps is None:
                fps = DEFAULT_FPS
            
            logger.info(f"使用 API 模式评估视频: {video_path}")
            
            try:
                # 1. 抽取视频帧
                frames = self.extract_frames(video_path, fps)
                
                if len(frames) == 0:
                    raise ValueError("未能从视频中抽取任何帧")
                
                # 2. 创建 API payload
                payload = self.create_api_payload(prompt_text, frames)
                
                # 3. 调用 API（带重试机制，不会抛出异常）
                output_text = self.call_api(payload)
                
            except Exception as e:
                logger.error(f"处理视频 {video_path} 时出错: {str(e)}")
                # 返回错误结果
                error_result = {
                    "error": "视频处理失败",
                    "error_message": str(e),
                    "evaluations": [
                        {"criterion": "冲突场景", "score": 0, "reason": f"视频处理失败: {str(e)}"},
                        {"criterion": "激情/擦边画面", "score": 0, "reason": f"视频处理失败: {str(e)}"},
                        {"criterion": "夸张表情或动作", "score": 0, "reason": f"视频处理失败: {str(e)}"},
                        {"criterion": "精彩打斗场景", "score": 0, "reason": f"视频处理失败: {str(e)}"},
                        {"criterion": "吸引力对白", "score": 0, "reason": f"视频处理失败: {str(e)}"}
                    ],
                    "average_score": 0.0,
                    "is_highlight": False,
                    "summary": f"视频处理失败，无法进行评估。错误信息: {str(e)}"
                }
                output_text = json.dumps(error_result, ensure_ascii=False)
            
            # 解析结果
            evaluation_result = self.parse_evaluation_result(output_text)
            
            return {
                "video_path": video_path,
                "parsed_result": evaluation_result
            }
            
        except Exception as e:
            # 最后的异常捕获，确保无论如何都返回一个结果
            logger.error(f"评估视频 {video_path} 时发生未知错误: {str(e)}")
            error_result = {
                "error": "评估失败",
                "error_message": str(e),
                "raw_text": f"评估过程中发生未知错误: {str(e)}"
            }
            return {
                "video_path": video_path,
                "parsed_result": error_result
            }

    def description_video(self, video_path: str, fps: float = None) -> Dict[str, Any]:
        """
        描述单个视频片段
        
        Args:
            video_path: 视频文件路径
            fps: 抽取帧率（默认使用 DEFAULT_FPS）
            
        Returns:
            描述结果字典。即使出现错误，也会返回包含错误信息的结果，不会抛出异常
        """
        try:
            prompt_text = self.create_description_prompt()
            
            # API 模式
            if fps is None:
                fps = DEFAULT_FPS
            
            logger.info(f"使用 API 模式描述视频: {video_path}")
            
            try:
                # 1. 抽取视频帧
                frames = self.extract_frames(video_path, fps)
                
                if len(frames) == 0:
                    raise ValueError("未能从视频中抽取任何帧")
                
                # 2. 创建 API payload
                payload = self.create_api_payload(prompt_text, frames)
                
                # 3. 调用 API（带重试机制，不会抛出异常）
                output_text = self.call_api(payload)
                
            except Exception as e:
                logger.error(f"处理视频 {video_path} 时出错: {str(e)}")
                # 返回错误结果
                error_result = {
                    "error": "视频处理失败",
                    "error_message": str(e),
                    "visual_summary": None,
                    "characters": [],
                    "actions": [],
                    "environment": None,
                    "emotion_tone": None,
                    "camera_movement": None,
                    "dialogue_or_text": None
                }
                output_text = json.dumps(error_result, ensure_ascii=False)
            
            # 解析结果
            description_result = self.parse_description_result(output_text)
            
            return {
                "video_path": video_path,
                "parsed_result": description_result
            }
            
        except Exception as e:
            # 最后的异常捕获，确保无论如何都返回一个结果
            logger.error(f"描述视频 {video_path} 时发生未知错误: {str(e)}")
            error_result = {
                "error": "描述失败",
                "error_message": str(e),
                "raw_text": f"描述过程中发生未知错误: {str(e)}"
            }
            return {
                "video_path": video_path,
                "parsed_result": error_result
            }

    def parse_description_result(self, desc_text: str) -> Dict[str, Any]:
        """解析视频描述结果"""
        try:
            # 清理文本，移除可能的markdown标记和多余字符
            cleaned_text = desc_text.strip()
            
            # 移除 <think>, <analysis> 等标签内容
            cleaned_text = re.sub(r'<think>.*?</think>', '', cleaned_text, flags=re.DOTALL)
            cleaned_text = re.sub(r'<analysis>.*?</analysis>', '', cleaned_text, flags=re.DOTALL)
            
            # 移除可能的markdown代码块标记
            if cleaned_text.startswith('```json'):
                cleaned_text = cleaned_text[7:]
            if cleaned_text.startswith('```'):
                cleaned_text = cleaned_text[3:]
            if cleaned_text.endswith('```'):
                cleaned_text = cleaned_text[:-3]
            
            cleaned_text = cleaned_text.strip()
            
            # 查找JSON格式的描述结果
            json_patterns = [
                r'\{[\s\S]*?"visual_summary"[\s\S]*?\}',  # 匹配包含visual_summary的完整JSON
                r'\{[\s\S]*?"characters"[\s\S]*?\}',  # 匹配包含characters的JSON
                r'\{[\s\S]*?\}',  # 匹配任何 {...} 格式
            ]
            
            json_str = None
            for pattern in json_patterns:
                match = re.search(pattern, cleaned_text, re.DOTALL)
                if match:
                    json_str = match.group(0)
                    try:
                        # 尝试解析，如果成功则使用
                        parsed = json.loads(json_str)
                        logger.info(f"成功解析JSON，模式: {pattern[:50]}")
                        return parsed
                    except:
                        # 解析失败，尝试下一个模式
                        continue
            
            # 如果所有模式都失败
            logger.warning("未找到有效的JSON格式，尝试从原始文本提取")
                
        except json.JSONDecodeError as e:
            logger.warning(f"JSON解析失败: {e}")
        except Exception as e:
            logger.warning(f"解析描述结果失败: {e}")
        
        # 如果解析失败，返回原始文本
        logger.error(f"所有解析方法失败，返回原始文本（前500字符）: {desc_text[:500]}")
        return {
            "raw_text": desc_text,
            "error": "JSON解析失败"
        }

    def parse_evaluation_result(self, eval_text: str) -> Dict[str, Any]:
        """解析评估结果"""
        import json
        
        try:
            # 清理文本，移除可能的markdown标记和多余字符
            cleaned_text = eval_text.strip()
            
            # 移除 <think>, <analysis> 等标签内容
            cleaned_text = re.sub(r'<think>.*?</think>', '', cleaned_text, flags=re.DOTALL)
            cleaned_text = re.sub(r'<analysis>.*?</analysis>', '', cleaned_text, flags=re.DOTALL)
            
            # 移除可能的markdown代码块标记
            if cleaned_text.startswith('```json'):
                cleaned_text = cleaned_text[7:]
            if cleaned_text.startswith('```'):
                cleaned_text = cleaned_text[3:]
            if cleaned_text.endswith('```'):
                cleaned_text = cleaned_text[:-3]
            
            cleaned_text = cleaned_text.strip()
            
            # 查找JSON格式的评估结果（更精确的模式）
            json_patterns = [
                r'\{[^{}]*"evaluations"[^{}]*\[[\s\S]*?\][^{}]*\}',  # 匹配包含evaluations的完整JSON
                r'\{[\s\S]*?"average_score"[\s\S]*?\}',  # 匹配包含average_score的JSON
                r'\{[\s\S]*?\}',  # 匹配任何 {...} 格式
                r'\[[\s\S]*?\]',  # 匹配 [...] 格式
            ]
            
            json_str = None
            for pattern in json_patterns:
                match = re.search(pattern, cleaned_text, re.DOTALL)
                if match:
                    json_str = match.group(0)
                    try:
                        # 尝试解析，如果成功则使用
                        parsed = json.loads(json_str)
                        logger.info(f"成功解析JSON，模式: {pattern[:50]}")
                        return parsed
                    except:
                        # 解析失败，尝试下一个模式
                        continue
            
            # 如果所有模式都失败，尝试从原始文本中提取
            logger.warning("未找到有效的JSON格式，尝试从原始文本提取")
            
            # 尝试提取评分信息（手动解析）
            # if '"score"' in eval_text or 'score' in eval_text.lower():
            #     logger.info("检测到评分信息，但JSON格式无效")
                
        except json.JSONDecodeError as e:
            logger.warning(f"JSON解析失败: {e}")
        except Exception as e:
            logger.warning(f"解析评估结果失败: {e}")
        
        # 如果解析失败，返回原始文本
        logger.error(f"所有解析方法失败，返回原始文本（前500字符）: {eval_text[:500]}")
        return {
            "raw_text": eval_text,
            "error": "JSON解析失败"
        }