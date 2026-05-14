#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文本结构化 + TTS  →  输出只在「脚本同级 cache/{dj}/」
"""
import json
import re
import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional
import sys

sys.path.append(str(Path(__file__).parent.parent))  # 导包
from video_commentary_text.model_client import ModelClient
from tools.video_tts import VideoTTSProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TextStructuredDescription:
    def __init__(self,
                 api_url: str = "https://api.360.cn/v1/chat/completions",
                 api_key: str = "fk1379989049.aEmK59LGEcoiwxat0pGE18JPvtSBkrybaf481994",
                 model: str = "google/gemini-3-pro-preview",
                 timeout: int = 300):
        self.model_client = ModelClient(api_url=api_url, api_key=api_key, model=model, timeout=timeout)
        self.tts_processor = VideoTTSProcessor()
        logger.info(f"初始化 TextStructuredDescription: model={model}")

    # ========== 唯一改动：输出目录 = 「脚本同级 cache/{dj}/」 ==========
    def _output_base(self, dj: str) -> Path:
        """只在 ./text_video_match/cache/{dj}/ 里干活"""
        base = Path(__file__).parent / "cache" / dj
        base.mkdir(parents=True, exist_ok=True)
        return base

    # -------------------- 以下全部原封不动 --------------------
    def _create_system_prompt(self) -> str:
        return """角色设定

            你是一位拥有 10 年经验的短视频解说导演。你的特长是根据解说文案（Script）还原画面视觉逻辑，并将其拆解为适合自动化剪辑的最小任务单元。

            任务背景

            我有一段影视解说文案，以及一组预先处理好的视频片段索引。你的任务是将长文案进行智能分段，并针对每一段文案生成用于视频检索的"视觉增强 Query"。

            任务指令

            1. 智能分段：

            按叙事逻辑切分，每段字数控制在 30-60 字。

            确保每一段有明确的视觉主体（人、物或场景）。

            2. 维度提取（关键）： 针对每一个分段，提取以下特征：

            核心角色 (Characters)： 明确指出谁在画面中（需区分"张志焱/哥哥"与"张志磊/弟弟"）。

            视觉行为 (Actions)： 将抽象动词转化为物理动作（如：将"残忍杀害"转化为"持刀、争执、倒地"）。

            氛围色调 (Tone/Environment)： 描述光影和环境（如：阴雨、豪宅、昏暗）。

            检索增强 (Query Expansion)： 将文学化表达转化为物理描述词。例如：文案说"厉鬼索命"，Query 应包含"面色苍白、惊恐、对视、特写"。

            3. 输出格式：

            严格以 JSON 数组 格式输出，方便程序解析。

            [
            {
                "segment_sequence": 1,
                "script_text": "女人亲手将丈夫推下万丈深渊，可第二天，男人竟毫发无伤地回到了家。",
                "visual_query": {
                "characters": ["丁佳柔", "张志磊"],
                "action_keywords": ["推搡", "坠落", "悬崖/高处", "回家", "推门", "特写"],
                "environment": "户外/深夜 或 室内门口",
                "expanded_tags": "对比镜头, 震惊的表情, 完好无损的西装, 诡异的氛围"
                }
            },
            {
                "segment_sequence": 2,
                "script_text": "就在几小时前，妻子丁佳柔伙同情夫，残忍杀害了丈夫张志磊，只为霸占张氏集团的百亿家产。",
                "visual_query": {
                "characters": ["丁佳柔", "情夫", "张志磊"],
                "action_keywords": ["密谋", "争吵", "控制", "合同/遗嘱", "伤害动作"],
                "environment": "光线昏暗的室内, 豪宅",
                "expanded_tags": "冷色调, 阴谋, 贪婪的神情, 黑色西装"
                }
            }
            ]

            说明：

            1. 解决语义鸿沟：文学词汇 vs 物理画面

            文案常写"付出代价"，视频里可能只是"被打倒"或"被警察带走"。通过 expanded_tags 强制 LLM 进行联想映射，能大幅提高 Embedding 检索的召回率。

            2. 身份逻辑前置处理

            在解说中，双胞胎是核心矛盾。我们在提示词中要求 LLM 必须在 characters 中明确区分身份（如：哥哥张志焱），这会引导后续检索系统去寻找那些标注了"霸气、自信"特征的片段，而非"被捆绑"的弟弟片段。

            重要提示：
            - 必须严格按照 JSON 格式输出，不要包含任何额外的文字说明
            - 确保 JSON 格式正确，可以被程序直接解析
            - 每个分段必须包含所有必需字段"""

    def _read_text_file(self, file_path: Path) -> str:
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        content = file_path.read_text(encoding="utf8").strip()
        if not content:
            raise ValueError(f"文件内容为空: {file_path}")
        logger.info(f"成功读取文件: {file_path}，内容长度: {len(content)} 字符")
        return content

    def _parse_json_response(self, raw: str) -> List[Dict[str, Any]]:
        m = re.search(r'\[[\s\S]*\]', raw)
        json_str = m.group(0) if m else raw.strip()
        data = json.loads(json_str)
        if not isinstance(data, list):
            raise ValueError("解析结果不是数组格式")
        return data

    def _validate_segment(self, seg: Dict[str, Any], seq: int) -> bool:
        required = ["segment_sequence", "script_text", "visual_query"]
        for k in required:
            if k not in seg:
                logger.warning(f"分段 {seq} 缺少字段 {k}")
                return False
        vq = seg["visual_query"]
        for k in ("characters", "action_keywords", "environment", "expanded_tags"):
            if k not in vq:
                logger.warning(f"分段 {seq} visual_query 缺少 {k}")
                return False
        return True

    def _post_process_segments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        for seq, seg in enumerate(segments, 1):
            seg["segment_sequence"] = seq
            if self._validate_segment(seg, seq):
                out.append(seg)
        return out

    # ---------- TTS：只落在统一 cache ----------
    async def _tts_for_segments(self, segments: List[Dict[str, Any]], dj: str) -> List[Dict[str, Any]]:
        tts_dir = self._output_base(dj) / "tts"
        tts_dir.mkdir(parents=True, exist_ok=True)
        ok, fail = 0, 0
        for seg in segments:
            seq = seg["segment_sequence"]
            text = seg["script_text"]
            if not text.strip():
                seg["audio_path"] = None
                seg["audio_duration"] = None
                fail += 1
                continue
            audio_file = tts_dir / f"segment_{seq:04d}.mp3"
            dur = await self.tts_processor.text_to_speech_transfer(text, audio_file)
            if dur and dur > 0:
                seg["audio_path"] = f"tts/segment_{seq:04d}.mp3"   # 相对路径
                seg["audio_duration"] = dur
                ok += 1
            else:
                seg["audio_path"] = None
                seg["audio_duration"] = None
                fail += 1
        logger.info(f"TTS 完成: 成功 {ok}  | 失败 {fail}")
        return segments

    # ---------- 主入口：只写统一 cache ----------
    def process(self, dj: str, input_text_file: Path) -> Dict[str, Any]:
        try:
            logger.info(f"开始文本结构化描述，输入: {input_text_file}")
            content = self._read_text_file(input_text_file)
            user_prompt = f"请对以下影视解说文案进行智能分段和结构化提取：\n\n{content}\n\n严格输出 JSON 数组。"
            raw, _ = self.model_client.call(self._create_system_prompt(), user_prompt)
            segments = self._parse_json_response(raw)
            segments = self._post_process_segments(segments)
            if not segments:
                raise ValueError("没有有效分段")
            segments = asyncio.run(self._tts_for_segments(segments, dj))
            out_json = self._output_base(dj) / f"{input_text_file.stem}_structured.json"
            out_json.write_text(json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf8")
            tts_ok = sum(1 for s in segments if s.get("audio_path"))
            return {"status": "success", "dj": dj, "output_file": str(out_json),
                    "total_segments": len(segments), "tts_success": tts_ok,
                    "tts_fail": len(segments) - tts_ok, "segments": segments}
        except Exception as e:
            logger.error(f"处理失败: {e}")
            return {"status": "error", "error_message": str(e)}


# -------------------- CLI --------------------
if __name__ == "__main__":
    if len(sys.argv) < 3:          # 现在需要两个参数
        print("用法: python skill_text_structured_description.py <dj> <txt_file_path>")
        sys.exit(1)
    dj        = sys.argv[1]        # 显式指定 dj
    txt_path  = Path(sys.argv[2])
    result    = TextStructuredDescription().process(dj, txt_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))