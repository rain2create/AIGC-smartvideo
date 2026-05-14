#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文本结构化 + TTS + 集数索引融合（单句处理版）
- 输入: 解说文案txt
- 处理: 
  1. 先用suoying映射每句话→集数
  2. 每个句子单独做结构化描述（不再合并）
  3. 每段保留episode_target（目标集数）
- 输出: {txt}_structured.json（每个segment含episode_target字段，且严格单集）
"""

import json
import re
import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import sys

sys.path.append(str(Path(__file__).parent.parent))
from video_commentary_text.model_client import ModelClient
from tools.video_tts import VideoTTSProcessor
from video_commentary_text import suoying
# 导入suoying的映射功能
from video_commentary_text.suoying import (
    map_commentary_to_episodes_with_llm,
    load_asr_files
)

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

    def _output_base(self, dj: str) -> Path:
        """输出目录: ./text_video_match/cache/{dj}/"""
        base = Path(__file__).parent / "cache" / dj
        base.mkdir(parents=True, exist_ok=True)
        return base

    # ==================== 新增：集数映射（融合suoying）====================
    def _map_sentences_to_episodes(self, 
                                   commentary: str, 
                                   dj: str,
                                   start_ep: int = 1,
                                   end_ep: int = 20,
                                   asr_base_path: str = "/mnt_liuhuan/search_public/aigc/smart_video_v2/audio/videos") -> Dict[int, List[Tuple[int, str, float, str]]]:
        """
        调用suoying逻辑，将解说文案的每句话映射到集数
        
        Returns:
            {episode: [(sentence_idx, sentence, confidence, reason), ...]}
        """
        logger.info(f"开始集数映射: {dj}, 范围 {start_ep}-{end_ep}")
        
        # 加载ASR数据
        asr_data = load_asr_files(dj, start_ep, end_ep, asr_base_path)
        
        if not asr_data:
            raise RuntimeError(
                f"未找到ASR数据，无法进行集数映射。"
                f"剧名={dj}，路径={asr_base_path}/{dj}/"
            )
        
        # 调用suoying的大模型映射
        episode_mapping = map_commentary_to_episodes_with_llm(
            model_client=self.model_client,
            commentary=commentary,
            asr_data=asr_data,
            temperature=0.1,
            max_utterances_per_episode=20,
        )
        
        if not episode_mapping:
            raise RuntimeError(
                f"集数映射失败(suoying JSON解析错误)，请检查 LLM 返回格式。"
                f"剧名={self.dj}，建议检查 suoying 日志或重跑。"
            )
        
        logger.info(f"映射完成，句子分布在 {len(episode_mapping)} 个集数")
        return episode_mapping

    def _split_sentences(self, text: str) -> List[str]:
        """将文本按句分割，并过滤表格/SOP 噪音。"""
        noise_keywords = [
            "原始大纲剧情", "核心关键字", "凝练输出", "SOP", "时序", "细/爆点标记"
        ]
        cleaned_lines: List[str] = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("|") and "|" in line:
                continue
            if re.fullmatch(r"[-|:\s]+", line):
                continue
            if any(k in line for k in noise_keywords):
                continue
            cleaned_lines.append(line)

        normalized = " ".join(cleaned_lines)
        sentences = re.split(r"[。！？]", normalized)
        out: List[str] = []
        for s in sentences:
            s = re.sub(r"^\s*[-*•]+\s*", "", s.strip())
            s = re.sub(r"^\s*\d+\s*[).、:：-]\s*", "", s)
            s = s.replace("※", "")
            s = re.sub(r"\s+", "", s)
            if not s:
                continue
            if not re.search(r"[\u4e00-\u9fff]", s):
                continue
            out.append(s)
        return out

    def _sanitize_tts_text(self, text: str) -> str:
        """TTS 前兜底清洗，避免符号/表格碎片进入语音模型。"""
        t = (text or "").strip()
        if not t:
            return ""
        t = t.replace("|", " ").replace("※", " ")
        t = re.sub(r"[`#*_~\[\]{}<>]", " ", t)
        # 去掉常见提纲编号，避免“报数字”
        t = re.sub(r"(^|[\s:：])\d+\s*[.、]\s*", " ", t)
        t = re.sub(r"\s+", "", t)
        if len(t) < 2:
            return ""
        if not re.search(r"[\u4e00-\u9fff]", t):
            return ""
        return t

    def _group_sentences_by_episode(self, 
                                    episode_mapping: Dict[int, List[Tuple[int, str, float, str]]]
                                    ) -> List[Dict[str, Any]]:
        """将按集数分组的句子，整理成连续的段落列表（保持原文顺序）"""
        all_sentences = []
        for episode, items in episode_mapping.items():
            for idx, sentence, confidence, reason in items:
                all_sentences.append({
                    "sentence_idx": idx,
                    "sentence": sentence,
                    "episode": episode,
                    "confidence": confidence,
                    "reason": reason
                })
        
        # 按原文顺序排序
        all_sentences.sort(key=lambda x: x["sentence_idx"])
        return all_sentences

    def _create_system_prompt(self) -> str:
        return """角色设定

你是一位拥有 10 年经验的短视频解说导演。你的特长是根据解说文案（Script），
从【导演与剪辑视角】还原画面视觉逻辑，
并将其拆解为适合自动化剪辑和视频检索的最小任务单元。

任务背景

我有一段影视解说文案，以及一组预先处理好的视频片段索引。
系统整体采用【四阶段匹配机制】：

- 第一阶段（粗排）：仅基于整体语义进行大致召回
- 第二阶段（精排）：基于「人物类型 + 动作 + 环境」判断画面是否可用
- 第三阶段（深排）：基于「互动关系 + 情绪强度 + 导演意图」判断氛围
- 第四阶段（LLM精排）：综合判断最合适的片段

你的任务是：
为每一段生成【第二阶段至第四阶段】所需的结构化视觉 Query。

任务指令


1. 第二阶段精排用：硬视觉要素（参与加权点积）

以下字段 **仅用于第二阶段精排**，
判断视频片段在"画面层面是否匹配"，
不涉及情绪或人物关系判断：

- characters（画面主体-姓名）
明确指出谁在画面中，输出角色姓名
例如：["张志焱", "江婉宁"]

- character_types（画面主体-类型）
输出角色的通用类型，用于与视频画面匹配
可选值：["男人", "女人", "男孩", "女孩", "老人", "老妇", "多人", "群体"]
例如：["男人", "女人"]

- action_keywords（可拍摄动作）
将抽象或文学描述转化为【明确可见的物理行为】
例如：将"残忍杀害"转化为"持刀、争执、倒地"

- environment（环境与空间）
描述场景类型、空间属性、光影状态
例如：室内/室外、夜晚、昏暗、雨天、狭窄空间

- expanded_tags（视觉具象补充）
用于弥合"文字 → 画面"的语义差距，
提供可感知、可拍摄的视觉线索，
例如：文案说"厉鬼索命"，Query 应包含"面色苍白、惊恐、对视、特写镜头"

⚠️ 注意：
- characters 输出具体姓名，character_types 输出通用类型（用于与视频侧对齐）
- 第二阶段 **不使用人物关系**
- 不判断情绪、叙事功能或导演意图

2. 第三阶段深排专用：人物关系 + 模糊视觉语义

以下字段 **仅用于第三阶段深排**，
用于判断"画面是否在情绪与关系层面真正合适"：

- interaction_type（人物互动关系）
描述画面中人物之间的互动关系（可多选）：
["无互动","对话","对峙/冲突","亲密互动","群体互动","追逐/打斗"]

interaction_type 属于【结构化关系信号】，
不参与第二阶段点积，
仅在第三阶段与情绪、叙事功能联合判断。

3. 第三阶段深排专用：人物关系+模糊视觉语义标签（fuzzy_tags）

fuzzy_tags **仅用于第三阶段深排**，
用于判断"画面感觉是否对"，
不参与任何基础语义召回或第二阶段点积计算。

该阶段重点关注：
人物关系张力、情绪强度、叙事位置与导演功能意图。

fuzzy_tags 必须从【导演 / 剪辑 / 观众感知】角度出发，
描述观众在观看画面时的主观体验，
而不是具体剧情或动作。

fuzzy_tags 结构如下（每段必须完整给出）：

- audience_state（观众心理状态）
最多选择 3 个：
["意外","不安","期待","疑惑","压迫感","冷漠","同情"]

- emotion_intensity（观众情绪强度）
单值枚举："低" | "中" | "高"

- director_intent（导演意图）
最多选择 2 个：
["制造悬疑","制造紧张","情绪压迫","情绪释放","强化反差","引发同情","引发期待"]

约束要求：

- fuzzy_tags 必须抽象、模糊、可泛化
- 不得出现具体人物姓名或具体动作
- 不得与 action_keywords / expanded_tags 重复
- 每个分段必须包含 fuzzy_tags
- fuzzy_tags 只服务于第三阶段深排

4. 输出格式

请严格以 JSON 数组格式输出，不要包含任何额外说明文字。

示例格式：
[
{
    "segment_sequence": 1,
    "script_text": "女人亲手将丈夫推下万丈深渊，可第二天，男人竟毫发无伤地回到了家。",
    "visual_query": {
        "characters": ["女人", "男人"],
        "character_types": ["女人", "男人"],
        "action_keywords": ["推搡", "坠落", "回家", "推门"],
        "environment": "户外悬崖 / 昏暗门口",
        "expanded_tags": "前后对比镜头, 完好无损, 表情异常",
        "interaction_type": ["对峙/冲突"],
        "fuzzy_tags": {
            "audience_state": ["意外","不安"],
            "emotion_intensity": "中",
            "director_intent": ["制造悬疑","强化反差"]
        }
    }
}
]

说明：

1. characters 和 character_types 必须同时输出：
   - characters：具体人物姓名（如"张志焱"）
   - character_types：通用类型（如"男人"），用于与视频侧对齐

2. expanded_tags 用于解决"文学表述 → 物理画面"的语义鸿沟，
提高前两阶段 Embedding 检索的召回率。

3. fuzzy_tags 仅用于第三阶段深排，
用于衡量画面情绪、节奏、导演意图是否一致，
不应影响基础语义匹配结果。

重要提示：
- 必须严格按照 JSON 格式输出，不要包含任何额外文字
- 确保 JSON 可被程序直接解析
- 每个分段必须包含所有必需字段（包括 character_types）
"""

    # ==================== 改造：每个句子单独做结构化描述（不再合并）====================
    def _structure_single_sentence(self, sentence: str, segment_sequence: int) -> Dict[str, Any]:
        """
        对单句话做结构化描述（不再合并句子）
        """
        user_prompt = f"""请对以下句子进行结构化视觉提取：

句子：{sentence}

要求：
1. 直接输出单个segment的JSON格式（不是数组）
2. 包含所有必需字段：characters, character_types, action_keywords, environment, expanded_tags, interaction_type, fuzzy_tags
3. script_text字段填入原句
4. characters 输出具体姓名，character_types 输出通用类型（如"男人"、"女人"）

输出格式：
{{
    "script_text": "{sentence}",
    "visual_query": {{
        "characters": [...],
        "character_types": [...],
        "action_keywords": [...],
        "environment": "...",
        "expanded_tags": "...",
        "interaction_type": [...],
        "fuzzy_tags": {{
            "audience_state": [...],
            "emotion_intensity": "...",
            "director_intent": [...]
        }}
    }}
}}"""
        
        raw, _ = self.model_client.call(self._create_system_prompt(), user_prompt)
        
        try:
            seg = json.loads(raw.strip())
        except json.JSONDecodeError:
            m = re.search(r'\{[\s\S]*\}', raw)
            seg = json.loads(m.group(0)) if m else {}
        
        seg["segment_sequence"] = segment_sequence
        seg["script_text"] = sentence
        
        if "visual_query" in seg:
            if "character_types" not in seg["visual_query"]:
                seg["visual_query"]["character_types"] = []
        
        return seg

    def _structure_by_episode_groups(self, sentence_episodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        主入口：每个句子单独处理，不再按集数合并
        """
        all_segments = []
        
        for idx, sent_info in enumerate(sentence_episodes, 1):
            try:
                # 单句做结构化
                seg = self._structure_single_sentence(sent_info["sentence"], idx)
                
                # 注入集数信息
                seg["episode_target"] = sent_info["episode"]
                seg["episode_confidence"] = round(sent_info["confidence"], 3)
                seg["_sentence_idx"] = sent_info["sentence_idx"]  # 调试用
                
                all_segments.append(seg)
                logger.info(f"Segment {idx}: EP{sent_info['episode']}, {len(sent_info['sentence'])}字, {sent_info['sentence'][:30]}...")
                
            except Exception as e:
                logger.error(f"句子 {idx} 处理失败: {e}")
                all_segments.append({
                    "segment_sequence": idx,
                    "script_text": sent_info["sentence"],
                    "episode_target": sent_info["episode"],
                    "episode_confidence": round(sent_info["confidence"], 3),
                    "visual_query": {
                        "characters": [],
                        "character_types": [],
                        "action_keywords": [],
                        "environment": "",
                        "expanded_tags": "",
                        "interaction_type": [],
                        "fuzzy_tags": {
                            "audience_state": [],
                            "emotion_intensity": "中",
                            "director_intent": []
                        }
                    },
                    "error": str(e)
                })
        
        return all_segments

    # ==================== 以下保持原有逻辑完全不变 ====================
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
        required_vq = ["characters", "character_types", "action_keywords", "environment", "expanded_tags", "interaction_type", "fuzzy_tags"]
        for k in required_vq:
            if k not in vq:
                logger.warning(f"分段 {seq} visual_query 缺少 {k}")
                if k == "character_types":
                    vq["character_types"] = []
        return True

    def _post_process_segments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        for seq, seg in enumerate(segments, 1):
            seg["segment_sequence"] = seq
            if self._validate_segment(seg, seq):
                out.append(seg)
        return out

    async def _tts_for_segments(self, segments: List[Dict[str, Any]], dj: str) -> List[Dict[str, Any]]:
        """保持原有TTS逻辑"""
        tts_dir = self._output_base(dj) / "tts"
        tts_dir.mkdir(parents=True, exist_ok=True)
        ok, fail = 0, 0
        batch_requests = []
        for seg in segments:
            seq = seg["segment_sequence"]
            text = self._sanitize_tts_text(seg["script_text"])
            seg["script_text"] = text
            if not text.strip():
                seg["audio_path"] = None
                seg["audio_duration"] = None
                fail += 1
                continue
            audio_file = tts_dir / f"segment_{seq:04d}.mp3"
            batch_requests.append(
                {
                    "id": str(seq),
                    "text": text,
                    "output": str(audio_file),
                }
            )

        duration_map = await self.tts_processor.batch_text_to_speech_transfer(batch_requests)

        for seg in segments:
            seq = seg["segment_sequence"]
            if not seg.get("script_text", "").strip():
                continue
            dur = duration_map.get(str(seq))
            if dur and dur > 0:
                seg["audio_path"] = f"tts/segment_{seq:04d}.mp3"
                seg["audio_duration"] = dur
                ok += 1
            else:
                seg["audio_path"] = None
                seg["audio_duration"] = None
                fail += 1
        logger.info(f"TTS 完成: 成功 {ok}  | 失败 {fail}")
        return segments

    # ==================== 主入口：单句处理版 ====================
    def process(self, 
                dj: str, 
                input_text_file: Path,
                start_ep: int = 1,
                end_ep: int = 20,
                asr_base_path: str = "/mnt_liuhuan/search_public/aigc/smart_video_v2/audio/videos") -> Dict[str, Any]:
        """
        主流程：每个句子单独处理 → 分别结构化 → 严格单集匹配
        """
        try:
            logger.info(f"开始文本结构化（单句处理版），输入: {input_text_file}")
            
            # 1. 读取文案
            content = self._read_text_file(input_text_file)
            
            # 2. suoying映射（句子级别）
            logger.info("步骤1: 集数映射（suoying）...")
            episode_mapping = self._map_sentences_to_episodes(
                content, dj, start_ep, end_ep, asr_base_path
            )
            sentence_episodes = self._group_sentences_by_episode(episode_mapping)
            logger.info(f"共 {len(sentence_episodes)} 句话，分布在 {len(episode_mapping)} 个集数")
            
            # 3. 【核心改造】每个句子单独做结构化描述
            logger.info("步骤2: 单句结构化...")
            segments = self._structure_by_episode_groups(sentence_episodes)
            segments = self._post_process_segments(segments)
            
            if not segments:
                raise ValueError("没有有效分段")
            
            logger.info(f"生成 {len(segments)} 个segment，每个句子一个segment")
            
            # 4. TTS
            logger.info("步骤3: TTS生成...")
            segments = asyncio.run(self._tts_for_segments(segments, dj))
            
            # 5. 保存结果
            out_json = self._output_base(dj) / f"{input_text_file.stem}_structured.json"
            out_json.write_text(json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf8")
            
            # 统计
            ep_distribution = {}
            for seg in segments:
                ep = seg.get("episode_target", 1)
                ep_distribution[ep] = ep_distribution.get(ep, 0) + 1
            
            tts_ok = sum(1 for s in segments if s.get("audio_path"))
            
            logger.info(f"完成！集数分布: {ep_distribution}")
            
            return {
                "status": "success",
                "dj": dj,
                "output_file": str(out_json),
                "total_segments": len(segments),
                "episode_distribution": ep_distribution,
                "tts_success": tts_ok,
                "tts_fail": len(segments) - tts_ok,
            }
            
        except Exception as e:
            logger.error(f"处理失败: {e}", exc_info=True)
            return {"status": "error", "error_message": str(e)}


# -------------------- CLI --------------------
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python skill_text_structured_description.py <dj> <txt_file_path> [start_ep] [end_ep]")
        print("示例: python skill_text_structured_description.py buxi5 ./buxi5.txt 1 15")
        sys.exit(1)
    
    dj = sys.argv[1]
    txt_path = Path(sys.argv[2])
    start_ep = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    end_ep = int(sys.argv[4]) if len(sys.argv) > 4 else 20
    
    result = TextStructuredDescription().process(dj, txt_path, start_ep, end_ep)
    print(json.dumps(result, ensure_ascii=False, indent=2))
