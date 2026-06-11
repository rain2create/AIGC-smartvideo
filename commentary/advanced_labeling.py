#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
高级多维度标签打标系统

支持：
- 主标签 + 子标签 + 强度标签 + 写法标签
- 事件级标签（ASR分段打标）
- 全局标签聚合
"""

import json
import re
import logging
from typing import Dict, List, Optional, Any, Tuple
from model_client import ModelClient

logger = logging.getLogger(__name__)

# ============================================================================
# 标签体系定义
# ============================================================================

PRIMARY_LABELS = ["走心催泪", "悬疑解谜", "搞笑吐槽", "爽快打脸", "温情叙事"]

SECONDARY_LABELS = {
    "走心催泪": ["生离死别", "亲情救赎", "误会遗憾", "自我牺牲", "绝境翻盘", "忍辱负重"],
    "爽快打脸": ["绿茶陷害", "豪门羞辱", "追妻火葬场", "身份反转", "恶人受罚", "强势护短", "反杀复仇"],
    "温情叙事": ["亲情日常", "爱情治愈", "成长陪伴", "旧友重逢", "小人物善意", "家庭和解"],
    "悬疑解谜": ["线索递进", "局中局", "身份谜团", "误导反转", "证据翻盘", "暗线追踪"],
    "搞笑吐槽": ["反差吐槽", "尴尬社死", "离谱夸张", "嘴炮互怼", "误会喜剧", "沙雕反转"]
}

WRITING_STYLE_POV = ["旁白叙述", "吐槽旁白", "强代入"]
WRITING_STYLE_SENTENCE = ["短句为主", "长短交错"]
WRITING_STYLE_HOOK_TYPE = ["反差断言", "疑问悬念", "未解释因果", "强转折"]
WRITING_STYLE_TONE = ["克制", "爽文", "温柔", "冷幽默"]
WRITING_STYLE_DIALOGUE_RATIO = ["低", "中", "高"]

# 子标签相近映射（用于检索时的加权）
SECONDARY_LABEL_SIMILARITY = {
    "走心催泪": {
        "生离死别": ["自我牺牲", "忍辱负重"],
        "亲情救赎": ["家庭和解", "成长陪伴"],
        "误会遗憾": ["绝境翻盘"],
    },
    "爽快打脸": {
        "绿茶陷害": ["恶人受罚"],
        "身份反转": ["反杀复仇"],
        "豪门羞辱": ["强势护短"],
    },
    "温情叙事": {
        "亲情日常": ["家庭和解", "成长陪伴"],
        "爱情治愈": ["小人物善意"],
    },
    "悬疑解谜": {
        "线索递进": ["暗线追踪"],
        "误导反转": ["证据翻盘"],
        "身份谜团": ["局中局"],
    },
    "搞笑吐槽": {
        "反差吐槽": ["沙雕反转"],
        "尴尬社死": ["误会喜剧"],
    }
}


# ============================================================================
# 系统提示词
# ============================================================================

LABELING_SYSTEM_PROMPT = """你是短剧/短视频解说文案的【多维度标签打标专家】。

你的任务：基于输入的剧情简介+对白片段（ASR），输出完整的多维度标签体系。

【标签体系说明】

1. 主标签（primary_label）：必须从以下5类中选择一个
   - 走心催泪 / 悬疑解谜 / 搞笑吐槽 / 爽快打脸 / 温情叙事

2. 子标签（secondary_label）：必须从对应主标签的子标签列表中选择一个
   - 走心催泪：生离死别 / 亲情救赎 / 误会遗憾 / 自我牺牲 / 绝境翻盘 / 忍辱负重
   - 爽快打脸：绿茶陷害 / 豪门羞辱 / 追妻火葬场 / 身份反转 / 恶人受罚 / 强势护短 / 反杀复仇
   - 温情叙事：亲情日常 / 爱情治愈 / 成长陪伴 / 旧友重逢 / 小人物善意 / 家庭和解
   - 悬疑解谜：线索递进 / 局中局 / 身份谜团 / 误导反转 / 证据翻盘 / 暗线追踪
   - 搞笑吐槽：反差吐槽 / 尴尬社死 / 离谱夸张 / 嘴炮互怼 / 误会喜剧 / 沙雕反转

3. 强度标签（intensity）：每个维度1-5分（整数）
   - pace：节奏/信息密度强度（1=慢/稀疏，5=快/密集）
   - hook_density：钩子密度强度（1=少，5=多）
   - emotion：情绪强度（1=平淡，5=强烈）
   - conflict：冲突强度（1=弱，5=强）
   - twist：反转强度（1=无反转，5=强反转）

4. 写法标签（writing_style）
   - POV：旁白叙述 / 吐槽旁白 / 强代入（默认：旁白叙述）
   - sentence：短句为主 / 长短交错
   - hook_type：可多选，从 ["反差断言", "疑问悬念", "未解释因果", "强转折"] 中选择
   - tone：克制 / 爽文 / 温柔 / 冷幽默
   - dialogue_ratio：低 / 中 / 高（默认：低，尽量避免直接台词）

5. 事件级标签（events）：将ASR按事件切分为若干段（每150-250字或按场景/人物变化），对每个事件分别打标

【输出要求】
- 严格输出JSON格式，不要Markdown代码块标记
- 不要额外说明文字
- 所有字段必须齐全
- confidence 为 0.0-1.0 的浮点数
- evidence 为来自ASR的证据句列表（每句不超过50字）
"""


# ============================================================================
# 事件切分函数
# ============================================================================

def segment_asr_into_events(asr_text: str, min_chars: int = 150, max_chars: int = 250) -> List[Dict[str, Any]]:
    """
    将ASR文本按事件切分为若干段
    
    Args:
        asr_text: ASR文本
        min_chars: 每段最小字符数
        max_chars: 每段最大字符数
        
    Returns:
        List[Dict]: 事件列表，每个事件包含 {start_char, end_char, text}
    """
    events = []
    current_start = 0
    current_text = ""
    
    # 按句号、问号、感叹号切分
    sentences = re.split(r'[。！？\n]', asr_text)
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # 如果当前段落加上新句子超过max_chars，保存当前段落
        if len(current_text) + len(sentence) > max_chars and len(current_text) >= min_chars:
            events.append({
                "start_char": current_start,
                "end_char": current_start + len(current_text),
                "text": current_text.strip()
            })
            current_start = current_start + len(current_text)
            current_text = sentence
        else:
            current_text += sentence + "。"
    
    # 保存最后一段
    if current_text.strip():
        events.append({
            "start_char": current_start,
            "end_char": current_start + len(current_text),
            "text": current_text.strip()
        })
    
    return events


# ============================================================================
# 核心打标函数
# ============================================================================

def label_content_advanced(
    model_client: ModelClient,
    content: str,
    asr_text: Optional[str] = None,
    enable_event_labeling: bool = True
) -> Dict[str, Any]:
    """
    高级多维度标签打标
    
    Args:
        model_client: 模型客户端
        content: 剧情简介
        asr_text: ASR文本（可选，用于事件级打标）
        enable_event_labeling: 是否启用事件级打标
        
    Returns:
        Dict: 完整的标签JSON
    """
    logger.info("开始高级多维度标签打标...")
    
    # 构建用户提示词
    user_prompt = f"""请对以下内容进行多维度标签打标。

【剧情简介】
{content[:5000]}

"""
    
    if asr_text and enable_event_labeling:
        user_prompt += f"""【ASR对白文本】
{asr_text[:8000]}

请先对ASR文本进行事件切分（每150-250字或按场景/人物变化），然后对每个事件分别打标。
最后再聚合为全局标签（全局标签=事件标签加权多数，权重：冲突/反转事件更高）。
"""
    else:
        user_prompt += "\n请直接输出全局标签（不进行事件级打标）。\n"
    
    user_prompt += f"""
请严格按照以下JSON Schema输出：

{{
  "global": {{
    "primary_label": "主标签（五选一）",
    "secondary_label": "子标签（从对应主标签的子标签列表中选择）",
    "intensity": {{
      "pace": 1-5,
      "hook_density": 1-5,
      "emotion": 1-5,
      "conflict": 1-5,
      "twist": 1-5
    }},
    "writing_style": {{
      "POV": "旁白叙述|吐槽旁白|强代入",
      "sentence": "短句为主|长短交错",
      "hook_type": ["反差断言", "疑问悬念", "未解释因果", "强转折"],
      "tone": "克制|爽文|温柔|冷幽默",
      "dialogue_ratio": "低|中|高"
    }},
    "confidence": 0.0-1.0,
    "evidence": ["证据句1", "证据句2"]
  }},
  "events": [
    {{
      "event_id": 1,
      "span": {{"start_char": 0, "end_char": 300}},
      "summary": "一句话概括本事件",
      "primary_label": "...",
      "secondary_label": "...",
      "intensity": {{"pace": 1-5, "hook_density": 1-5, "emotion": 1-5, "conflict": 1-5, "twist": 1-5}},
      "confidence": 0.0-1.0,
      "evidence": ["证据句..."]
    }}
  ]
}}

只输出JSON，不要Markdown代码块标记，不要额外说明文字。"""
    
    try:
        raw_response, _ = model_client.call(
            system_prompt=LABELING_SYSTEM_PROMPT,
            user_prompt=user_prompt
        )
        
        raw_response = (raw_response or "").strip()
        logger.info(f"打标响应长度: {len(raw_response)} 字符")
        
        # 提取JSON
        json_str = _extract_json_from_response(raw_response)
        if not json_str:
            logger.warning("无法从响应中提取JSON")
            return _create_fallback_label(content)
        
        parsed = json.loads(json_str)
        
        # 验证和规范化
        validated = _validate_and_normalize_labels(parsed)
        
        logger.info(f"✅ 打标成功: primary={validated['global']['primary_label']}, secondary={validated['global']['secondary_label']}")
        return validated
        
    except Exception as e:
        logger.error(f"打标失败: {e}", exc_info=True)
        return _create_fallback_label(content)


def _extract_json_from_response(response: str) -> Optional[str]:
    """从模型响应中提取JSON对象"""
    if not response:
        return None
    
    # 去除代码块标记
    response = re.sub(r'^```(?:json)?\s*', '', response, flags=re.MULTILINE)
    response = re.sub(r'\s*```$', '', response)
    response = response.strip()
    
    # 尝试整体解析
    try:
        json.loads(response)
        return response
    except:
        pass
    
    # 提取第一个 {...}
    match = re.search(r'\{[\s\S]*\}', response)
    if match:
        return match.group(0)
    
    return None


def _validate_and_normalize_labels(label_data: Dict[str, Any]) -> Dict[str, Any]:
    """验证和规范化标签数据"""
    # 验证全局标签
    if "global" not in label_data:
        raise ValueError("缺少 global 字段")
    
    global_label = label_data["global"]
    
    # 验证主标签
    if "primary_label" not in global_label:
        raise ValueError("缺少 primary_label")
    if global_label["primary_label"] not in PRIMARY_LABELS:
        logger.warning(f"主标签 {global_label['primary_label']} 不在候选列表中，使用默认值")
        global_label["primary_label"] = PRIMARY_LABELS[0]
    
    # 验证子标签
    primary = global_label["primary_label"]
    if "secondary_label" not in global_label:
        raise ValueError("缺少 secondary_label")
    if global_label["secondary_label"] not in SECONDARY_LABELS.get(primary, []):
        logger.warning(f"子标签 {global_label['secondary_label']} 不在候选列表中，使用默认值")
        global_label["secondary_label"] = SECONDARY_LABELS.get(primary, [""])[0] if SECONDARY_LABELS.get(primary) else ""
    
    # 验证强度标签
    if "intensity" not in global_label:
        global_label["intensity"] = {}
    for key in ["pace", "hook_density", "emotion", "conflict", "twist"]:
        if key not in global_label["intensity"]:
            global_label["intensity"][key] = 3  # 默认值
        else:
            # 确保在1-5范围内
            global_label["intensity"][key] = max(1, min(5, int(global_label["intensity"][key])))
    
    # 验证写法标签
    if "writing_style" not in global_label:
        global_label["writing_style"] = {}
    
    ws = global_label["writing_style"]
    if "POV" not in ws or ws["POV"] not in WRITING_STYLE_POV:
        ws["POV"] = "旁白叙述"
    if "sentence" not in ws or ws["sentence"] not in WRITING_STYLE_SENTENCE:
        ws["sentence"] = "长短交错"
    if "hook_type" not in ws or not isinstance(ws["hook_type"], list):
        ws["hook_type"] = []
    if "tone" not in ws or ws["tone"] not in WRITING_STYLE_TONE:
        ws["tone"] = "克制"
    if "dialogue_ratio" not in ws or ws["dialogue_ratio"] not in WRITING_STYLE_DIALOGUE_RATIO:
        ws["dialogue_ratio"] = "低"
    
    # 验证confidence
    if "confidence" not in global_label:
        global_label["confidence"] = 0.5
    else:
        global_label["confidence"] = max(0.0, min(1.0, float(global_label["confidence"])))
    
    # 验证evidence
    if "evidence" not in global_label or not isinstance(global_label["evidence"], list):
        global_label["evidence"] = []
    
    # 验证事件标签
    if "events" not in label_data:
        label_data["events"] = []
    
    for event in label_data["events"]:
        if "primary_label" not in event:
            event["primary_label"] = primary
        if "secondary_label" not in event:
            event["secondary_label"] = global_label["secondary_label"]
        if "intensity" not in event:
            event["intensity"] = global_label["intensity"].copy()
        if "confidence" not in event:
            event["confidence"] = global_label["confidence"]
        if "evidence" not in event or not isinstance(event["evidence"], list):
            event["evidence"] = []
    
    return label_data


def _create_fallback_label(content: str) -> Dict[str, Any]:
    """创建回退标签（当打标失败时）"""
    logger.warning("使用回退标签")
    return {
        "global": {
            "primary_label": "温情叙事",
            "secondary_label": "亲情日常",
            "intensity": {
                "pace": 3,
                "hook_density": 3,
                "emotion": 3,
                "conflict": 3,
                "twist": 2
            },
            "writing_style": {
                "POV": "旁白叙述",
                "sentence": "长短交错",
                "hook_type": [],
                "tone": "克制",
                "dialogue_ratio": "低"
            },
            "confidence": 0.3,
            "evidence": [content[:50] if content else ""]
        },
        "events": []
    }

