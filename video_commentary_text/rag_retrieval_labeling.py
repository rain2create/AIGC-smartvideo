#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
高级RAG检索排序系统

基于多维度标签进行匹配和排序
"""

import json
import re
import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

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


def calculate_secondary_label_score(
    query_secondary: str,
    candidate_secondary: str,
    primary_label: str
) -> int:
    """
    计算子标签匹配得分
    
    Returns:
        int: 0=不同, 1=相近, 2=相同
    """
    if query_secondary == candidate_secondary:
        return 2
    
    # 检查是否相近
    similarity_map = SECONDARY_LABEL_SIMILARITY.get(primary_label, {})
    for key, similar_list in similarity_map.items():
        if key == query_secondary and candidate_secondary in similar_list:
            return 1
        if key == candidate_secondary and query_secondary in similar_list:
            return 1
    
    return 0


def calculate_intensity_distance(
    query_intensity: Dict[str, int],
    candidate_intensity: Dict[str, int]
) -> float:
    """
    计算强度标签距离（越小越好）
    
    Returns:
        float: 距离值
    """
    keys = ["pace", "hook_density", "emotion", "conflict", "twist"]
    distance = 0.0
    for key in keys:
        q_val = query_intensity.get(key, 3)
        c_val = candidate_intensity.get(key, 3)
        distance += abs(q_val - c_val)
    return distance


def calculate_style_match_score(
    query_style: Dict[str, Any],
    candidate_style: Dict[str, Any]
) -> float:
    """
    计算写法标签匹配得分
    
    Returns:
        float: 0.0-2.0
    """
    score = 0.0
    
    # POV匹配（优先，0.5分）
    if query_style.get("POV") == candidate_style.get("POV"):
        score += 0.5
    
    # dialogue_ratio匹配（优先，0.5分）
    if query_style.get("dialogue_ratio") == candidate_style.get("dialogue_ratio"):
        score += 0.5
    
    # hook_type交并比（Jaccard，最多1.0分）
    query_hooks = set(query_style.get("hook_type", []))
    candidate_hooks = set(candidate_style.get("hook_type", []))
    if query_hooks or candidate_hooks:
        intersection = len(query_hooks & candidate_hooks)
        union = len(query_hooks | candidate_hooks)
        if union > 0:
            jaccard = intersection / union
            score += jaccard
    
    return min(2.0, score)


def retrieve_by_advanced_labels(
    rag_entries: List[Dict[str, Any]],
    query_labels: Dict[str, Any],
    k: int = 3
) -> List[Tuple[Dict[str, Any], float]]:
    """
    基于多维度标签进行RAG检索和排序
    
    Args:
        rag_entries: RAG条目列表（每个条目应包含完整的标签信息）
        query_labels: 查询标签（格式同label_content_advanced的输出）
        k: 返回TopK个结果
        
    Returns:
        List[Tuple]: [(条目, 得分), ...]，按得分降序排列
    """
    query_global = query_labels.get("global", {})
    query_primary = query_global.get("primary_label")
    query_secondary = query_global.get("secondary_label")
    query_intensity = query_global.get("intensity", {})
    query_style = query_global.get("writing_style", {})
    
    if not query_primary:
        logger.warning("查询标签缺少primary_label，无法检索")
        return []
    
    # 第一步：按主标签过滤
    filtered_entries = []
    for entry in rag_entries:
        entry_labels = entry.get("labels", {})
        if isinstance(entry_labels, str):
            # 兼容旧格式（只有label字段）
            entry_primary = entry.get("label") or entry_labels
        else:
            entry_global = entry_labels.get("global", {}) if isinstance(entry_labels, dict) else {}
            entry_primary = entry_global.get("primary_label") or entry.get("label")
        
        if entry_primary == query_primary:
            filtered_entries.append(entry)
    
    if not filtered_entries:
        logger.warning(f"未找到主标签为 '{query_primary}' 的条目")
        return []
    
    logger.info(f"主标签过滤后剩余 {len(filtered_entries)} 条")
    
    # 第二步：计算综合得分
    scored_entries = []
    for entry in filtered_entries:
        entry_labels = entry.get("labels", {})
        
        # 解析条目标签
        if isinstance(entry_labels, str):
            # 旧格式，只有主标签
            entry_primary = entry_labels
            entry_secondary = ""
            entry_intensity = {}
            entry_style = {}
        else:
            entry_global = entry_labels.get("global", {}) if isinstance(entry_labels, dict) else {}
            entry_primary = entry_global.get("primary_label", entry.get("label", ""))
            entry_secondary = entry_global.get("secondary_label", "")
            entry_intensity = entry_global.get("intensity", {})
            entry_style = entry_global.get("writing_style", {})
        
        # 计算各项得分
        primary_match = 1  # 已过滤，恒为1
        
        secondary_match = calculate_secondary_label_score(
            query_secondary, entry_secondary, query_primary
        )
        
        intensity_distance = calculate_intensity_distance(
            query_intensity, entry_intensity
        )
        
        style_match = calculate_style_match_score(
            query_style, entry_style
        )
        
        # 综合得分公式
        score = (
            3 * primary_match +
            2 * secondary_match -
            0.5 * intensity_distance +
            1 * style_match
        )
        
        scored_entries.append((entry, score))
    
    # 按得分降序排序
    scored_entries.sort(key=lambda x: x[1], reverse=True)
    
    # 第三步：去重（同secondary_label最多取1条；同hook_type组合最多取1条）
    deduplicated = []
    seen_secondary = set()
    seen_hook_combo = set()
    
    for entry, score in scored_entries:
        entry_labels = entry.get("labels", {})
        if isinstance(entry_labels, dict):
            entry_global = entry_labels.get("global", {})
            entry_secondary = entry_global.get("secondary_label", "")
            entry_hooks = tuple(sorted(entry_global.get("writing_style", {}).get("hook_type", [])))
        else:
            entry_secondary = ""
            entry_hooks = tuple()
        
        # 检查是否重复
        if entry_secondary and entry_secondary in seen_secondary:
            continue
        if entry_hooks and entry_hooks in seen_hook_combo:
            continue
        
        deduplicated.append((entry, score))
        if entry_secondary:
            seen_secondary.add(entry_secondary)
        if entry_hooks:
            seen_hook_combo.add(entry_hooks)
        
        if len(deduplicated) >= k:
            break
    
    logger.info(f"检索完成，返回 {len(deduplicated)} 条结果")
    return deduplicated


def de_plotize_content(content: str) -> str:
    """
    去剧情化处理：将人名/地名替换为占位符，避免内容迁移带来幻觉
    
    Args:
        content: 原始内容
        
    Returns:
        str: 去剧情化后的内容
    """
    # 简单实现：使用正则替换常见人名模式
    # 实际可以更复杂，使用NER识别
    
    # 替换常见人名模式（2-4字，可能包含常见姓氏）
    # 常见姓氏列表（部分）
    common_surnames = r'[张李王刘陈杨黄赵吴周徐孙马朱胡郭何高林罗郑梁谢宋唐许韩冯邓曹彭曾肖田董袁潘于蒋蔡余杜叶程苏魏吕丁任沈姚卢姜崔钟谭陆汪范金石廖贾夏韦付方白邹孟熊秦邱江尹薛闫段雷侯龙史陶黎贺顾毛郝龚邵万钱严覃武戴莫孔向汤]'
    content = re.sub(common_surnames + r'[一-龥]{1,3}', 'A', content)
    
    # 替换常见地名（可以扩展）
    # 这里只是示例，实际应该更智能，可以使用NER工具
    
    return content

