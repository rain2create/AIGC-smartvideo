#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S4：匹配算法模块（四阶段匹配 + 动态权重 + LLM精排）
核心原则：每句文案必须在对应集数内找到匹配，不跳过、不回退
四阶段流程：
  Stage1: 语义召回 → TOP 20
  Stage2: 视觉精排 → TOP 10
  Stage3: 情感深排 → TOP 5
  Stage4: LLM精排 → TOP 1
"""

import json
import pickle
import logging
import requests
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import sys

sys.path.append(str(Path(__file__).parent.parent))
from commentary.model_client import ModelClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def classify_script_type(visual_query: Dict[str, Any]) -> str:
    """
    根据文案特征判断类型
    类型：
      - emotion_action: 情绪高 + 动作多（优先）
      - emotion_high: 情绪强度高
      - action_heavy: 动作词多（>=2个）
      - interaction_heavy: 复杂互动关系
      - balanced: 平衡型
    """
    action_keywords = visual_query.get("action_keywords", [])
    fuzzy_tags = visual_query.get("fuzzy_tags", {})
    emotion = fuzzy_tags.get("emotion_intensity", "中")
    interaction_type = visual_query.get("interaction_type", [])
    
    is_emotion_high = emotion == "高"
    is_action_heavy = len(action_keywords) >= 2
    
    complex_interactions = ["对峙/冲突", "追逐/打斗", "亲密互动"]
    is_interaction_heavy = any(it in complex_interactions for it in interaction_type)
    
    if is_emotion_high and is_action_heavy:
        return "emotion_action"
    elif is_emotion_high:
        return "emotion_high"
    elif is_action_heavy:
        return "action_heavy"
    elif is_interaction_heavy:
        return "interaction_heavy"
    else:
        return "balanced"


def get_stage2_weights(script_type: str, has_dialogue: bool = True) -> Dict[str, float]:
    """
    返回第二阶段动态权重
    Stage2关注视觉要素：characters, actions, environment, dialogue
    has_dialogue: 视频侧是否有dialogue，无则重新分配权重
    """
    if has_dialogue:
        weights_map = {
            "emotion_action": {"characters": 0.25, "actions": 0.35, "environment": 0.15, "dialogue": 0.25},
            "emotion_high": {"characters": 0.35, "actions": 0.15, "environment": 0.25, "dialogue": 0.25},
            "action_heavy": {"characters": 0.15, "actions": 0.45, "environment": 0.15, "dialogue": 0.25},
            "interaction_heavy": {"characters": 0.3, "actions": 0.3, "environment": 0.15, "dialogue": 0.25},
            "balanced": {"characters": 0.25, "actions": 0.3, "environment": 0.2, "dialogue": 0.25}
        }
    else:
        weights_map = {
            "emotion_action": {"characters": 0.3, "actions": 0.4, "environment": 0.3, "dialogue": 0.0},
            "emotion_high": {"characters": 0.4, "actions": 0.2, "environment": 0.4, "dialogue": 0.0},
            "action_heavy": {"characters": 0.2, "actions": 0.6, "environment": 0.2, "dialogue": 0.0},
            "interaction_heavy": {"characters": 0.4, "actions": 0.4, "environment": 0.2, "dialogue": 0.0},
            "balanced": {"characters": 0.33, "actions": 0.34, "environment": 0.33, "dialogue": 0.0}
        }
    return weights_map.get(script_type, weights_map["balanced"])


def get_stage3_weights(script_type: str) -> Dict[str, float]:
    """
    返回第三阶段动态权重
    Stage3关注情感意图：interaction, emotion, intent
    """
    weights_map = {
        "emotion_action": {"interaction": 0.35, "emotion": 0.35, "intent": 0.3},
        "emotion_high": {"interaction": 0.2, "emotion": 0.5, "intent": 0.3},
        "action_heavy": {"interaction": 0.5, "emotion": 0.2, "intent": 0.3},
        "interaction_heavy": {"interaction": 0.5, "emotion": 0.2, "intent": 0.3},
        "balanced": {"interaction": 0.33, "emotion": 0.34, "intent": 0.33}
    }
    return weights_map.get(script_type, weights_map["balanced"])


class SkillMatchAndRank:
    def __init__(self, 
                 emb_api_url: str = "http://11.93.209.140:17998/emb/",
                 llm_api_url: str = "https://api.360.cn/v1/chat/completions",
                 llm_api_key: str = "fk1379989049.aEmK59LGEcoiwxat0pGE18JPvtSBkrybaf481994",
                 llm_model: str = "google/gemini-3-pro-preview"):
        self.emb_api_url = emb_api_url
        self._emb_cache = {}
        self.model_client = ModelClient(api_url=llm_api_url, api_key=llm_api_key, model=llm_model, timeout=120)

    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        if not text or not text.strip():
            return None
        text_key = text[:100]
        if text_key in self._emb_cache:
            return self._emb_cache[text_key]
        try:
            r = requests.get(self.emb_api_url, params={"title": text}, timeout=30)
            if r.status_code == 200 and isinstance(r.json(), list):
                emb = np.array(r.json(), dtype=np.float32)
                self._emb_cache[text_key] = emb
                return emb
            return None
        except Exception:
            return None

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        if not text1 or not text2:
            return 0.0
        emb1 = self._get_embedding(text1)
        emb2 = self._get_embedding(text2)
        if emb1 is None or emb2 is None:
            return 0.0
        return self._cosine(emb1, emb2)

    def _check_structured_description(self, dj: str) -> Tuple[bool, Optional[List], Optional[str]]:
        try:
            cache_dir = Path(__file__).parent / "cache" / dj
            json_file_path = cache_dir / "structured_description.json"

            if not json_file_path.exists():
                return False, None, f"结构化描述文件不存在: {json_file_path}"

            logger.info(f"读取结构化描述文件: {json_file_path}")
            data = json.loads(json_file_path.read_text(encoding="utf-8"))

            if isinstance(data, dict):
                segments = data.get("segments", [])
            elif isinstance(data, list):
                segments = data
            else:
                return False, None, "结构化描述文件格式错误"

            if not segments:
                return False, None, "未找到片段数据"

            logger.info(f"检查通过: 共 {len(segments)} 个视频片段")
            return True, segments, None

        except Exception as e:
            logger.error("检查结构化描述文件异常", exc_info=True)
            return False, None, str(e)

    def _check_script_json(self, script_json_path: str) -> Tuple[bool, Optional[List], Optional[str]]:
        try:
            script_json_path = Path(script_json_path)
            if not script_json_path.exists():
                return False, None, f"文案 JSON 文件不存在: {script_json_path}"

            logger.info(f"读取文案 JSON 文件: {script_json_path}")
            script_data = json.loads(script_json_path.read_text(encoding="utf-8"))

            if not isinstance(script_data, list):
                return False, None, "文案 JSON 文件格式错误，应为 list"
            if not script_data:
                return False, None, "文案数据为空"

            for i, item in enumerate(script_data):
                if "episode_target" not in item:
                    item["episode_target"] = 1

            logger.info(f"检查通过: 文案片段数={len(script_data)}")
            return True, script_data, None

        except Exception as e:
            logger.error("检查文案 JSON 异常", exc_info=True)
            return False, None, str(e)

    def _build_episode_index(self, video_segments: List[Dict], embedding_file_path: str, cache_dir: Path):
        emb_full = cache_dir / embedding_file_path
        try:
            raw = pickle.loads(emb_full.read_bytes())
            if isinstance(raw, dict):
                raw = raw.get("embeddings", [])
            if not isinstance(raw, list):
                raise ValueError("embedding 文件格式错误，应为 list 或包含 embeddings 的 dict")
        except Exception as e:
            logger.error(f"加载embedding失败: {e}")
            return {}, {}

        emb_by_episode = {}
        segments_by_episode = {}
        seg_by_episode_and_id = {}
        seg_by_id = {}
        seg_by_path = {}

        for seg in video_segments:
            ep = seg.get("episode", 1)
            seg_id = seg.get("segment_id")

            if ep not in segments_by_episode:
                segments_by_episode[ep] = []
            segments_by_episode[ep].append(seg)

            if seg_id is None:
                continue

            seg_id_int = int(seg_id)
            seg_by_id.setdefault(seg_id_int, []).append(seg)
            seg_by_episode_and_id[(int(ep), seg_id_int)] = seg

            seg_path = seg.get("segment_file_path")
            if seg_path:
                seg_by_path[str(seg_path)] = seg

        unresolved_count = 0
        ambiguous_legacy_count = 0
        legacy_without_metadata_count = 0

        for item in raw:
            if "segment_id" not in item or "embedding" not in item:
                continue

            seg_id_int = int(item["segment_id"])
            record_episode = item.get("episode")
            record_path = item.get("segment_file_path")
            segment = None

            if record_episode is not None:
                segment = seg_by_episode_and_id.get((int(record_episode), seg_id_int))

            if segment is None and record_path:
                segment = seg_by_path.get(str(record_path))

            if segment is None:
                legacy_without_metadata_count += 1
                id_matches = seg_by_id.get(seg_id_int, [])
                if len(id_matches) == 1:
                    segment = id_matches[0]
                elif id_matches:
                    ambiguous_legacy_count += 1

            if segment is None:
                unresolved_count += 1
                continue

            ep = int(segment.get("episode", 1))
            emb_by_episode.setdefault(ep, {})[seg_id_int] = np.array(item["embedding"], dtype=np.float32)

        logger.info(f"集数索引构建完成: {len(segments_by_episode)} 集")
        for ep in sorted(segments_by_episode.keys()):
            logger.info(f"  EP{ep}: {len(segments_by_episode[ep])} 片段, {len(emb_by_episode.get(ep, {}))} 个embedding")

        if ambiguous_legacy_count:
            logger.warning(
                "检测到 %s 条旧版 embedding 记录缺少 episode/path 且 segment_id 跨集重复，已跳过。"
                "建议先重跑 S2 生成新版 embeddings.pkl",
                ambiguous_legacy_count,
            )
        elif legacy_without_metadata_count:
            logger.info(
                "检测到 %s 条旧版 embedding 记录缺少 episode/path，已按唯一 segment_id 兼容加载",
                legacy_without_metadata_count,
            )

        if unresolved_count:
            logger.warning("有 %s 条 embedding 记录未能对齐到视频片段，已跳过", unresolved_count)
        
        return emb_by_episode, segments_by_episode

    def _find_segment_by_id(self, segments: List[Dict], seg_id: int) -> Optional[Dict]:
        for seg in segments:
            if seg.get("segment_id") == seg_id:
                return seg
        return None

    def _generate_match_reason(self, script_text: str, seg_info: Dict, score: float) -> str:
        desc = seg_info.get("structured_description", {})
        visual_summary = desc.get("visual_summary", "")
        reason = visual_summary[:80] if visual_summary else "基于语义相似度匹配"
        if len(visual_summary) > 80:
            reason += "..."
        return reason

    def _stage1_semantic_recall(self,
                                 script_item: Dict,
                                 target_embs: Dict[int, np.ndarray],
                                 target_segments: List[Dict],
                                 top_n: int = 20) -> List[Dict]:
        """
        第一阶段：语义召回
        Query: script_text + expanded_tags
        输出: TOP 20
        """
        script_text = script_item.get("script_text", "")
        vq = script_item.get("visual_query", {})
        expanded_tags = vq.get("expanded_tags", "")
        
        query = script_text
        if expanded_tags:
            query = f"{script_text} {expanded_tags}"
        
        qemb = self._get_embedding(query)
        
        candidates = []
        for seg_id, emb in target_embs.items():
            seg_info = self._find_segment_by_id(target_segments, seg_id)
            if not seg_info:
                continue
            
            if qemb is not None:
                score = self._cosine(qemb, emb)
            else:
                score = 0.3
            
            candidates.append({
                "segment_id": seg_id,
                "stage1_score": score,
                "seg_info": seg_info
            })
        
        candidates.sort(key=lambda x: x["stage1_score"], reverse=True)
        return candidates[:top_n]

    def _stage2_visual_rerank(self,
                                script_item: Dict,
                                stage1_candidates: List[Dict],
                                script_type: str,
                                top_n: int = 10) -> List[Dict]:
        """
        第二阶段：视觉特征精排
        Query: character_types + actions + environment + dialogue
        权重：动态（根据是否有dialogue调整）
        输出: TOP 10
        """
        vq = script_item.get("visual_query", {})
        script_text = script_item.get("script_text", "")
        
        query_character_types = "、".join(vq.get("character_types", []))
        query_actions = "、".join(vq.get("action_keywords", []))
        query_env = vq.get("environment", "")
        query_dialogue = script_text
        
        scored_candidates = []
        for c in stage1_candidates:
            seg_info = c["seg_info"]
            desc = seg_info.get("structured_description", {})
            
            video_characters = "、".join(desc.get("characters", []))
            video_actions = "、".join(desc.get("actions", []))
            video_env = desc.get("environment", "")
            video_dialogue = desc.get("dialogue_or_text", "")
            
            has_dialogue = bool(video_dialogue and video_dialogue.strip() not in ["null", "None", ""])
            weights = get_stage2_weights(script_type, has_dialogue)
            
            sim_characters = self._calculate_similarity(query_character_types, video_characters) if query_character_types else 0.0
            sim_actions = self._calculate_similarity(query_actions, video_actions) if query_actions else 0.0
            sim_env = self._calculate_similarity(query_env, video_env) if query_env else 0.0
            
            if has_dialogue:
                sim_dialogue = self._calculate_similarity(query_dialogue, video_dialogue)
            else:
                sim_dialogue = 0.0
            
            stage2_score = (weights["characters"] * sim_characters +
                          weights["actions"] * sim_actions +
                          weights["environment"] * sim_env +
                          weights["dialogue"] * sim_dialogue)
            
            final_score = 0.4 * c["stage1_score"] + 0.6 * stage2_score
            
            scored_candidates.append({
                "segment_id": c["segment_id"],
                "stage1_score": c["stage1_score"],
                "stage2_score": stage2_score,
                "combined_score": final_score,
                "seg_info": seg_info,
                "breakdown": {
                    "sim_characters": round(sim_characters, 3),
                    "sim_actions": round(sim_actions, 3),
                    "sim_env": round(sim_env, 3),
                    "sim_dialogue": round(sim_dialogue, 3),
                    "has_dialogue": has_dialogue,
                    "stage2_weighted": round(stage2_score, 3)
                }
            })
        
        scored_candidates.sort(key=lambda x: x["combined_score"], reverse=True)
        return scored_candidates[:top_n]

    def _stage3_emotion_rerank(self,
                                 script_item: Dict,
                                 stage2_candidates: List[Dict],
                                 script_type: str,
                                 top_n: int = 5) -> List[Dict]:
        """
        第三阶段：情感意图深排
        Query: interaction + emotion + intent
        权重：动态
        输出: TOP 5
        
        注意：文案侧director_intent和视频侧functional_intent枚举值不完全对应
        文案侧：制造悬疑/制造紧张/情绪压迫/情绪释放/强化反差/引发同情/引发期待
        视频侧：引发好奇/制造冲突/信息铺垫/人物亮相/制造反转预期/情绪渲染/氛围建立/视觉刺激
        需要做映射后计算相似度
        """
        weights = get_stage3_weights(script_type)
        vq = script_item.get("visual_query", {})
        fuzzy_tags = vq.get("fuzzy_tags", {})
        
        query_interaction = "、".join(vq.get("interaction_type", []))
        query_emotion = fuzzy_tags.get("emotion_intensity", "")
        if fuzzy_tags.get("audience_state"):
            query_emotion = f"{query_emotion} {' '.join(fuzzy_tags['audience_state'])}"
        
        script_intent = fuzzy_tags.get("director_intent", [])
        intent_mapping = {
            "制造悬疑": "引发好奇 制造反转预期",
            "制造紧张": "制造冲突 紧张氛围",
            "情绪压迫": "情绪渲染 氛围建立",
            "情绪释放": "情绪渲染",
            "强化反差": "制造反转预期 强化对比",
            "引发同情": "情绪渲染 引发共鸣",
            "引发期待": "引发好奇 悬念设置"
        }
        mapped_intents = []
        for intent in script_intent:
            if intent in intent_mapping:
                mapped_intents.append(intent_mapping[intent])
            else:
                mapped_intents.append(intent)
        query_intent = "、".join(mapped_intents) if mapped_intents else ""
        
        scored_candidates = []
        for c in stage2_candidates:
            seg_info = c["seg_info"]
            desc = seg_info.get("structured_description", {})
            
            video_interaction = "、".join(desc.get("interaction_type", []))
            video_emotion = str(desc.get("emotion_intensity", ""))
            video_intent = "、".join(desc.get("functional_intent", []))
            
            sim_interaction = self._calculate_similarity(query_interaction, video_interaction) if query_interaction else 0.0
            sim_emotion = self._calculate_similarity(query_emotion, video_emotion) if query_emotion else 0.5
            sim_intent = self._calculate_similarity(query_intent, video_intent) if query_intent else 0.0
            
            stage3_score = (weights["interaction"] * sim_interaction +
                          weights["emotion"] * sim_emotion +
                          weights["intent"] * sim_intent)
            
            final_score = 0.3 * c["combined_score"] + 0.7 * stage3_score
            
            scored_candidates.append({
                "segment_id": c["segment_id"],
                "stage1_score": c["stage1_score"],
                "stage2_score": c["stage2_score"],
                "stage3_score": stage3_score,
                "final_score": final_score,
                "seg_info": seg_info,
                "breakdown": {
                    **c["breakdown"],
                    "sim_interaction": round(sim_interaction, 3),
                    "sim_emotion": round(sim_emotion, 3),
                    "sim_intent": round(sim_intent, 3),
                    "stage3_weighted": round(stage3_score, 3)
                }
            })
        
        scored_candidates.sort(key=lambda x: x["final_score"], reverse=True)
        return scored_candidates[:top_n]

    def _stage4_llm_final(self,
                           script_item: Dict,
                           stage3_candidates: List[Dict]) -> Tuple[Dict, str]:
        """
        第四阶段：LLM精排（当前已禁用，直接返回Stage3 TOP1）
        Input: 5个候选片段完整描述
        输出: TOP 1 + 选择理由
        """
        # ========== 当前禁用LLM精排，直接返回Stage3 TOP1 ==========
        if not stage3_candidates:
            return None, "无候选"
        
        top1 = stage3_candidates[0]
        logger.info(f"  [Stage4跳过] 直接使用第三阶段TOP1: 片段{top1['segment_id']}, 分数={top1['final_score']:.3f}")
        return top1, "[Stage3] 第三阶段TOP1"
        
        # ========== 以下是LLM精排代码（暂时禁用）==========
        # script_text = script_item.get("script_text", "")
        # vq = script_item.get("visual_query", {})
        # fuzzy_tags = vq.get("fuzzy_tags", {})
        # 
        # character_types = "、".join(vq.get("character_types", []))
        # action_keywords = "、".join(vq.get("action_keywords", []))
        # environment = vq.get("environment", "")
        # interaction_type = "、".join(vq.get("interaction_type", []))
        # emotion_intensity = fuzzy_tags.get("emotion_intensity", "")
        # director_intent = "、".join(fuzzy_tags.get("director_intent", []))
        # 
        # candidates_desc = []
        # for i, c in enumerate(stage3_candidates, 1):
        #     seg_info = c["seg_info"]
        #     desc = seg_info.get("structured_description", {})
        #     dialogue_text = desc.get('dialogue_or_text', '')
        #     dialogue_display = dialogue_text if dialogue_text and dialogue_text.strip() not in ["null", "None", ""] else "无"
        #     candidates_desc.append(f"""
        # 片段{i}（综合得分：{c['final_score']:.3f}）：
        # - 画面概括：{desc.get('visual_summary', '')}
        # - 出场角色：{'、'.join(desc.get('characters', []))}
        # - 主要动作：{'、'.join(desc.get('actions', []))}
        # - 环境：{desc.get('environment', '')}
        # - 对白/字幕：{dialogue_display}
        # - 互动关系：{'、'.join(desc.get('interaction_type', []))}
        # - 情绪强度：{desc.get('emotion_intensity', '')}
        # - 画面功能：{'、'.join(desc.get('functional_intent', []))}
        # """)
        # 
        # prompt = f"""你是影视剪辑师，需要为以下解说文案选择你认为最合适且匹配度最高的视频片段。
        # 
        # 【解说文案】
        # {script_text}
        # 
        # 【文案视觉要求】
        # - 人物类型：{character_types}
        # - 动作关键词：{action_keywords}
        # - 环境：{environment}
        # - 互动关系：{interaction_type}
        # - 情绪强度：{emotion_intensity}
        # - 导演意图：{director_intent}
        # 
        # 【候选片段】共{len(stage3_candidates)}个
        # {''.join(candidates_desc)}
        # 
        # 【判断标准】
        # 1. 画面内容是否匹配文案描述
        # 2. 人物类型是否对齐
        # 3. 动作/环境是否契合
        # 4. 对白/字幕是否与文案相关
        # 5. 情绪氛围是否一致
        # 
        # 【输出格式】严格JSON，不要包含任何其他文字
        # {{
        #   "selected_index": 1,
        #   "confidence": 0.9,
        #   "reason": "选择理由（50字以内）"
        # }}"""
        # 
        # max_retries = 3
        # for attempt in range(max_retries):
        #     try:
        #         raw, _ = self.model_client.call("", prompt)
        #         
        #         raw = raw.strip()
        #         if raw.startswith("```json"):
        #             raw = raw[7:]
        #         if raw.startswith("```"):
        #             raw = raw[3:]
        #         if raw.endswith("```"):
        #             raw = raw[:-3]
        #         raw = raw.strip()
        #         
        #         import re
        #         m = re.search(r'\{[\s\S]*\}', raw)
        #         if m:
        #             result = json.loads(m.group(0))
        #             selected_idx = result.get("selected_index", 1)
        #             confidence = result.get("confidence", 0.5)
        #             reason = result.get("reason", "LLM选择")
        #             
        #             if 1 <= selected_idx <= len(stage3_candidates):
        #                 selected = stage3_candidates[selected_idx - 1]
        #                 logger.info(f"  LLM精排成功: 选择片段{selected_idx}, 置信度={confidence}")
        #                 return selected, f"[LLM]{reason} (置信度:{confidence})"
        #     except Exception as e:
        #         logger.warning(f"  LLM精排尝试 {attempt + 1} 失败: {e}")
        # 
        # logger.warning("  LLM精排失败，使用第三阶段TOP1")
        # top1 = stage3_candidates[0] if stage3_candidates else None
        # return top1, "第三阶段TOP1（LLM调用失败）"

    def _match_in_episode_force(self,
                                  script_item: Dict,
                                  target_ep: int,
                                  emb_by_episode: Dict[int, Dict[int, np.ndarray]],
                                  segments_by_episode: Dict[int, List[Dict]],
                                  top_k: int = 5) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        四阶段强制匹配：在目标集数内找出最合适的片段
        Stage1 → Stage2 → Stage3 → Stage4
        """
        seq = script_item.get("segment_sequence", 0)
        script_text = script_item.get("script_text", "")
        
        target_embs = emb_by_episode.get(target_ep, {})
        target_segments = segments_by_episode.get(target_ep, [])
        
        logger.info(f"Segment {seq}: 在 EP{target_ep} 内四阶段匹配 ({len(target_embs)} 个候选)")
        
        mapping_info = {
            "sentence_idx": seq,
            "sentence_content": script_text,
            "target_episode": target_ep,
            "confidence": 0.0,
            "match_reason": "",
            "best_segment_id": None,
            "final_score": 0.0
        }
        
        if not target_embs:
            mapping_info["match_reason"] = f"EP{target_ep}无embedding数据"
            return {
                "segment_sequence": seq,
                "status": "success",
                "episode_target": target_ep,
                "top_matches": [],
                "match_breakdown": {"warning": f"EP{target_ep}无embedding数据"},
                "best_match": None
            }, mapping_info
        
        vq = script_item.get("visual_query", {})
        script_type = classify_script_type(vq)
        logger.info(f"  文案类型: {script_type}")
        
        total_candidates = len(target_embs)
        stage1_top = min(max(20, top_k), total_candidates)
        stage2_top = min(top_k, stage1_top)
        stage3_top = min(top_k, stage2_top)
        
        stage1_results = self._stage1_semantic_recall(script_item, target_embs, target_segments, stage1_top)
        if not stage1_results:
            mapping_info["match_reason"] = "第一阶段无候选"
            return {"segment_sequence": seq, "status": "success", "episode_target": target_ep,
                    "top_matches": [], "match_breakdown": {"error": "第一阶段无候选"}, "best_match": None}, mapping_info
        logger.info(f"  Stage1完成: {len(stage1_results)} 个候选")
        
        stage2_results = self._stage2_visual_rerank(script_item, stage1_results, script_type, stage2_top)
        if not stage2_results:
            mapping_info["match_reason"] = "第二阶段无候选"
            return {"segment_sequence": seq, "status": "success", "episode_target": target_ep,
                    "top_matches": [], "match_breakdown": {"error": "第二阶段无候选"}, "best_match": None}, mapping_info
        logger.info(f"  Stage2完成: {len(stage2_results)} 个候选")
        
        stage3_results = self._stage3_emotion_rerank(script_item, stage2_results, script_type, stage3_top)
        if not stage3_results:
            mapping_info["match_reason"] = "第三阶段无候选"
            return {"segment_sequence": seq, "status": "success", "episode_target": target_ep,
                    "top_matches": [], "match_breakdown": {"error": "第三阶段无候选"}, "best_match": None}, mapping_info
        logger.info(f"  Stage3完成: {len(stage3_results)} 个候选")
        
        best_candidate, llm_reason = self._stage4_llm_final(script_item, stage3_results)
        
        if best_candidate:
            reason = self._generate_match_reason(script_text, best_candidate["seg_info"], best_candidate["final_score"])
            full_reason = f"{llm_reason} | {reason}"
            
            mapping_info.update({
                "confidence": round(best_candidate["final_score"], 3),
                "match_reason": full_reason[:100],
                "best_segment_id": best_candidate["segment_id"],
                "final_score": round(best_candidate["final_score"], 3)
            })
            
            logger.info(f"  Segment {seq} (EP{target_ep}) 匹配完成: 片段ID={best_candidate['segment_id']}, 分数={best_candidate['final_score']:.3f}")
            
            return {
                "segment_sequence": seq,
                "status": "success",
                "episode_target": target_ep,
                "top_matches": [
                    {"segment_id": c["segment_id"], "episode": c["seg_info"].get("episode"), "similarity": round(c["final_score"], 3)}
                    for c in stage3_results
                ],
                "match_breakdown": best_candidate.get("breakdown", {}),
                "best_match": {
                    "segment_id": best_candidate["segment_id"],
                    "score": round(best_candidate["final_score"], 3),
                    "episode": best_candidate["seg_info"].get("episode"),
                    "segment_file_path": best_candidate["seg_info"].get("segment_file_path"),
                    "video_info": {
                        "start_time": best_candidate["seg_info"].get("start_time"),
                        "end_time": best_candidate["seg_info"].get("end_time"),
                        "episode": best_candidate["seg_info"].get("episode"),
                        "segment_file_path": best_candidate["seg_info"].get("segment_file_path"),
                        "video_path": best_candidate["seg_info"].get("video_path"),
                    }
                }
            }, mapping_info
        else:
            mapping_info["match_reason"] = "无最终候选"
            return {
                "segment_sequence": seq,
                "status": "success",
                "episode_target": target_ep,
                "top_matches": [],
                "match_breakdown": {"error": "无最终候选"},
                "best_match": None
            }, mapping_info

    def _save_mapping_report(self, dj: str, mapping_list: List[Dict], cache_dir: Path):
        report_path = cache_dir / f"{dj}_mapping_report.txt"
        
        lines = []
        lines.append("=" * 100)
        lines.append(f"文案-视频集数映射报告（四阶段匹配）| {dj}")
        lines.append("=" * 100)
        lines.append("")
        lines.append(f"{'句子索引':<10} {'句子内容':<50} {'目标集数':<10} {'置信度':<10} {'匹配理由'}")
        lines.append("-" * 100)
        
        for item in mapping_list:
            content = item["sentence_content"]
            if len(content) > 48:
                content = content[:45] + "..."
            lines.append(
                f"{item['sentence_idx']:<10} "
                f"{content:<50} "
                f"EP{item['target_episode']:<8} "
                f"{item['confidence']:<10.3f} "
                f"{item['match_reason']}"
            )
        
        lines.append("-" * 100)
        lines.append(f"总计: {len(mapping_list)} 句文案")
        lines.append("")
        lines.append("匹配详情:")
        for item in mapping_list:
            lines.append(f"\n【句子 {item['sentence_idx']}】")
            lines.append(f"  内容: {item['sentence_content']}")
            lines.append(f"  匹配到: EP{item['target_episode']} 片段{item['best_segment_id']} (分数: {item['final_score']})")
            lines.append(f"  理由: {item['match_reason']}")
        
        report_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"匹配映射报告已保存: {report_path}")
        return report_path

    def process(self, dj: str, script_json_path: str, top_k: int = 5) -> Dict[str, Any]:
        try:
            logger.info(f"开始四阶段匹配流程 dj={dj}")

            cache_dir = Path(__file__).parent / "cache" / dj
            
            ok, video_segments, err = self._check_structured_description(dj)
            if not ok:
                return {"status": "error", "dj": dj, "error_message": err}

            ok, script_data, err = self._check_script_json(script_json_path)
            if not ok:
                return {"status": "error", "dj": dj, "error_message": err}

            emb_by_episode, segments_by_episode = self._build_episode_index(
                video_segments, "embeddings.pkl", cache_dir
            )
            
            if not emb_by_episode:
                return {"status": "error", "dj": dj, "error_message": "构建集数索引失败"}

            results = []
            mapping_list = []
            accumulated_time = 0.0
            
            for idx, item in enumerate(script_data, 1):
                audio_dur = item.get("audio_duration") or 3
                
                if accumulated_time >= 20:
                    logger.info(f"段落 {idx} 超过20秒，停止处理")
                    break
                
                accumulated_time += audio_dur
                target_ep = item.get("episode_target", 1)
                
                match_result, mapping_info = self._match_in_episode_force(
                    item, target_ep, emb_by_episode, segments_by_episode, top_k
                )
                
                results.append(match_result)
                mapping_list.append(mapping_info)
                
                if idx % 5 == 0:
                    logger.info(f"匹配进度: {idx}句, 累计时长: {accumulated_time:.1f}s")

            report_path = self._save_mapping_report(dj, mapping_list, cache_dir) 
            
            for item, r in zip(script_data, results):
                item["match_result"] = r

            Path(script_json_path).write_text(
                json.dumps(script_data, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            success_count = sum(1 for r in results if r.get("best_match") is not None)

            return {
                "status": "success", 
                "dj": dj, 
                "total_segments": len(results),
                "success_count": success_count,
                "failed_count": len(results) - success_count,
                "mapping_report": str(report_path)
            }

        except Exception as e:
            logger.error("匹配流程异常", exc_info=True)
            return {"status": "error", "dj": dj, "error_message": str(e)}


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python skill_match_and_rank.py <dj> <script_json_path> [top_k]")
        sys.exit(1)

    dj = sys.argv[1]
    script_json = sys.argv[2]
    top_k = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    result = SkillMatchAndRank().process(dj, script_json, top_k)
    print(json.dumps(result, ensure_ascii=False, indent=2))
