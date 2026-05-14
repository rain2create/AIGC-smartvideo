#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S4：匹配算法模块（召回和排序）- 前20秒多字段加权匹配版
改进：前20秒使用 script_text/actions/environment 三字段加权匹配，20秒后恢复单字段匹配
"""

import os
import json
import pickle
import logging
import requests
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import sys

# ---------------- 日志 ----------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SkillMatchAndRank:
    """技能匹配和排序模块"""

    def __init__(self, emb_api_url: str = "http://10.163.166.76:17998/emb/"):
        self.emb_api_url = emb_api_url
        # 缓存embedding避免重复调用API
        self._emb_cache = {}

    # ====================================================================== 1️⃣ 检查与加载（保持不变）
    def _check_structured_description(
        self, dj: str
    ) -> Tuple[bool, Optional[Dict], Optional[str]]:
        try:
            cache_dir = Path(__file__).parent / "cache" / dj
            json_file_path = cache_dir / "structured_description.json"

            if not json_file_path.exists():
                return False, None, f"结构化描述文件不存在: {json_file_path}"

            logger.info(f"读取结构化描述文件: {json_file_path}")
            data = json.loads(json_file_path.read_text(encoding="utf-8"))

            if isinstance(data, dict):
                segments = data.get("segments", [])
                embedding_file_path = data.get("embedding_file_path", "embeddings.pkl")
            elif isinstance(data, list):
                segments = data
                embedding_file_path = "embeddings.pkl"
            else:
                return False, None, "结构化描述文件格式错误"

            if not segments:
                return False, None, "未找到片段数据"

            missing_files = []
            for seg in segments:
                seg_path = seg.get("segment_file_path")
                if not seg_path:
                    missing_files.append(f"片段 {seg.get('segment_id', 'unknown')} 缺少 segment_file_path")
                    continue

                clean_path = seg_path
                if Path(clean_path).is_absolute():
                    full_path = Path(clean_path)
                else:
                    full_path = cache_dir / clean_path
                    if not full_path.exists():
                        full_path = Path(__file__).parent / clean_path
                    if not full_path.exists():
                        clean_path = clean_path.lstrip("./").lstrip(".\\")
                        patterns_to_remove = [
                            f"cache/{dj}/",
                            f"text_video_match/cache/{dj}/",
                            f"smart_video/text_video_match/cache/{dj}/"
                        ]
                        for pattern in patterns_to_remove:
                            if pattern in clean_path:
                                clean_path = clean_path.replace(pattern, "")
                                break
                        full_path = Path(__file__).parent / clean_path

                if not full_path.exists():
                    if "mnt_liuhuan" in seg_path:
                        parts = seg_path.split("/")
                        try:
                            smart_video_index = parts.index("smart_video")
                            relative_path = "/".join(parts[smart_video_index:])
                            full_path = Path("/mnt_liuhuan/search_public/xby/smart_video") / relative_path
                        except ValueError:
                            pass
                    
                    if not full_path.exists():
                        missing_files.append(f"片段 {seg.get('segment_id', 'unknown')} 的文件不存在: {full_path}")

            if missing_files:
                return False, None, "以下文件不存在:\n" + "\n".join(missing_files[:5])

            emb_full = cache_dir / embedding_file_path if not Path(embedding_file_path).is_absolute() else Path(embedding_file_path)
            if not emb_full.exists():
                return False, None, f"向量文件不存在: {emb_full}"

            logger.info(f"检查通过: segments={len(segments)}, embedding={emb_full}")
            return True, data, None

        except Exception as e:
            logger.error("检查结构化描述文件异常", exc_info=True)
            return False, None, str(e)

    def _check_script_json(
        self, script_json_path: str
    ) -> Tuple[bool, Optional[List], Optional[str]]:
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

            script_dir = script_json_path.parent
            missing_audio = []
            for i, item in enumerate(script_data):
                audio_path = item.get("audio_path")
                if not audio_path:
                    missing_audio.append(f"第 {i+1} 个片段缺少 audio_path")
                    continue
                full_audio = script_dir / audio_path if not Path(audio_path).is_absolute() else Path(audio_path)
                if not full_audio.exists():
                    missing_audio.append(f"第 {i+1} 个音频不存在: {full_audio}")

            if missing_audio:
                return False, None, "以下音频文件不存在:\n" + "\n".join(missing_audio[:5])

            logger.info(f"检查通过: 文案片段数={len(script_data)}")
            return True, script_data, None

        except Exception as e:
            logger.error("检查文案 JSON 异常", exc_info=True)
            return False, None, str(e)

    # ====================================================================== 2️⃣ Embedding基础方法
    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """获取文本的embedding向量，带内存缓存"""
        if not text or not text.strip():
            return None
        
        text_key = text[:100]  # 缓存key截断（仅用于查缓存，不影响传入API的完整文本）
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
            logger.error("获取 embedding 失败", exc_info=True)
            return None

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """计算两段文本的embedding余弦相似度 - 修正：不截断语义"""
        if not text1 or not text2:
            return 0.5  # 中性分
        
        # 修正：传入完整文本，不做截断，保留完整语义
        emb1 = self._get_embedding(text1)
        emb2 = self._get_embedding(text2)
        
        if emb1 is None or emb2 is None:
            return 0.5
            
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def _load_embeddings(self, embedding_file_path: str, cache_dir: Path) -> Optional[Dict[int, np.ndarray]]:
        try:
            full_path = cache_dir / embedding_file_path if not Path(embedding_file_path).is_absolute() else Path(embedding_file_path)
            logger.info(f"加载向量文件: {full_path}")
            raw = pickle.loads(full_path.read_bytes())
            emb_dict = {}
            for item in raw:
                if "segment_id" in item and "embedding" in item:
                    emb_dict[item["segment_id"]] = np.array(item["embedding"], dtype=np.float32)
            return emb_dict
        except Exception:
            logger.error("加载 embedding 失败", exc_info=True)
            return None

    def _load_video_segments(self, cache_dir: Path) -> List[Dict]:
        """加载视频结构化描述"""
        json_file = cache_dir / "structured_description.json"
        if not json_file.exists():
            return []
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data.get("segments", [])
            return data if isinstance(data, list) else []
        except Exception as e:
            logger.warning(f"加载视频片段数据失败: {e}")
            return []

    def _find_segment_by_id(self, segments: List[Dict], seg_id: int) -> Optional[Dict]:
        for seg in segments:
            if seg.get("segment_id") == seg_id:
                return seg
        return None

    # ====================================================================== 3️⃣ 20秒后：原有单字段匹配逻辑
    def _compose_search_text(self, script_item: Dict[str, Any]) -> str:
        """20秒后使用的简单匹配文本"""
        parts = []
        if script_item.get("script_text"):
            parts.append(f"内容：{script_item['script_text']}")
        vq = script_item.get("visual_query", {})
        if vq.get("characters"):
            parts.append(f"人物：{'、'.join(vq['characters'])}")
        if vq.get("action_keywords"):
            parts.append(f"动作：{'、'.join(vq['action_keywords'])}")
        if vq.get("environment"):
            parts.append(f"环境：{vq['environment']}")
        if vq.get("expanded_tags"):
            parts.append(f"标签：{vq['expanded_tags']}")
        return "，".join(parts)

    # ====================================================================== 4️⃣ 前20秒：多字段加权匹配（核心改动）
    def _match_hook_segment_weighted(self, script_item: Dict, 
                                     emb_dict: Dict[int, np.ndarray],
                                     video_segments: List[Dict],
                                     prev_matched_seg: Optional[Dict],  # 参数保留但不用
                                     top_k: int = 10) -> Dict[str, Any]:
        """
        前20秒专用：多字段加权匹配（纯语义，无时间连续性约束）
        """
        seq = script_item.get("segment_sequence", 0)
        
        # Step 1: 用基础embedding召回Top50候选（粗排）
        base_query = self._compose_search_text(script_item)
        qemb = self._get_embedding(base_query)
        
        if qemb is None:
            return {
                "segment_sequence": seq,
                "status": "error",
                "error_message": "查询向量化失败"
            }
        
        rough_candidates = sorted(
            ((sid, self._cosine(qemb, v)) for sid, v in emb_dict.items()),
            key=lambda x: x[1],
            reverse=True
        )[:50]  # 召回Top50做精排
        
        # Step 2: 精细化加权计算
        scored_candidates = []
        
        for seg_id, base_sim in rough_candidates:
            seg_info = self._find_segment_by_id(video_segments, seg_id)
            if not seg_info:
                continue
                
            desc = seg_info.get("structured_description", {})
            
            # 字段提取
            video_summary = desc.get("visual_summary", "")
            video_actions = desc.get("actions", [])
            video_env = desc.get("environment", "")
            
            vq = script_item.get("visual_query", {})
            script_text = script_item.get("script_text", "")
            script_actions = vq.get("action_keywords", [])
            script_env = vq.get("environment", "")
            
            # 计算各维度相似度（纯语义，无截断）
            sim_summary = self._calculate_similarity(script_text, video_summary)
            
            if script_actions and video_actions:
                script_act_str = "、".join(script_actions) if isinstance(script_actions, list) else str(script_actions)
                video_act_str = "、".join(video_actions) if isinstance(video_actions, list) else str(video_actions)
                sim_actions = self._calculate_similarity(script_act_str, video_act_str)
            else:
                sim_actions = 0.5 if not (script_actions or video_actions) else 0.3
            
            if script_env and video_env:
                sim_env = self._calculate_similarity(str(script_env), str(video_env))
            else:
                sim_env = 0.5
            
            # 加权计算最终分数（纯语义，无时间奖励）
            final_score = 0.5 * sim_summary + 0.3 * sim_actions + 0.2 * sim_env
            
            scored_candidates.append({
                "segment_id": seg_id,
                "final_score": float(final_score),
                "base_similarity": float(base_sim),
                "breakdown": {
                    "sim_summary": round(sim_summary, 3),
                    "sim_actions": round(sim_actions, 3),
                    "sim_env": round(sim_env, 3)
                    # 删除continuity_bonus相关
                },
                "seg_info": seg_info
            })
        
        # Step 3: 排序取Top
        scored_candidates.sort(key=lambda x: x["final_score"], reverse=True)
        best = scored_candidates[:top_k]
        
        if not best:
            return {
                "segment_sequence": seq,
                "status": "error",
                "error_message": "无候选片段"
            }
        
        top1 = best[0]
        logger.info(f"  Hook段落 {seq} 加权匹配: 总分={top1['final_score']:.3f} "
                   f"[摘要{top1['breakdown']['sim_summary']:.2f} "
                   f"动作{top1['breakdown']['sim_actions']:.2f} "
                   f"环境{top1['breakdown']['sim_env']:.2f}]")
        
        return {
            "segment_sequence": seq,
            "status": "success",
            "is_hook": True,
            "top_matches": [{"segment_id": b["segment_id"], "similarity": b["final_score"]} for b in best],
            "match_breakdown": best[0]["breakdown"]
        }
    # ====================================================================== 5️⃣ 主流程
    def process(self, dj: str, script_json_path: str, top_k: int = 10) -> Dict[str, Any]:
        try:
            logger.info(f"开始匹配 dj={dj}")

            ok, structured, err = self._check_structured_description(dj)
            if not ok:
                return {"status": "error", "dj": dj, "error_message": err}

            ok, script_data, err = self._check_script_json(script_json_path)
            if not ok:
                return {"status": "error", "dj": dj, "error_message": err}

            cache_dir = Path(__file__).parent / "cache" / dj
            embedding_file = (
                structured.get("embedding_file_path")
                if isinstance(structured, dict)
                else "embeddings.pkl"
            )

            emb_dict = self._load_embeddings(embedding_file, cache_dir)
            if emb_dict is None:
                return {"status": "error", "dj": dj, "error_message": "加载向量失败"}

            video_segments = self._load_video_segments(cache_dir)
            
            results = []
            accumulated_time = 0.0
            prev_matched_seg = None
            
            for idx, item in enumerate(script_data, 1):
                audio_dur = item.get("audio_duration", 3)
                is_hook = accumulated_time < 20  # 前20秒判断
                accumulated_time += audio_dur
                
                if is_hook:
                    # 前20秒：使用多字段加权匹配
                    logger.info(f"【前20秒Hook】段落 {idx}，累计时长 {accumulated_time:.1f}s，启用加权匹配")
                    match_result = self._match_hook_segment_weighted(
                        item, emb_dict, video_segments, prev_matched_seg, top_k
                    )
                else:
                    # 20秒后：恢复原流程（单字段embedding匹配）
                    qtext = self._compose_search_text(item)
                    qemb = self._get_embedding(qtext)
                    
                    if qemb is None:
                        match_result = {
                            "segment_sequence": idx,
                            "status": "error",
                            "error_message": "向量化失败"
                        }
                    else:
                        sims = sorted(
                            ((sid, self._cosine(qemb, v)) for sid, v in emb_dict.items()),
                            key=lambda x: x[1],
                            reverse=True
                        )[:top_k]
                        match_result = {
                            "segment_sequence": idx,
                            "status": "success",
                            "top_matches": [{"segment_id": sid, "similarity": float(score)} for sid, score in sims]
                        }

                results.append(match_result)
                
                # 更新prev用于连续性检查
                if match_result.get("top_matches") and video_segments:
                    best_id = match_result["top_matches"][0]["segment_id"]
                    prev_matched_seg = self._find_segment_by_id(video_segments, best_id)

            # 写回JSON
            for item, r in zip(script_data, results):
                item["match_result"] = r

            Path(script_json_path).write_text(
                json.dumps(script_data, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            hook_count = sum(1 for r in results if r.get("is_hook"))
            logger.info(f"匹配完成，前20秒Hook段落数: {hook_count}，总段落数: {len(script_data)}")

            return {"status": "success", "dj": dj, "total_script_segments": len(script_data)}

        except Exception as e:
            logger.error("匹配流程异常", exc_info=True)
            return {"status": "error", "dj": dj, "error_message": str(e)}


# ====================================================================== CLI
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python skill_match_and_rank.py <dj> <script_json_path> [top_k]")
        sys.exit(1)

    dj = sys.argv[1]
    script_json = sys.argv[2]
    top_k = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    result = SkillMatchAndRank().process(dj, script_json, top_k)
    print(json.dumps(result, ensure_ascii=False, indent=2))