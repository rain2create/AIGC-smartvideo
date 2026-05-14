#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S2：索引化模块
读取 S1 阶段保存的每个片段的结构化描述，然后组成成一个字符串，
调用 API 获取向量，保存到本地，然后更新 S1 阶段的 json 文件，
把向量文件（pkl 文件）路径加进去
"""
import os
import json
import pickle
import logging
import requests
from pathlib import Path
from typing import Dict, List, Any, Optional
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StructInfoEmbedding:
    def __init__(self, emb_api_url: str = "http://10.163.166.76:17998/emb/"):
        self.emb_api_url = emb_api_url

    # -------------------- 以下逻辑与你原文件完全一致，仅把路径改成「相对于本脚本」 --------------------
    def _compose_description_text(self, structured_description: Dict[str, Any]) -> str:
        if not structured_description:
            return ""
        if "error" in structured_description:
            return f"错误: {structured_description.get('error_message', '未知错误')}"
        parts = []
        if structured_description.get("visual_summary"):
            parts.append(f"画面概括: {structured_description['visual_summary']}")
        if structured_description.get("characters"):
            chars = structured_description["characters"]
            parts.append(f"出场角色: {'、'.join(chars) if isinstance(chars, list) else str(chars)}")
        if structured_description.get("actions"):
            acts = structured_description["actions"]
            parts.append(f"主要动作: {'、'.join(acts) if isinstance(acts, list) else str(acts)}")
        if structured_description.get("environment"):
            parts.append(f"环境: {structured_description['environment']}")
        if structured_description.get("emotion_tone"):
            parts.append(f"情绪: {structured_description['emotion_tone']}")
        if structured_description.get("dialogue_or_text"):
            parts.append(f"对白/文字: {structured_description['dialogue_or_text']}")
        return " | ".join(parts)

    def _get_embedding(self, text: str) -> Optional[List[float]]:
        if not text or not text.strip():
            logger.warning("文本为空，跳过向量化")
            return None
        try:
            rsp = requests.get(self.emb_api_url, params={"title": text}, timeout=30)
            if rsp.status_code == 200:
                emb = rsp.json()
                if isinstance(emb, list):
                    logger.info(f"成功获取向量，维度: {len(emb)}")
                    return emb
            logger.error(f"API 失败: {rsp.status_code}  {rsp.text[:200]}")
            return None
        except Exception as e:
            logger.error(f"获取向量异常: {e}")
            return None

    def _save_embeddings(self, embeddings: List[Dict[str, Any]], output_path: Path) -> bool:
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("wb") as f:
                pickle.dump(embeddings, f)
            logger.info(f"向量已保存: {output_path}  共 {len(embeddings)} 条")
            return True
        except Exception as e:
            logger.error(f"保存向量失败: {e}")
            return False

    def _update_json_file(self, json_file: Path, embedding_file: Path, original_data: Any) -> bool:
        try:
            if isinstance(original_data, list):
                data = {"segments": original_data, "embedding_file_path": embedding_file.name}
            else:
                data = original_data.copy()
                data["embedding_file_path"] = embedding_file.name
            json_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf8")
            logger.info(f"已更新 JSON: {json_file}")
            return True
        except Exception as e:
            logger.error(f"更新 JSON 失败: {e}")
            return False

    # -------------------- 主入口 --------------------
    def process(self, dj: str) -> Dict[str, Any]:
        """
        与原来完全一致，只是所有路径都用「本脚本所在目录」推导
        """
        base_cache = Path(__file__).parent / "cache" / dj
        s1_json = base_cache / "structured_description.json"
        if not s1_json.exists():
            return {"status": "error", "dj": dj,
                    "error_message": f"结构化描述文件不存在: {s1_json}"}

        logger.info(f"读取结构化描述: {s1_json}")
        segment_data = json.loads(s1_json.read_text(encoding="utf8"))
        original_data = segment_data

        # 解析 segments
        if isinstance(segment_data, list):
            segments = segment_data
        elif isinstance(segment_data, dict):
            segments = segment_data.get("segments") or segment_data.get("segment_infos") or []
        else:
            segments = []

        if not segments:
            return {"status": "error", "dj": dj, "error_message": "未找到片段数据"}

        # 向量化
        embeddings = []
        ok, fail = 0, 0
        for idx, seg in enumerate(segments, 1):
            seg_id = seg.get("segment_id", idx)
            desc = self._compose_description_text(seg.get("structured_description", {}))
            if not desc:
                logger.warning(f"片段 {seg_id} 描述为空，跳过")
                fail += 1
                continue
            emb = self._get_embedding(desc)
            if emb:
                embeddings.append({"segment_id": seg_id, "embedding": emb})
                ok += 1
            else:
                fail += 1

        logger.info(f"向量化完成: 成功 {ok}  | 失败 {fail}")
        if not embeddings:
            return {"status": "error", "dj": dj, "error_message": "所有片段向量化失败"}

        # 保存 pkl
        pkl_path = base_cache / "embeddings.pkl"
        if not self._save_embeddings(embeddings, pkl_path):
            return {"status": "error", "dj": dj, "error_message": "保存向量文件失败"}

        # 更新 JSON
        if not self._update_json_file(s1_json, pkl_path, original_data):
            logger.warning("更新 JSON 失败，但向量文件已保存")

        return {"status": "success", "dj": dj,
                "total_segments": len(segments),
                "success_count": ok,
                "fail_count": fail,
                "embedding_file_path": str(pkl_path),
                "json_file_path": str(s1_json)}


# -------------------- CLI --------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python skill_struct_info_emb.py <dj>")
        sys.exit(1)
    dj = sys.argv[1]
    print(json.dumps(StructInfoEmbedding().process(dj), ensure_ascii=False, indent=2))