#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
带 RAG 示例的解说文案生成系统

工作流程：
1. 对输入剧情进行打标（使用 rag-label 逻辑）
2. 从 rag.jsonl 中检索相同标签的解说（2条）
3. 使用 examples.py 为这些解说生成 SOP1-5 步骤
4. 将生成的 SOP 案例拼接到 system prompt 的 #7.参考案例 部分
5. 使用增强后的 system prompt 生成解说


新增：
1、将plotgraph事件之间分的更细，要有桥接事件
2、不再把 PlotGraph/BeatSheet 拼到skill后面。
改为：rules 模式下，把 【PlotGraph】…【/PlotGraph】 和 【BeatSheet】…【/BeatSheet】 作为“本次任务专属实例数据”，拼到 user_prompt 里。

"""

import os
import json
import logging
import argparse
import random
import re
import difflib
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from model_client import ModelClient
# 高级标签与检索
from advanced_labeling import label_content_advanced
from rag_retrieval_labeling import retrieve_by_advanced_labels


from fewshot7 import (  # type: ignore
        load_rag_jsonl,
        retrieve_by_label,
        load_sop_doc,
        generate_forward_sop,
    )


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# 步骤 1: 剧情打标
# ============================================================================

LABEL_SYSTEM_PROMPT = """你是短剧/短视频解说文案的"一级标签打标员"。
你的任务：阅读输入解说文案，只做打标与理由说明。

硬规则：
1) 标签只能从以下五个中选一个：走心催泪 / 悬疑解谜 / 搞笑吐槽 / 爽快打脸 / 温情叙事
2) 只输出严格 JSON，不要输出 Markdown、不要输出多余文字
3) reason 用一句话说明依据（<=60字），不要扩写剧情、不要复述整段文案
"""

LABEL_USER_PROMPT_TMPL = """请对下面这段【剧情简介】进行一级标签打标，并给出一句话理由。
只能从五选一：走心催泪 / 悬疑解谜 / 搞笑吐槽 / 爽快打脸 / 温情叙事

严格输出 JSON：
{{
  "label": "五选一",
  "reason": "一句话理由"
}}

【剧情简介】
{content}
"""

LABELS = ["走心催泪", "悬疑解谜", "搞笑吐槽", "爽快打脸", "温情叙事"]

# 子标签（冲突类型/子风格）候选：用于“主标签+子标签”打标与风格画像归一
SUBLABELS: Dict[str, List[str]] = {
    "爽快打脸": ["绿茶陷害", "霸总强占", "追妻火葬场", "身份反转", "复仇反杀", "豪门狗血", "虐渣打脸", "误会反转"],
    "悬疑解谜": ["线索递进", "误导反转", "多线并进", "身份谜团", "时间诡计", "真相揭晓"],
    "搞笑吐槽": ["反差吐槽", "沙雕日常", "社死名场面", "人设反转", "阴阳怪气"],
    "走心催泪": ["亲情救赎", "遗憾错过", "病痛生离", "牺牲成全", "重逢和解"],
    "温情叙事": ["治愈陪伴", "成长励志", "轻松温暖", "生活小确幸", "关系修复"],
}


def _strip_code_fences(s: str) -> str:
    """去除代码块标记"""
    s = (s or "").strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()


def _extract_first_json_object(s: str) -> Optional[str]:
    """从模型输出中提取第一个 JSON 对象"""
    if not s:
        return None
    s = _strip_code_fences(s)
    # 去掉思考标签
    s = re.sub(r"<think>[\s\S]*?</think>", "", s, flags=re.IGNORECASE)
    s = re.sub(r"<analysis>[\s\S]*?</analysis>", "", s, flags=re.IGNORECASE)
    s = s.strip()
    
    # 尝试整体解析
    try:
        json.loads(s)
        return s
    except Exception:
        pass
    
    # 提取第一个 {...}
    m = re.search(r"\{[\s\S]*?\}", s)
    if not m:
        return None
    return m.group(0).strip()


def _normalize_label(label: Any) -> Optional[str]:
    """规范化标签"""
    if not isinstance(label, str):
        return None
    label = label.strip()
    if label in LABELS:
        return label
    
    # 容错映射
    mapping = {
        "情绪共鸣": "走心催泪",
        "情感共鸣": "走心催泪",
        "温情": "温情叙事",
        "温暖治愈": "温情叙事",
        "治愈系": "温情叙事",
        "幽默风趣": "搞笑吐槽",
        "爆笑解构": "搞笑吐槽",
        "悬疑推理": "悬疑解谜",
        "爽文打脸": "爽快打脸",
        "打脸爽文": "爽快打脸",
    }
    return mapping.get(label)


STYLE_PROFILE_SYSTEM_PROMPT = """你是短视频解说文案的【风格画像抽取器】。
你的任务：只从输入文本中抽取“写法风格”，不要复述剧情，不要改写文本。

输出严格 JSON，不要 Markdown，不要多余文字。
"""


def _stable_case_id(case: Dict[str, Any]) -> str:
    cid = case.get("id")
    if isinstance(cid, str) and cid.strip():
        return cid.strip()
    content = (case.get("content") or "").encode("utf-8", errors="ignore")
    return "hash_" + hashlib.md5(content).hexdigest()[:12]


def _load_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        return None
    return None


def _save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def extract_style_profile(
    model_client: ModelClient,
    text: str,
    main_label: Optional[str],
    cache_dir: Optional[Path] = None,
    cache_key: Optional[str] = None,
    temperature: float = 0.2,
) -> Dict[str, Any]:
    """
    抽取 StyleProfile（可缓存）。用于：子风格匹配、稳定 RAG、生成 StyleSpec。
    """
    if cache_dir is not None and cache_key:
        cached = _load_json_if_exists(cache_dir / "style_profiles" / f"{cache_key}.json")
        if cached:
            return cached

    user_prompt = f"""请从下面文本中抽取风格画像，输出 JSON。
要求：
- 只描述写法/节奏/句式/钩子/转场，不要复述剧情，不要把具体人名当“风格”
- 数值字段尽量客观估计

输出 JSON 字段：
{{
  "main_label": "{main_label or ""}",
  "sub_label_guess": "一句话子风格（可空）",
  "short_sentence_ratio": 0.0,          // 0-1，短句占比（<=15字）
  "avg_sentence_len": 0.0,              // 平均句长（字）
  "hook_interval": 0,                   // 平均每 N 句一个钩子/转折
  "transition_preference": ["时间", "因果", "反差"], // 常用转场类型
  "tone_keywords": ["冷嘲", "怒怼"],     // 语气关键词（5个以内）
  "banned_phrases": ["画面", "旁白"],     // 明确不该出现在最终文案里的字眼（5个以内）
  "templates": {{
    "opening": ["一句开头模板A", "一句开头模板B"],
    "transition": ["一句转场模板"],
    "ending": ["一句收尾钩子模板"]
  }}
}}

【文本】
{(text or "")[:12000]}
"""
    raw, _ = model_client.call(
        system_prompt=STYLE_PROFILE_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=temperature,
        top_p=0.4,
    )
    js = _extract_first_json_object(raw or "")
    profile: Dict[str, Any]
    if not js:
        profile = {"main_label": main_label, "error": "style_profile_parse_failed"}
    else:
        try:
            profile = json.loads(js)
        except Exception:
            profile = {"main_label": main_label, "error": "style_profile_json_failed"}

    if cache_dir is not None and cache_key:
        _save_json(cache_dir / "style_profiles" / f"{cache_key}.json", profile)
    return profile


def build_style_spec(
    label_info: Dict[str, Any],
    input_style_profile: Dict[str, Any],
    example_profiles: List[Dict[str, Any]],
) -> str:
    """
    生成 StyleSpec（规则注入为主，示例为辅）。
    """
    # 取一个“合并视角”：优先 input_profile，其次从 examples 补全
    def _pick(key: str, default: Any = None) -> Any:
        v = input_style_profile.get(key)
        if v not in [None, "", [], {}]:
            return v
        for p in example_profiles:
            v2 = p.get(key)
            if v2 not in [None, "", [], {}]:
                return v2
        return default

    main_label = label_info.get("main_label")
    sub_label = label_info.get("sub_label")
    suang_intensity = label_info.get("suang_intensity")
    rhythm_speed = label_info.get("rhythm_speed")
    dogblood_level = label_info.get("dogblood_level")

    short_ratio = _pick("short_sentence_ratio", 0.6)
    hook_interval = _pick("hook_interval", 6)
    transitions = _pick("transition_preference", ["时间", "因果", "反差"])
    tone_kws = _pick("tone_keywords", [])
    banned = _pick("banned_phrases", ["（画面", "旁白", "（念白", "**"])
    templates = _pick("templates", {})

    return (
        "【StyleSpec｜参数化风格规范（规则优先）】\n"
        f"- 主标签: {main_label}\n"
        f"- 子标签: {sub_label or '（未指定，按主标签）'}\n"
        f"- 爽点强度(1-5): {suang_intensity}\n"
        f"- 节奏快慢(1-5): {rhythm_speed}\n"
        f"- 狗血程度(1-5): {dogblood_level}\n"
        f"- 短句占比(<=15字)目标: ≥ {short_ratio}\n"
        f"- 钩子密度: 平均每 {hook_interval} 句至少 1 个钩子/转折\n"
        f"- 转场偏好: {', '.join(transitions) if isinstance(transitions, list) else transitions}\n"
        f"- 语气关键词: {('、'.join(tone_kws) if isinstance(tone_kws, list) else tone_kws) or '（按主标签）'}\n"
        f"- 禁用输出: {('、'.join(banned) if isinstance(banned, list) else banned)}\n"
        "- 重要：示例只学节奏/结构/句式，不得迁移具体情节或新增人物动作。\n"
        "【/StyleSpec】\n"
        "\n"
        "【句式模板（可选参考）】\n"
        f"- 开头模板: {json.dumps((templates or {}).get('opening', []), ensure_ascii=False)}\n"
        f"- 转场模板: {json.dumps((templates or {}).get('transition', []), ensure_ascii=False)}\n"
        f"- 结尾模板: {json.dumps((templates or {}).get('ending', []), ensure_ascii=False)}\n"
        "【/句式模板】\n"
    )


def score_style_match(input_label: Dict[str, Any], cand_profile: Dict[str, Any]) -> float:
    """
    简单可解释的风格匹配打分（越大越匹配）。
    """
    score = 0.0
    if input_label.get("main_label") and cand_profile.get("main_label") == input_label.get("main_label"):
        score += 2.0

    # 子标签一致强加分（如果候选没给就不扣）
    sub = input_label.get("sub_label")
    sub_guess = cand_profile.get("sub_label_guess")
    if sub and isinstance(sub_guess, str) and sub_guess.strip():
        if sub in sub_guess:
            score += 2.0
        else:
            score -= 0.5

    # 节奏与爽点强度（候选如缺失则不计）
    def _num(x: Any) -> Optional[float]:
        try:
            return float(x)
        except Exception:
            return None

    in_speed = _num(input_label.get("rhythm_speed"))
    in_suang = _num(input_label.get("suang_intensity"))
    if in_speed is not None:
        cand_hook = _num(cand_profile.get("hook_interval"))
        # hook_interval 越小越快；把 speed(1-5) 映射成期望 interval(8->3)
        expected_interval = 9 - 1.2 * in_speed
        if cand_hook is not None and cand_hook > 0:
            score += max(0.0, 2.0 - abs(cand_hook - expected_interval) * 0.4)

    if in_suang is not None:
        # 用 tone_keywords/转场偏好做弱信号
        trans = cand_profile.get("transition_preference") or []
        if isinstance(trans, list) and ("反差" in trans or "冲突" in trans):
            score += 0.4

    return score


def retrieve_by_style(
    model_client: ModelClient,
    rag_entries: List[Dict[str, Any]],
    label_info: Dict[str, Any],
    k: int,
    cache_dir: Optional[Path],
) -> List[Dict[str, Any]]:
    """
    先按主标签过滤，再按风格画像打分，最后做“子风格去重”（同一 sub_label_guess 最多 1 条）。
    """
    main_label = label_info.get("main_label")
    if not main_label:
        # 回退：用旧逻辑
        return retrieve_by_label(rag_entries, label_info.get("label") or label_info.get("main_label") or "温情叙事", k=k)

    candidates = [e for e in rag_entries if e.get("label") == main_label]
    if not candidates:
        return []

    scored: List[Tuple[float, Dict[str, Any], Dict[str, Any]]] = []
    for c in candidates:
        cid = _stable_case_id(c)
        profile = extract_style_profile(
            model_client=model_client,
            text=c.get("content", "") or "",
            main_label=main_label,
            cache_dir=cache_dir,
            cache_key=f"rag_{cid}",
            temperature=0.1,
        )
        profile["main_label"] = profile.get("main_label") or main_label
        s = score_style_match(label_info, profile)
        scored.append((s, c, profile))

    scored.sort(key=lambda x: x[0], reverse=True)

    selected: List[Dict[str, Any]] = []
    used_sub: set = set()
    for s, c, profile in scored:
        if len(selected) >= k:
            break
        sg = profile.get("sub_label_guess")
        sg_key = sg.strip() if isinstance(sg, str) else ""
        if sg_key and sg_key in used_sub:
            continue
        selected.append(c)
        if sg_key:
            used_sub.add(sg_key)

    return selected


def _split_sentences_cn(text: str) -> List[str]:
    t = (text or "").strip()
    if not t:
        return []
    # 优先按换行；否则按中文句号/叹号/问号
    if "\n" in t:
        parts = [p.strip() for p in t.splitlines() if p.strip()]
    else:
        parts = [p.strip() for p in re.split(r"[。！？!?]+", t) if p.strip()]
    return parts


def extract_snippets(text: str) -> Dict[str, str]:
    sents = _split_sentences_cn(text)
    if not sents:
        return {"opening": "", "mid": "", "ending": ""}
    opening = "\n".join(sents[:2])
    mid_start = max(0, len(sents) // 2 - 1)
    mid = "\n".join(sents[mid_start:mid_start + 3])
    ending = "\n".join(sents[-2:])
    return {"opening": opening, "mid": mid, "ending": ending}


EXAMPLE_COMPILER_SYSTEM_PROMPT = """你是【Example Compiler｜示例编译器】。
你的任务：把“黄金解说原文”编译成“严格对齐 SOP 的过程示例 + 去剧情化句式资产”，用于 few-shot 教学。

硬规则（极其重要）：
1) 只输出 Markdown（不要输出解释、不要输出多余废话）
2) 禁止复读原文剧情：不得输出任何具体人名/地名/组织/职业/具体数值（用 A/B/C、地点X、金额N、时间T 代替）
3) 禁止粘贴原文句子：不要出现与原文连续相同的长片段（>10个汉字的连续片段视为违规）
4) SOP1-5 示例必须严格依据“爆款解说文案创作技能文档（skill_video_commentary_text_1230.md）”的规则与格式
5) SOP4 的主体串联：每段末尾必须标注钩子类型（例如： (钩子:反差)/(钩子:悬念)/(钩子:冲突升级)/(钩子:身份线索)）
6) SOP5：只留疑问/悬念，不新增事实
"""


def compile_example(
    model_client: ModelClient,
    sop_doc: str,
    golden_content: str,
    label_info: Dict[str, Any],
    style_spec: str,
    temperature: float = 0.15,
) -> str:
    """
    示例编译器：黄金解说原文 -> (去剧情化句式资产) + (SOP1-5过程示例占位符)
    """
    # 句式资产：抽“模板/连接词/节奏规则”，不带剧情
    user_prompt = f"""请把下面【黄金解说原文】编译为 #7 参考案例的两部分内容：

你必须严格遵循的 SOP 文档（以此为格式标准）：
{sop_doc}

本次风格参数（StyleSpec，已给出，直接沿用，不要改写它的字段含义）：
{style_spec}

主标签/子标签/强度参数：
{json.dumps(label_info, ensure_ascii=False)}

【输出格式要求｜必须严格按下面结构输出】：

### 句式资产（去剧情化）

#### 开头钩子句式（不含地点/职业/组织/具体数值）
- 模板1：……
- 模板2：……
- 模板3：……

#### 转场连接词清单（可直接复用）
直到 / 就在这时 / 可谁知 / 下一秒 / 然而 / 结果 / 更要命的是 / 偏偏 / 但他不知道的是 / 于是

#### 中段推进句式（不含具体人名地名）
- 模板1：……
- 模板2：……
- 模板3：……

#### 悬念收尾句式（说了但不说破，不新增事实）
- 模板1：……
- 模板2：……
- 模板3：……

---

### SOP1-5 过程示例（占位符版｜严格对齐 SOP 文档）

#### SOP 1: 列出大纲（剧情分析阶段）
| 时序 | 人 | 地 | 事 | 因 | 情 | 细/爆点标记 |
|------|-----|-----|-----|-----|-----|-------------|
| 1 | 角色A | 地点X | …… | …… | …… | 爆点※ |
| 2 | 角色B | 地点Y | …… | …… | …… | 细节/反差※ |
| 3 | 角色A/角色C | 地点Z | …… | …… | …… | 反转※ |

**【极其重要】时序列格式要求**：
- ✅ 必须使用数字序号：1、2、3、4、5...
- ❌ 严禁使用文字描述："开篇"、"发展"、"高潮"、"结尾"、"冲突"、"转折"等
- 原因：文字描述容易产生幻觉，数字序号更客观准确

#### SOP 2: 提取核心剧情和关键字（精简阶段）
| 原始大纲剧情（大白话） | 提取的核心关键字 | 最终凝练输出（文案架构基础） |
|----------------------|----------------|---------------------------|
| ……（保持大白话） | 关键词1、关键词2 | ……（精简后的一句话） |
| …… | …… | …… |

#### SOP 3: 构思爆款开头（虎头）
- Version 1（技巧：反差）：……
- Version 2（技巧：悬念）：……
- Version 3（技巧：夸张）：……
- Version 4（技巧：细节）：……
- 最终选择（综合爆点最强）：……（理由：一句话）

#### SOP 4: 串联核心剧情（豹身）
（每句话一行；每段末尾标注钩子类型）
……

#### SOP 5: 悬念结尾（凤尾）
（每句话一行；只留悬念，不新增事实）
……

【黄金解说原文｜仅用于抽规则/模板，禁止复读】：
{(golden_content or "")[:12000]}
"""

    compiled, _ = model_client.call(
        system_prompt=EXAMPLE_COMPILER_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=temperature,
        top_p=0.4,
    )
    return (compiled or "").strip()

def label_content(
    model_client: ModelClient,
    content: str,
    temperature: float = 0.3
) -> Tuple[Optional[str], str]:
    """
    对内容进行打标
    
    Args:
        model_client: 模型客户端
        content: 待打标内容（剧情简介或ASR文本）
        temperature: 采样温度
        
    Returns:
        tuple: (标签, 原始响应)
    """
    logger.info("开始对内容进行打标...")
    
    user_prompt = LABEL_USER_PROMPT_TMPL.format(content=content[:12000])
    
    try:
        raw, _ = model_client.call(
            system_prompt=LABEL_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=temperature,
            top_p=0.5
        )
        
        raw = (raw or "").strip()
        logger.info(f"打标响应长度: {len(raw)} 字符")
        
        # 提取 JSON
        js = _extract_first_json_object(raw)
        if not js:
            logger.warning("无法从响应中提取 JSON")
            return None, raw
        
        parsed = json.loads(js)
        if not isinstance(parsed, dict):
            logger.warning("解析的 JSON 不是字典类型")
            return None, raw
        
        label = _normalize_label(parsed.get("label"))
        reason = parsed.get("reason", "")
        
        logger.info(f"打标结果: {label}, 理由: {reason}")
        return label, raw
        
    except Exception as e:
        logger.error(f"打标失败: {e}")
        return None, str(e)


LABELV2_SYSTEM_PROMPT = """你是短剧/短视频解说文案的【标签+子风格画像】打标员。
你的任务：基于输入的剧情简介+对白片段，输出主标签、子标签，以及3个强度参数。

硬规则：
1) main_label 只能从以下五个中选一个：走心催泪 / 悬疑解谜 / 搞笑吐槽 / 爽快打脸 / 温情叙事
2) sub_label 必须从该 main_label 的子标签候选里选一个（如果不确定，选最接近的一个）
3) 输出严格 JSON，不要 Markdown，不要多余文字
4) 三个强度参数都在 1-5 之间（整数）：
   - suang_intensity（爽点强度）
   - rhythm_speed（节奏快慢，越大越快）
   - dogblood_level（狗血程度）
5) reason 一句话（<=60字）
"""


def label_content_v2(
    model_client: ModelClient,
    content: str,
    temperature: float = 0.2,
) -> Tuple[Dict[str, Any], str]:
    """
    扩展打标：主标签 + 子标签 + 强度参数（用于 StyleSpec / RAG 稳定检索）
    """
    # 先用旧模板减少变更成本：主标签仍按既有 LABELS；子标签候选注入 prompt
    sublabels_json = json.dumps(SUBLABELS, ensure_ascii=False, indent=2)
    user_prompt = f"""请对下面内容进行【主标签+子标签+强度】打标。

子标签候选如下（必须从对应 main_label 的列表里选）：
{sublabels_json}

严格输出 JSON：
{{
  "main_label": "五选一",
  "sub_label": "从候选里选",
  "suang_intensity": 1,
  "rhythm_speed": 1,
  "dogblood_level": 1,
  "reason": "一句话理由"
}}

【输入内容】
{content[:12000]}
"""
    raw, _ = model_client.call(
        system_prompt=LABELV2_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=temperature,
        top_p=0.4,
    )
    raw = (raw or "").strip()
    js = _extract_first_json_object(raw)
    if not js:
        # 回退：至少返回主标签（保持可用）
        return {
            "main_label": None,
            "sub_label": None,
            "suang_intensity": 3,
            "rhythm_speed": 3,
            "dogblood_level": 3,
            "reason": "label_v2_parse_failed",
        }, raw
    try:
        parsed = json.loads(js)
    except Exception:
        return {
            "main_label": None,
            "sub_label": None,
            "suang_intensity": 3,
            "rhythm_speed": 3,
            "dogblood_level": 3,
            "reason": "label_v2_json_failed",
        }, raw

    main_label = _normalize_label(parsed.get("main_label") or parsed.get("label"))
    sub_label = parsed.get("sub_label")
    if isinstance(sub_label, str):
        sub_label = sub_label.strip()
    else:
        sub_label = None

    # 兜底：若子标签不在候选里，置空，后续会回退到“主标签匹配”
    if main_label and sub_label and sub_label not in SUBLABELS.get(main_label, []):
        sub_label = None

    def _clamp_1_5(x: Any, default: int) -> int:
        try:
            v = int(x)
            return max(1, min(5, v))
        except Exception:
            return default

    info = {
        "main_label": main_label,
        "sub_label": sub_label,
        "suang_intensity": _clamp_1_5(parsed.get("suang_intensity"), 3),
        "rhythm_speed": _clamp_1_5(parsed.get("rhythm_speed"), 3),
        "dogblood_level": _clamp_1_5(parsed.get("dogblood_level"), 3),
        "reason": parsed.get("reason", ""),
    }
    return info, raw


# ============================================================================
# 步骤 2-4: 生成 SOP 案例并拼接
# ============================================================================

def _clean_sop_output(sop_text: str) -> str:
    """
    清理 SOP 输出，移除可能包含的文档头部
    
    Args:
        sop_text: LLM 生成的 SOP 文本
        
    Returns:
        str: 清理后的 SOP 文本（只保留 SOP1-5 部分）
    """
    # 如果包含完整文档结构（以 "# 爆款解说文案创作技能文档" 开头）
    if "# 爆款解说文案创作技能文档" in sop_text:
        # 尝试找到第一个 "### SOP 1" 或 "## SOP 1" 的位置
        import re
        match = re.search(r'###?\s*SOP\s*1', sop_text, re.IGNORECASE)
        if match:
            # 从这里开始截取
            sop_text = sop_text[match.start():]
            logger.info("✂️ 检测到文档头部，已自动清理")
    
    return sop_text


def generate_rag_examples(
    model_client: ModelClient,
    label: str,
    rag_jsonl_path: str,
    sop_doc_path: str,
    num_examples: int = 2,
    mode: str = "rules",
    label_info: Optional[Dict[str, Any]] = None,
    cache_dir: Optional[Path] = None,
    compiler_sop_doc_path: Optional[str] = None,
    advanced_labels: Optional[Dict[str, Any]] = None,
) -> str:
    """
    根据标签生成 RAG 示例的 SOP 步骤
    
    Args:
        model_client: 模型客户端
        label: 标签
        rag_jsonl_path: rag.jsonl 文件路径
        sop_doc_path: SOP 文档路径
        num_examples: 示例数量
        
    Returns:
        str: 格式化的 SOP 案例文本
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"开始生成 RAG 示例（标签: {label}，mode={mode}）")
    logger.info(f"{'='*60}\n")
    
    # 1. 加载 rag.jsonl
    logger.info(f"加载 RAG 数据: {rag_jsonl_path}")
    rag_entries = load_rag_jsonl(rag_jsonl_path)
    
    # 2. 检索：优先使用高级标签；否则 rules 用风格画像；最后回退旧逻辑
    logger.info(f"检索标签 '{label}' 的案例（数量: {num_examples}）")
    selected_cases: List[Dict[str, Any]] = []

    # 2.1 高级标签检索（带 primary / secondary / intensity / writing_style）
    if advanced_labels:
        try:
            scored = retrieve_by_advanced_labels(
                rag_entries=rag_entries,
                query_labels=advanced_labels,
                k=max(1, num_examples)
            )
            if scored:
                selected_cases = [entry for entry, _score in scored[:num_examples]]
                logger.info(f"高级标签检索成功，命中 {len(selected_cases)} 条")
            else:
                logger.warning("高级标签检索未命中，回退风格检索")
        except Exception as e:
            logger.warning(f"高级标签检索失败，回退风格检索: {e}")

    # 2.2 风格画像检索
    if not selected_cases and mode == "rules" and label_info is not None:
        selected_cases = retrieve_by_style(
            model_client=model_client,
            rag_entries=rag_entries,
            label_info=label_info,
            k=num_examples,
            cache_dir=cache_dir,
        )

    # 2.3 回退旧逻辑
    if not selected_cases:
        selected_cases = retrieve_by_label(rag_entries, label, k=num_examples)
    
    if not selected_cases:
        logger.warning(f"未找到标签 '{label}' 的案例，返回空字符串")
        return ""
    
    logger.info(f"成功检索到 {len(selected_cases)} 个案例")
    
    # 3. 加载 SOP 文档（sop 模式需要；rules 模式也可用于补充规则）
    logger.info(f"加载 SOP 文档: {sop_doc_path}")
    sop_doc = load_sop_doc(sop_doc_path)
    
    # 4. 生成参考：rules=规则+片段 few-shot；sop=反推 SOP1-5
    examples_text = "\n\n## 7. 参考案例\n\n---\n\n"

    if mode == "rules" and label_info is not None:
        # 为输入内容/案例抽取风格画像，构建 StyleSpec（规则注入）
        example_profiles: List[Dict[str, Any]] = []
        for c in selected_cases:
            cid = _stable_case_id(c)
            p = extract_style_profile(
                model_client=model_client,
                text=c.get("content", "") or "",
                main_label=label_info.get("main_label"),
                cache_dir=cache_dir,
                cache_key=f"rag_{cid}",
                temperature=0.1,
            )
            example_profiles.append(p)

        # input_style_profile 先用 label_input 近似（这里不再重复传入长 ASR；由 main 负责传）
        input_style_profile = {"main_label": label_info.get("main_label")}
        style_spec = build_style_spec(label_info, input_style_profile, example_profiles)
        
        # 保存 StyleSpec 和风格参数，稍后移到 ##8
        style_spec_content = style_spec
        
        # 提取风格参数（从 example_profiles 中获取）
        style_params_content = ""
        if example_profiles:
            first_profile = example_profiles[0]
            style_params_dict = {
                "main_label": first_profile.get("main_label", label_info.get("main_label")),
                "sub_label": first_profile.get("sub_label_guess", label_info.get("sub_label")),
                "suang_intensity": label_info.get("suang_intensity"),
                "rhythm_speed": label_info.get("rhythm_speed"),
                "dogblood_level": label_info.get("dogblood_level"),
                "reason": label_info.get("reason", ""),
            }
            style_params_content = json.dumps(style_params_dict, ensure_ascii=False, indent=2)

        # Example Compiler：把黄金原文"编译"为去剧情化资产 + SOP对齐过程示例（占位符）
        # 只需要 1 条案例即可（更稳、更少迁移风险）
        case = selected_cases[0]
        golden_content = case.get("content", "") or ""
        compiler_doc_path = compiler_sop_doc_path or sop_doc_path
        compiler_doc = load_sop_doc(compiler_doc_path)

        compiled_md = compile_example(
            model_client=model_client,
            sop_doc=compiler_doc,
            golden_content=golden_content,
            label_info=label_info,
            style_spec=style_spec,
        )

        # 分离"句式资产"和"SOP1-5 过程示例"
        # 查找分隔符 "---" 的位置，通常"句式资产"在前，"SOP1-5"在后
        sentence_assets = ""
        sop_examples = ""
        
        # 尝试用正则表达式分离两部分
        # 第一部分：从开头到第一个 "---" 之前（句式资产）
        # 第二部分：从第一个 "---" 之后到结尾（SOP1-5）
        parts = compiled_md.split("\n---\n", 1)
        if len(parts) == 2:
            sentence_assets = parts[0].strip()
            sop_examples = parts[1].strip()
        else:
            # 如果没有找到分隔符，尝试用 "### SOP1-5" 作为分隔点
            sop_match = re.search(r"###\s*SOP1-5", compiled_md, re.IGNORECASE)
            if sop_match:
                sentence_assets = compiled_md[:sop_match.start()].strip()
                sop_examples = compiled_md[sop_match.start():].strip()
            else:
                # 如果都找不到，整个作为 SOP1-5
                sop_examples = compiled_md.strip()

        # ##7. 参考案例：只包含 SOP1-5 过程示例和完整文案输出
        examples_text += "### SOP1-5 过程示例（占位符版｜严格对齐 SOP 文档）\n\n"
        examples_text += sop_examples + "\n"
        examples_text += "\n---\n\n"
        examples_text += "## 完整文案输出\n\n"
        examples_text += "**【爆款短视频解说文案】**\n\n"
        examples_text += golden_content + "\n\n"
        examples_text += "---\n\n"
        
        # ##8. 参考结构：包含句式资产、StyleSpec 和风格参数
        examples_text += "## 8. 参考结构\n\n"
        
        # 添加句式资产（去剧情化）
        if sentence_assets:
            examples_text += "### 句式资产（去剧情化）\n\n"
            examples_text += sentence_assets + "\n\n"
            examples_text += "---\n\n"
        
        # 添加风格规范（StyleSpec）
        examples_text += "### 风格规范（StyleSpec）\n\n"
        examples_text += style_spec_content + "\n\n"
        
        # 添加风格参数
        if style_params_content:
            examples_text += "### 风格参数\n\n"
            examples_text += "```json\n"
            examples_text += style_params_content + "\n"
            examples_text += "```\n\n"
        examples_text += "---\n"
        
        return examples_text
    
    for idx, case in enumerate(selected_cases, 1):
        case_id = case.get('id', f'case_{idx}')
        golden_content = case.get('content', '')
        
        logger.info(f"\n处理案例 {idx}/{len(selected_cases)}: {case_id}")
        
        try:
            # 生成 SOP
            sop_output = generate_forward_sop(
                model_client=model_client,
                sop_doc=sop_doc,
                golden_content=golden_content,
                max_retries=2
            )
            
            # 清理 SOP 输出（移除可能的文档头部）
            sop_output = _clean_sop_output(sop_output)
            
            # 构建案例文本（不添加案例标题，直接使用 SOP 内容）
            case_text = sop_output
            case_text += f"\n\n**【完整文案输出】**\n\n{golden_content}\n\n"
            
            # 如果不是最后一个案例，添加分隔线
            if idx < len(selected_cases):
                case_text += "---\n\n"
            
            examples_text += case_text
            
            logger.info(f"✅ 案例 {case_id} 生成成功")
            
        except Exception as e:
            logger.error(f"❌ 案例 {case_id} 生成失败: {e}")
            continue
    
    logger.info(f"\n✅ RAG 示例生成完成，共 {len(selected_cases)} 个案例\n")
    return examples_text


def build_enhanced_system_prompt(
    base_prompt_path: str,
    rag_examples: str
) -> str:
    """
    构建增强的 system prompt
    
    Args:
        base_prompt_path: 基础 system prompt 文件路径
        rag_examples: RAG 示例文本
        
    Returns:
        str: 增强后的 system prompt
    """
    logger.info(f"构建增强的 system prompt...")
    
    # 读取基础 prompt
    with open(base_prompt_path, 'r', encoding='utf-8') as f:
        base_prompt = f.read()
    
    # 检查是否已有 "## 7. 参考案例" 部分
    if "## 7. 参考案例" in base_prompt or "## 7.参考案例" in base_prompt:
        logger.info("检测到已有参考案例部分，将替换")
        # 移除已有的参考案例部分（从 ## 7 开始到文档结尾或下一个 ## 开始）
        base_prompt = re.sub(
            r'## 7\.?\s*参考案例[\s\S]*$',
            '',
            base_prompt
        ).rstrip()
    
    # 拼接
    enhanced_prompt = base_prompt + "\n\n" + rag_examples
    
    logger.info(f"✅ 增强 system prompt 构建完成，总长度: {len(enhanced_prompt)} 字符")
    return enhanced_prompt


PLOTGRAPH_SYSTEM_PROMPT = """你是【剧情事实锚点抽取器 PlotGraph】。
你的任务：从输入的剧情简介与对白片段中，抽取"确定发生的事实"与"未知/悬念点"。

硬规则：
1) 只输出严格 JSON，不要 Markdown
2) 只能写"输入中明确支持"的事实，不要脑补，不要写推测
3) 人物名若不确定，可以用"男主/女主/婆婆/恶少"等角色名
4) events 必须按对白/简介出现顺序排列（时序优先）
5) 必须抽取“过渡事件”：凡是地点切换、人物会面、回到住处、出发去某地、拿出某物/提到关键资产等，都要作为独立事件写入（只要输入能支持）
6) 【邻接连贯性校验】对每一对相邻事件 Ei->Ei+1：
   - 若两者之间缺少“可解释的因果/过渡”（不共享人物/地点/动机），必须：
     a) 优先在 events 中插入一条来自输入证据的桥接事件；
     b) 若输入没有证据可支撑桥接事件，则在 unknowns 中写：GAP(Ei->Ei+1): 缺失了什么环节（只能基于输入未说清）
7) 【新名词触发器】当某事件首次出现关键名词/资产/道具（如房契、溥仪、古董、银票等），其前必须有一条事件交代触发语境（如缺钱/换房/被质疑等），否则按规则6处理

"""


def extract_key_asr_segments(
    drama_info: str,
    asr_text: str,
    max_chars: int = 8000,
    context_window: int = 3,
) -> str:
    """
    智能提取 ASR 文本的关键片段，避免简单截断导致桥接证据丢失
    
    Args:
        drama_info: 剧情简介
        asr_text: 完整 ASR 文本
        max_chars: 最大字符数限制
        context_window: 关键词匹配句子的上下文窗口（前后各 N 句）
        
    Returns:
        str: 提取的关键片段
    """
    if not asr_text or len(asr_text) <= max_chars:
        return asr_text or ""
    
    # 从剧情简介中提取关键词（人物、地点、物品、动作等）
    keywords = []
    
    # 提取剧情简介中的关键名词（2-4字），过滤掉常见停用词
    stopwords = {'一个', '这个', '那个', '什么', '怎么', '为什么', '因为', '所以', '但是', '然而', '然后', '接着', '最后', '开始', '结束'}
    drama_keywords = re.findall(r'[\u4e00-\u9fa5]{2,4}', drama_info)
    keywords.extend([kw for kw in drama_keywords if len(kw) >= 2 and kw not in stopwords])
    
    # 提取包含重要动作/信息传递的词汇（这些通常是桥接事件的关键）
    important_patterns = [
        r'[看听认发收得][到知出见]',  # 看到/听到/认出/发现/收到/得知
        r'[上报递交展示告知]',  # 上报/递交/展示/告知
        r'[转述传达通知]',  # 转述/传达/通知
        r'监控[画面]?|视频|照片|证据|信物|文件|报告',  # 监控画面/视频/照片/证据/信物/文件/报告
        r'[柜员|服务员|医生|警察|老师|董事长|经理|主任]',  # 常见角色
    ]
    for pattern in important_patterns:
        matches = re.findall(pattern, drama_info)
        keywords.extend(matches)
    
    # 去重
    keywords = list(set(keywords))
    
    # 将 ASR 文本按句分割（支持多种分隔符）
    sentences = re.split(r'[。！？\n]+', asr_text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # 标记包含关键词的句子
    matched_indices = set()
    for i, sentence in enumerate(sentences):
        for keyword in keywords:
            if keyword in sentence:
                # 包含该句子及其上下文
                start = max(0, i - context_window)
                end = min(len(sentences), i + context_window + 1)
                matched_indices.update(range(start, end))
                break
    
    # 如果匹配的句子太少，使用更宽松的策略：保留前中后各一部分
    if len(matched_indices) < 10:
        # 保留开头、中间、结尾各一部分
        total = len(sentences)
        if total > 0:
            # 开头 30%
            start_indices = set(range(0, max(1, total // 3)))
            # 中间 40%
            mid_start = total // 3
            mid_end = mid_start + total // 2
            mid_indices = set(range(mid_start, min(total, mid_end)))
            # 结尾 30%
            end_start = total - total // 3
            end_indices = set(range(max(0, end_start), total))
            
            matched_indices.update(start_indices)
            matched_indices.update(mid_indices)
            matched_indices.update(end_indices)
    
    # 按顺序提取匹配的句子
    selected_sentences = []
    for i in sorted(matched_indices):
        if i < len(sentences):
            selected_sentences.append(sentences[i])
    
    result = '。'.join(selected_sentences)
    
    # 如果还是太长，按比例截断（保留开头和结尾）
    if len(result) > max_chars:
        # 保留开头 60% 和结尾 40%
        head_len = int(max_chars * 0.6)
        tail_len = max_chars - head_len
        result = result[:head_len] + '\n...\n' + result[-tail_len:]
    
    return result


def build_plot_graph(
    model_client: ModelClient,
    drama_info: str,
    asr_text: str,
    temperature: float = 0.1,
) -> Dict[str, Any]:
    # 智能提取 ASR 关键片段，避免截断导致桥接证据丢失
    asr_segments = extract_key_asr_segments(drama_info, asr_text, max_chars=8000)
    
    user_prompt = f"""请生成 PlotGraph（事实锚点），严格输出 JSON：
{{
  "characters": [{{"name":"角色名","role":"关系/定位"}}],
  "relations": ["A-关系-B", "..."],
  "events": [{{"id":"E1","fact":"确定事实（短句，原子事件）","evidence":"来自剧情简介/对白的完整原文片段（必须完整保存，不要省略，用于后续去幻觉检查）"}}],
  "beats": [{{"id":"B1","type":"爆点/冲突/反差/反转/钩子","fact_ref":["E1","E2"]}}],
  "unknowns": ["仍未揭晓/可做悬念的问题（只能基于输入中的未知）"]
}}

重要要求：
- events 中的 evidence 字段必须完整保存原文片段，不要省略或简写
- evidence 将用于后续去幻觉检查，需要完整的原文支持
- 事件必须是原子事件：一句话只做一件事，包含清晰动词
- 认知/信息获得类事件必须包含信息来源桥接事件，如果没有证据则只能写入 unknowns

【剧情简介】
{drama_info}

【对白片段】
{asr_segments}

补充要求：
- 如果发现事件链条存在缺口（例如人物如何会面/如何从A到B/为何突然谈到某物），必须用“桥接事件”补齐；若无法从输入找到证据，则写入 unknowns，格式：GAP(Ei->Ej): ...

"""
    raw, _ = model_client.call(
        system_prompt=PLOTGRAPH_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=temperature,
        top_p=0.4,
    )
    js = _extract_first_json_object(raw or "")
    if not js:
        return {"error": "plotgraph_parse_failed", "raw": raw}
    try:
        return json.loads(js)
    except Exception:
        return {"error": "plotgraph_json_failed", "raw": raw}


BEATSHEET_SYSTEM_PROMPT = """你是【节奏分镜 BeatSheet 规划器】。
目标：在不增删事实的前提下，规划 6 段节奏与信息披露，控制推进速度与钩子分布。
硬规则：只输出严格 JSON，不要 Markdown。
"""


def build_beat_sheet(
    model_client: ModelClient,
    label_info: Dict[str, Any],
    plot_graph: Dict[str, Any],
    temperature: float = 0.1,
) -> Dict[str, Any]:
    user_prompt = f"""请基于 PlotGraph 与风格参数，输出 6 段 BeatSheet（严格 JSON）：
{{
  "segments": [
    {{"seg":1,"goal":"开头钩子","max_new_facts":2,"must_hook":true,"notes":"...","fact_refs":["E1"]}},
    {{"seg":2,"goal":"铺垫推进","max_new_facts":2,"must_hook":true,"notes":"...","fact_refs":["E2","E3"]}},
    {{"seg":3,"goal":"冲突升级","max_new_facts":2,"must_hook":true,"notes":"...","fact_refs":["E4"]}},
    {{"seg":4,"goal":"反转/反差","max_new_facts":2,"must_hook":true,"notes":"...","fact_refs":["E5"]}},
    {{"seg":5,"goal":"高潮爆点","max_new_facts":2,"must_hook":true,"notes":"...","fact_refs":["E6"]}},
    {{"seg":6,"goal":"结尾悬念","max_new_facts":2,"must_hook":true,"notes":"卡在70%-99%完成度","fact_refs":["E7"]}}
  ]
}}

风格参数：
{json.dumps(label_info, ensure_ascii=False)}

PlotGraph：
{json.dumps(plot_graph, ensure_ascii=False)[:12000]}
"""
    raw, _ = model_client.call(
        system_prompt=BEATSHEET_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=temperature,
        top_p=0.4,
    )
    js = _extract_first_json_object(raw or "")
    if not js:
        return {"error": "beatsheet_parse_failed", "raw": raw}
    try:
        return json.loads(js)
    except Exception:
        return {"error": "beatsheet_json_failed", "raw": raw}


def format_constraints_block(
    style_spec: str,
    plot_graph: Optional[Dict[str, Any]],
    beat_sheet: Optional[Dict[str, Any]],
) -> str:
    out = "\n\n【硬约束｜生成前置】\n"
    out += style_spec.strip() + "\n"
    if plot_graph is not None:
        out += "\n【PlotGraph｜事实锚点（不得编造）】\n"
        out += json.dumps(plot_graph, ensure_ascii=False, indent=2)[:12000] + "\n"
        out += "规则：最终解说中的“确定事实”必须能映射到 events.fact（或剧情简介/对白明确支持）。不确定内容只能用悬念句，不得编细节。\n"
        out += "【/PlotGraph】\n"
    if beat_sheet is not None:
        out += "\n【BeatSheet｜节奏分镜（信息预算）】\n"
        out += json.dumps(beat_sheet, ensure_ascii=False, indent=2)[:12000] + "\n"
        out += "规则：每段最多引入 max_new_facts 个新事实；每段结尾必须有 must_hook 的钩子/转折。\n"
        out += "【/BeatSheet】\n"
    out += "【/硬约束】\n"
    return out


def format_instance_blocks(
    plot_graph: Optional[Dict[str, Any]],
    beat_sheet: Optional[Dict[str, Any]],
    max_json_chars: int = 12000,
) -> str:
    """
    将每次任务的“实例数据”（PlotGraph/BeatSheet）格式化为输入块，用于拼接到 user message。

    设计目标：
    - skill/system prompt 保持通用能力与规则，不塞入每次变化的实例数据，避免串戏与注意力稀释
    - 实例数据放到 user message，作为本次任务的显式输入
    """
    out = ""
    if plot_graph is not None:
        out += "\n\n【PlotGraph】\n"
        out += json.dumps(plot_graph, ensure_ascii=False, indent=2)[:max_json_chars] + "\n"
        out += "【/PlotGraph】"
    if beat_sheet is not None:
        out += "\n\n【BeatSheet】\n"
        out += json.dumps(beat_sheet, ensure_ascii=False, indent=2)[:max_json_chars] + "\n"
        out += "【/BeatSheet】"
    return out.strip()


def detect_invalid_script_output(raw_response: str) -> Tuple[bool, str]:
    """
    检测模型输出是否为不可用的“模板化梗概文”，而非可直接用于 TTS 的正文。
    """
    text = (raw_response or "").strip()
    final_text = extract_final_commentary(text) if text else ""
    target_text = final_text.strip()

    if not target_text:
        return True, "final_commentary_missing"

    invalid_markers = [
        "根据提供的信息，我们可以初步勾勒出",
        "由于缺乏详细的情节描述",
        "请注意，以上内容基于提供的信息",
        "如果需要更准确的故事复述",
        "故事大致走向",
        "较为模糊的故事框架",
        "根据提供的短剧对白内容和指令",
        "首先，我们需要梳理剧情",
        "以下是根据对白内容整理的大纲",
        "接下来，我们将提取",
        "整体输出风格控制参考",
        "参考案例",
    ]
    for marker in invalid_markers:
        if marker in target_text:
            return True, f"template_marker:{marker}"

    # 明显提纲/方法论文本，非可直接配音正文
    if re.search(r"\n\s*\d+\s*[.、]\s*", target_text):
        return True, "outline_numbered_list"
    if "SOP" in target_text.upper():
        return True, "sop_style_output"

    if len(target_text) < 120:
        return True, f"too_short:{len(target_text)}"

    return False, "ok"


def check_required_output_format(raw_response: str) -> Tuple[bool, str]:
    """
    检查是否满足固定输出结构：
    SOP1 -> SOP2 -> SOP3 -> SOP4 -> SOP5 -> 最终解说文案
    """
    text = (raw_response or "")
    if not text.strip():
        return False, "empty_output"

    patterns = [
        r"^##\s*SOP\s*1\b",
        r"^##\s*SOP\s*2\b",
        r"^##\s*SOP\s*3\b",
        r"^##\s*SOP\s*4\b",
        r"^##\s*SOP\s*5\b",
        r"^##\s*最终解说文案\b",
    ]
    positions = []
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE | re.MULTILINE)
        if not m:
            return False, f"missing_section:{p}"
        positions.append(m.start())

    if positions != sorted(positions):
        return False, "section_order_invalid"

    return True, "ok"


# ============================================================================
# 步骤 5: 生成解说（复用 fewshot1231.py 的逻辑）
# ============================================================================

def generate_commentary_with_enhanced_prompt(
    model_client: ModelClient,
    eval_skill: Any,
    enhanced_system_prompt: str,
    user_prompt: str,
    dj: str,
    max_iterations: int = 3,
    trace_dir: Optional[Path] = None,
    trace_prefix: str = "trace",
    gate_enabled: bool = True,
    plot_graph: Optional[Dict[str, Any]] = None,
    beat_sheet: Optional[Dict[str, Any]] = None,
) -> Tuple[str, List[Dict[str, str]]]:
    """
    使用增强的 system prompt 生成解说（支持多轮迭代）
    
    Args:
        model_client: 模型客户端
        eval_skill: 评估技能模块
        enhanced_system_prompt: 增强后的 system prompt
        user_prompt: 用户 prompt
        dj: 短剧名（用于保存评估结果）
        max_iterations: 最大迭代轮数
        
    Returns:
        tuple: (最终解说文本, 对话历史)
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"开始生成解说（最大迭代: {max_iterations}）")
    logger.info(f"{'='*60}\n")
    
    # 初始化对话历史
    messages = [
        {"role": "system", "content": enhanced_system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    # 如果未提供评估模块，直接单轮生成并返回，不做评估/Gate
    if eval_skill is None:
        retry_limit = 5
        logger.info(f"未提供评估模块，进入单轮重试模式（最多 {retry_limit} 次），不进行评估与 Gate")
        last_reason = "unknown"
        for attempt in range(1, retry_limit + 1):
            raw_response, _ = model_client.call(messages=messages, temperature=0.45, top_p=0.85)
            messages.append({"role": "assistant", "content": raw_response})

            # 先检查固定输出结构，不满足则直接重试
            format_ok, format_reason = check_required_output_format(raw_response or "")
            if not format_ok:
                last_reason = format_reason
                logger.warning(f"单轮生成第 {attempt} 次格式不符合要求，原因: {format_reason}")
                if attempt < retry_limit:
                    retry_prompt = (
                        "你上一轮输出不符合固定结构要求。"
                        "请严格按固定结构重写并完整输出：SOP1、SOP2、SOP3、SOP4、SOP5、最终解说文案。"
                        "其中“最终解说文案”必须是可直接用于TTS的连续中文旁白正文。"
                    )
                    messages.append({"role": "user", "content": retry_prompt})
                continue

            is_invalid, reason = detect_invalid_script_output(raw_response or "")
            if not is_invalid:
                logger.info(f"单轮生成第 {attempt} 次成功，响应长度: {len(raw_response or '')} 字符")
                return raw_response or "", messages

            logger.warning(f"单轮生成第 {attempt} 次命中无效正文，原因: {reason}")
            last_reason = reason
            if attempt < retry_limit:
                retry_prompt = (
                    "你上一轮输出不符合格式要求。"
                    "请严格按固定结构重写并完整输出：SOP1、SOP2、SOP3、SOP4、SOP5、最终解说文案。"
                    "其中“最终解说文案”必须是可直接用于TTS的连续中文旁白正文。"
                    "不要免责声明，不要“根据提供的信息”等套话，不要提纲编号列表。"
                )
                messages.append({"role": "user", "content": retry_prompt})

        raise RuntimeError(f"连续 {retry_limit} 次未生成符合要求的文案，终止流程。最后原因: {last_reason}")

    iteration = 0
    raw_response = ""
    last_assistant_text: Optional[str] = None
    
    while iteration < max_iterations:
        iteration += 1
        logger.info(f"第 {iteration} 轮对话开始")
        
        try:
            # 调用模型
            # 生成阶段：显式降温，减少漂移/幻觉
            raw_response, _ = model_client.call(messages=messages, temperature=0.45, top_p=0.85)
            messages.append({"role": "assistant", "content": raw_response})
            
            logger.info(f"第 {iteration} 轮模型调用成功，响应长度: {len(raw_response)} 字符")

            # 保存本轮模型输出（含 SOP1-5 / 最终文案等原始内容）
            if trace_dir is not None:
                trace_dir.mkdir(parents=True, exist_ok=True)
                iter_out_path = trace_dir / f"{trace_prefix}_iter{iteration:02d}_assistant.md"
                with open(iter_out_path, "w", encoding="utf-8") as f:
                    f.write(raw_response or "")

                # 保存 diff（与上一轮 assistant 输出对比，便于看“怎么改的”）
                if last_assistant_text:
                    diff_lines = difflib.unified_diff(
                        last_assistant_text.splitlines(keepends=True),
                        (raw_response or "").splitlines(keepends=True),
                        fromfile=f"iter{iteration-1:02d}",
                        tofile=f"iter{iteration:02d}",
                    )
                    diff_path = trace_dir / f"{trace_prefix}_iter{iteration:02d}_diff.patch"
                    with open(diff_path, "w", encoding="utf-8") as f:
                        f.writelines(diff_lines)
                last_assistant_text = raw_response or ""
            
            # 调用评估
            logger.info(f"第 {iteration} 轮：调用评估模块")
            eval_result = eval_skill.execute(raw_response, dj=dj)
            eval_response = eval_result.get("raw_response", "")

            # 保存本轮评估反馈
            if trace_dir is not None:
                iter_eval_path = trace_dir / f"{trace_prefix}_iter{iteration:02d}_eval.md"
                with open(iter_eval_path, "w", encoding="utf-8") as f:
                    f.write(eval_response or "")
            
            # 检查评估结果
            if "SUPER-优秀" in eval_response.upper():
                logger.info(f"✅ 第 {iteration} 轮评估通过，停止迭代")
                break
            else:
                logger.info(f"⚠️  第 {iteration} 轮评估未通过，需要修订")
                
                if iteration < max_iterations:
                    revision_prompt = f"这一版解说文案的问题和优化建议如下：\n\n{eval_response}\n\n请按照优化建议进行修改"

                    # Gate：硬拦截（幻觉/节奏预算/结尾钩子），并生成“定点修复”指令
                    if gate_enabled:
                        gate_report = run_gate_check(
                            model_client=model_client,
                            assistant_text=raw_response,
                            plot_graph=plot_graph,
                            beat_sheet=beat_sheet,
                        )
                        if trace_dir is not None:
                            gate_path = trace_dir / f"{trace_prefix}_iter{iteration:02d}_gate.json"
                            _save_json(gate_path, gate_report)

                        if gate_report.get("pass") is False:
                            fix_targets = gate_report.get("fix_targets") or []
                            fix_plan = gate_report.get("fix_plan") or ""
                            revision_prompt = (
                                "【Gate 硬拦截未通过】\n"
                                + (gate_report.get("summary") or "")
                                + "\n\n【只做定点修复（不要整篇重写）】\n"
                                + f"- 只允许修改这些章节：{fix_targets}\n"
                                + "- 其他章节保持原样（尽量逐句不动）\n"
                                + "- 必须输出完整 SOP1-5 + 最终解说文案，顺序不变\n"
                                + "\n【修复计划】\n"
                                + fix_plan
                                + "\n\n【评估建议】\n"
                                + eval_response
                            )

                    messages.append({"role": "user", "content": revision_prompt})
                    logger.info(f"已添加修订请求，准备第 {iteration + 1} 轮对话")
                else:
                    logger.info(f"已达到最大轮数 {max_iterations}，停止迭代")
                    break
        
        except Exception as e:
            logger.error(f"第 {iteration} 轮失败: {e}")
            raise
    
    logger.info(f"\n✅ 解说生成完成，共 {iteration} 轮\n")
    return raw_response, messages


GATE_SYSTEM_PROMPT = """你是【生成质量 Gate 审核器】。
你的任务：基于 PlotGraph/BeatSheet，对模型输出做硬性拦截检查，并给出可执行的定点修复计划。

硬规则：
1) 只输出严格 JSON，不要 Markdown
2) pass 必须是 true/false
3) fix_targets 只能从 ["SOP1","SOP2","SOP3","SOP4","SOP5","最终解说文案"] 中选择若干
"""


def run_gate_check(
    model_client: ModelClient,
    assistant_text: str,
    plot_graph: Optional[Dict[str, Any]],
    beat_sheet: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Gate：把“幻觉/节奏预算/结尾钩子”变成硬拦截，并生成定点修复建议。
    说明：这里用低温 LLM 做规则核对，避免纯正则误伤（尤其中文事实对齐）。
    """
    user_prompt = f"""请检查下面输出是否满足 Gate：

Gate 规则：
G1 幻觉：输出中出现 PlotGraph 之外的“确定事实/人物动作/关键事件” -> fail
G2 节奏预算：BeatSheet 每段 max_new_facts 约束被明显突破 -> fail
G3 结尾钩子：最终解说文案结尾缺乏悬念/钩子 -> fail

输出 JSON：
{{
  "pass": true,
  "summary": "一句话结论",
  "fail_reasons": ["..."],
  "fix_targets": ["SOP4","最终解说文案"],
  "fix_plan": "按条列出：改哪几句、为什么、怎么改（尽量给替换句式）"
}}

PlotGraph：
{json.dumps(plot_graph, ensure_ascii=False)[:12000] if plot_graph else "(none)"}

BeatSheet：
{json.dumps(beat_sheet, ensure_ascii=False)[:12000] if beat_sheet else "(none)"}

模型输出：
{assistant_text[:12000]}
"""
    raw, _ = model_client.call(
        system_prompt=GATE_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.1,
        top_p=0.4,
    )
    js = _extract_first_json_object(raw or "")
    if not js:
        return {"pass": True, "summary": "gate_parse_failed_fallback_pass", "raw": raw}
    try:
        parsed = json.loads(js)
    except Exception:
        return {"pass": True, "summary": "gate_json_failed_fallback_pass", "raw": raw}

    # 轻度兜底校验
    if parsed.get("pass") not in [True, False]:
        parsed["pass"] = True
    if not isinstance(parsed.get("fix_targets"), list):
        parsed["fix_targets"] = []
    return parsed


# ============================================================================
# 去幻觉流程：检查冲突 → 局部修补
# ============================================================================

def extract_plotgraph_facts(plot_graph: Dict[str, Any]) -> Dict[str, Any]:
    """
    从 PlotGraph 中提取用于检查的关键字段
    
    Args:
        plot_graph: PlotGraph 数据
        
    Returns:
        Dict: 包含 characters, relations, events, unknowns 的字典
    """
    return {
        "characters": plot_graph.get("characters", []),
        "relations": plot_graph.get("relations", []),
        "events": plot_graph.get("events", []),
        "unknowns": plot_graph.get("unknowns", []),
    }


def extract_final_commentary(raw_response: str) -> str:
    """
    从 raw_response 中提取"最终解说文案"部分，避免 SOP 过程干扰检查
    
    Args:
        raw_response: 包含 SOP1-5 和最终解说文案的完整输出
        
    Returns:
        str: 提取的最终解说文案（如果找不到，返回空字符串）
    """
    # 尝试多种匹配模式
    patterns = [
        r"##\s*完整文案输出[\s\S]*?【爆款短视频解说文案】[\s\S]*?(?=##|\Z)",
        r"【爆款短视频解说文案】[\s\S]*?(?=##|\Z)",
        r"##\s*最终解说文案[\s\S]*?(?=##|\Z)",
        r"##\s*最终生成的精彩解说文案[\s\S]*?(?=##|\Z)",
        r"最终生成的精彩解说文案[\s\S]*?(?=##|\Z)",
        r"##\s*最终文案输出[\s\S]*?(?=##|\Z)",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, raw_response, re.IGNORECASE)
        if match:
            text = match.group(0)
            # 移除标题行
            text = re.sub(r"^.*?【爆款短视频解说文案】.*?\n", "", text, flags=re.IGNORECASE)
            text = re.sub(r"^.*?最终解说文案.*?\n", "", text, flags=re.IGNORECASE)
            text = re.sub(r"^.*?最终生成的精彩解说文案.*?\n", "", text, flags=re.IGNORECASE)
            text = re.sub(r"^.*?最终文案输出.*?\n", "", text, flags=re.IGNORECASE)
            text = re.sub(r"^.*?完整文案输出.*?\n", "", text, flags=re.IGNORECASE)
            return text.strip()
    
    # 不再回退到完整 raw_response，避免 SOP/流程文本污染后续去幻与提取
    logger.warning("未能提取最终解说文案，返回空字符串")
    return ""


CHECK_CONFLICTS_SYSTEM_PROMPT = """你是【解说冲突检查员】。
你的任务：检查解说中与 PlotGraph 的冲突，只标记相悖的内容，不标记 PlotGraph 中没有的内容。

检查规则（必须标记）：
1. contradicted：与 events/characters/relations 中的事实相悖
   - 每个 event 包含 fact（事实）和 evidence（完整原文证据）
   - 如果解说与 fact 相悖，或与 evidence 中的原文不符，必须标记
   - 可以使用 evidence 中的完整原文来判断解说是否正确
2. over_confident_on_unknown：将 unknowns 中的内容说成确定事实（需要降格为不确定表达）

允许的情况（不标记）：
- 解说中有 PlotGraph 没有的内容（允许补充/扩展）

硬规则：
1) 只输出严格 JSON，不要 Markdown
2) 只定位冲突的句子/片段，给出具体位置
3) 必须提供 suggested_fix（如何修改）
4) 在 conflict_detail 中，可以引用 events 中的 evidence 来说明为什么是冲突
"""


def check_hallucination_conflicts(
    model_client: ModelClient,
    final_commentary: str,
    plotgraph_facts: Dict[str, Any],
    temperature: float = 0.1,
) -> Dict[str, Any]:
    """
    检查解说中与 PlotGraph 的冲突
    
    Args:
        model_client: 模型客户端
        final_commentary: 最终解说文案
        plotgraph_facts: PlotGraph 关键字段（characters, relations, events, unknowns）
        temperature: 采样温度
        
    Returns:
        Dict: 冲突检查报告
    """
    user_prompt = f"""请检查下面解说中与 PlotGraph 的冲突。

检查规则：
1. contradicted：与 events/characters/relations 中的事实相悖 → 必须标记
   - events 中每个事件包含 fact（事实）和 evidence（完整原文证据）
   - 如果解说与 fact 相悖，或与 evidence 中的原文不符，必须标记
   - 在 conflict_detail 中，可以引用 evidence 来说明冲突原因
2. over_confident_on_unknown：将 unknowns 中的内容说成确定事实 → 必须标记
3. 解说中有 PlotGraph 没有的内容 → 允许，不标记

输出格式（严格 JSON）：
{{
  "has_conflicts": true,
  "issues": [
    {{
      "type": "contradicted",
      "location": {{"sentence_idx": 5, "span": "原文片段"}},
      "conflict_with": "E1",
      "conflict_detail": "与事件E1的事实相悖：事件E1的fact是'...'，evidence原文是'...'，但解说中写的是'...'，与原文不符",
      "original_text": "冲突的原文",
      "suggested_fix": "修改建议（替换为与PlotGraph一致的表达，可参考evidence中的原文）"
    }},
    {{
      "type": "over_confident_on_unknown",
      "location": {{"sentence_idx": 8, "span": "原文片段"}},
      "conflict_with": "unknown_1",
      "conflict_detail": "将未知内容说成确定事实：...",
      "original_text": "过度确定的原文",
      "suggested_fix": "降格为不确定表达（如'似乎'、'可能'、'看起来'）"
    }}
  ],
  "summary": "发现 X 处冲突：Y 处相悖，Z 处过度确定"
}}

【PlotGraph 关键字段】
注意：events 中的 evidence 字段包含完整的原文证据，请仔细比对。
{{
  "characters": {json.dumps(plotgraph_facts['characters'], ensure_ascii=False, indent=2)},
  "relations": {json.dumps(plotgraph_facts['relations'], ensure_ascii=False, indent=2)},
  "events": {json.dumps(plotgraph_facts['events'], ensure_ascii=False, indent=2)},
  "unknowns": {json.dumps(plotgraph_facts['unknowns'], ensure_ascii=False, indent=2)}
}}

【最终解说文案】
{final_commentary[:8000]}

输出：仅输出 JSON。
"""
    
    raw, _ = model_client.call(
        system_prompt=CHECK_CONFLICTS_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=temperature,
        top_p=0.3,
    )
    
    js = _extract_first_json_object(raw or "")
    if not js:
        logger.warning("冲突检查解析失败，返回无冲突报告")
        return {"has_conflicts": False, "issues": [], "summary": "检查失败，默认无冲突"}
    
    try:
        parsed = json.loads(js)
        # 确保 has_conflicts 字段存在
        if "has_conflicts" not in parsed:
            parsed["has_conflicts"] = len(parsed.get("issues", [])) > 0
        return parsed
    except Exception:
        logger.warning("冲突检查 JSON 解析失败，返回无冲突报告")
        return {"has_conflicts": False, "issues": [], "summary": "检查失败，默认无冲突"}


PATCH_CONFLICTS_SYSTEM_PROMPT = """你是【局部冲突修补器】。
你的任务：只根据冲突报告修改解说中被标记的冲突部分，其他内容一字不动。

修补策略：
1. contradicted → 替换为与 PlotGraph 一致的表达
2. over_confident_on_unknown → 降格为不确定表达（如"似乎"、"可能"、"看起来"、"他意识到..."）

硬规则：
1) 只改 issues 中标记的句子/片段；未标记内容一字不动
2) 必须保持原文节奏和风格
3) 只输出"修补后的最终解说文案全文"，不要输出 JSON，不要输出解释
"""


def patch_commentary_local(
    model_client: ModelClient,
    final_commentary: str,
    conflict_report: Dict[str, Any],
    plotgraph_facts: Dict[str, Any],
    temperature: float = 0.2,
) -> str:
    """
    根据冲突报告进行局部修补
    
    Args:
        model_client: 模型客户端
        final_commentary: 原始最终解说文案
        conflict_report: 冲突检查报告
        plotgraph_facts: PlotGraph 关键字段
        temperature: 采样温度
        
    Returns:
        str: 修补后的最终解说文案
    """
    issues = conflict_report.get("issues", [])
    if not issues:
        return final_commentary
    
    # 构建修补指令
    patch_instructions = []
    for issue in issues:
        issue_type = issue.get("type", "")
        original_text = issue.get("original_text", "")
        suggested_fix = issue.get("suggested_fix", "")
        conflict_detail = issue.get("conflict_detail", "")
        
        if issue_type == "contradicted":
            instruction = f"- 类型：相悖冲突\n"
            instruction += f"  原文：{original_text}\n"
            instruction += f"  冲突详情：{conflict_detail}\n"
            instruction += f"  修改建议：{suggested_fix}\n"
        elif issue_type == "over_confident_on_unknown":
            instruction = f"- 类型：过度确定未知内容\n"
            instruction += f"  原文：{original_text}\n"
            instruction += f"  冲突详情：{conflict_detail}\n"
            instruction += f"  修改建议：{suggested_fix}\n"
        else:
            continue
        
        patch_instructions.append(instruction)
    
    user_prompt = f"""请根据冲突清单对解说文案进行局部修补。

修补策略：
1. contradicted → 替换为与 PlotGraph 一致的表达
2. over_confident_on_unknown → 降格为不确定表达（如"似乎"、"可能"、"看起来"、"他意识到..."）

编辑硬规则：
- 只改 issues 中标记的句子/片段；未标记内容一字不动
- 必须保持原文节奏和风格
- 短分句优先，口播友好

【冲突清单】
{chr(10).join(patch_instructions)}

【PlotGraph 关键字段（供参考）】
{json.dumps(plotgraph_facts, ensure_ascii=False, indent=2)[:4000]}

【final_commentary 原文】
{final_commentary[:8000]}

输出：只输出"修补后的最终解说文案全文"，不要输出 JSON，不要输出解释。
"""
    
    raw, _ = model_client.call(
        system_prompt=PATCH_CONFLICTS_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=temperature,
        top_p=0.4,
    )
    
    return (raw or "").strip()


def format_output_with_dehallucination(
    original_response: str,
    original_commentary: str,
    patched_commentary: Optional[str],
    conflict_report: Dict[str, Any],
) -> str:
    """
    格式化最终输出，包含原始版本和去幻觉后的版本
    
    Args:
        original_response: 原始完整输出（包含 SOP1-5）
        original_commentary: 原始最终解说文案
        patched_commentary: 修补后的解说文案（如有）
        conflict_report: 冲突检查报告
        
    Returns:
        str: 格式化后的完整输出
    """
    output_parts = []
    
    # 1. 原始生成的解说文案
    output_parts.append("## 原始生成的解说文案\n")
    output_parts.append(original_response)
    output_parts.append("\n\n---\n\n")
    
    # 2. 冲突检查报告
    output_parts.append("## 冲突检查报告\n\n")
    has_conflicts = conflict_report.get("has_conflicts", False)
    summary = conflict_report.get("summary", "")
    
    if has_conflicts:
        output_parts.append(f"**检查结果**：发现冲突\n\n")
        output_parts.append(f"**摘要**：{summary}\n\n")
        
        issues = conflict_report.get("issues", [])
        if issues:
            output_parts.append("**详细问题列表**：\n\n")
            for idx, issue in enumerate(issues, 1):
                issue_type = issue.get("type", "")
                conflict_detail = issue.get("conflict_detail", "")
                original_text = issue.get("original_text", "")
                suggested_fix = issue.get("suggested_fix", "")
                
                output_parts.append(f"{idx}. **{issue_type}**\n")
                output_parts.append(f"   - 冲突详情：{conflict_detail}\n")
                output_parts.append(f"   - 原文：{original_text}\n")
                output_parts.append(f"   - 修改建议：{suggested_fix}\n\n")
    else:
        output_parts.append("**检查结果**：未发现冲突\n\n")
        output_parts.append("解说文案与 PlotGraph 无冲突，无需修改。\n\n")
    
    output_parts.append("---\n\n")
    
    # 3. 去幻觉后的解说文案（如有修改）
    if patched_commentary:
        output_parts.append("## 去幻觉后的解说文案\n\n")
        output_parts.append(patched_commentary)
        output_parts.append("\n\n---\n\n")
    
    # 4. 说明
    output_parts.append("## 说明\n\n")
    if has_conflicts and patched_commentary:
        output_parts.append("- 已检测到冲突并完成修补\n")
        output_parts.append("- 只修改了与 PlotGraph 相悖的内容，其他内容保持不变\n")
        output_parts.append("- 允许解说中有 PlotGraph 没有的内容（允许补充/扩展）\n")
    else:
        output_parts.append("- 已检查，未发现冲突\n")
        output_parts.append("- 解说文案与 PlotGraph 一致，无需修改\n")
        output_parts.append("- 允许解说中有 PlotGraph 没有的内容（允许补充/扩展）\n")
    
    return "".join(output_parts)


# ============================================================================
# 主流程
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='带 RAG 示例的解说文案生成系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
工作流程：
  1. 对输入剧情打标
  2. 从 rag.jsonl 检索相同标签的解说（2条）
  3. 生成这些解说的 SOP1-5 步骤
  4. 将 SOP 案例拼接到 system prompt
  5. 使用增强的 prompt 生成解说

示例用法：
  python commentary_with_rag_examples.py \\
    --dj aijiufangzhiqm \\
    --start 1 \\
    --end 10
        """
    )
    
    # 输入参数
    parser.add_argument('--dj', type=str, default="buxi5", help='短剧名')
    #     parser.add_argument('--dj', type=str, default="aijiufangzhiqm", help='短剧名')
    parser.add_argument('--start', type=int, default=1, help='开始集数')
    parser.add_argument('--end', type=int, default=15, help='结束集数')
    
    # 文件路径
    # RAG 相关参数暂时保留但不使用，方便以后扩展
    parser.add_argument(
        '--rag-jsonl',
        type=str,
        default='/home/jovyan/mllm-nlp-dev/search_public/humiao/video_commentary_text/rag-jsonl/newrag0105.jsonl',
        help='[暂未使用] rag.jsonl 文件路径'
    )
    parser.add_argument(
        '--system-prompt',
        type=str,
        default='video_commentary_text/prompts/skill_video_commentary_text_formatted.md',
        help='System Prompt / SOP 文档路径（用于直接构建最终 prompt，不再通过 RAG 注入案例）'
    )
    
    # 模型参数
    parser.add_argument(
        '--api-url',
        type=str,
        default='https://api.360.cn/v1/chat/completions',
        help='API 地址'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        default='fk1379989049.aEmK59LGEcoiwxat0pGE18JPvtSBkrybaf481994',
        help='API Key'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='google/gemini-3-pro-preview',
        help='模型名称'
    )
    
    # 其他参数
    # RAG 模式相关参数暂时保留但不使用
    parser.add_argument(
        '--num-examples',
        type=int,
        default=1,
        help='[暂未使用] RAG 示例数量'
    )
    parser.add_argument(
        '--rag-mode',
        type=str,
        default='rules',
        choices=['rules', 'sop'],
        help='[暂未使用 RAG 示例] 预留参数，当前仅用于控制是否注入 PlotGraph/BeatSheet 约束'
    )
    parser.add_argument(
        '--compiler-sop-doc',
        type=str,
        default='prompts/skill_video_commentary_text_1230.md',
        help='[暂未使用] Example Compiler 使用的 SOP 文档路径'
    )
    parser.add_argument(
        '--enable-plotgraph',
        action='store_true',
        help='开启 PlotGraph 事实锚点（推荐，与 rules 模式配合）'
    )
    parser.add_argument(
        '--enable-beatsheet',
        action='store_true',
        help='开启 BeatSheet 节奏分镜（推荐，与 rules 模式配合）'
    )
    parser.add_argument(
        '--disable-gate',
        action='store_true',
        help='关闭 Gate 硬拦截（默认开启，建议保留）'
    )
    parser.add_argument(
        '--max-iterations',
        type=int,
        default=3,
        help='最大迭代轮数'
    )
    parser.add_argument(
        '--trace-sop',
        action='store_true',
        help='调试模式：保存每轮迭代的 SOP1-5/最终文案输出、评估反馈与 diff，便于查看每轮怎么改的'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results_with_rag',
        help='输出目录'
    )
    
    args = parser.parse_args()
    
    logger.info(f"""
{'='*60}
带 RAG 示例的解说文案生成系统
{'='*60}

输入信息:
  - 短剧: {args.dj}
  - 集数: {args.start} - {args.end}
  - RAG 示例数: {args.num_examples}
  - 最大迭代: {args.max_iterations}
    """)
    
    try:
        # 初始化模型客户端
        logger.info("初始化模型客户端...")
        model_client = ModelClient(
            api_url=args.api_url,
            api_key=args.api_key,
            model=args.model,
            timeout=300
        )
        
        # 评估模块暂时禁用：直接设为 None
        logger.info("评估模块暂时禁用，直接返回首轮生成结果")
        eval_skill = None
        
        # 步骤 1: 获取剧情信息、ASR 文本并打标
        logger.info("\n" + "="*60)
        logger.info("步骤 1: 获取剧情信息、ASR 文本并打标")
        logger.info("="*60 + "\n")
        
        # 获取剧情简介和 ASR 文本（使用 fewshot1231.py 的逻辑）
        from skill_video_commentary_text import SkillVideoCommentaryText
        temp_skill = SkillVideoCommentaryText()
        drama_info = temp_skill._get_drama_info(args.dj)
        
        logger.info(f"剧情简介: {drama_info}")
        
        # 加载 ASR 文本（用于更准确的打标）
        try:
            asr_text = temp_skill._load_asr_texts(args.dj, args.start, args.end)
            logger.info(f"ASR 文本加载成功，长度: {len(asr_text)} 字符")
            # 用 ASR 文本进行打标（更准确）
            label_input = f"{drama_info}\n\n【实际对白片段】\n{asr_text[:3000]}"  # 取前 3000 字符
        except Exception as e:
            logger.warning(f"ASR 文本加载失败: {e}，将只使用剧情简介打标")
            label_input = drama_info
        
        # 打标（优先使用高级多维标签；失败回退 v2）
        advanced_labels: Optional[Dict[str, Any]] = None
        label = None
        label_raw = ""
        try:
            advanced_labels = label_content_advanced(
                model_client=model_client,
                content=drama_info,
                asr_text=asr_text if 'asr_text' in locals() else None,
                enable_event_labeling=False  # 这里仅用全局标签做检索
            )
            label = (advanced_labels.get("global") or {}).get("primary_label")
            label_raw = json.dumps(advanced_labels, ensure_ascii=False)
            logger.info(f"✅ 高级打标结果: primary={label}, secondary={(advanced_labels.get('global') or {}).get('secondary_label')}")
        except Exception as e:
            logger.warning(f"高级打标失败，回退到 label_v2：{e}")
        
        # 回退/补充 v2：主标签+子标签+强度
        label_v2, label_raw_v2 = label_content_v2(model_client, label_input)
        if not label:
            label = label_v2.get("main_label")
            label_raw = label_raw_v2
        # 若高级标签缺少子标签/强度，补齐
        if advanced_labels is None:
            advanced_labels = {}
        if "global" not in advanced_labels:
            advanced_labels["global"] = {}
        g = advanced_labels["global"]
        g.setdefault("primary_label", label_v2.get("main_label"))
        g.setdefault("secondary_label", label_v2.get("sub_label"))
        g.setdefault("intensity", {
            "pace": label_v2.get("rhythm_speed", 3),
            "hook_density": label_v2.get("suang_intensity", 3),
            "emotion": label_v2.get("suang_intensity", 3),
            "conflict": label_v2.get("dogblood_level", 3),
            "twist": 3,
        })
        g.setdefault("writing_style", {
            "POV": "旁白叙述",
            "sentence": "短句为主",
            "hook_type": [],
            "tone": "克制",
            "dialogue_ratio": "低",
        })
        g.setdefault("confidence", 0.7)
        g.setdefault("evidence", [])
        
        # 兼容旧字段 label_info 供 style 检索使用
        label_info = {
            "main_label": g.get("primary_label"),
            "sub_label": g.get("secondary_label"),
            "suang_intensity": (g.get("intensity") or {}).get("hook_density", 3),
            "rhythm_speed": (g.get("intensity") or {}).get("pace", 3),
            "dogblood_level": (g.get("intensity") or {}).get("conflict", 3),
            "reason": (g.get("evidence") or [""])[0] if g.get("evidence") else "",
        }
        
        logger.info(f"✅ 最终使用主标签: {label}")
        
        # 准备输出目录（提前创建，用于保存中间结果）
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{args.start}_{args.end}_{timestamp}"
        cache_dir = output_dir / "_cache"

        # 规则驱动：生成 PlotGraph / BeatSheet（作为硬约束注入）
        plot_graph = None
        beat_sheet = None
        plot_graph_path = output_dir / f"{base_filename}_plot_graph.json"
        beat_sheet_path = output_dir / f"{base_filename}_beat_sheet.json"
        if args.enable_plotgraph or (args.rag_mode == "rules"):
            try:
                plot_graph = build_plot_graph(model_client, drama_info, asr_text if 'asr_text' in locals() else "")
                # 始终保存，方便排查与复现（不依赖 --trace-sop）
                _save_json(plot_graph_path, plot_graph if isinstance(plot_graph, dict) else {"plot_graph": plot_graph})
                logger.info(f"✅ PlotGraph 已保存: {plot_graph_path}")
            except Exception as e:
                logger.warning(f"PlotGraph 生成失败（继续运行）: {e}")
                plot_graph = None

        if (args.enable_beatsheet or (args.rag_mode == "rules")) and plot_graph is not None:
            try:
                beat_sheet = build_beat_sheet(model_client, label_v2, plot_graph)
                # 始终保存，方便排查与复现（不依赖 --trace-sop）
                _save_json(beat_sheet_path, beat_sheet if isinstance(beat_sheet, dict) else {"beat_sheet": beat_sheet})
                logger.info(f"✅ BeatSheet 已保存: {beat_sheet_path}")
            except Exception as e:
                logger.warning(f"BeatSheet 生成失败（继续运行）: {e}")
                beat_sheet = None
        
        # 步骤 2: 构建系统提示词（不再通过 RAG 注入案例）
        logger.info("\n" + "="*60)
        logger.info("步骤 2: 构建系统提示词（不使用 RAG 示例，仅使用方法论文档）")
        logger.info("="*60 + "\n")

        # 直接读取系统提示词 markdown（爆款解说文案创作技能文档）
        try:
            with open(args.system_prompt, 'r', encoding='utf-8') as f:
                enhanced_system_prompt = f.read()
            logger.info(f"系统提示词读取成功，长度: {len(enhanced_system_prompt)} 字符")
        except Exception as e:
            logger.error(f"读取系统提示词失败: {e}")
            raise

        # 重要：不要把 PlotGraph/BeatSheet 这类“实例数据”塞进 system prompt，避免串戏与注意力稀释。
        # rules 模式下，PlotGraph/BeatSheet 会作为“本次任务输入”拼到 user message（见后续 user_prompt 构建）。

        # 添加 SOP1 时序数字序号要求（防止幻觉）+ 禁止直接台词的输出规范
        enhanced_system_prompt = (
            enhanced_system_prompt
            + "\n\n"
            + "【最终输出格式硬约束｜必须遵守】\n"
            + "你必须输出且只输出以下固定结构，缺少任一部分都视为失败：\n"
            + "## SOP 1\n"
            + "## SOP 2\n"
            + "## SOP 3\n"
            + "## SOP 4\n"
            + "## SOP 5\n"
            + "## 最终解说文案\n"
            + "要求：\n"
            + "1. “最终解说文案”必须是可直接用于TTS的连续中文旁白正文。\n"
            + "2. 禁止在“最终解说文案”段中输出SOP、教程说明、免责声明、提纲编号列表。\n"
            + "3. 即使前文包含SOP过程，也必须单独完整给出“最终解说文案”段。\n"
            + "【/最终输出格式硬约束】\n"
            + "\n"
            + "【重要格式要求】\n"
            + "SOP1 的时序列必须使用数字序号（1、2、3、4...），严禁使用'开篇'、'发展'、'高潮'、'结尾'等文字描述。\n"
            + "正确示例：| 1 | 角色A | 地点X | ... |\n"
            + "错误示例：| 开篇 | 角色A | 地点X | ... |（禁止）\n"
            + "【/重要格式要求】\n"
            + "\n"
            + "【对话禁止规则｜只允许旁白转述】\n"
            + "1. 最终解说文案中【禁止】出现任何人物直接说出的台词/对话，不要输出带引号的句子，也不要输出“某人说：”后面的原话。\n"
            + "2. 具体禁止形式示例（都不要出现在最终文案里）：\n"
            + "   - \"你这么不想我走，该不会是假戏真做，爱上我了吧？\"\n"
            + "   - 「你给我滚出去」\n"
            + "   - 男人说：\"……\" / 女人骂道：\"……\" / 他冷笑着说：\"……\" 等。\n"
            + "3. 如果需要表达说话内容，一律改用第三人称旁白转述，例如：\n"
            + "   - 把“女人冷笑着说：\"你给我滚\"”改写为“女人冷笑着，让他滚出自己的世界”。\n"
            + "   - 把“男人大喊：\"这是我的孩子\"”改写为“男人当众宣称那是自己的孩子”。\n"
            + "4. 总原则：\n"
            + "   - 解说=旁白叙事，不是角色对话脚本。\n"
            + "   - 任何台词内容都必须被你消化成叙事性描述，而不是原封不动地写出来。\n"
            + "【/对话禁止规则｜只允许旁白转述】\n"
        )
        
        # 调试：强制每轮输出包含 SOP1-5 + 最终文案，方便追踪"每一步生成了什么、怎么改的"
        if args.trace_sop:
            enhanced_system_prompt = (
                enhanced_system_prompt
                + "\n\n"
                + "【调试模式｜必须输出完整过程】\n"
                + "本次为调试追踪，你每一轮输出都必须包含：SOP 1-5 的完整产出 + 最终解说文案。\n"
                + "严禁只输出'优化后的解说文案'而省略 SOP1-5。\n"
                + "输出顺序固定：SOP1 → SOP2 → SOP3 → SOP4 → SOP5 → 最终解说文案。\n"
                + "【/调试模式】\n"
            )
        
        # 【保存中间结果】：增强后的完整 system prompt
        enhanced_prompt_path = output_dir / f"{base_filename}_enhanced_system_prompt.md"
        with open(enhanced_prompt_path, 'w', encoding='utf-8') as f:
            f.write(enhanced_system_prompt)
        logger.info(f"✅ 增强后的 System Prompt 已保存: {enhanced_prompt_path}")
        
        # 步骤 5: 生成解说
        logger.info("\n" + "="*60)
        logger.info("步骤 5: 生成解说")
        logger.info("="*60 + "\n")
        
        # 构建 user prompt（使用 fewshot1231.py 的逻辑）
        input_data = {
            "dj": args.dj,
            "start": args.start,
            "end": args.end
        }
        
        user_prompt = temp_skill.get_user_prompt(input_data, additional_context=drama_info)

        # rules 模式：将 PlotGraph/BeatSheet 作为“实例输入块”放到 user message，避免污染 system prompt
        if args.rag_mode == "rules":
            instance_blocks = format_instance_blocks(plot_graph, beat_sheet)
            if instance_blocks:
                user_prompt = (
                    user_prompt.rstrip()
                    + "\n\n"
                    + "【实例输入｜本次任务专属】\n"
                    + "说明：下面的 PlotGraph/BeatSheet 是本次任务的实例数据，不要迁移到下一次任务。\n"
                    + "优先级：事实边界以 PlotGraph 为准；BeatSheet 只约束信息预算/节奏，不得覆盖 PlotGraph 的事实边界。\n"
                    + "【/实例输入｜本次任务专属】\n"
                    + "\n"
                    + instance_blocks
                    + "\n"
                )
        
        # 生成解说
        trace_dir = None
        trace_prefix = base_filename
        if args.trace_sop:
            trace_dir = output_dir / f"{base_filename}_trace"

        final_commentary, conversation_history = generate_commentary_with_enhanced_prompt(
            model_client=model_client,
            eval_skill=eval_skill,
            enhanced_system_prompt=enhanced_system_prompt,
            user_prompt=user_prompt,
            dj=args.dj,
            max_iterations=args.max_iterations,
            trace_dir=trace_dir,
            trace_prefix=trace_prefix,
            gate_enabled=False,
            plot_graph=plot_graph,
            beat_sheet=beat_sheet,
        )

        if not (final_commentary or "").strip():
            raise RuntimeError("解说正文生成失败：连续 5 次重试仍未得到有效正文，流程终止")
        
        # 【新增】步骤 6: 去幻觉检查与修补
        if plot_graph is not None and not plot_graph.get("error"):
            logger.info("\n" + "="*60)
            logger.info("步骤 6: 去幻觉检查与修补")
            logger.info("="*60 + "\n")
            
            # 1. 提取最终解说文案
            logger.info("提取最终解说文案...")
            original_commentary = extract_final_commentary(final_commentary)
            if not original_commentary.strip():
                raise RuntimeError("去幻前最终解说文案提取失败（缺少“最终解说文案”段），流程终止")
            logger.info(f"✅ 提取完成，长度: {len(original_commentary)} 字符")
            
            # 2. 提取 PlotGraph 关键字段
            logger.info("提取 PlotGraph 关键字段...")
            plotgraph_facts = extract_plotgraph_facts(plot_graph)
            logger.info(f"✅ 提取完成：{len(plotgraph_facts['characters'])} 个角色，{len(plotgraph_facts['events'])} 个事件，{len(plotgraph_facts['unknowns'])} 个未知点")
            
            # 3. 检查冲突
            logger.info("检查解说中的冲突...")
            conflict_report = check_hallucination_conflicts(
                model_client=model_client,
                final_commentary=original_commentary,
                plotgraph_facts=plotgraph_facts,
                temperature=0.1,
            )
            
            # 4. 判断是否需要修补
            has_conflicts = conflict_report.get("has_conflicts", False)
            issues = conflict_report.get("issues", [])
            
            if has_conflicts and issues:
                logger.info(f"⚠️  检测到 {len(issues)} 处冲突，开始修补")
                
                # 5. 局部修补
                patched_commentary = patch_commentary_local(
                    model_client=model_client,
                    final_commentary=original_commentary,
                    conflict_report=conflict_report,
                    plotgraph_facts=plotgraph_facts,
                    temperature=0.2,
                )
                logger.info(f"✅ 修补完成，长度: {len(patched_commentary)} 字符")
                
                # 6. 组装最终输出
                final_commentary = format_output_with_dehallucination(
                    original_response=final_commentary,
                    original_commentary=original_commentary,
                    patched_commentary=patched_commentary,
                    conflict_report=conflict_report,
                )
                
                # 保存冲突报告和修补结果
                if args.trace_sop:
                    _save_json(output_dir / f"{base_filename}_conflict_report.json", conflict_report)
                    with open(output_dir / f"{base_filename}_patched_commentary.md", "w", encoding="utf-8") as f:
                        f.write(patched_commentary)
            else:
                logger.info("✅ 未检测到冲突，使用原始解说")
                # 即使没有冲突，也格式化输出（标注"已检查，无冲突"）
                final_commentary = format_output_with_dehallucination(
                    original_response=final_commentary,
                    original_commentary=original_commentary,
                    patched_commentary=None,
                    conflict_report=conflict_report,
                )
                
                # 保存冲突报告
                if args.trace_sop:
                    _save_json(output_dir / f"{base_filename}_conflict_report.json", conflict_report)
        else:
            logger.info("PlotGraph 缺失或无效，跳过去幻觉流程")
        
        # 保存最终结果
        logger.info("\n" + "="*60)
        logger.info("保存最终结果")
        logger.info("="*60 + "\n")
        
        # 【保存最终结果】：按照 skill_video_commentary_text.py 的格式
        # 1. 保存完整 JSON（包含所有元信息）
        result = {
            "raw_response": final_commentary,
            "status": "success",
            "input_data": input_data,
            "additional_context": drama_info,
            "conversation_history": conversation_history,
            # 保存约束实例数据，便于回溯（可能为 None）
            "plot_graph": plot_graph,
            "beat_sheet": beat_sheet,
            "plot_graph_path": str(plot_graph_path) if plot_graph_path is not None else "",
            "beat_sheet_path": str(beat_sheet_path) if beat_sheet_path is not None else "",
            "model_input": {
                "system_prompt_length": len(enhanced_system_prompt),
                "enhanced_system_prompt_path": str(enhanced_prompt_path)
            },
            "rag_info": {
                "label": label,
                "label_raw_response": label_raw,
                "advanced_labels": advanced_labels,
                "enabled": False
            },
            "constraint_info": {
                "plot_graph_enabled": bool(args.enable_plotgraph or (args.rag_mode == "rules")),
                "beat_sheet_enabled": bool((args.enable_beatsheet or (args.rag_mode == "rules")) and plot_graph is not None),
                "plot_graph_present": plot_graph is not None,
                "beat_sheet_present": beat_sheet is not None,
            }
        }
        
        json_path = output_dir / f"{base_filename}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ 完整 JSON 已保存: {json_path}")
        
        # 2. 保存 raw_response 为 Markdown（和 skill_video_commentary_text.py 一致）
        md_path = output_dir / f"{base_filename}.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(final_commentary)
        
        logger.info(f"✅ 解说文案 Markdown 已保存: {md_path}")
        
        logger.info(f"""
{'='*60}
🎉 全部完成！
{'='*60}

结果摘要:
  - 标签: {label}
  - RAG 示例: {args.num_examples} 个
  - 解说长度: {len(final_commentary)} 字符
  - 输出目录: {output_dir}

生成的文件:
  【中间结果】
  - 增强后的 System Prompt: {enhanced_prompt_path}
  
  【最终结果】
  - 完整 JSON: {json_path}
  - 解说文案 Markdown: {md_path}
        """)
        
        return 0
        
    except Exception as e:
        logger.error(f"\n❌ 执行失败: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
