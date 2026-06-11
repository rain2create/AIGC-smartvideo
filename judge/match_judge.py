#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM-as-Judge 单条评估核心模块

评估一条 (文案, 候选视频片段) 的匹配质量，输出 5 维度评分 + 详细推理。

用法:
    # 快速测试单条评估
    python judge/match_judge.py \
        --dj weinishiguit4 \
        --segment-id 417 \
        --script-text "结巴女孩被当众逼着吃剩蛋糕"

    # 启用一致性检查（annotation 模式）
    python judge/match_judge.py \
        --dj weinishiguit4 \
        --segment-id 417 \
        --script-text "..." \
        --consistency
"""

import os
import re
import json
import base64
import logging
import time
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict

import requests

logger = logging.getLogger(__name__)

# ─── 维度权重（与 judge.md 第二节一致）───
DIMENSION_WEIGHTS = {
    "entity_match": 0.20,
    "event_match": 0.30,
    "scene_match": 0.15,
    "tone_match": 0.25,
    "editing_suitability": 0.10,
}
VALID_DIMENSIONS = list(DIMENSION_WEIGHTS.keys())


@dataclass
class JudgeInput:
    """Judge 输入结构"""
    # 文案侧
    script_text: str
    visual_query: Dict[str, Any]
    episode_target: int

    # 候选视频侧
    segment_id: int
    episode: int
    video_segment_path: str
    structured_description: Dict[str, Any]
    duration: float

    # 可选上下文
    prev_script_text: Optional[str] = None
    prev_segment_id: Optional[int] = None
    audio_duration: Optional[float] = None


@dataclass
class JudgeOutput:
    """Judge 输出结构"""
    evaluation_id: str
    query: Dict[str, Any]
    candidate: Dict[str, Any]
    scores: Dict[str, Any]
    reasoning: Dict[str, str]
    match_highlights: List[str]
    match_issues: List[str]
    confidence: float
    would_use: bool
    metadata: Dict[str, Any]


class MatchJudge:
    """Judge 核心评估器"""

    DEFAULT_API_URL = "https://api.360.cn/v1/chat/completions"
    DEFAULT_API_KEY = "fk1379989049.aEmK59LGEcoiwxat0pGE18JPvtSBkrybaf481994"

    def __init__(
        self,
        api_url: str = None,
        api_key: str = None,
        model: str = "volcengine/doubao-seed-2-0-lite",
        temperature: float = 0.0,
        timeout: int = 300,
    ):
        """
        Args:
            api_url: 多模态 API 地址
            api_key: API Key
            model: 模型名称（eval 模式用 flash，annotation 模式用 pro）
            temperature: 采样温度，建议 0 以保证一致性
            timeout: API 超时秒数
        """
        self.api_url = api_url or self.DEFAULT_API_URL
        self.api_key = api_key or self.DEFAULT_API_KEY
        self.model = model
        self.temperature = temperature
        self.timeout = timeout

        self._system_prompt = self._build_system_prompt()

    # ─── Prompt 构建 ───

    @staticmethod
    def _build_system_prompt() -> str:
        return """你是一位拥有 15 年经验的资深影视剪辑师和内容审核专家。
你的核心能力是判断"解说文案"与"视频画面"的匹配质量。

你将收到：
- 一段解说文案及其结构化视觉需求（visual_query）
- 一个候选视频片段（你将直接观看视频画面）
- 视频片段的结构化文本描述（辅助参考）

你的任务是：直接观看视频片段，从"观众同时听到解说+看到画面"的视角，
按 5 个维度独立评估匹配质量，给出 1-5 分和详细推理。

评估原则：
- 优先基于视频画面进行判断，文本描述作为辅助参考
- 如果视频画面与文本描述不一致，以视频画面为准
- reasoning 中要引用具体画面特征（"视频中可以看到..."）

【五个评估维度】

维度1 - entity_match（实体匹配）：
画面里有谁？人物类型和数量是否与文案需求对齐？
只关注"谁在画面里"，不判断动作、环境或情绪。

维度2 - event_match（事件匹配）：
画面里发生了什么？动作/事件是否与文案描述一致？
只关注"发生了什么事"，不判断人物类型、场景环境或情绪氛围。

维度3 - scene_match（场景匹配）：
在什么环境下？场景类型、空间、光线、时间/天气是否匹配？
只关注"在什么场景下"，不判断谁在画面里、发生了什么或感觉如何。

维度4 - tone_match（基调匹配）：
画面的情绪感觉对不对？情绪强度、互动关系、氛围是否一致？
只关注"感觉是什么"，不重复评判人物、动作或环境的具体匹配。

维度5 - editing_suitability（剪辑适用性）：
整体好不好用？从剪辑实操角度判断：
- 节奏是否匹配（画面节奏 vs 解说节奏）
- 信息是否互补（画面补充文案，而非重复文案）
- 观众注意力（搭配能否抓住观众）
- 实操可用性（时长是否够、镜头质量是否可用）
注意：这个维度不重复前四个维度的判断，而是从"最终成片效果"的角度给出实操层面的评估。

【评估要求】
1. 必须先对每个维度独立分析推理，再给出分数
2. 每个维度的推理必须≥50字，具体指出哪些元素匹配、哪些不匹配、为什么
3. 各维度独立评分：不要因为事件匹配好就给情绪加分，也不要因为场景不对就给事件扣分
4. 分数严格按评分标准给出，不能"差不多就给4分"
5. 最终给出综合推理总结（≥100字）

【通用评分标准】
1分 = 完全不匹配：该维度描述的内容与候选毫无关系
2分 = 严重偏离：仅有一个次要元素勉强沾边
3分 = 部分匹配：主要元素有匹配但存在明显缺失或错位
4分 = 较好匹配：核心内容一致，仅有微小偏差
5分 = 完美匹配：精准吻合，无任何违和感

【综合分计算】
你只需要输出五个维度的 1-5 分，不要输出 overall 分数。
程序会按以下权重计算 overall：
- entity_match（实体匹配）：20%
- event_match（事件匹配）：30%
- scene_match（场景匹配）：15%
- tone_match（基调匹配）：25%
- editing_suitability（剪辑适用性）：10%
评分示例中的 overall 仅为按上述权重计算后的参考值，用于帮助校准分数档位。

请严格以 JSON 格式输出，不要包含任何其他文字。

【评分示例】

示例1 - 完美匹配（overall = 5.0）：
解说文案："假名媛不仅霸占了结巴校花的情书和礼物，甚至抢走玉佩，顶替了首富孙女的身份。"
视频画面：两名年轻女子因一块玉佩产生激烈争执。一名穿珍珠白裙、戴蝴蝶结的女孩展示玉佩并强硬对峙；另一名穿校服风马甲的女孩捂住胸口，结巴地说玉佩是父母遗物，随后被对方抓住手腕指责。
评分：entity_match=5（两名女性分别对应假名媛和结巴校花，人物类型和数量完全对齐），event_match=5（玉佩展示、争执、抓手腕、指责都直接对应抢走玉佩和身份顶替），scene_match=5（明亮室内公共空间符合校园/高档场所的对峙场景），tone_match=5（嚣张压迫与委屈结巴的强弱对比非常明确），editing_suitability=5（玉佩特写、结巴台词和抓手腕动作都能增强文案信息）
overall_reasoning: 这是全维度高度匹配的标准高分样例。画面中的人物设定、核心道具、抢夺动作、身份压迫和情绪张力都精准支撑文案，且画面信息不是简单重复文案，而是用玉佩特写、手腕拉扯和结巴对话强化了观众理解。作为剪辑素材可以直接使用。confidence=0.95, would_use=true

示例2 - 高分但场景不完美（overall ≈ 4.4）：
解说文案："这不是我30年前生活的小渔村吗。"
视频画面：一名穿白背心的年轻男子在 80 年代风格的简陋室内从床上惊醒，猛地坐起并环顾四周，看到墙上的 1985 年历书后露出极度震惊的表情。
评分：entity_match=5（单一年轻男性主角完全匹配），event_match=4（惊醒、环顾、震惊高度符合发现回到过去的反应，但没有室外眺望），scene_match=3（画面是复古室内，不是海边/渔村；但 1985 年历书和简陋陈设强力支撑"30年前"），tone_match=5（意外、疑惑、难以置信的情绪完全对齐），editing_suitability=5（历书把"30年前"视觉化，画面给文案提供增量信息）
overall_reasoning: 这类样例不能因为局部场景偏差而一票否决。虽然画面没有直接展示海边小渔村，但男主震惊反应、复古室内陈设和 1985 年历书共同支撑了"回到30年前"的核心叙事。画面与文案形成信息互补，适合作为重生开场素材，因此整体应给高分。confidence=0.95, would_use=true

示例3 - 中分边界（overall ≈ 3.0）：
解说文案："起初，丈夫林涛根本不信，甚至当众发誓，就算老婆怀了个妖怪他也认。"
视频画面：白天农村院落中，多名长辈情绪激动地驱赶一对年轻男女；年轻男子愤怒欲反击，被身旁怀孕女子拉住，女子神情委屈。
评分：entity_match=5（丈夫、怀孕妻子、多名围观长辈都出现，人物配置很准），event_match=2（画面是被驱赶和对峙，没有发誓、举手或坚定表态），scene_match=4（农村院落和多人围观符合"当众"），tone_match=2（画面是烦躁、不忿、被驱赶，不是坚定护妻的信念感），editing_suitability=3（孕妇大肚子可辅助文案，但缺少"发誓"主动作，直接搭配会割裂）
overall_reasoning: 这是典型的中分边界样例。人物和公开场景都很像，但文案最关键的信息是"丈夫当众发誓、坚定认下妻子和孩子"，候选画面没有给出这个核心动作和情绪，只是冲突驱赶场面。可作为辅助镜头裁剪使用，但不能作为这句文案的主匹配，因此 overall 约 3 分且 would_use=false。confidence=0.90, would_use=false

示例4 - Hard negative：实体对但事件/情绪错（overall ≈ 1.8）：
解说文案："就在他迷茫之际，一个身影出现在眼前，是江清雪，好像是她，真的是她。"
视频画面：室内暖色调环境中，一名年轻男子和一名年轻女子深情对视、紧握双手。男子许下承诺，女子温柔微笑并抚摸他的颈部，两人处于非常亲密的互动状态。
评分：entity_match=5（画面确实有一男一女，基础人物配置完全对），event_match=1（文案是发呆、突然发现、难以置信；画面是牵手告白和亲密互动），scene_match=1（文案需要空旷/光影模糊的出现感，画面是生活化室内），tone_match=1（文案是期待、意外、震惊，画面是温馨、深情、关系确定），editing_suitability=1（解说说"真的是她"时画面已经在告白牵手，会让观众出戏）
overall_reasoning: 这是需要特别警惕的 hard negative。它容易骗过粗召回，因为人物实体完全对齐；但剪辑上完全不可用。文案讲的是带悬念的"第一次看见她出现"，候选画面已经进入亲密告白阶段，事件时点、互动关系和情绪方向全部错位。实体匹配不能抵消事件和情绪的严重错误。confidence=0.95, would_use=false

示例5 - 完全错配（overall = 1.0）：
解说文案："为了报答曾孙三年的悉心照料，老人随手掏出一张泛黄的房契。"
视频画面：一名穿粉色丝绸睡袍的年轻女子在豪华室内打电话，语气讥讽地告诉对方亲王府已经改成人民公园；镜头穿插室外公园空镜和一名年轻男子的错愕表情。
评分：entity_match=1（文案需要老人和曾孙，画面是年轻女子和年轻男子，核心老人缺失），event_match=1（文案是掏出房契，画面是打电话讥讽和公园空镜），scene_match=1（文案偏居家/温馨/陈旧，画面是豪华室内和室外公园），tone_match=1（文案是报恩和惊喜，画面是嘲讽、打压和错愕），editing_suitability=1（原视频台词会强烈干扰解说，观众会完全困惑）
overall_reasoning: 这是全维度完全错配的标准低分样例。人物、动作、场景、情绪和视频自带台词都与文案冲突，没有任何可用的核心元素。即使某些词如"房契/亲王府"在语义上有一点关联，画面本身仍无法支撑"老人报恩掏出房契"的视听逻辑，必须给极低分。confidence=0.95, would_use=false"""

    def _build_user_prompt(self, judge_input: JudgeInput) -> str:
        vq = judge_input.visual_query
        sd = judge_input.structured_description
        fuzzy = vq.get("fuzzy_tags", {}) or {}

        # ── 文案部分 ──
        script_section = f"""【解说文案】
{judge_input.script_text}

【文案视觉需求（visual_query）】
- 人物姓名：{', '.join(vq.get('characters', []) or []) or '未指定'}
- 人物类型：{', '.join(vq.get('character_types', []) or []) or '未指定'}
- 动作关键词：{', '.join(vq.get('action_keywords', []) or []) or '未指定'}
- 环境：{vq.get('environment', '未指定')}
- 视觉补充：{vq.get('expanded_tags', '未指定')}
- 互动关系：{', '.join(vq.get('interaction_type', []) or []) or '未指定'}
- 观众心理：{', '.join(fuzzy.get('audience_state', []) or []) or '未指定'}
- 情绪强度：{fuzzy.get('emotion_intensity', '未指定')}
- 导演意图：{', '.join(fuzzy.get('director_intent', []) or []) or '未指定'}"""

        # ── 视频描述部分 ──
        desc_section = f"""【视频片段文本描述】（辅助参考，如与视频画面不一致以画面为准）
- 画面概括：{sd.get('visual_summary', '无')}
- 出场角色：{', '.join(sd.get('characters', []) or []) or '无'}
- 主要动作：{', '.join(sd.get('actions', []) or []) or '无'}
- 环境：{sd.get('environment', '无')}
- 互动关系：{', '.join(sd.get('interaction_type', []) or []) or '无'}
- 画面功能：{', '.join(sd.get('functional_intent', []) or []) or '无'}
- 情绪基调：{sd.get('emotion_intensity', '无')}
- 镜头运动：{sd.get('camera_movement', '无')}
- 对白/字幕：{sd.get('dialogue_or_text', '无')}"""

        # ── 上下文（可选）──
        context_section = ""
        if judge_input.prev_script_text:
            context_section = f"""
【上下文】
上一句文案：{judge_input.prev_script_text}
上一句匹配的片段ID：{judge_input.prev_segment_id}
当前文案音频时长：{judge_input.audio_duration}秒"""

        # ── JSON 输出模板 ──
        json_template = """请按以下 JSON 格式严格输出：
注意：scores 中只输出五个维度分数，不要添加 overall；overall 会由程序按权重计算。

{
  "analysis": {
    "entity_match": "分析画面中有谁，人物类型和数量是否与文案需求对齐...",
    "event_match": "分析画面中发生了什么，动作/事件是否与文案描述一致...",
    "scene_match": "分析场景环境，空间/光线/时间是否与文案描述匹配...",
    "tone_match": "分析情绪氛围，情绪强度/互动关系/导演意图是否一致...",
    "editing_suitability": "从剪辑实操角度分析：节奏是否匹配、信息是否互补、能否抓住观众..."
  },
  "scores": {
    "entity_match": 1-5,
    "event_match": 1-5,
    "scene_match": 1-5,
    "tone_match": 1-5,
    "editing_suitability": 1-5
  },
  "match_highlights": ["具体匹配好的点1", "具体匹配好的点2"],
  "match_issues": ["具体不匹配的点1", "具体不匹配的点2"],
  "overall_reasoning": "综合总结：整体匹配质量如何，哪些维度强、哪些弱，作为剪辑师你会怎么选择（≥100字）",
  "confidence": 0.0-1.0,
  "would_use": true或false
}"""

        ep_label = f"第 {judge_input.episode} 集" if judge_input.episode else "未知集数"
        header = f"【候选视频片段】（{ep_label}，片段ID {judge_input.segment_id}，时长 {judge_input.duration:.1f}秒）\n以下为视频帧序列："

        return f"""{script_section}

{header}

{desc_section}{context_section}

{json_template}"""

    # ─── 视频抽帧 ───

    FRAME_FPS = 1.0  # 每秒抽 1 帧

    @staticmethod
    def _extract_frames(video_path: str) -> List[str]:
        """将视频按 fps=1 抽帧，返回 base64 JPEG data URI 列表"""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")

        try:
            import cv2
        except ImportError:
            raise ImportError("opencv-python 未安装，无法抽帧")

        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / max(video_fps, 1.0)
        interval = max(1, int(video_fps / MatchJudge.FRAME_FPS))

        logger.info(
            f"抽帧: {video_path} ({total_frames}frames {duration:.1f}s fps={video_fps:.1f} interval={interval})"
        )

        frame_uris = []
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % interval == 0:
                _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                b64 = base64.b64encode(buf).decode("utf-8")
                frame_uris.append(f"data:image/jpeg;base64,{b64}")
            idx += 1
        cap.release()

        logger.info(f"抽帧完成: {len(frame_uris)} 帧")
        return frame_uris

    # ─── API 调用 ───

    def _build_api_payload(
        self,
        system_prompt: str,
        user_prompt: str,
        frame_uris: List[str],
    ) -> Dict[str, Any]:
        """构建多模态 API payload：system text + user (text + images)"""
        user_content = [{"type": "text", "text": user_prompt}]
        for uri in frame_uris:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": uri},
            })

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            "temperature": self.temperature,
        }
        return payload

    def _call_api(self, payload: Dict[str, Any], max_retries: int = 3) -> str:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        last_error = None
        for attempt in range(max_retries):
            try:
                resp = requests.post(
                    self.api_url, json=payload, headers=headers, timeout=self.timeout
                )
                if resp.status_code == 200:
                    result = resp.json()
                    content = result["choices"][0]["message"]["content"]
                    logger.info(f"API 成功 (attempt {attempt + 1})")
                    return content

                if resp.status_code == 429:
                    delay = 5 * (attempt + 1)
                    logger.warning(f"429 限流，{delay}s 后重试...")
                    time.sleep(delay)
                    continue

                last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                logger.warning(f"API 失败 (attempt {attempt + 1}): {last_error}")
                if 400 <= resp.status_code < 500:
                    break

            except requests.exceptions.Timeout:
                last_error = f"超时 ({self.timeout}s)"
                logger.warning(f"API 超时 (attempt {attempt + 1})")
            except requests.exceptions.RequestException as e:
                last_error = str(e)
                logger.warning(f"API 异常 (attempt {attempt + 1}): {e}")

            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

        raise RuntimeError(f"API 调用失败 ({max_retries} 次): {last_error}")

    # ─── 输出解析 ───

    @staticmethod
    def _parse_response(raw_text: str) -> Dict[str, Any]:
        """从模型原始输出中提取 JSON"""
        cleaned = raw_text.strip()

        # 移除 thinking / analysis 标签
        cleaned = re.sub(r'<thinking>.*?</thinking>', '', cleaned, flags=re.DOTALL)
        cleaned = re.sub(r'<analysis>.*?</analysis>', '', cleaned, flags=re.DOTALL)

        # 移除 markdown 代码块
        for prefix in ['```json', '```']:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):]
        if cleaned.endswith('```'):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        # 提取 JSON（贪婪匹配完整 JSON 对象）
        match = re.search(r'\{[\s\S]*\}', cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        raise ValueError(f"无法解析 JSON，原始输出: {raw_text[:500]}")

    def _validate_and_build_output(
        self,
        parsed: Dict[str, Any],
        judge_input: JudgeInput,
        call_metadata: Dict[str, Any],
    ) -> JudgeOutput:
        """校验 5 维度分数，计算 overall，构建 JudgeOutput"""
        scores_raw = parsed.get("scores", {})
        analysis = parsed.get("analysis", {})

        # 校验 & clamp 每个维度
        validated = {}
        for dim in VALID_DIMENSIONS:
            val = scores_raw.get(dim)
            if val is None:
                raise ValueError(f"维度 {dim} 缺失，LLM 未输出该维度分数")
            score = int(val)
            if score < 1 or score > 5:
                raise ValueError(f"维度 {dim} 分数超出范围: {score}，应为 1-5")
            validated[dim] = score

        # 加权总分
        overall = sum(validated[d] * DIMENSION_WEIGHTS[d] for d in VALID_DIMENSIONS)
        validated["overall"] = round(overall, 2)

        # reasoning
        reasoning = {}
        for dim in VALID_DIMENSIONS:
            reasoning[dim] = analysis.get(dim, "未提供推理")
        reasoning["overall"] = parsed.get("overall_reasoning", "未提供综合推理")

        confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.5))))
        would_use = bool(parsed.get("would_use", False))

        evaluation_id = (
            f"{call_metadata.get('dj', 'unknown')}_"
            f"seg{judge_input.episode_target}_"
            f"cand{judge_input.segment_id}"
        )

        return JudgeOutput(
            evaluation_id=evaluation_id,
            query={
                "script_text": judge_input.script_text,
                "episode_target": judge_input.episode_target,
            },
            candidate={
                "segment_id": judge_input.segment_id,
                "episode": judge_input.episode,
            },
            scores=validated,
            reasoning=reasoning,
            match_highlights=parsed.get("match_highlights", []),
            match_issues=parsed.get("match_issues", []),
            confidence=confidence,
            would_use=would_use,
            metadata={
                "judge_model": self.model,
                "judge_version": "v2",
                "video_segment_path": judge_input.video_segment_path,
                "temperature": self.temperature,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "run_id": call_metadata.get("run_id", ""),
            },
        )

    # ─── 公开接口 ───

    def evaluate_single(
        self,
        judge_input: JudgeInput,
        dj: str = "",
        run_id: str = "",
    ) -> JudgeOutput:
        """
        单条评估（一次调用，无一致性检查）

        Args:
            judge_input: 评估输入
            dj: 剧名（写入 metadata）
            run_id: 批次 ID

        Returns:
            JudgeOutput
        """
        logger.info(
            f"评估: seg_seq={judge_input.episode_target}, "
            f"candidate_id={judge_input.segment_id}"
        )

        # 1. 抽帧
        frame_uris = self._extract_frames(judge_input.video_segment_path)

        # 2. 构建 prompt + payload
        user_prompt = self._build_user_prompt(judge_input)
        payload = self._build_api_payload(self._system_prompt, user_prompt, frame_uris)

        # 3. 调用 API + 解析（含格式校验重试）
        max_parse_retries = 5
        for attempt in range(1, max_parse_retries + 1):
            raw_text = self._call_api(payload)
            try:
                parsed = self._parse_response(raw_text)
                return self._validate_and_build_output(
                    parsed, judge_input, {"dj": dj, "run_id": run_id}
                )
            except (ValueError, KeyError, TypeError) as e:
                logger.warning(
                    f"输出格式错误（尝试 {attempt}/{max_parse_retries}）: {e}"
                )
                if attempt < max_parse_retries:
                    continue
                raise RuntimeError(
                    f"Judge 评估失败（{max_parse_retries} 次格式校验不通过）: {e}"
                ) from e

    def evaluate_with_consistency(
        self,
        judge_input: JudgeInput,
        n_calls: int = 3,
        dj: str = "",
        run_id: str = "",
    ) -> Tuple[JudgeOutput, Dict[str, Any]]:
        """
        带一致性检查的评估（annotation 模式）

        调用 n_calls 次，检查 overall 分数标准差：
        - std <= 0.5: 一致，取中位数结果
        - std > 0.5: 追加 2 次调用
        - 5 次仍不一致: 标记 low_consistency

        Returns:
            (JudgeOutput, consistency_info)
        """
        results: List[JudgeOutput] = []

        for i in range(n_calls):
            out = self.evaluate_single(judge_input, dj=dj, run_id=run_id)
            results.append(out)

        overalls = [r.scores["overall"] for r in results]
        std = statistics.stdev(overalls) if len(overalls) > 1 else 0.0

        consistency_info = {
            "n_calls": n_calls,
            "overall_scores": overalls,
            "std": round(std, 3),
            "status": "consistent",
        }

        if std > 0.5:
            logger.warning(f"一致性差 (std={std:.2f})，追加 2 次调用")
            for _ in range(2):
                out = self.evaluate_single(judge_input, dj=dj, run_id=run_id)
                results.append(out)

            overalls = [r.scores["overall"] for r in results]
            std = statistics.stdev(overalls) if len(overalls) > 1 else 0.0
            consistency_info.update({
                "n_calls": len(results),
                "overall_scores": overalls,
                "std": round(std, 3),
            })

            if std > 0.5:
                consistency_info["status"] = "low_consistency"
                logger.warning(f"5 次调用仍不一致 (std={std:.2f})")
            else:
                consistency_info["status"] = "consistent_after_extra"

        # 取中位数那次的结果
        median_idx = _median_index(overalls)
        final_output = results[median_idx]
        final_output.metadata["consistency"] = consistency_info

        return final_output, consistency_info


# ─── 工具函数 ───

def _median_index(values: List[float]) -> int:
    """找到中位数对应的原始索引"""
    sorted_pairs = sorted(enumerate(values), key=lambda x: x[1])
    mid = len(sorted_pairs) // 2
    return sorted_pairs[mid][0]


def resolve_video_path(
    segment_id: int,
    dj: str,
    cache_dir: str = None,
) -> Optional[str]:
    """
    根据 segment_id 解析视频片段实际路径

    查找策略：
    1. 从 structured_description.json 的 segment_file_path 查找
       （可能指向旧路径，自动修正）
    2. 在 segment/ 目录下按文件名前缀匹配

    Returns:
        视频文件的绝对路径，找不到返回 None
    """
    base_dir = Path(cache_dir) if cache_dir else (
        Path(__file__).resolve().parent.parent / "text_video_match" / "cache" / dj
    )

    # 策略 1: structured_description.json
    sd_path = base_dir / "structured_description.json"
    if sd_path.exists():
        try:
            data = json.loads(sd_path.read_text(encoding="utf-8"))
            for seg in data.get("segments", []):
                if seg.get("segment_id") == segment_id:
                    path = seg.get("segment_file_path", "")
                    if path and os.path.exists(path):
                        return path
                    # 旧路径修正
                    if path:
                        fixed = _fix_stale_path(path, base_dir)
                        if fixed:
                            return fixed
        except Exception as e:
            logger.warning(f"读取 structured_description.json 失败: {e}")

    # 策略 2: 按文件名前缀搜索 segment/ 目录
    seg_dir = base_dir / "segment"
    if not seg_dir.exists():
        return None

    prefix = f"{segment_id:03d}"
    # 搜 ep_* 子目录（新版 indexer）
    for ep_dir in sorted(seg_dir.glob("ep_*")):
        for f in ep_dir.glob(f"{prefix}_*.mp4"):
            return str(f.resolve())
    # 搜平级目录（旧版）
    for f in seg_dir.glob(f"{prefix}_*.mp4"):
        return str(f.resolve())

    return None


def _fix_stale_path(old_path: str, base_dir: Path) -> Optional[str]:
    """修正指向旧挂载点的路径"""
    filename = Path(old_path).name

    # 先在 segment/ 下找
    candidate = base_dir / "segment" / filename
    if candidate.exists():
        return str(candidate.resolve())

    # 在 segment/ep_*/ 下找
    for ep_dir in (base_dir / "segment").glob("ep_*"):
        candidate = ep_dir / filename
        if candidate.exists():
            return str(candidate.resolve())

    return None


def load_segment_info(
    segment_id: int,
    dj: str,
    cache_dir: str = None,
) -> Optional[Dict[str, Any]]:
    """
    从 structured_description.json 加载指定 segment 的完整信息

    Returns:
        包含 segment_id, duration, structured_description, episode 等字段的字典
    """
    base_dir = Path(cache_dir) if cache_dir else (
        Path(__file__).resolve().parent.parent / "text_video_match" / "cache" / dj
    )
    sd_path = base_dir / "structured_description.json"

    if not sd_path.exists():
        return None

    try:
        data = json.loads(sd_path.read_text(encoding="utf-8"))
        for seg in data.get("segments", []):
            if seg.get("segment_id") == segment_id:
                return seg
    except Exception as e:
        logger.warning(f"读取 structured_description.json 失败: {e}")

    return None


# ─── CLI 入口 ───

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s"
    )

    parser = argparse.ArgumentParser(description="LLM-as-Judge 单条评估")
    parser.add_argument("--dj", required=True, help="剧名")
    parser.add_argument("--segment-id", type=int, required=True, help="候选视频片段 ID")
    parser.add_argument("--script-text", required=True, help="解说文案原文")
    parser.add_argument("--model", default="google/gemini-3-flash-preview", help="模型")
    parser.add_argument("--consistency", action="store_true", help="启用一致性检查")
    parser.add_argument("--cache-dir", default=None, help="缓存目录（默认自动推导）")
    args = parser.parse_args()

    # 加载 segment 信息
    seg_info = load_segment_info(args.segment_id, args.dj, args.cache_dir)
    if not seg_info:
        print(f"ERROR: 未找到 segment_id={args.segment_id}", flush=True)
        exit(1)

    # 解析视频路径
    video_path = resolve_video_path(args.segment_id, args.dj, args.cache_dir)
    if not video_path:
        print(f"ERROR: 未找到视频文件 segment_id={args.segment_id}", flush=True)
        exit(1)

    sd = seg_info["structured_description"]
    ep = seg_info.get("episode", 0)

    judge_input = JudgeInput(
        script_text=args.script_text,
        visual_query={},  # CLI 简化测试，不传 visual_query
        episode_target=ep,
        segment_id=args.segment_id,
        episode=ep,
        video_segment_path=video_path,
        structured_description=sd,
        duration=seg_info["duration"],
    )

    judge = MatchJudge(model=args.model)

    if args.consistency:
        output, info = judge.evaluate_with_consistency(judge_input, dj=args.dj)
        print(f"\n一致性信息: {json.dumps(info, ensure_ascii=False)}", flush=True)
    else:
        output = judge.evaluate_single(judge_input, dj=args.dj)

    print(json.dumps(asdict(output), ensure_ascii=False, indent=2))
