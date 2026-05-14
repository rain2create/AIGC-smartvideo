# LLM-as-Judge 评估系统设计文档

> **定位**：V4 匹配系统的质量基准（benchmark），同时为 Phase 3 Reranker 微调提供高可信度的偏好数据。
> **原则**：严格、可复现、可追溯、面向训练。

---

## 一、为什么 Judge 如此重要

Judge 承担两个完全不同但都极其关键的角色：

| 角色 | 用途 | 质量要求 |
|------|------|----------|
| **系统评估器** | 衡量当前匹配系统的匹配质量，A/B 对比，回归检测 | 聚合指标稳定、可复现 |
| **训练数据标注器** | 为 Phase 3 Reranker 的 DPO 训练构造 (query, positive, negative) 偏好对 | 单条标注必须精确、可校准、有推理链路 |

两者共用同一套评分体系和输出格式，但运行配置不同（详见第六节）。

---

## 二、评估维度设计（5 维度 × 5 分制）

### 2.1 维度设计原则：严格正交

五个维度各自回答一个**独立问题**，两两之间无交叉：

```
entity_match  = "画面里有谁"       → 纯人物维度
event_match   = "画面里发生了什么" → 纯动作/事件维度
scene_match   = "在什么环境下"     → 纯环境/空间维度
tone_match    = "感觉是什么"       → 纯情绪/氛围维度
editing_suitability = "整体好不好用" → 剪辑实操维度
```

前四个维度（entity / event / scene / tone）与当前匹配系统的权重体系直接对齐：

```
Stage2（视觉精排）: characters → entity,  actions → event,  environment → scene
Stage3（情感深排）: interaction + emotion + intent → tone
```

第五个维度 `editing_suitability` 是唯一的全局维度，评估其他四个维度无法覆盖的**剪辑实操层面**的问题。

### 2.2 五个评估维度

#### D1: entity_match（实体匹配）权重 0.20

**回答的问题**：画面里有谁？人物类型和数量对不对？

**与匹配系统的对齐**：Stage2 的 characters + character_types 维度

| 分数 | 含义 | 判定标准 |
|------|------|----------|
| 1 | 完全不匹配 | 画面中无人物，或人物类型与文案需求完全不符（文案要"男女对话"，画面是空镜/动物） |
| 2 | 严重偏离 | 人物类型有重大偏差（文案要"女人+老人"，画面是两个年轻男人）；或数量差距极大（文案要"多人"，画面只有一个人） |
| 3 | 部分匹配 | 人物类型基本对，但存在明显偏差（文案要"老人"，画面是中年人；或文案要两个人，画面只有一个人） |
| 4 | 较好匹配 | 人物类型和数量都对，仅有微小差异（文案要"3-4人的群体"，画面有2-3人；或年龄层略有偏差） |
| 5 | 完美匹配 | 人物类型、数量、大致年龄/性别特征完全吻合 |

#### D2: event_match（事件匹配）权重 0.30

**回答的问题**：画面里正在发生什么？动作/事件与文案描述是否一致？

**与匹配系统的对齐**：Stage2 的 actions 维度 + expanded_tags 维度

> 这是权重最高的维度。因为观众感知匹配质量的第一信号就是"画面里在发生的事"是否与"解说在说的事"一致。

| 分数 | 含义 | 判定标准 |
|------|------|----------|
| 1 | 完全不匹配 | 画面中的事件/动作与文案描述毫无关系（文案说"推下悬崖"，画面是坐在办公室喝茶） |
| 2 | 严重偏离 | 仅有一个次要动作勉强沾边（文案说"残忍杀害"，画面只是两个人站着说话；或核心动作方向反了） |
| 3 | 部分匹配 | 事件大类对了但具体动作有偏差（文案说"激烈争吵推搡"，画面是两人在对话但气氛平和；或动作对了但强度/烈度不够） |
| 4 | 较好匹配 | 核心动作/事件一致，仅有微小差异（文案说"推搡"，画面是"抓住衣领对峙"，动作强度略有差异；或次要动作不完全对） |
| 5 | 完美匹配 | 画面精准呈现文案描述的动作和事件，动作类型、强度、主体关系完全吻合 |

#### D3: scene_match（场景匹配）权重 0.15

**回答的问题**：在什么环境下？场景/空间/光影/时间对不对？

**与匹配系统的对齐**：Stage2 的 environment 维度

| 分数 | 含义 | 判定标准 |
|------|------|----------|
| 1 | 完全不匹配 | 场景类型完全不对（文案要"深夜街道"，画面是明亮办公室） |
| 2 | 严重偏离 | 室内/室外大类都反了，或时间/天气完全不对（文案要"雨夜户外"，画面是晴天室内） |
| 3 | 部分匹配 | 大类对了但具体环境有偏差（文案要"昏暗的地下室"，画面是普通室内；或文案要"豪华办公室"，画面是普通办公室） |
| 4 | 较好匹配 | 场景基本吻合，仅有细节差异（文案要"雨天街道"，画面是阴天街道；或空间感略有不同） |
| 5 | 完美匹配 | 场景类型、空间、光线、时间/天气完全吻合 |

#### D4: tone_match（基调匹配）权重 0.25

**回答的问题**：画面的情绪感觉对不对？情绪强度、互动关系、氛围是否一致？

**与匹配系统的对齐**：Stage3 的 interaction + emotion + intent 维度

| 分数 | 含义 | 判定标准 |
|------|------|----------|
| 1 | 完全不匹配 | 情绪方向完全相反（文案要"紧张对峙"，画面是温馨拥抱；或文案要"悲伤沉重"，画面是欢快庆祝） |
| 2 | 严重偏离 | 情绪方向大致对但强度差距极大（文案要"高压紧迫、观众不安"，画面只有轻微的日常感）；或互动关系完全不对（文案要"对峙/冲突"，画面是"无互动"） |
| 3 | 部分匹配 | 情绪类型和强度基本对，但互动关系或导演意图不匹配（文案要"激烈冲突"，画面是"平和对话"——类型对了但烈度不够）；或情绪对了但画面功能不对（文案要"制造悬疑"，画面是"信息铺垫"） |
| 4 | 较好匹配 | 情绪强度、互动关系基本一致，仅在导演意图的细微层面有差异（如文案要"引发期待"，画面是"引发好奇"，方向一致但力度略弱） |
| 5 | 完美匹配 | 情绪强度、互动关系、观众感受、导演意图完全一致，画面的"感觉"与文案高度统一 |

#### D5: editing_suitability（剪辑适用性）权重 0.10

**回答的问题**：假设你是剪辑师，把这段文案和这个画面放在一起，整体效果好不好？

**与其他维度的区别**：前四个维度评估的是"匹配不匹配"（客观对齐），D5 评估的是"好不好用"（实操判断）。D5 关注的是其他维度不覆盖的剪辑实操层面：

- **节奏适配**：画面节奏与解说节奏是否匹配（快速剪辑配快语速？慢镜头配沉重文案？）
- **信息互补**：画面是补充了文案信息还是重复了文案信息？（好的搭配应该是画面和文案互补，而非画面只是"翻译"文案）
- **观众注意力**：这个搭配能否在前3秒抓住观众？还是会让人想划走？
- **实操可用性**：画面时长够不够？镜头质量是否可用？是否需要额外处理？

| 分数 | 含义 | 判定标准 |
|------|------|----------|
| 1 | 完全不可用 | 放在一起让观众困惑或出戏，绝对不能用（画面与文案完全冲突，或者画面本身质量极差） |
| 2 | 勉强可用 | 有明显违和感，观众需要脑补才能接受。专业剪辑师不会选择这个搭配（画面节奏与文案严重不匹配，或画面信息与文案完全重复无增量的场景） |
| 3 | 可用但平庸 | 没有明显错误但缺乏吸引力。"凑合能用"但不是好的选择（画面和文案"各自为政"，没有形成合力） |
| 4 | 较好 | 画面和文案搭配协调，观众能自然接受。节奏匹配，信息互补，是专业水准的搭配 |
| 5 | 优秀 | 画面和文案形成 1+1>2 的效果。情绪、节奏、信息量完美互补，观众会被这种搭配吸引。去掉任何一个都会大幅降低效果 |

### 2.3 overall_score 计算规则

```
overall_score = 0.20 × entity_match
              + 0.30 × event_match
              + 0.15 × scene_match
              + 0.25 × tone_match
              + 0.10 × editing_suitability
```

权重设计理由：

| 维度 | 权重 | 理由 |
|------|------|------|
| event_match | 0.30 | 最高。观众判断"匹不匹配"的第一信号——画面里在发生什么。核心动作/事件对不对直接决定观众是否出戏 |
| tone_match | 0.25 | 第二高。情绪氛围是短视频的灵魂。画面感觉对了观众才"买账"，感觉不对即使内容对了也会违和 |
| entity_match | 0.20 | 中等。人物类型对不对是匹配的前提条件，类型偏差会直接导致观众困惑（比如文案说女主，画面是个老头） |
| scene_match | 0.15 | 较低。环境匹配有影响但不是决定性的。场景偏差观众能容忍（比如文案说办公室，画面是会议室），但方向不能反 |
| editing_suitability | 0.10 | 最低。实操维度的判断已部分隐含在前四个维度中，单独给低权重避免重复计算 |

> 注：这是初始权重。积累 Judge 数据后，可以通过分析各维度与人工标注的相关性来校准权重。

---

## 三、Judge 的输入输出设计

### 3.1 输入结构

Judge 直接看视频片段 + 读取结构化文本描述，两者结合评估：

```python
JudgeInput = {
    # ─── 文案侧（query）───
    "query": {
        "script_text": str,          # 解说文案原文
        "visual_query": {            # S3 生成的结构化视觉需求
            "characters": list,      # 具体姓名
            "character_types": list,  # 通用类型
            "action_keywords": list,
            "environment": str,
            "expanded_tags": str,
            "interaction_type": list,
            "fuzzy_tags": {
                "audience_state": list,
                "emotion_intensity": str,
                "director_intent": list,
            }
        },
        "episode_target": int,       # 目标集数
    },

    # ─── 视频候选侧（candidate）───
    "candidate": {
        "segment_id": int,
        "episode": int,
        "video_segment_path": str,   # S1 切好的视频片段路径（直接传给模型）
        "structured_description": {  # S1 生成的文本描述（辅助参考）
            "visual_summary": str,
            "characters": list,
            "actions": list,
            "environment": str,
            "interaction_type": list,
            "functional_intent": list,
            "emotion_intensity": str,
            "camera_movement": str,
            "dialogue_or_text": str,
        },
        "duration": float,
    },

    # ─── 可选：上下文信息 ───
    "context": {
        "prev_script_text": str | None,    # 上一句文案（连贯性参考）
        "prev_segment_id": int | None,     # 上一句匹配的片段ID
        "audio_duration": float | None,    # 当前文案的音频时长
    }
}
```

**为什么直接传视频而不抽关键帧**：

1. S1 已经切好了片段：`cache/{dj}/segment/ep_{n}/xxx.mp4`，每个 5-10 秒，直接拿来用
2. Gemini 原生支持视频输入，不需要额外抽帧，保留完整的时序信息（动作连贯性、节奏、画面切换）
3. 抽关键帧有信息损失：3-5 帧无法还原动作过程（"推搡"在关键帧里就是两张静止人像）
4. 实现更简单：不需要写关键帧抽取逻辑，复用 `video_evaluator.py` 的视频上传方式即可

### 3.2 输出结构

```python
JudgeOutput = {
    "evaluation_id": str,           # 唯一ID，格式: {dj}_{seg_seq}_{candidate_seg_id}
    "query": {                      # 原始输入的 query 部分（回写，方便追溯）
        "script_text": str,
        "episode_target": int,
    },
    "candidate": {                  # 候选摘要
        "segment_id": int,
        "episode": int,
    },

    "scores": {
        "entity_match": int,         # D1: 1-5
        "event_match": int,          # D2: 1-5
        "scene_match": int,          # D3: 1-5
        "tone_match": int,           # D4: 1-5
        "editing_suitability": int,  # D5: 1-5
        "overall": float,            # 加权总分 1.0-5.0
    },

    "reasoning": {
        "entity_match": str,          # D1 的详细推理，≥50字
        "event_match": str,           # D2 的详细推理，≥50字
        "scene_match": str,           # D3 的详细推理，≥50字
        "tone_match": str,            # D4 的详细推理，≥50字
        "editing_suitability": str,   # D5 的详细推理，≥50字
        "overall": str,               # 综合推理总结，≥100字
    },

    "match_highlights": [str],       # 匹配亮点列表（具体指出哪些元素匹配得好）
    "match_issues": [str],           # 不匹配问题列表（具体指出哪些元素有问题）

    "confidence": float,             # 0.0-1.0，评估置信度
    "would_use": bool,               # 如果你是剪辑师，你会用这个片段吗？

    "metadata": {
        "judge_model": str,          # 使用的模型
        "judge_version": str,        # 评分标准版本号
        "video_segment_path": str,   # 视频片段路径
        "temperature": float,
        "timestamp": str,
        "run_id": str,               # 批次ID，方便追溯
    }
}
```

### 3.3 为什么 reasoning 要求这么严格

Reranker DPO 训练需要偏好数据，而偏好数据的**质量上限**取决于标注的精确度。

如果 Judge 只给一个笼统的分数和一句"画面匹配较好"：
- 无法判断分数波动是模型随机性还是评估标准不明确
- 无法定位某个维度为什么得分低
- 无法构造高质量的 hard negative

详细的 reasoning 保证：
1. **可审计**：每条评分都能追溯到具体的推理链路
2. **可校准**：人工抽查时可以逐条对齐 reasoning 和评分
3. **可构造高质量 DPO 数据**：可以按维度筛选 hard negative
   - 例如：entity=4, event=4, scene=4, tone=2 的候选，是非常好的"基调不匹配"负样本
   - 这种细粒度负样本比笼统的"不好"有价值得多

---

## 四、Prompt 设计

### 4.1 设计原则

1. **角色锚定**：让 LLM 扮演具体角色（资深影视剪辑师），而非抽象的"评估者"
2. **先推理后打分**：强制 chain-of-thought，先分析再给分，减少随意性
3. **评分锚定**：每个分数等级都有可观察的客观标准（见第二节 rubric）
4. **维度独立**：强调每个维度独立评估，不要因为某个维度好就影响另一个维度的判断
5. **输出结构化**：严格要求 JSON 输出，无多余文字
6. **温度=0**：追求确定性，同一输入应产生相同输出

### 4.2 System Prompt

```
你是一位拥有 15 年经验的资深影视剪辑师和内容审核专家。
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

请严格以 JSON 格式输出，不要包含任何其他文字。
```

### 4.3 User Prompt（模板）

```
【解说文案】
{script_text}

【文案视觉需求（visual_query）】
- 人物姓名：{characters}
- 人物类型：{character_types}
- 动作关键词：{action_keywords}
- 环境：{environment}
- 视觉补充：{expanded_tags}
- 互动关系：{interaction_type}
- 观众心理：{audience_state}
- 情绪强度：{emotion_intensity}
- 导演意图：{director_intent}

【候选视频片段】（第 {episode} 集，片段ID {segment_id}，时长 {duration}秒）
<video>   ← 直接传入视频片段文件

【视频片段文本描述】（辅助参考，如与视频画面不一致以画面为准）
- 画面概括：{visual_summary}
- 出场角色：{characters}
- 主要动作：{actions}
- 环境：{environment}
- 互动关系：{interaction_type}
- 画面功能：{functional_intent}
- 情绪强度：{emotion_intensity}
- 镜头运动：{camera_movement}
- 对白/字幕：{dialogue_or_text}

{context_block}

请按以下 JSON 格式严格输出：

{{
  "analysis": {{
    "entity_match": "分析画面中有谁，人物类型和数量是否与文案需求对齐...",
    "event_match": "分析画面中发生了什么，动作/事件是否与文案描述一致...",
    "scene_match": "分析场景环境，空间/光线/时间是否与文案描述匹配...",
    "tone_match": "分析情绪氛围，情绪强度/互动关系/导演意图是否一致...",
    "editing_suitability": "从剪辑实操角度分析：节奏是否匹配、信息是否互补、能否抓住观众..."
  }},
  "scores": {{
    "entity_match": 1-5,
    "event_match": 1-5,
    "scene_match": 1-5,
    "tone_match": 1-5,
    "editing_suitability": 1-5
  }},
  "match_highlights": ["具体匹配好的点1", "具体匹配好的点2"],
  "match_issues": ["具体不匹配的点1", "具体不匹配的点2"],
  "overall_reasoning": "综合总结：整体匹配质量如何，哪些维度强、哪些弱，作为剪辑师你会怎么选择（≥100字）",
  "confidence": 0.0-1.0,
  "would_use": true/false
}}
```

其中 `{context_block}` 是可选的上下文信息：

```
【上下文】
上一句文案：{prev_script_text}
上一句匹配的片段ID：{prev_segment_id}
当前文案音频时长：{audio_duration}秒
```

### 4.4 Few-shot 示例（嵌入 system prompt 尾部）

提供 3 个评分示例，覆盖不同分数段，帮助模型校准评分尺度。

示例选择原则：
- 一个高分示例（overall ≥ 4.0）：展示"好的匹配"长什么样
- 一个中分示例（overall 2.5-3.5）：展示"部分匹配"的判断细节
- 一个低分示例（overall ≤ 2.0）：展示"完全不匹配"的判断细节

> 示例需要从真实数据中选取，人工校准后固化到 prompt 中。
> 首批开发时先用 placeholder，跑几轮后替换为真实校准数据。

---

## 五、质量保障机制

### 5.1 一致性校验（Self-Consistency Check）

对同一组 (query, candidate)，用 temperature=0 调用 3 次：

```
consistency_check:
  如果 3 次的 overall 分数标准差 > 0.5:
    标记为 "low_consistency"
    追加调用 2 次取众数
  如果 5 次仍不一致:
    标记为 "unstable"，标记为需人工审核
```

### 5.2 维度间相关性监控

每批次评估后统计各维度间的 Pearson 相关系数：

```
correlation_check:
  如果任意两个维度的相关系数 > 0.85:
    说明这两个维度在 LLM 看来区分度不够
    需要：
    1. 检查 prompt 中维度定义是否足够独立
    2. 考虑增加维度特异性的 few-shot 示例
    3. 如果确实无法区分，考虑合并维度
```

这是保证 5 维度正交性的关键监控手段。

### 5.3 分数分布监控

每批次评估后统计：

```
distribution_report:
  - 各维度的 mean / std / min / max
  - overall 分数直方图（1.0-5.0，0.5为一档）
  - 分数集中在 3 分左右的占比（如果>60%，说明评分缺乏区分度）
  - 分数<2 的占比（如果>30%，说明候选质量太差，不是 Judge 的问题）
  - 各维度分数方差（方差太小说明该维度缺乏区分度，需要调整 rubric）
```

### 5.4 人工校准基准（Human Calibration Baseline）

定期选取 50 条评估结果，人工重新打分，计算：

```
calibration_metrics:
  - 各维度的 Pearson / Spearman 相关系数（LLM vs 人工）
  - overall 的 Cohen's Kappa（一致性）
  - 平均绝对误差（MAE）
```

基准要求：
- overall MAE < 0.5（LLM 和人工的平均偏差不超过半分）
- Cohen's Kappa > 0.6

当偏差过大时：
1. 检查 prompt 是否需要增加示例
2. 检查是否需要切换更强的模型
3. 检查评分标准是否需要更新

### 5.5 跨模型交叉验证

对于"训练数据标注"场景（非快速评估），使用两个不同模型分别打分：

```
cross_validation:
  model_A: gemini-3-pro-preview（主力）
  model_B: gpt-4o 或 claude-sonnet（交叉验证）

  如果两模型 overall 差异 > 1.0:
    标记为 "disputed"
    进入人工审核队列
```

---

## 六、两种运行模式

### 6.1 快速评估模式（Evaluation Mode）

**用途**：评估当前系统匹配质量、A/B 对比、回归测试

| 配置 | 值 |
|------|-----|
| 单次调用 | temperature=0，调 1 次 |
| 模型 | gemini-3-flash（快且便宜） |
| 候选范围 | 每条文案只评 best_match |
| 一致性检查 | 不做 |
| 输出 | 聚合报告（平均分、分布、趋势） |

### 6.2 数据标注模式（Annotation Mode）

**用途**：构造 Reranker 的 DPO 训练数据

| 配置 | 值 |
|------|-----|
| 一致性检查 | 每条调 3 次，取中位数 |
| 模型 | gemini-3-pro（最强的） |
| 候选范围 | 每条文案评 top 5 候选（S4 输出的 top_matches） |
| 跨模型验证 | 10% 随机抽样做交叉验证 |
| 人工校准 | 5% 随机抽样做人工审核 |
| 输出 | 逐条完整 JudgeOutput JSON |

---

## 七、数据产出与 Reranker 训练接口

### 7.1 评估结果存储结构

```
judge/
├── outputs/
│   └── {dj}/
│       ├── eval_{run_id}/
│       │   ├── config.json              # 本次评估的配置
│       │   ├── results.jsonl            # 逐条评估结果（每行一个 JudgeOutput）
│       │   ├── distribution_report.json # 分数分布统计
│       │   └── summary.md               # 人类可读的评估报告
│       └── annotation_{run_id}/
│           ├── config.json
│           ├── results.jsonl            # 高质量标注数据
│           ├── calibration_report.json  # 一致性+交叉验证报告
│           └── disputed.jsonl           # 有争议的样本（需人工审核）
```

### 7.2 Reranker DPO 数据构造规则

从标注模式的 results.jsonl 构造 DPO 训练数据：

```
构造规则（按优先级排序）：

1. 正样本定义：
   - overall >= 4.0 且 would_use == true

2. 负样本定义（三种类型，混合使用）：

   a) Easy Negative（容易区分）：
      overall <= 2.0 的候选

   b) Hard Negative（难区分，高价值）：
      overall 在 2.5-3.5 之间
      且至少有一个维度 >= 4（说明"有部分匹配但整体不行"）
      这种负样本对 Reranker 训练最有价值

   c) Dimension-Specific Negative（维度特异性负样本）：
      针对特定维度的弱项构造
      例如：tone_match <= 2 但 entity/event/scene >= 4
      含义："内容和场景都对了但情绪完全不对"
      需要标注 weak_dimensions 字段

3. 数据格式：
{
    "query_text": "script_text + visual_query 展开为自然语言",
    "query_structured": { visual_query 的 JSON },
    "positive": {
        "segment_id": 123,
        "description_text": "视频描述展开为自然语言",
        "description_structured": { structured_description 的 JSON },
        "judge_scores": { 各维度分数 },
        "judge_overall": 4.2,
    },
    "negative": {
        "segment_id": 456,
        "description_text": "...",
        "description_structured": { ... },
        "judge_scores": { 各维度分数 },
        "judge_overall": 2.8,
        "negative_type": "hard" | "easy" | "dimension_specific",
        "weak_dimensions": ["tone_match"],
    },
    "preference_gap": 1.4,
    "source_run_id": "annotation_20260417_001"
}
```

### 7.3 数据质量门槛

不是所有标注数据都能用于训练，必须通过以下门槛：

```
数据准入检查：
  ✓ Judge confidence >= 0.7（低置信度的不用）
  ✓ preference_gap >= 1.0（正负样本差距太小的不用）
  ✓ 不是 "disputed" 样本（跨模型不一致的不用）
  ✓ 至少通过一次一致性检查（3次调用的标准差 < 0.5）
```

---

## 八、Judge 演进计划

### 当前方案：多模态 Judge（视频片段 + 文本描述）

**设计决策**：Judge 从一开始就直接输入视频片段，不做 text-only 版本。

理由：
1. S1 已经切好了视频片段（`cache/{dj}/segment/ep_{n}/xxx.mp4`），素材现成的
2. Gemini 原生支持视频输入，技术上零门槛
3. Benchmark 从第一天就一致，不用迁移数据
4. DPO 训练数据从一开始就是高质量的，不存在"先用低质量数据凑合"的阶段
5. 文本描述是视频理解模型的二次加工，不可避免有信息损失或偏差，直接看视频是最准的

**Judge 看到的信息**：

```
输入 = {
    script_text,                  # 文案原文
    visual_query,                 # 文案侧结构化需求
    video_segment (mp4),          # 直接观看视频片段
    video_structured_description, # 文本描述（辅助参考）
}
```

**每个维度 Judge 实际看到了什么**：

| 维度 | Judge 的信息来源 | 视频带来的独特价值 |
|------|-----------------|-------------------|
| entity_match | 视频中的人物 + 文本描述的人物 | 直接看到人数/性别/年龄，不依赖描述准确性 |
| event_match | 视频中的动作过程 + 文本描述 | **完整动作轨迹**（推→倒→起身），强度/节奏/微动作文本无法传达 |
| scene_match | 视频中的环境 + 文本描述 | 镜头移动揭示空间感，光影变化 |
| tone_match | 视频中的表情变化/肢体语言 + 文本描述 | **情绪动态**（平静→愤怒→崩溃），文本只能说"情绪强度高"但看不到表情 |
| editing_suitability | 视频的画面质量/节奏/视觉冲击力 | 镜头稳定性、剪辑节奏、动态效果，纯文本完全无法评估 |

---

### 未来扩展：片段连贯性评估

**触发时机**：当 Reranker 训练完成（Phase 3）后，Judge 需要评估"整段解说的连续匹配质量"，而不仅仅是单条匹配。

**新增能力**：

```
1. 片段连贯性评估（当前完全没有考虑）
   - 前后两个片段是否来自同一集的连续场景？
   - 是否出现重复片段（连续两段用了同一个视频片段）？
   - 跳集是否合理（前一句说第3集，后一句突然跳到第8集）？

2. 叙事流畅度评分
   - 整段解说（20秒）的所有片段拼接后，视觉上是否流畅？
   - 画面切换是否自然（色调、场景连续性）？

3. 高吞吐打分模式
   - 用于 RL 训练时需要大量快速打分
   - 可以用 flash 模型 + 简化 prompt，只输出分数不做详细 reasoning
```

**接口扩展**：

```python
# 在现有 JudgeOutput 基础上新增（不改动现有字段）
JudgeOutputExtended = {
    ... # 现有全部字段不变

    "coherence": {
        "prev_segment_similarity": float,  # 与前一片段的视觉相似度
        "is_repeated": bool,               # 是否重复使用同一片段
        "episode_jump": bool,              # 是否跳集
        "scene_continuity": int,           # 1-5 场景连续性
    },

    "narrative_flow": {
        "visual_smoothness": int,    # 1-5 视觉流畅度
        "pacing_match": int,         # 1-5 画面节奏与解说节奏匹配度
    }
}
```

**用途变化**：

| 用途 | 当前 | 扩展后 |
|------|------|--------|
| 系统质量评估 | 逐条评 best_match | 逐条评 + 整段评 |
| A/B 对比 | 支持 | 支持 |
| DPO 训练数据 | 核心用途 | 不再是主要用途（Reranker 已训练好） |
| RL reward signal | 不支持 | 新增核心用途 |
| 回归测试 | 支持 | 支持 + 连贯性回归 |

---

## 九、实现计划

### 文件结构

```
judge/
├── judge.md                  # 本设计文档
├── match_judge.py            # Judge 核心模块（单条评估逻辑）
├── batch_judge.py            # 批量评估脚本（评估模式/标注模式）
├── judge_report.py           # 评估报告生成（分布统计、相关性分析）
├── rl_data_builder.py        # RL / Reranker 训练数据构造（从 Judge 结果生成样本与偏好对）
├── prompts/
│   ├── judge_system.md       # System Prompt
│   └── judge_user.py         # User Prompt 模板（代码动态拼接）
├── configs/
│   └── judge_config.yaml     # 模型、温度、维度权重等配置
└── outputs/                  # 评估结果输出目录
    └── {dj}/
```

### 开发顺序

```
Step 1: match_judge.py —— 单条评估核心逻辑
        - 输入解析
        - Prompt 拼接
        - 模型调用
        - 输出解析 + 校验
        - 一致性检查

Step 2: batch_judge.py —— 批量评估
        - 读取 cache/{dj}/*_structured.json（含 match_result 和 top_matches）
        - 对每个 segment 的 top_matches 逐条跑 Judge
        - 支持评估模式和标注模式
        - 断点续评（已评估的跳过）

Step 3: judge_report.py —— 报告生成
        - 分数分布统计
        - 维度间相关性分析
        - 跨集数/跨剧对比

Step 4: rl_data_builder.py —— 训练数据构造
        - 正/负样本筛选
        - Hard negative 策略
        - 数据质量过滤
        - 输出标准 DPO 格式

Step 5: 少量剧跑通全流程 + 人工校准
        - 选 2-3 部剧
        - 每部剧跑标注模式
        - 50 条人工校准
        - 根据 校准结果 调整 prompt / 示例
```

### 与现有系统的集成方式

```
集成点 1: 批量评估入口
  在 run.sh / run.py 的 S4 完成后，可选调用 batch_judge.py

  python judge/batch_judge.py \
    --dj buxi5 \
    --mode eval \          # eval 或 annotation
    --model gemini-3-flash \
    --output-dir judge/outputs/buxi5

集成点 2: 训练数据构造入口
  积累足够多剧的标注数据后：

  python judge/rl_data_builder.py \
    --input-dirs judge/outputs/buxi5/annotation_*/ judge/outputs/weinishiguit4/annotation_*/ \
    --output-dir reranker/training_data/ \
    --min-gap 1.0 \
    --min-confidence 0.7
```

---

## 十、评分标准维护规范

### 何时更新评分标准

1. **人工校准发现系统性偏差**（如 LLM 总是给 tone_match 偏高分）
2. **维度间相关性过高**（两个维度相关系数 > 0.85，说明 LLM 区分不开）
3. **某维度缺乏区分度**（某维度分数方差极小，集中在 3 分左右）
4. **新增评估维度**（如 v3 新增 coherence_score）
5. **评分标准不够区分**（分数集中在一个狭窄区间）

### 更新流程

1. 在 judge.md 中更新评分标准（修改 rubric 表格）
2. 更新 judge_version 版本号（v1 → v2）
3. 用新标准重新标注一个小样本集
4. 对比新旧标准的分数分布变化
5. 确认新标准更优后，全量重跑

### 版本兼容

- `match_judge.py` 根据 config 中的 `judge_version` 选择对应的 prompt 和 rubric
- 不同版本的评估结果分目录存储，不互相覆盖
- DPO 数据构造时只使用同一版本的数据，不混用

---

## 附录：维度正交性检查清单

每次修改评分标准或 prompt 后，跑一个小批次（~50条）并检查：

```
正交性检查：
  [ ] 任意两个维度间的 Pearson 相关系数 < 0.85
  [ ] 每个维度的分数方差 > 0.5（有区分度）
  [ ] 各维度的分数分布不完全一致
  [ ] 人工抽查 10 条，各维度分数确实反映不同方面
  [ ] 不存在"某维度总是最高/最低"的系统性偏差
```

如果检查不通过，回到 prompt 调整阶段。
