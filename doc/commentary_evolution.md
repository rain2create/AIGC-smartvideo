# 解说文案生成 — 演进总结

> 从 V1 的"调 Gemini 直接出稿"到 V6 的"RAG + PlotGraph 事实锚定 + BeatSheet 节奏规划 + 反幻觉闭环"，解说文案生成经历了从粗放到精细的完整进化。

## 1. 概述

解说文案是下游"文案-视频匹配"的起点——文案质量直接决定匹配精度的天花板。整个演进围绕一个核心命题：

**如何让 LLM 稳定产出高质量、低幻觉、可操作的解说文案？**

V1 给出了基础框架（SOP prompt + 迭代修订），V6 在此基础上补了四块短板：内容理解、事实约束、节奏控制、幻觉检测。

---

## 2. 两版对照总览

```
V1 (smart_video)                         V6 (smart_videoV6)
═══════════════════════════════════════════════════════════════
输入: 剧名 + 集数 + ASR文本                  同 V1
                                              │
模型: google/gemini-3-pro-preview            同模型，API 不变
                                              │
生成: text0204.py                            text0204.py (大幅增强)
  ├─ SOP prompt (5步法)                         ├─ 同 SOP prompt
  ├─ 3轮迭代 (生成→评估→修订)                     ├─ 同迭代逻辑
  ├─ 简单 RAG (单标签检索)  ←─已废弃─→           ├─ 多维标签 + 加权检索
  └─ 无幻觉检测                                 ├─ PlotGraph 事实锚定
                                                ├─ BeatSheet 节奏规划
                                                ├─ 反幻觉检测 + 局部修补
                                                └─ fail-fast 格式校验
                                              │
后处理: extract_final_script_to_text.py       同，改为 fail-fast
                                              │
下游:  .txt → 视频匹配                        同
```

---

## 3. 核心模块演进

### 3.1 基础生成：SOP 五步法始终是骨架

两个版本共享同一套 SOP prompt (`prompts/skill_video_commentary_text_formatted.md`)：

```
SOP1 大纲：六要素表（人地事因情细）+ 爆点标注
SOP2 精简：关键词提取 + 冗余裁剪
SOP3 虎头：钩子开头，对比/夸张/细节三技法，元素控制法则
SOP4 豹身：细节+钩子+解释+连接+剧情 循环，单视角叙事，反平铺
SOP5 凤尾：大钩子收尾，悬念完成度 70-99%，拆钩法
```

这套方法论本身在 V1→V6 期间没有变化——它已经是经过验证的产出标准。变化发生在**围绕 SOP 的约束和保障机制**上。

### 3.2 内容理解：从无到多维标签体系

| | V1 | V6 |
|---|---|---|
| 内容分析 | 无，LLM 直接读 ASR | `advanced_labeling.py` 先分析 |
| 标签维度 | 无 | 5 主标签(走心催泪/悬疑解谜/搞笑吐槽/爽快打脸/温情叙事) + 子标签 + 强度打分(节奏/钩子密度/情感/冲突/反转, 各1-5) + 写作风格(POV/句式/钩子类型/语态/对白比) |
| 粒度 | - | 剧级标签 + 事件级标签(150-250字一个事件) |

**为什么引入**：V1 中 LLM 面对长 ASR 文本容易"平均用力"，产出千篇一律。多维标签体系让系统先"读懂"这部剧的类型和节奏特征，后续的 RAG 检索、PlotGraph 构建、节奏规划都基于标签做决策。

### 3.3 示例检索：从简单匹配到多维度加权

| | V1 | V6 |
|---|---|---|
| 检索方式 | 单标签匹配 | `rag_retrieval_labeling.py` 多维度加权打分 |
| 打分公式 | 命中/不命中 | `3×主标签 + 2×子标签 - 0.5×强度距离 + 1×风格匹配` |
| 去重 | 无 | 按子标签+钩子类型去重 |

**为什么变**：V1 的简单匹配经常召回"标签对但风格不对"的示例（比如同为悬疑，但一个是快节奏揭秘型、一个是慢热氛围型），反而误导生成。加权打分+风格匹配+去重三管齐下，召回的示例更精准。

> 注：RAG 在当前版本中默认关闭（`--num-examples` 默认 0），因为 PlotGraph + BeatSheet 已经提供了足够的结构约束。RAG 作为可选增强保留。

### 3.4 事实约束：PlotGraph（V6 新增）

**V1 的问题**：LLM 生成解说时经常捏造剧情细节——人物关系错乱、事件顺序颠倒、凭空编造冲突。

**V6 的解法**：`build_plot_graph()` 在生成前先用 LLM 从 ASR 中提取结构化事实图谱：

```
PlotGraph 输出结构:
├── characters: [{name, aliases, role, traits}]     ← 人物表
├── relations: [{from, to, type}]                   ← 人物关系
├── atomic_events: [{id, desc, evidence, episode}]  ← 原子事件+原文证据
├── beats: [{id, type, events}]                     ← 情节节拍
├── unknowns: [{question, why_important}]           ← 已知的未知
└── adjacency_check: {gap, bridge}                  ← 邻接连贯性校验
```

关键设计：
- **evidence 字段**：每个原子事件必须引用 ASR 原文证据（防止 LLM 自己编事件）
- **adjacency_check**：检测连续事件之间是否有逻辑断档（如 A 打了 B 然后 B 哭了，中间缺少"B 的反应"桥接）
- **unknowns**：明确记录"我们不知道什么"（如"男女主什么时候认识的"），后续反幻觉检测会用到

PlotGraph 生成后会注入到用户 prompt 中，让主生成 LLM 在创作时有事实参照。

### 3.5 节奏控制：BeatSheet（V6 新增）

**V1 的问题**：生成的解说"平铺直叙"，开头不够炸、中段没起伏、结尾缺钩子。

**V6 的解法**：`build_beat_sheet()` 在生成前规划 6 段式节奏结构：

```
BeatSheet 分段:
  段1 开头钩子   max_new_facts=1   must_hook=True    → 一句话引爆
  段2 铺垫       max_new_facts=3   must_hook=False   → 人物+背景
  段3 冲突升级   max_new_facts=2   must_hook=True    → 矛盾激化
  段4 反转       max_new_facts=1   must_hook=True    → 剧情翻转
  段5 高潮爆点   max_new_facts=2   must_hook=True    → 情绪顶峰
  段6 结尾悬念   max_new_facts=1   must_hook=True    → 卡悬念续集
```

每段指定了"最多引入几个新事实"和"是否需要钩子"，主生成 LLM 在这套约束下创作，节奏感有骨架保证。

### 3.6 质量评估：从迭代修订到 Gate 硬拦截

| | V1 | V6 |
|---|---|---|
| 评估方式 | `quality_eval` 打分 (虎头20+豹身60+凤尾20) | 同一套 |
| 迭代逻辑 | 3轮: 生成→评估→修订 | 同3轮 |
| Gate 硬拦截 | 无 | 有，检查幻觉/节奏/钩子 |

V6 的 Gate 模块在评估模块之后增加了硬性拦截规则：
- **幻觉 Gate**：检查是否与 PlotGraph 冲突
- **节奏 Gate**：检查是否符合 BeatSheet 分段要求
- **钩子 Gate**：检查开头和结尾是否有钩子

触发 Gate 拦截后不是"建议修订"，而是直接替换为针对性 retry prompt，迫使 LLM 改正特定问题再继续。

### 3.7 反幻觉：V6 最关键的增量

**V1 的问题**：LLM 生成完后直接输出，不做事实核查。结果有大量"剧情合理但实际没发生"的内容。

**V6 的解法**：三步闭环——

```
1. 提取: extract_final_commentary()
   从完整输出中提取"最终解说文案"部分（剥离 SOP1-5 分析过程）

2. 检测: check_hallucination_conflicts()
   逐句比对 PlotGraph:
   ├── contradicted: 与已有事实矛盾 → 标记
   └── over_confident_on_unknown: 对标记为 "unknown" 的事做了断言 → 标记

3. 修补: patch_commentary_local()
   只替换被标记的句子，不改动其他内容
   用 LLM 对每个标记句子生成 "同义但尊重事实" 的替换版本
```

最终输出同时包含 `raw_response`（原始生成）和 `dehallucinated_text`（反幻觉后版本），`extract_final_script_to_text.py` 优先提取反幻觉版本。

### 3.8 格式校验：从容错到 fail-fast

**V1 的做法**：格式不满足要求时降级使用原始输出，下游遇到含 SOP 分析文本的 txt 出现噪音。

**V6 的做法**：
- SOP1-5 + 最终解说文案必须按严格顺序出现
- 最终解说文案中不能包含 SOP 模板标记或分析文字
- 任一检查不通过 → 重试，重试 5 次仍失败 → 直接报 RuntimeError

改为 fail-fast 的原因：下游匹配管线后面还有 judge 评估和训练数据构造，一条脏文案会污染整个 pipeline。宁可生成阶段报错，也不要让脏数据流下去。

### 3.9 集数映射：suoying.py（V6 新增）

**V1 的做法**：解说文案与剧集集数的对应关系是隐含的——LLM 在生成时"大概知道"当前在讲第几集，但没有显式标注。

**V6 新增**：`suoying.py` 将解说文案的每一句话显式映射到对应的剧集集数：

```
输入：解说文案 + ASR 文本（含集数信息）
输出：每句话 → {sentence_idx, episode, confidence, reason}
```

这为下游的 `skill_text_structured_description.py` (S3) 提供了精确的 episode_target 标注，使得匹配阶段可以只在目标集数范围内检索视频片段，大幅缩小候选空间。

---

## 4. 关键设计决策回顾

| 决策 | 为什么 | 效果 |
|------|--------|------|
| 引入 PlotGraph | LLM 编造剧情细节，人物关系混乱 | 以 evidence 驱动的事实图谱锚定生成，减少幻觉 |
| 引入 BeatSheet | 产出节奏平铺，开头不炸结尾缺钩 | 6 段式节奏骨架约束，每段设新事实上限 |
| 引入反幻觉检测 | 生成内容"听起来合理但实际没发生" | 逐句比对+局部修补，不破坏整体流畅度 |
| 引入多维标签 | 单一标签匹配不精准 | 5 维标签+强度+风格，RAG 召回更准 |
| 引入 suoying | 集数映射不精确 | 下游匹配在小范围检索，减少误匹配 |
| fail-fast 格式校验 | 容错让脏数据流入下游 | 宁可报错，不污染 judge 和训练链路 |
| 关闭 RAG 默认 | PlotGraph + BeatSheet 已足够 | 减少 LLM 调用次数和 token 消耗 |

---

## 5. 产出与下游承接

```
text0204.py 产出:
  results/{dj}/
    ├── {start}_{end}_{ts}_result.json     ← 完整元数据
    ├── {start}_{end}_{ts}_commentary.md   ← 解说文案 (SOP1-5 + 最终文案)
    ├── plot_graph.json                    ← 事实图谱
    └── beat_sheet.json                    ← 节奏规划

extract_final_script_to_text.py:
    .md → .txt                             ← 提取纯文案文本

suoying.py:
    .txt + ASR → episode_mapping.json      ← 句子→集数映射

下游:
    run_mul_qwen.py 中的 find_or_create_txt()
    → txt 文案喂给 text_video_match/ (S3: skill_text_structured_description.py)
    → episode_target 限制匹配范围
    → Qwen3-VL Embedding 检索 + Reranker 精排
```

---

## 6. 代码资产索引

| 文件 | 职责 | 版本 |
|------|------|:----:|
| `commentary/text0204.py` | 主生成管线 (RAG+PlotGraph+BeatSheet+反幻觉) | V6 |
| `commentary/skill_video_commentary_text.py` | 基础生成技能 (SOP+迭代) | V1 |
| `commentary/skill_video_commentary_quality_eval.py` | 质量评估 (虎头豹身凤尾) | V1 |
| `commentary/advanced_labeling.py` | 多维标签 + 事件级标注 | V6 |
| `commentary/rag_retrieval_labeling.py` | 加权 RAG 检索 | V6 |
| `commentary/fewshot7.py` | Forward SOP 生成器 | V6 |
| `commentary/suoying.py` | 句子→集数映射 | V6 |
| `commentary/extract_final_script_to_text.py` | MD→TXT 提取 | V1+ |
| `commentary/model_client.py` | LLM API 客户端 (360.cn) | V1 |
| `commentary/prompts/skill_video_commentary_text_formatted.md` | SOP 方法论 prompt | V1 |
| `commentary/prompts/skill_video_commentary_quality_eval.md` | 评估维度 prompt | V1 |
