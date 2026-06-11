# LLM-as-Judge 评估系统：从 0 到 Benchmark

> 面向中文短剧解说场景，设计一套自动化、可复现、可批量的视频匹配质量评估体系。
> 一句话：让 LLM 看视频+读文案，代替人判断"这个片段配这句解说合不合适"。

---

## 1. 问题定义：为什么需要 Judge

### 1.1 任务背景

短剧解说视频的生产链路是：

```
解说文案（LLM生成） → 视频片段匹配（检索+精排） → 合成输出
```

其中"视频片段匹配"这一步，需要从目标集数的几十个候选片段中，找到最配这句文案的那一个。

### 1.2 核心矛盾

```
每部短剧: 15-40 句文案 × 每句 20-50 个候选片段 = 约 1000 次匹配判断
100 部剧: 约 100,000 次判断
```

V1 时代没有自动化质量评估。视频匹配效果好坏全凭肉眼拉片——这在 6 部剧的早期验证阶段勉强可行，但要扩展到 100 部、同时作为微调训练的数据基础，人工评估完全不可规模化。

### 1.3 需求拆解

| 需求 | 具体要求 |
|---|---|
| 客观性 | 同一对（文案, 视频）多次评估分数稳定，不同人/模型评估结果可比 |
| 细粒度 | 能区分"人物对但动作错"和"全对"这种微妙差异 |
| 可批量 | 能跑 100 部剧 × 1000 次 = 10 万次评估，无需人工介入 |
| 信息充分 | 评估必须基于视频画面本身，不能只靠文本描述中转 |
| 可操作 | 评估结果能直接指导"这个片段能不能用"和"为什么不能用" |

---

## 2. 方法论：为什么用 LLM-as-Judge

### 2.1 放弃传统 IR 指标

传统文本检索用 Hit@K、MRR、NDCG 等指标衡量排序质量——但文案-视频匹配的核心问题是**跨模态语义对齐**，不是排序精度。

```
Hit@1 能回答: "系统排第一的是不是 Judge 最高分那个？"
Hit@1 不能回答: "系统排第一的那个，画面和文案到底合不合适？"
```

系统可以把画面拍得最像的排第一，但那个画面可能是"人物对/动作对/情绪全错"。传统指标对此完全不可见。

### 2.2 放弃文本描述中转

V1 的做法是：视频 → Gemini 描述成文本 → BERT Embedding → 和文案做文本匹配。匹配效果受限于 Gemini 描述的质量——微表情、光影变化、镜头运动这些东西，文本永远描述不全。

Judge 必须**直接看视频画面**做判断，不能走文本描述中转这条路。

### 2.3 LLM-as-Judge 的三重优势

| 能力 | 为什么关键 |
|---|---|
| 多模态理解 | 直接看视频帧 + 读文案，跳过文本描述的信息损失 |
| 结构化输出 | 分维度打分 + 推理 + would_use，结果可解析、可追溯 |
| Few-shot 校准 | system prompt 中嵌入评分示例，把评分标准"刻"进模型行为 |

### 2.4 模型选型

| 模型 | 用途 | 选择理由 |
|---|---|---|
| `volcengine/doubao-seed-2-0-lite` | 全量批量标注（主力） | 性价比高，100剧×50K评估能跑完 |
| `google/gemini-3-pro-preview` | 早期验证 + 高精度标注 | Pro 级推理，一致性更好 |
| `google/gemini-3-flash-preview` | 快速实验 | 成本低，适合开发迭代 |

---

## 3. 五维度评估体系

### 3.1 设计原则

直接提问"这个片段好不好"是无效的——模型缺乏判断标准。我们借鉴电影剪辑的工作方法，把"好匹配"拆成 5 个独立维度，每个维度只问一件事：

```
维度1 - entity_match（实体匹配，权重 0.20）
  问题：画面里有谁？人物类型和数量是否与文案需求对齐？

维度2 - event_match（事件匹配，权重 0.30）
  问题：画面里发生了什么？动作/事件是否与文案描述一致？

维度3 - scene_match（场景匹配，权重 0.15）
  问题：在什么环境下？空间/光线/时间是否匹配？

维度4 - tone_match（基调匹配，权重 0.25）
  问题：画面的情绪感觉对不对？情绪强度/互动关系/氛围是否一致？

维度5 - editing_suitability（剪辑适用性，权重 0.10）
  问题：好不好用？节奏/信息互补/观众注意力/时长与镜头质量。
```

### 3.2 权重设计的理由

```python
DIMENSION_WEIGHTS = {
    "entity_match": 0.20,
    "event_match": 0.30,      # 最高：动作/事件是画面核心
    "scene_match": 0.15,      # 最低之一：场景可容忍偏差
    "tone_match": 0.25,       # 次高：情绪错了一切都错
    "editing_suitability": 0.10, # 最低：实操兜底项
}
```

- **event_match (0.30)**：短剧解说的核心是"讲剧情"，事件/动作是画面的主体
- **tone_match (0.25)**：情绪错位的片段，人物和动作再对也没法用——观众出戏
- **entity_match (0.20)**：人物对了是基础门槛，但不是充分条件
- **scene_match (0.15)**：场景偏差可以容忍（办公室 vs 会议室问题不大）
- **editing_suitability (0.10)**：实操层面的兜底，不重复前四维的判断

### 3.3 通用评分 Rubric

```
1分 = 完全不匹配：该维度描述的内容与候选毫无关系
2分 = 严重偏离：仅有一个次要元素勉强沾边
3分 = 部分匹配：主要元素有匹配但存在明显缺失或错位
4分 = 较好匹配：核心内容一致，仅有微小偏差
5分 = 完美匹配：精准吻合，无任何违和感
```

### 3.4 System Prompt 中的方法论要点

System prompt（`match_judge.py:118-217`）中植入了几个关键约束：

**画面优先原则**："如果视频画面与文本描述不一致，以视频画面为准"。这避免了模型直接采信"S1 结构化的文本描述"而不看视频。

**维度独立原则**："各维度独立评分：不要因为事件匹配好就给情绪加分，也不要因为场景不对就给事件扣分"。这是防"喜欢就全给高分"的 halo effect。

**推理具体化要求**：每个维度的推理必须 >=50 字，引用具体画面特征（"视频中可以看到..."），防止模型给一个笼统分就跑。

**Few-shot 校准**：嵌入 5 个覆盖全部分数区间的评分示例（overall 5.0 / 4.4 / 3.0 / 1.8 / 1.0），用具体案例"教"模型怎么打分。

---

## 4. 单条评估的完整链路

### 4.1 输入准备

```python
JudgeInput(
    script_text:  "总裁一把将她按在墙上，眼神危险",
    visual_query: {characters:["总裁","女主"], action_keywords:["壁咚"], ...},
    episode_target: 3,
    segment_id: 417,
    video_segment_path: "segment/ep_3/417_总裁壁咚.mp4",
    structured_description: {visual_summary:"...", characters:[...], ...},
    duration: 3.5,
)
```

输入的三层信息：
- **文案侧**（必须）：script_text + visual_query（S3 产出）
- **视频侧**（必须）：视频文件本身（抽帧注入） + structured_description（S1 产出，辅助参考）
- **上下文**（可选）：上一句文案 + 上一句片段，帮助判断叙事连续性

### 4.2 视频抽帧策略

```python
FRAME_FPS = 1.0       # 每秒抽 1 帧
JPEG_QUALITY = 80     # 压缩质量，平衡清晰度和 API 传输
```

- 为什么 fps=1.0 而非更密：Judge 需要理解"这个片段在讲什么"，不需要逐帧分析
- 帧以 base64 data URI 直接注入 API 请求，不走文件存储

### 4.3 API 调用与输出解析

```
video_frames (base64) ──┐
                         ├──→ LLM API → JSON response
user_prompt (text)  ────┘

JSON 格式:
{
  "analysis": {
    "entity_match": "...≥50字推理...",
    "event_match": "...",
    "scene_match": "...",
    "tone_match": "...",
    "editing_suitability": "..."
  },
  "scores": {entity_match:5, event_match:4, scene_match:3, tone_match:4, editing_suitability:5},
  "match_highlights": [...],
  "match_issues": [...],
  "overall_reasoning": "...≥100字综合推理...",
  "confidence": 0.90,
  "would_use": true
}
```

解析链路：
1. 移除 thinking/analysis 标签（部分模型自带）
2. 移除 markdown 代码块（```json ... ```）
3. 正则提取完整 JSON 对象
4. 校验 5 维度分数全部在 1-5 范围内
5. 加权计算 overall = Σ(score × weight)
6. **格式重试**：解析失败或字段缺失 → 重新调用 API，最多 5 次

### 4.4 一致性检查

Annotation 模式下启用（`evaluate_with_consistency`）：

```
调用 3 次，计算 overall 标准差:
  std ≤ 0.5: 一致，取中位数那次的结果
  std > 0.5: 追加 2 次调用
  5 次仍 > 0.5: 标记 low_consistency，仍取中位数
```

一致性检查的定位：不是为了追求绝对零方差（human annotator 自己也有波动），而是**检测和标记高不确定性样本**——这些样本往往处于评分边界，为后续的 quality check 提供线索。

---

## 5. Canonical Judge：一次评估，多方复用

### 5.1 核心思路

```
传统做法（低效）:
  V4匹配 → Judge评估V4结果
  Qwen V1匹配 → Judge评估Qwen V1结果
  Qwen B3匹配 → Judge评估Qwen B3结果
  ...每个新匹配方法都要重跑Judge

Canonical Judge（高效）:
  对固定 (文案集合 × 目标集数内所有视频片段) 做一次全量评分矩阵
  → 各 variant 只提供自己的 best_match
  → 从矩阵里查这个 best_match 的 Judge 分数
  → 做比较
```

### 5.2 为什么这样设计

| 收益 | 说明 |
|---|---|
| 消除 Judge 随机性 | 各 variant 比较的是同一个评分矩阵，不存在"这次打分偏高"的噪音 |
| 大幅节省成本 | 100 剧全量 Judge 跑一次 ~50K 次 LLM 调用。每增加一个 variant 只查表，零额外 LLM 调用 |
| 便于增量实验 | 新检索方案只需产出匹配结果，直接对表比较，秒级得到结论 |

### 5.3 前提约束

Canonical Judge 的复用有严格前提：
- 文案拆分必须固定
- episode_target 必须固定
- 目标集数内视频片段池必须固定

如果后续某个方案改变了文案拆分逻辑或集数映射，就需要重新跑 Canonical Judge。

---

## 6. Benchmark 构建全过程

### 6.1 Phase A：6 剧摸底

最初在 6 部验证剧上跑通完整链路，建立基线认知：

| 指标 | 数字 |
|---|---|
| 总样本 | 849 |
| 正样本 (overall >= 4.0 & would_use=True) | 89 (10.5%) |
| 负样本 | 651 (76.7%) |
| 无正样本的 query | 7/28 (25%) |
| 在 tie_group 内的样本 | 70% |

关键发现：
- **正样本极少**：849 个候选里只有 89 个真能用，匹配本身就很难
- **easy_negative 压倒性多数**：2 分以下的片段占 60%+，训练数据构造必须强制采 hard
- **query 间差异大**：最好的 query（如打斗场景）30%+ 高分候选，最差的 query 一个正样本都没有

### 6.2 全因子实验：选定微调底座

在 6 剧上跑全因子比较（V4 × Qwen V1/B1/B2/B3 × Q1/Q3 × Reranker）：

```
控制变量:
  - Embedding 方法: V4 (BGM API) vs Qwen (纯视觉 V1 / 联合 B1 / 双路 B2 / 双路加权 B3)
  - Query 格式: Q1 (纯文本) vs Q3 (文本+visual_query结构化字段)
  - 是否精排: 无 Reranker vs Reranker Q1

评估指标:
  - Hit@K (K=1,3,5,10,20): 系统 top-1 是不是 judge top-1
  - Judge overall 均值: 系统 best_match 的平均评分
  - Score >= 4.5 / >= 4.0 占比: 高分片段比例
  - 距最优分差: 系统 best_match 与同 query 最优候选的分差
```

核心结论：

| 变体 | Hit@1 | 均分 | >=4.5 | 选择理由 |
|------|:---:|:---:|:---:|------|
| qwen_b1_q3 | 39.3% | 4.03 | 42.9% | 综合最佳，但依赖 B1 sidecar 文本 |
| **qwen_q3_rerank_q1** | **39.3%** | **3.96** | **42.9%** | 零样本 Reranker 追平 B1 Hit@1，且无 B1 依赖 |
| qwen_b1_q3_rerank_q1 | 39.3% | 3.96 | 42.9% | 与上相同，多了 B1 依赖无增益 |

**为什么选 `qwen_q3_rerank_q1`**：
- Hit@1 已追平综合最佳，Reranker 还是零样本——微调后有明确提升空间
- 不依赖 B1 sidecar（B1 需要每个片段先生成联合 embedding，多一个故障依赖）
- Embedder + Reranker 两阶段独立，可以只微调 Reranker（emb 冻住，训练更简单）

### 6.3 Phase X：100 剧扩展

全因子实验在 6 剧上定了底座，接下来在 100 部新剧上跑全链路，扩大数据覆盖：

```
流程:
  S1: video_episode_indexer → 视频结构化索引
  S3: text_structured_description → 文案拆分 + visual_query
  S2(qwen): index_builder → Qwen3-VL 视觉 embedding (embeddings_vl.pkl)
  S4(qwen): retriever V1+Q3 → 召回 top_matches
  S4.5(qwen): reranker Q1 → 精排重排 top_matches
  S5: postprocess → 最终视频 mp4
  S6: final.json

→ Canonical Judge 全量标注 (50,051 次评估)
→ Builder v3 构造训练数据
```

落地结果：

| 项目 | 数量 |
|---|---|
| 成功完成的剧 | 93/100 |
| 总 Judge 评估 | 50,051 次 |
| 训练样本 (samples) | 50,900 |
| 失败 | 5 部 LLM 输出格式不对 + 1 部视频编码 + 1 部视频文件损坏 |

### 6.4 全量数据分布

```
总样本: 50,900 (106 剧)

  strong_positive     3,365 ( 6.6%)    强正
  fallback_positive   3,810 ( 7.5%)    回退正
  ambiguous_positive     18 ( 0.0%)    弱正（极少）
  hard_negative      13,841 (27.2%)    难负
  easy_negative      29,866 (58.7%)    简单负
  unusable                0 ( 0.0%)    已全部归桶

  正样本合计: 7,193 (14.1%)
  负样本合计: 43,707 (85.9%)
```

50K 样本的分布验证了最初的判断：**匹配确实很难**。60% 的候选根本不能看（easy_negative），只有 14% 是可用的正样本。

---

## 7. 从 Judge 到训练数据

### 7.1 5-Bucket 分类策略

基于 Judge 的 overall / would_use / confidence 三个信号分层：

| bucket | 规则 | 核心逻辑 |
|--------|------|------|
| strong_positive | overall>=4.5, would_use=True, confidence>=0.8 | Judge 高信心判为"真匹配" |
| fallback_positive | overall>=3.5, would_use=True (仅无 strong 的 query) | 凑合能用的底线，覆盖无强正 query |
| ambiguous_positive | overall>=4.0, would_use=False | Judge 给高分但认为不可实操（训练中作中位样本） |
| hard_negative | 2.0 <= overall < 4.0, would_use=False | 像但不对——最容易迷惑模型的样本 |
| easy_negative | overall < 2.0 | 完全不相关，一眼假 |

**分类是 context-independent 的**：每个样本只看自己的分数，不看同 query 下其他候选。这样分类规则简单稳定、不依赖跨样本比较。

**hard_negative 的区间选择 (2.0-4.0)**：这是经历了多次数据摸底后定下来的。overall 在 2.0-3.5 的样本——人物基本对、但动作或情绪错了——是最容易迷惑 reranker 的。把区间扩到 4.0 能把"would_use=False 但 score 不低"的临界样本也包进来，让 DPO 有更丰富的对比信号。

### 7.2 三层数据模型

```
samples.jsonl ←── 底座：50,900 条 (query, candidate) 完整评估
   │
   ├──→ pairs.jsonl ←── DPO 偏好对：23,741 条 (query, chosen, rejected)
   │
   └──→ groups.jsonl ←── Listwise AWR 候选组：1,669 组 (每组 15-19 候选)
```

**samples 是唯一真相源**：pairs 和 groups 都从它派生，不做额外 Judge 评估、不引入新数据源。这样保证训练数据的可追溯性——每条 pair/group 都能回溯到具体的 sample。

### 7.3 DPO Pairs 采样设计

| 配对 | chosen | rejected | 每 chosen 配几个 rejected | 实际 pairs |
|------|:--:|:--:|:--:|:--:|
| strong vs hard | strong | hard | 4 | 11,706 |
| strong vs easy | strong | easy | 3 | 8,496 |
| fallback vs hard | fallback | hard | 4 | 3,539 |

采样逻辑：
- **只用 strong 和 fallback 做 chosen**，ambiguous 不参与 DPO（would_use=False 作为 chosen 信号矛盾）
- **hard 是 rejected 主力**：它们"像但不对"，是最有价值的对比信号
- **easy 配少一点**：容易区分，不需要大量样本
- **strong 优先 vs fallback 回退**：fallback vs hard 只在无 strong 的 query 生成
- **总计 23,741 pairs**

### 7.4 Listwise AWR Groups 设计

每组 (1 个 query) 的候选构成：

```
strong_positive      1-3 个
ambiguous_positive   0-1 个
fallback_positive    0-1 个
hard_negative        3-4 个
easy_negative        2-3 个
─────────────────────────────
总计                 15-19 候选
```

**Bucket-aware Reward**：
```python
# 正样本 = overall (4.0~5.0，正数)
if bucket in (strong, fallback, ambiguous):
    reward = overall

# 负样本 = overall - 4.0 (hard → -2.0~0.0, easy → 约 -3.0~-2.0)
else:
    reward = overall - 4.0
```

这样在正负两区间之间制造约 3.5 分的断层——hard 最高分（3.99→-0.01）与最低 fallback（3.5→3.5）之间差了约 3.5 分，正负样本 reward 号分离。

Per-group z-score 归一化由训练框架在 forward 阶段完成，groups.jsonl 只存原始 reward。

**边界处理**：
| 情况 | 处理 |
|------|------|
| query 候选 < 5 | 跳过 |
| 无 overall >= 4.0 候选 | 跳过 |
| 候选 > 25 | 截断到 25，优先截 easy |

### 7.5 Train/Val/Test 切分

```
策略: 按剧切分（同剧 query 不跨 split，避免泄漏）

val:   固定 6 部旧剧 (judge/tasks.txt)
train: 90 部 (100部 × 90%)
test:  10 部

结果:
  train: 21,596 pairs / 1,521 groups / 46,425 samples
  val:     578 pairs /    25 groups /    849 samples
  test:  1,567 pairs /   123 groups /  3,626 samples
```

---

## 8. 系统架构与代码组织

### 8.1 模块关系

```
judge/
  ├── match_judge.py                       ← 单条评估核心引擎
  │     ├─ MatchJudge.evaluate_single()    → 单次评估
  │     ├─ MatchJudge.evaluate_with_consistency() → 带一致性检查
  │     ├─ 5 维度 system prompt (118-217行)
  │     ├─ 视频抽帧 + base64 注入
  │     └─ 输出解析 + 格式重试
  │
  ├── batch_judge.py                       ← 批量编排 + 报告生成
  │     ├─ 按剧遍历: 加载文案 → 找目标集数 → 对所有片段跑 Judge
  │     ├─ 断点续评 (已有 run_id 自动跳过)
  │     └─ 自动生成 report.json + summary.md
  │
  ├── compare_variants_with_canonical.py   ← 变体效果对比
  │     ├─ 读 Canonical Judge 矩阵
  │     ├─ 各 variant best_match 查表比较
  │     └─ 输出汇总 md + 每剧/每变体详细 json
  │
  ├── rl_data_builder.py                   ← 训练数据构造 (v3)
  │     ├─ 5-bucket 分类
  │     ├─ samples.jsonl 底座
  │     ├─ pairs.jsonl 导出 (DPO采样)
  │     └─ groups.jsonl 导出 (AWR采样)
  │
  ├── split_dataset.py                     ← 按剧切分 train/val/test
  └── doc/                                 ← Schema 设计文档
```

### 8.2 产出路径全景

```
judge/
  outputs/
    canonical/{dj}/{run_id}/
      results.jsonl            ← 全量 Judge 评分矩阵（50K+ 条）
      report.json              ← 汇总统计
      summary.md               ← 可读报告
      config.json              ← 运行配置

    {variant}/{dj}/{run_id}/
      results.jsonl            ← 各变体的 best_match 评估

    compare_*.md               ← 版本对比报告

  training_data/v3/
    {dj}/{run_id}/
      samples.jsonl            ← 样本底座
      stats.json               ← 统计摘要
      dropped.jsonl            ← 被过滤样本及原因

    splits/
      train_samples.jsonl      ← 切分后的训练样本
      val_samples.jsonl
      test_samples.jsonl
      train_pairs.jsonl        ← DPO 训练对
      val_pairs.jsonl
      test_pairs.jsonl
      train_groups.jsonl       ← Listwise AWR 组
      val_groups.jsonl
      test_groups.jsonl
      split_manifest.json      ← 切分元信息
```

---

## 9. 关键设计决策回顾

| 决策 | 为什么 | 影响 |
|------|--------|------|
| LLM-as-Judge，不靠文本描述中转 | 文本描述丢视觉细节（微表情/光影/镜头运动） | 评估基于视频画面本身，文本描述仅作辅助参考 |
| 5 维度独立评分 + 加权 overall | 单一总分太粗糙，拆维才能知道"哪里对/哪里错" | 可分析每个维度的缺陷，指导检索改进方向 |
| 画面优先于文本描述 | LLM 可能偷懒只读文本描述不看画面 | System prompt 强制"以画面为准" |
| Canonical Judge 一次跑，多方复用 | 每个 variant 重跑 Judge 浪费成本 + 引入分数波动 | 100 剧 Judge 只跑一次，新方案查表秒级对比 |
| fps=1.0 抽帧 | 3 秒片段 3 帧，Judge 需要的是"讲的是什么"，不需要逐帧分析 | API 传输量可控，足够做判断 |
| 5 次格式重试 + fail-fast | LLM 有时输出非标准 JSON 或缺字段 | 格式脏数据绝不流入下游 |
| 一致性检查 std≤0.5 | 非追求绝对零方差，只标记高不确定性样本 | 为后续 quality check 提供线索 |
| 按剧切分 train/val/test | 同剧 query 放不同 split 会导致数据泄漏 | 训练评估更保守，泛化结论更可信 |
| 正样本宁缺毋滥 (strong>=4.5) | DPO 需要明确偏好信号 | strong 仅 6.6%，但每个都是高质量 |
| fallback 阈值 3.5 而非 4.0 | 4.0 太严，会丢掉大量 covering 无正 query 的回退样本 | 全覆盖无强正的 query |

---

## 10. 从 V1 到 V6 的演进总结

```
V1 (smart_video):
  无 Judge 系统
  视频质量靠人看
  匹配效果无客观度量

V6 (smart_videoV6):
  LLM-as-Judge 5维度评估
  100剧 × 50K样本 Canonical Benchmark
  5-Bucket 自动分类
  samples → pairs → groups 训练数据管线
```

Judge 系统是 V6 最关键的增量之一。它从 0 开始搭建了一整套"评估→分出好坏→构造训练数据"的闭环，使得 Reranker 微调从此有了稳定、高质量、大规模的训练数据底座。
