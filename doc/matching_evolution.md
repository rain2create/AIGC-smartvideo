# 文案-视频片段匹配 — 演进总结

> 从 V1 的"文本→文本余弦检索"到 V6 的"Qwen3-VL 多模态多路召回 + Reranker 精排"，匹配管线经历了从间接描述到直接视觉理解、从单路到多路融合、从召回即终点到两阶段精排的三次革命。

## 1. 核心命题

```
输入: 一句解说文案（如 "总裁一把将她按在墙上，眼神危险"）
      + 目标剧集的若干视频片段（每段 2-5 秒，scene-detect 切割）
输出: 最匹配那句文案的视频片段
```

这个任务难在三个地方：
- **跨模态**：文案是文本，视频是像素，语义鸿沟大
- **细粒度**：同一集内的视频片段之间差异很小（同一场景、同一角色、不同机位/动作微差）
- **生产级覆盖**：每部剧 15-40 句文案，每句对应 20-50 个待选片段，总评估量 ~1000 次/剧

---

## 2. 两代架构对照

```
V1: 文本→文本 间接匹配                    V6: 文本→视觉 直接理解
═════════════════════════                  ════════════════════════

视频片段                                   视频片段
  │                                         │
  ├─ Gemini 描述成文字                       ├─ Qwen3-VL-Embedding
  │  "总裁走向女主，眼神凌厉..."              │   直接看视频帧 → visual_emb
  │                                         │
  ├─ BGM/BERT API → text_emb                │
  │                                         │
  └─ 余弦 → 文案_emb vs 视频_emb             └─ 余弦 → 文案_emb vs visual_emb

模型: 外挂 API (10.163.166.76)             模型: 本地 Qwen3-VL-Embedding-8B
匹配空间: 文本→文本                          匹配空间: 跨模态 文本→视觉
精排: 无                                     精排: Qwen3-VL-Reranker-8B
覆盖: 前 20 秒, 4-6 段/剧                    覆盖: 全量文案, 15-40 段/剧
```

---

## 3. V1：纯文本匹配时代

### 3.1 为什么走"描述文本"的弯路

V1 时代的 LLM 没有视觉理解能力（或者说有，但没接入本地模型）。要让"视频片段"和"文案"做匹配，必须先走一步"翻译"：

```
视频片段 → Gemini (1fps 抽帧 → API 调用) → 结构化文字描述
         "characters: [总裁, 女主]
          actions: [走向, 壁咚]
          environment: 办公室
          emotion: 压迫感+暧昧"
```

然后把这段文字向量化（BGM/BERT embedding API），和文案的向量做余弦相似度检索。本质上做的是 **text-to-text 匹配**，不是 text-to-video 匹配。

### 3.2 四阶段匹配设计

V1 的 `skill_match_and_rank.py` (S4) 设计了四阶段流水线：

```
Stage 1: 语义召回 (TOP 20)
  文案 embedding vs 所有视频片段 embedding → 余弦 Top20

Stage 2: 视觉精排 (TOP 10)
  对 Top20 做细粒度加权: characters + actions + environment + dialogue 各维度加权相似度

Stage 3: 情感深排 (TOP 5)
  interaction_type + emotion_intensity + functional_intent 加权

Stage 4: LLM 精排 (TOP 1)
  调用大模型对 Top5 做最终选择
  ↑ 在实际使用中被禁用 —— LLM 精排开销大、结果波动大，不如 Stage3 直接取 Top1 稳定
```

四个阶段从粗到细逐层过滤：语义 → 视觉字段 → 情感 → 最终选取。这个设计思路是对的，问题是**每个阶段都在文本空间操作**，Gemini 描述过程中的信息损失从第一步就开始积累。

### 3.3 前 20 秒截断

V1 每部剧只处理前 20 秒解说词（4-6 句），超过 20 秒的直接丢弃。这是早期快速验证阶段的取舍——先跑通小规模链路，确认匹配逻辑可行后再扩展到全量。代价是训练数据极少（6 部剧仅 849 个样本），正样本占比仅 10.5%。

### 3.4 结果与瓶颈

V1 匹配结果在 6 部验证剧上的效果（Canonical Judge 评估）足以为方案可行性提供证据，但在以下维度受限：

- 文本 embedding 丢失视觉细节（Gemini 描述不出来"微表情"、"光影变化"、"镜头运动"）
- 外挂 API 有网络延迟和稳定性问题
- 前 20 秒覆盖太窄，正样本太少，不具备训练条件

---

## 4. V6：Qwen3-VL 时代的三次革命

### 4.1 革命一：从"读描述"到"看视频"

V6 的核心变化是**跨模态 embedding**：

```
V1: 视频 → Gemini描述 → 文本 → BERT API → text_vector
V6: 视频 → Qwen3-VL-Embedding → visual_vector (直接)
```

`Qwen3-VL-Embedding-8B` 是本地部署的多模态模型，直接把视频帧序列编码成向量。不再需要 Gemini 描述这个"中间翻译"环节，避免了信息损失。

```python
# embedder.py 核心配置
fps=3.0          # 每秒抽 3 帧
max_frames=30    # 最多 30 帧（支持最长 10 秒片段）
normalize=True   # L2 归一化
output_dtype=float16
```

**效果**：视频 embedding 直接捕获视觉特征——人物姿态、场景构图、光影变化、镜头运动——这些是文本描述永远无法完整传达的。

### 4.2 革命二：从"召回即终点"到"召回+精排"两阶段

V1 只有召回（检索），没有精排。V6 引入了 `Qwen3-VL-Reranker-8B`：

```
Stage 1: 召回 (embedder)
  query_text → Qwen3-VL-Embedding → query_vector
  所有视频片段的 visual_emb 依次余弦 → Top-K candidate list

Stage 2: 精排 (reranker)
  (query_text, video_frames) → Qwen3-VL-Reranker → cross-attention score
  对 Top-K 逐对打分 → 按 score 重排 → 最终 Top1
```

为什么需要精排：
- **Embedding 是双塔模型**：query 和 video 独立编码，交互只在最后的余弦乘积中发生，丢失细粒度语义
- **Reranker 是交叉编码**：query_text 和 video frames 一起走 cross-attention，每一帧可以与 query 的每个 token 交互
- 典型的"召回保证覆盖，精排保证精度"模式

```python
# reranker.py 核心配置
fps=1.0          # 更少的帧（精排精度要求更高，单帧信息量更大）
max_frames=8     # 最多 8 帧
# 打分方式: sigmoid(logit_yes - logit_no)
```

### 4.3 检索方案的探索与定稿

V1 只有一条检索路径。V6 时代尝试了四种检索方案，最终定稿在**双路加权融合（B3）**：视频侧用多帧视觉编码、文本侧用结构化语义编码，分数级加权融合。

#### 最终主方案：双路加权融合 (B3)

```
                     Query 侧                                  Document 侧
         ┌──────────────────────────┐            ┌──────────────────────────────┐
视觉路    │ 文案文本 + visual_query   │  余弦      │ 视频多帧 → Qwen3-VL-Embedding │
(α=0.8)  │ (Q3: script + 角色/动作   │ ────────→  │          → visual_emb         │
         │  /环境/互动标签)          │            │                              │
         └──────────────────────────┘            └──────────────────────────────┘

         ┌──────────────────────────┐            ┌──────────────────────────────┐
语义路    │ visual_query 结构化字段   │  余弦      │ S1结构化描述文本               │
(1-α=0.2)│ (角色/动作/环境/互动/情绪  │ ────────→  │ → Qwen3-VL Embedding text     │
         │  /意图)                  │            │          → text_emb           │
         └──────────────────────────┘            └──────────────────────────────┘

                    两路分数各自 min-max normalize 后:
                    combined = 0.8 × vis_norm + 0.2 × txt_norm
```

核心设计：
- **视觉路是主路径（α=0.8）**：视频片段的多帧画面直接经 Qwen3-VL-Embedding 编码成 visual_emb。Query 用 Q3 版本（文案纯文本 + visual_query 结构化字段），同样经 Embedding 的 text encoder 编码。两边的 embedding 在同一隐空间做余弦检索。
- **语义路是辅助路径（α=0.2）**：Query 侧从 visual_query 中提取纯结构化语义（角色、动作、环境、互动、情绪、意图），不含文案原文。Document 侧用 S1 产出的视频结构化描述文本（`characters: [...] actions: [...] environment: ...`）经 Embedding text encode。两路各自 min-max 归一化后加权求和。

为什么语义路的 query **不包含文案原文**：语义路的目的是结构层面的匹配——"这场戏是什么角色在什么环境做什么动作"。加文案原文会让语义路变成视觉路的弱副本，失去互补价值。

为什么 α=0.8 倾斜视觉：视觉帧包含的细粒度信息（微表情、光影、镜头运动）远多于结构化描述文本，实验验证视觉主导收敛更快、Hit@1 更高。

#### 探索过的其他方案

| 方案 | 做法 | 尝试结果 |
|------|------|------|
| **V1** (纯视觉) | 单路视觉 embedding，query Q1 纯文本 | 基线。忽略了视频的结构化语义信息 |
| **B1** (联合 embedding) | 每个视频片段 embedding = concat(visual_emb, text_sidecar_emb)，单路余弦检索 | Hit@1 在 6 剧验证中最高（39.3%），但依赖 S1 的 sidecar 文本作为一个整体的联合向量，微调 reranker 时多一个故障依赖 |
| **B2** (双路 RRF) | 与 B3 同样的双路架构，但融合方式用 RRF (k=60) 而非加权。两路完全平等投票，不偏袒任一模态 | RRF 零调参，但实验发现它无法体现视觉路的主路径地位。当两路分歧时（视觉说第 3 好、语义说第 8 好），RRF 取调和结果，不如 α=0.8 加权直接信任视觉路 |

### 4.4 Query 版本的演进

```
Q1: query = 文案纯文本
    例: "总裁一把将她按在墙上，眼神危险"

Q3: query = 文案纯文本 + visual_query 结构化字段
    例: "总裁一把将她按在墙上，眼神危险
         角色: 总裁, 女主 | 场景: 办公室 | 动作: 壁咚 | 氛围: 压迫感暧昧"
```

Q3 的 visual_query 字段来自 S3 (`skill_text_structured_description.py`) 的输出——S3 用 LLM 对每句文案做结构化分析，提取角色、场景、动作、氛围等。这些额外信息拉近了 query 和 visual embedding 之间的语义距离。

---

## 5. 变体效果对比

基于 Canonical Judge 对 6 部验证剧的全量评估（Judge 模型：Doubao-seed-2-0-lite，5 维度打分）：

| 变体 | Hit@1 | Score均值 | Score >=4.5 | 说明 |
|------|:-----:|:---------:|:-----------:|------|
| `qwen_b1_q3` | 39.3% | 4.03 | 42.9% | 综合最佳，但依赖 B1 sidecar |
| `qwen_q3_rerank_q1` | 39.3% | 3.96 | 42.9% | **微调底座**：零样本 Reranker 已追平 B1 Hit@1 |
| `qwen_b1_q3_rerank_q1` | 39.3% | 3.96 | 42.9% | 与上相同，多了 B1 依赖无增益 |

**选定 `qwen_q3_rerank_q1` 做微调底座的原因**：
- Hit@1 追平 B1 综合最佳，但不需要 B1 sidecar 文本（降低依赖风险）
- Reranker 是零样本结果——微调后有明确提升空间
- Embedder + Reranker 两阶段，可以只微调 Reranker（emb 冻住）

---

## 6. 全量数据覆盖

V6 在 100 部新剧上跑了 V1+Q3+Rerank Q1 全链路 Pipeline：

```
S1: video_episode_indexer → structured_description.json (视频结构化索引)
S3: text_structured_description → *_structured.json (文案拆分+visual_query)
S2(qwen): index_builder → embeddings_vl.pkl (Qwen3-VL 视觉 embedding)
S4(qwen): retriever V1+Q3 → top_matches (召回)
S4.5(qwen): reranker Q1 → 精排后的 top_matches
S5: postprocess → mp4 视频
S6: final.json
```

产出：
- 100 部剧 × 每剧 15-40 句文案 = 50,051 次 Judge 评估
- 训练样本 50,900 条（含 6 部旧剧）
- 7 部剧失败（5 部 LLM 格式不符 + 1 部视频编码 + 1 部文件损坏）

---

## 7. 关键设计决策

| 决策 | 为什么 | 效果 |
|------|--------|------|
| 选 Qwen3-VL-Embedding 替代 BGM/BERT API | 跨模态能力：视觉直接编码，跳过文本描述 | 信息损失减少，稳定性提升 |
| 引入 Reranker 做精排 | Embedding 双塔模型丢失细粒度交互 | 零样本追平 B1 最佳，微调后超越 |
| 检索定稿 B3 双路加权融合 | 视觉多帧做主路径(α=0.8)，结构化语义做辅助 | 互补：视觉捕获细粒度画面，语义捕获结构信息 |
| 选定 V1+Q3+Rerank Q1 为微调底座 | 单路检索更简洁，B1 依赖 sidecar 多一个故障点 | 成为后续所有训练实验的统一底座 |
| B2 RRF → B3 加权融合的转变 | RRF 零调参但视文歧义时取调和；加权直接信任视觉路 | α=0.8 视觉主导，语义路做补充不干扰 |
| fps=3.0(emb) vs fps=1.0(reranker) | emb 需要更多帧做全局表征，reranker 单帧信息量更大 | 平衡精度和显存 |
| Emb 和 Reranker 独立不共享 | 后续只微调 Reranker，emb 保持零样本泛化 | 降低训练复杂度，避免 emb 过拟合 |
| 前 20 秒 → 全量文案 | V1 演示级 → V6 生产级 | 正样本从 849 → 50,900，具备训练条件 |

---

## 8. 全链路产物流

```
S1: skill_video_episode_indexer.py
  → cache/{dj}/structured_description.json     (每集视频片段的结构化描述)

S3: skill_text_structured_description.py
  → cache/{dj}/{txt}_structured.json            (每句文案拆分 + episode_target + visual_query)

S2(qwen): qwen/index_builder.py
  → cache/{dj}/embeddings_vl.pkl               (每个片段的视觉 embedding 向量)

S4(qwen): qwen/retriever.py (V1/B1) / retriever_b2.py / retriever_b3.py
  → 写入 structured json 的 match_result.top_matches

S4.5(qwen): qwen/reranker.py
  → 重排 top_matches，写入 structured json

S5: skill_postprocess.py
  → finish/{variant}/{dj}_final.mp4

S6: final.json
  → cache/{dj}/{dj}_final.json
```

---

## 9. 代码资产索引

| 文件 | 职责 | 时代 |
|------|------|:----:|
| `text_video_match/skill_video_episode_indexer.py` | S1: 逐集视频结构化索引 | V1+ |
| `text_video_match/skill_struct_info_emb.py` | S2: BGM/BERT API embedding (V4) | V1 |
| `text_video_match/skill_text_structured_description.py` | S3: 文案拆分 + visual_query | V1+ |
| `text_video_match/skill_match_and_rank.py` | S4: 四阶段匹配 (V4) | V1 |
| `text_video_match/skill_postprocess.py` | S5: 视频合成 | V1+ |
| `text_video_match/qwen/embedder.py` | Qwen3-VL-Embedding 封装 | V6 |
| `text_video_match/qwen/index_builder.py` | 视频 visual embedding 建索引 | V6 |
| `text_video_match/qwen/retriever.py` | V1/B1 检索器 | V6 |
| `text_video_match/qwen/retriever_b2.py` | B2: 双路 RRF 融合检索器 | V6 |
| `text_video_match/qwen/retriever_b3.py` | B3: 双路加权融合检索器 | V6 |
| `text_video_match/qwen/reranker.py` | Qwen3-VL-Reranker 精排 | V6 |
| `run_mul_qwen.py` | V6 主流程编排 | V6 |
| `run_mul.py` | V4 主流程编排 | V1 |
