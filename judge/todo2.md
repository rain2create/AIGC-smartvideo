# B4 实验：Late Fusion 双路检索

> 最后更新: 2026-05-11
> 目标：找到最优 query 版本，在 B4 框架下跑通 6 部剧，加入 judge 对比

## 背景

b1 (Q3+B1) 目前效果最好（R@10=68.2%）。b1 的视频侧用了"视觉+文本 sidecar"来构建单个 embedding，但检索时仍是单路 cosine。

B4 的方案是**双路独立检索 + Late Fusion**：视觉和文本各建各的 embedding，各自检索 top-N，最后 RRF 合并。

## 方案设计

### 视频侧：两个独立 embedding

| 路 | embedding 内容 | 来源 |
|---|---|---|
| `visual_emb` | 纯视觉帧 → Qwen3-VL embedding | 已有（V1 pipeline） |
| `text_emb` | structured_description 文本 → text embedding | S1 的 `structured_description.json` |

每路各自建独立的 pkl 索引。text_emb 的输入是 segment 的 `structured_description` 中关键字段拼成的文本：
```
characters: [...]
actions: [...]
environment: ...
interaction_type: [...]
functional_intent: [...]
emotion_intensity: ...
```

### Query 侧：拆两路

| 路 | query 内容 | 目标 embedding |
|---|---|---|
| `visual_query` | 文案段 visual_query（角色、场景、动作） | `visual_emb` |
| `text_query` | 文案段 structured fields：characters + actions + environment + interaction + emotion + intent | `text_emb` |

### 检索：Late Fusion (RRF)

```
视觉路:  visual_query → visual_emb → top-N candidates
文本路:  text_query   → text_emb   → top-N candidates

RRF 合并:  score(d) = 1/(k + rank_vis(d)) + 1/(k + rank_txt(d))
          k = 60 (经典值)
```

不用加权 fusion 的理由：α/β 权重调不好会淹没弱势模态，RRF 每路独立贡献，零调参。

### 变体命名

| 变体名 | query_version | video_index_version | 说明 |
|--------|:---:|:---:|------|
| qwen_b4_q1 | Q1 | B4 | 视觉 query: Q1; 文本 query: structured fields |
| qwen_b4_q2 | Q2 | B4 | 同上，Q2 query |
| qwen_b4_q3 | Q3 | B4 | 同上，Q3 query |

B4 是新的 video_index_version，代表"双路独立 embedding + Late Fusion"。

## 实现计划

### 1. 新建 `qwen_match/embedder_b4.py`

- Video 侧：给每个 segment 生成 `text_emb`（基于 structured_description 文本）
- 存储为 `text_embeddings_b4.pkl` — `{segment_id: np.ndarray}`
- 可选：复用 Qwen3-VL 的 text encoder 部分，或用单独的 text embedding 模型

### 2. 改造 `qwen_match/retriever.py` 或新建 `retriever_b4.py`

- 加载 `visual_emb` (已有 pkl) + `text_emb` (新 pkl)
- 两路独立检索，各自返回 top-N
- RRF 合并，排序输出 top_matches

### 3. 集成到 `run_mul_qwen.py`

- 新增 `video_index_version="B4"` 支持
- S2 阶段：除 visual embedding 外，额外生成 text_emb
- S4 阶段：走 B4 retriever

### 4. 验证

- 先跑一部剧（如 luxiansheng），对比 b4_q1/q2/q3 vs 现有变体
- 选最优 query 版本后，全量跑 6 部剧
- 加入 judge 对比报告

## 待确定

- [ ] text_emb 用 Qwen3-VL 的 text encoder 还是单独模型（如 bge-m3）？
- [ ] RRF 的 N 取多少？（建议 50，保留足够覆盖）
- [ ] text_emb 需要 S2 重新跑还是可以离线生成？（依赖 `structured_description.json`，S1 已有）
