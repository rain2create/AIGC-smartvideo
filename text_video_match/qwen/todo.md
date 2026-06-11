# Qwen Match TODO

## 目标

在 `smart_videoV6/text_video_match/qwen_match/` 下落一条独立的 Qwen3-VL 多模态匹配开发线。

当前阶段只做：

1. 视频片段 -> 视觉 embedding 索引
2. 文案文本 -> 查询 embedding
3. 跨模态 TopK 检索
4. 用现有 Judge 做 benchmark

当前阶段不做：

1. 服务化部署
2. 旧 Stage2/Stage3 文本重排叠加
3. Reranker
4. RL 训练

## 开发顺序

| 顺序 | 要做什么 | 我的建议 |
|---|---|---|
| 1 | 先定部署形态 | 本地离线推理，先不做服务化 |
| 2 | 先定编码策略 | 视频侧优先走原生 `video` 输入，不要先手工 3 帧 mean-pooling |
| 3 | 先定初始参数 | `fps=3.0`、`max_frames=30`、`normalize=True`，输出先存 `float16` |
| 4 | 先定 query 形式 | `script_text + visual_query` 轻量展开一起编码，不只喂一句裸文案 |
| 5 | 先定第一版 benchmark | 只做跨模态 TopK 检索，先不叠旧 Stage2/Stage3 文本重排 |
| 6 | 先处理缓存风险 | 不要直接复用现在拷贝过来的 V6 cache，至少要重跑 S1/S2 或修正路径 |
| 7 | 后续是否上 reranker | 等裸检索 benchmark 出来再决定 |

核心要点：V6 先做一个干净版本，避免一开始就把“多模态召回 + 旧文本重排 + reranker”一起堆进去，导致问题难以归因。

## Phase 划分

| Phase | 目标 | 产物 |
|---|---|---|
| V6-A | 本地 Qwen3-VL 视频索引 | `embeddings_vl.pkl` |
| V6-B | 文案文本 -> 视频 emb 裸检索 | 新版 `match_result.top_matches` |
| V6-C | 用现有 Judge 跑 benchmark | `report.json`、`summary.txt`、对比结果 |
| V6-D | 如果 Top5 高、Top1 低，再考虑 reranker | 下一阶段方案 |

核心要点：先把“能不能召回对”这件事跑明白，再决定有没有必要上 reranker。

## 目录规划

| 路径 | 用途 |
|---|---|
| `text_video_match/qwen_match/todo.md` | 当前开发任务和阶段说明 |
| `text_video_match/qwen_match/embedder.py` | Qwen3-VL 本地加载与 `encode_text` / `encode_video` 封装 |
| `text_video_match/qwen_match/index_builder.py` | 视频侧 embedding 索引构建逻辑 |
| `text_video_match/qwen_match/query_builder.py` | 文案 query 展开与编码逻辑 |
| `text_video_match/qwen_match/retriever.py` | cosine 检索、TopK 召回、结果格式化 |
| `text_video_match/cache/{dj}/embeddings_vl.pkl` | 视频片段视觉 embedding 索引 |

核心要点：`qwen_match` 这条线和旧的 `skill_struct_info_emb.py / skill_match_and_rank.py` 先并存开发，等 benchmark 稳定后再决定是否替换主链路。

## 当前已确认

| 项目 | 状态 |
|---|---|
| 本地模型目录 | `/nlp_liuhuan_data03/search_public/xby/models/Qwen3-VL-Embedding-8B` |
| 模型形态 | 可本地 `transformers` / `sentence-transformers` 加载 |
| 机器显卡 | 远程服务器 `2 x L20 46GB`，足够先跑 8B |
| 当前 V6 现状 | 仍然是文本描述 -> 文本 embedding 的旧链路 |
| 当前视频片段源 | `structured_description.json` 里已有 `segment_file_path` 可直接编码 |

核心要点：环境上已经满足“先做本地 8B 多模态索引”的基本条件，当前缺的是代码实现，不是算力。

## 风险点

| 风险 | 说明 | 当前处理建议 |
|---|---|---|
| V6 cache 里路径仍指向 V4 | 复制目录后，历史 `segment_file_path` 仍可能落在 `smart_videoV4` | 重跑 S1，或统一修正路径 |
| Python 默认环境缺少 torch | 当前系统 `python3` 不能直接跑多模态模型 | 固定到专用 conda 环境后再开发执行 |
| 8B 视频输入显存开销大 | 多帧输入比文本大很多 | 当前先试 `fps=3.0` + `max_frames=30`，若显存或吞吐不稳再回调 |
| benchmark 结果不稳定 | 初版 query 展开和抽帧参数可能影响较大 | 先固定一个基线口径，不频繁改变量 |

核心要点：最大风险不是模型本身，而是“环境不统一”和“路径/缓存沿用旧版本”，这两件事如果不先收干净，后面排障成本会很高。

## 下一步

| 顺序 | 要做什么 | 产出 |
|---|---|---|
| 1 | 新建 `embedder.py` | 本地 Qwen3-VL 编码封装 |
| 2 | 新建 `index_builder.py` | 生成 `embeddings_vl.pkl` |
| 3 | 新建 `retriever.py` | query vs video cosine TopK |
| 4 | 再决定如何接入旧 `run.py` / `skill_match_and_rank.py` | 是否并行链路或替换主链路 |

核心要点：第一步不是改主流程，而是先把 `qwen_match` 子模块自己跑通。
