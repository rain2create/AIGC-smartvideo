# Judge 模块开发 TODO

> 最后更新: 2026-04-24

## 已完成

- [x] **judge.md** — 设计文档（5 维度评分、Prompt、DPO 数据格式、演进计划）
- [x] **match_judge.py** — 单条评估核心模块
  - 视频片段直接 base64 传 Gemini（非抽帧）
  - 5 维度打分 + 详细推理 + few-shot 示例
  - temperature=0，单次调用
  - 工具函数：resolve_video_path、load_segment_info
- [x] **batch_judge.py** — 批量评估脚本
  - 已改为全量标注模式：每条文案 vs 目标集内所有片段
  - 评估完自动生成系统质量报告（best_match 排名、分数分布、格式化 summary）
  - 支持断点续评、`--report-only`、`--max-segments`
- [x] **用新版 indexer 跑一部剧，验证 episode 字段 + batch_judge 全量标注**
  - 已用 `chongfan80` 新数据验证：`structured_description.json` 含 `episode`，`*_structured.json` 含 `episode_target`
  - `batch_judge` 已按目标集过滤评估，当前样例为每条文案仅评目标集内 38 个片段，而非全量片段
- [x] **rl_data_builder.py** — Reranker / RL 训练数据底座
  - 已完成 `samples.jsonl` / `dropped.jsonl` / `stats.json` 导出
  - 已完成 `query + candidate + judge + retrieval` 对齐
  - 已完成基础 `bucket` 标签：`strong_positive`、`fallback_positive`、`hard_negative`、`easy_negative`、`dimension_negative`、`unusable`
  - 已用 `chongfan80` 跑通 dry-run，当前样例共 228 条样本、0 条 dropped

## 进行中

## 待开发

- [ ] **pairs / groups 规则与导出**
  - 基于 `samples.jsonl` 的 `bucket` 标签派生训练视图，而不是直接重做清洗
  - `pairs`：服务 DPO / pairwise ranking，后续再定 `positive vs hard_negative / easy_negative` 的采样配比
  - `groups`：服务 GRPO / PPO / listwise，后续再定每条 query 保留哪些 candidate 进入训练组
  - 当前先不定死规则，等多跑几部剧后再统一看样本分布和正负比例

- [ ] **rl_data_builder.py 新模态 emb 数据适配**
  - 当前底座主要面向 V4 的 benchmark / 样本对齐口径
  - 后续如果 reranker 输入升级为 `文案 emb + 视频 emb` 或其他多模态特征，需要补一层 feature/adapter 逻辑
  - 重点包括：新输入字段映射、不同候选池口径兼容、旧 benchmark 与新候选集合的补标策略
  - 当前先不提前绑定输入格式，等未来 reranker 方案和数据形态稳定后再做

- [ ] **Few-shot 示例替换**
  - 当前 few-shot 是 placeholder（基于真实短剧场景手写）
  - 需要跑几轮后用模型真实打分的标定数据替换
  - 当前先保留，等多部剧的 Judge 结果和样本分布更稳定后再替换

