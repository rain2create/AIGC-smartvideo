# Data Builder Plan

## 1. 目标与边界

| 项目 | 内容 |
|---|---|
| 主目标 | 基于 Canonical Judge 标注结果构建高质量、可追溯、可复用的 RL 微调训练数据 |
| 支持方向 | DPO、PPO、Listwise AWR |
| 当前范围 | samples（底座）+ pairs（DPO 偏好对）+ groups（Listwise AWR 候选组）|
| 不做 | 不直接开训；不在 builder 阶段引入训练器；不修改 Judge 主流程 |

核心要点：先把数据层做对，再去试不同训练方法。

## 2. 输入与输出

| 类型 | 路径 | 作用 |
|---|---|---|
| Judge 原始结果 | `judge/outputs/canonical/{dj}/{run_id}/results.jsonl` | 主标签来源 |
| Judge 配置 | `judge/outputs/canonical/{dj}/{run_id}/config.json` | 提供 run 元数据 |
| 文案结构化 | `text_video_match/cache/{dj}/*_structured.json` | 提供 script_text、visual_query、系统 top_matches |
| 视频结构化 | `text_video_match/cache/{dj}/structured_description.json` | 提供候选片段的结构化描述与路径 |

| 输出文件 | 作用 | 下游用途 |
|---|---|---|
| `samples.jsonl` | 每行一个 `(query, candidate)` 标准样本 | pointwise / 质检 / 追溯 |
| `dropped.jsonl` | 被过滤掉的样本及原因 | 质检 |
| `stats.json` | 汇总统计 | 回归对比 |
| `pairs.jsonl` | DPO 偏好对 | DPO / ORPO / pairwise |
| `groups.jsonl` | 每条 query 的候选组 | Listwise AWR / PPO / listwise |

核心要点：`samples.jsonl` 是主资产，`pairs` 和 `groups` 都必须从它派生。

## 3. 样本分层规则（Phase C 定稿）

| bucket | 规则 | 说明 |
|---|---|---|
| `strong_positive` | `overall >= 4.5` 且 `would_use=true` 且 `confidence>=0.8` | 首选正样本，DPO chosen 主力 |
| `fallback_positive` | 该 query 无 strong_positive 时，`overall >= 4.0` 且 `would_use=true` | 覆盖无强正 query |
| `ambiguous_positive` | `overall >= 4.0` 且 `would_use=false` | Judge 高分但判为不可实操；DPO 弱偏好 chosen / Listwise AWR 中等分组 |
| `hard_negative` | `2.0 <= overall <= 3.5` | 原 hard_negative + dimension_negative + unusable 2.0-3.5 区间；DPO rejected 主力 |
| `easy_negative` | `overall < 2.0` | 边界稳定用；训练时强制采样，不全量进 |

> 注：`dimension_negative` 和 `unusable` 已在 Phase C 取消，全部拆解并入以上 5 个 bucket。

## 4. Tie 与切分规则

| 规则 | 默认值 | 作用 |
|---|---|---|
| 同分判定 | `overall` 完全相同 | 标记为 `tie_group`（仅统计用，不过滤配对） |
| 数据切分 | 第一优先按剧切 train / val / test | 避免 query 泄漏 |

## 5. Phase B 定稿的 v3 Schema 要点

- samples.jsonl v3：v2 最小增量升级，详见 `samples_jsonl_schema.md`
- 关键新增字段：`labels.preference_eligible`、`labels.sample_weight`
- retrieval 精简：`system_best_match` 对象拆分为两个标量字段，`system_top_matches` 条目精简
- query/candidate 删除训练无关字段（audio、duration 等），但 visual_query 和 structured_description 保留完整原始对象

## 6. 开发顺序

| 阶段 | 状态 | 产出 |
|------|------|------|
| Phase A: 数据摸底 | ✅ 完成 | 6 剧 stats 汇总 |
| Phase B: Schema 设计 | ✅ 完成 | `samples_jsonl_schema.md` v3 定稿 |
| Phase C: 正负样本策略 | ✅ 完成 | bucket 定义 + DPO/Listwise AWR 采样配比，见 todo.md §6 |
| Phase D: Builder 重构 | ✅ 完成 | `rl_data_builder.py` v3，批量 + pairs + groups，samples.v3 schema |
| Phase E: 导出与切分 | ✅ 完成 | 23,741 pairs + 1,669 groups + train/val/test 切分（`split_dataset.py`）|
| Phase F: 质量验证 | ✅ 跳过 | 由训练阶段反馈驱动 |
