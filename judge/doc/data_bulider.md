# Data Builder Plan

## 1. 目标与边界

| 项目 | 内容 |
|---|---|
| 主目标 | 基于 Canonical Judge 标注结果构建高质量、可追溯、可复用的 RL 微调训练数据 |
| 支持方向 | DPO、PPO、GRPO |
| 当前范围 | samples（底座）+ pairs（DPO 偏好对）+ groups（GRPO 候选组）|
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
| `groups.jsonl` | 每条 query 的候选组 | GRPO / PPO / listwise |

核心要点：`samples.jsonl` 是主资产，`pairs` 和 `groups` 都必须从它派生。

## 3. 样本分层规则 (Phase C 会重新审视)

| bucket | 规则 | 说明 |
|---|---|---|
| `strong_positive` | `overall >= 4.5` 且 `would_use=true` 且 `confidence>=0.8` | 首选正样本 |
| `fallback_positive` | 若该 query 无强正样本，则允许 `overall >= 4.0` 且 `would_use=true` | 防止整条 query 无正样本 |
| `hard_negative` | `2.5 <= overall <= 3.5`，且至少一个维度 `>=4` 或来自系统 `top_matches` | 最有训练价值 |
| `easy_negative` | `overall <= 2.0` | 用于稳定边界 |
| `dimension_negative` | 存在明确单维错配 | 打磨特定弱项 |
| `unusable` | 不满足以上任一条件 | 不导出训练对 |

## 4. Tie 与切分规则

| 规则 | 默认值 | 作用 |
|---|---|---|
| 同分判定 | `overall` 完全相同 | 标记为 `tie_group` |
| 最小偏好差 | `preference_gap >= 1.0` | 防止训练信号过噪 |
| `5 vs 5` | 禁止构造成 pair | 避免伪偏好 |
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
| Phase C: 正负样本策略 | 待做 | 更新 bucket 定义 + 采样配比 |
| Phase D: Builder 重构 | 待做 | 新版 `rl_data_builder.py`，支持批量 + pairs + groups |
| Phase E: 导出与切分 | 待做 | pairs/groups 导出 + train/val/test |
| Phase F: 质量验证 | 待做 | stats 检查 + 人工抽样 |
