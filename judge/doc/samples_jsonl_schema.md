# samples.jsonl Schema v3

> 从 v2 最小增量升级，不裁剪已有字段，仅新增 labels 字段 + 精简 retrieval 冗余。

## 1. 目标

| 项目 | 值 |
|---|---|
| 样本粒度 | 一行代表一个 `(query, candidate)` Judge 样本 |
| 主要用途 | 样本追溯、质检、pointwise/listwise 输入、后续派生 pair/group |
| 版本 | v3，builder 对应 `samples.v3` |

核心要点：`samples.jsonl` 是训练数据底座，后续所有 pair/group 导出都必须从它派生。

## 2. 顶层字段

| 字段路径 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `sample_id` | `string` | 是 | 单条样本唯一 ID，即 `results.jsonl.evaluation_id` |
| `query_id` | `string` | 是 | 格式 `{dj}_seg{segment_sequence}` |
| `candidate_id` | `string` | 是 | 格式 `ep{episode}_seg{segment_id}` |
| `source` | `object` | 是 | 运行与数据源信息 |
| `query` | `object` | 是 | 文案侧字段 |
| `candidate` | `object` | 是 | 候选片段字段 |
| `judge` | `object` | 是 | Judge 标签 |
| `retrieval` | `object` | 是 | 系统原排序信息 |
| `labels` | `object` | 是 | 样本分类标签与采样元信息 |

## 3. `source` 字段

| 字段路径 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `source.dj` | `string` | 是 | 剧名 |
| `source.run_id` | `string` | 是 | Judge 运行批次 |
| `source.structured_json` | `string` | 是 | 使用的 `*_structured.json` 文件名 |
| `source.judge_model` | `string` | 是 | Judge 模型名 |
| `source.judge_version` | `string` | 是 | Judge prompt/version |
| `source.judge_temperature` | `number` | 否 | Judge 采样温度 |
| `source.judge_timestamp` | `string` | 否 | 单条样本评估时间 |
| `source.dataset_version` | `string` | 是 | 数据集版本 |
| `source.builder_version` | `string` | 是 | builder 版本，当前 `samples.v3` |

## 4. `query` 字段

| 字段路径 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `query.segment_sequence` | `int` | 是 | 文案段序号 |
| `query.script_text` | `string` | 是 | 文案原文 |
| `query.episode_target` | `int` | 否 | 目标集数 |
| `query.visual_query` | `object` | 否 | 文案结构化查询（保留完整原始对象，不做字段裁剪） |

v2→v3 删除：`sentence_idx`、`episode_confidence`、`audio_path`、`audio_duration`（训练用不到）。

## 5. `candidate` 字段

| 字段路径 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `candidate.segment_id` | `int` | 是 | 片段 ID |
| `candidate.episode` | `int` | 否 | 所属集数 |
| `candidate.structured_description` | `object` | 否 | 片段结构化描述（保留完整原始对象，不做字段裁剪） |
| `candidate.video_segment_path` | `string` | 否 | 视频片段路径（多模态训练输入） |

v2→v3 删除：`duration`、`start_time`、`end_time`、`segment_file_path`（训练用不到）。

## 6. `judge` 字段

| 字段路径 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `judge.scores` | `object` | 是 | 5 维分数 + `overall` |
| `judge.reasoning` | `object` | 是 | 5 维解释 + `overall`（质检保留） |
| `judge.confidence` | `number` | 是 | Judge 置信度 |
| `judge.would_use` | `bool` | 是 | 是否愿意实际剪辑使用 |
| `judge.match_highlights` | `array[string]` | 否 | 亮点摘要 |
| `judge.match_issues` | `array[string]` | 否 | 问题摘要 |

v2→v3：无变化，完整保留 reasoning 用于人工质检。

## 7. `retrieval` 字段

| 字段路径 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `retrieval.in_system_top_matches` | `bool` | 是 | 当前候选是否在系统 top-20 中 |
| `retrieval.system_rank` | `int/null` | 是 | 在系统 top-20 中的 1-based 排名 |
| `retrieval.system_similarity` | `number/null` | 是 | 在系统 top-20 中的 similarity |
| `retrieval.is_system_best_match` | `bool` | 是 | 当前候选是否等于系统 top1 |
| `retrieval.system_best_match_segment_id` | `int/null` | 是 | 系统 top1 的 segment_id |
| `retrieval.system_best_match_overall` | `number/null` | 是 | 系统 top1 的 Judge overall 分 |
| `retrieval.system_top_matches` | `array[object]` | 否 | 精简版系统 top-20，每条仅 `{segment_id, episode, similarity}` |

v2→v3 变化：
- `system_best_match` 完整对象拆分精简为 `system_best_match_segment_id` + `system_best_match_overall`
- `system_top_matches` 条目精简为 `{segment_id, episode, similarity}`
- 删除 `match_breakdown`

## 8. `labels` 字段

| 字段路径 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `labels.sample_bucket` | `string` | 是 | `strong_positive` / `fallback_positive` / `hard_negative` / `easy_negative` / `dimension_negative` / `unusable` |
| `labels.is_positive` | `bool` | 是 | 是否为正样本（strong_positive 或 fallback_positive） |
| `labels.is_negative` | `bool` | 是 | 是否为负样本（hard/easy/dimension_negative） |
| `labels.tie_group` | `string/null` | 是 | 同 query 下 overall 相同的样本组名，格式 `{query_id}_overall_{score}` |
| `labels.preference_eligible` | `bool` | 是 | 是否可参与 DPO pair 构造（Phase C 根据 tie/confidence 判定） |
| `labels.sample_weight` | `float` | 是 | 采样权重，默认 1.0（Phase C 用于 easy_negative 降权等） |

v2→v3 变化：
- `is_positive_candidate` → `is_positive`
- `is_negative_candidate` → `is_negative`
- 新增 `preference_eligible`（默认 true，Phase C 根据规则覆写）
- 新增 `sample_weight`（默认 1.0，Phase C 根据采样策略覆写）
