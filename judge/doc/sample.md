# Sample Labeling Guide

## 1. 这份文档是干什么的

| 项目 | 值 |
|---|---|
| 目标 | 解释 `samples.jsonl` 里正负样本是怎么打标签的 |
| 当前实现文件 | `judge/rl_data_builder.py` |
| 当前统计样例 | `chongfan80 / annotation_20260423_104822` |
| 当前阶段 | 只定义 `sample bucket`，还没开始导出 `pairs/groups` |

核心要点：这份文档不讲未来模型输入，只讲当前训练数据底座里“什么算正样本、什么算负样本、比例是多少”。

## 2. 总体思路

| 层级 | 当前做法 | 目的 |
|---|---|---|
| Query 级 | 先看这条文案下有没有 `strong_positive` | 决定要不要启用 `fallback_positive` |
| Sample 级 | 给每个 `(query, candidate)` 打一个 `sample_bucket` | 固定样本语义 |
| 统计级 | 汇总整体 bucket 分布、每条 query 的正负样本数 | 判断数据是否可用于后续 pair/group |

核心要点：不是看到单个候选就直接判正负，而是先结合整条 `query` 的候选池情况，再决定单条样本怎么标。

## 3. 判定顺序

| 顺序 | 规则 | 说明 |
|---|---|---|
| 1 | 先判断该 `query` 是否存在 `strong_positive` | 如果存在，就不启用 `fallback_positive` |
| 2 | 命中 `strong_positive` 就直接记为正样本 | 最高优先级 |
| 3 | 只有当该 `query` 没有强正样本时，才允许命中 `fallback_positive` | 这是“兜底正样本” |
| 4 | 再判断 `dimension_negative` | 优先抓“局部很像但关键点错”的难负样本 |
| 5 | 再判断 `hard_negative` | 抓中间分数段、容易错排的样本 |
| 6 | 再判断 `easy_negative` | 收明显低分错配样本 |
| 7 | 其余全部记为 `unusable` | 暂时不参与训练对构造 |

核心要点：这是一个“有顺序的覆盖规则”。同一条样本只会落入一个 bucket，不会同时属于多个类别。

## 4. 正样本定义

| bucket | 是否算正样本 | 当前规则 | 设计原因 |
|---|---|---|---|
| `strong_positive` | 是 | `overall >= 4.5` 且 `would_use=true` 且 `confidence>=0.8` | 保留最干净、最可信、Judge 明确愿意使用的正例 |
| `fallback_positive` | 是 | 仅当该 `query` 没有 `strong_positive` 时，允许 `overall >= 4.0` 且 `would_use=true` | 防止整条 query 完全没有正样本 |

核心要点：正样本不是“高分就行”，还要求 `would_use=true`。也就是说，Judge 不只是觉得它像，还觉得它真的值得剪进去。

## 5. 负样本定义

| bucket | 是否算负样本 | 当前规则 | 设计原因 |
|---|---|---|---|
| `dimension_negative` | 是 | `overall` 在中间段，`would_use=false`，且至少 1 个维度 `<=2`、至少 2 个维度 `>=4` | 典型“局部很像，但关键点错了”的难负样本 |
| `hard_negative` | 是 | `2.5 <= overall <= 3.5`，且存在高维度，或该候选本来就在系统 `top_matches` 里 | 这是 reranker 最该学的错排样本 |
| `easy_negative` | 是 | `overall <= 2.0` | 明显不匹配，用来稳定边界 |

核心要点：负样本分层的目的，是区分“特别差的样本”和“容易迷惑模型的样本”。后面做 pair 时，`hard_negative` 会比 `easy_negative` 更有训练价值。

## 6. 非训练样本定义

| bucket | 当前含义 | 是否参与后续训练对 |
|---|---|---|
| `unusable` | 不够好做正样本，也不够典型做负样本 | 默认不参与 |

核心要点：`unusable` 不是坏数据，而是“暂时不给训练器，避免把信号做脏”的样本池。

## 7. `chongfan80` 整体比例

| 项目 | 数量 | 占比 |
|---|---:|---:|
| 总样本数 | 228 | 100.0% |
| 正样本总数 | 9 | 3.9% |
| 负样本总数 | 191 | 83.8% |
| `unusable` | 28 | 12.3% |

核心要点：这轮数据是“正样本很少、负样本很多”的典型结构，和当前 benchmark 里整体分数偏低是一致的。

## 8. `chongfan80` 各 bucket 分布

| bucket | 数量 | 占比 | 当前解读 |
|---|---:|---:|---|
| `strong_positive` | 9 | 3.9% | 干净正样本，数量少但质量高 |
| `fallback_positive` | 0 | 0.0% | 当前这轮没触发 |
| `dimension_negative` | 7 | 3.1% | 难负样本，适合后续重点利用 |
| `hard_negative` | 15 | 6.6% | 中间分段负样本，适合做主负样本 |
| `easy_negative` | 169 | 74.1% | 数量最多，说明候选池里明显错配很多 |
| `unusable` | 28 | 12.3% | 暂时不进训练 |

核心要点：当前最大头是 `easy_negative`，这意味着如果后面做 pair 不加采样，训练会过度偏向“区分特别差的样本”，而不是学会区分难例。

## 9. `chongfan80` 的 query 级情况

| query | 正样本数 | 负样本数 | `unusable` | 是否有强正样本 | 最佳 `overall` |
|---|---:|---:|---:|---|---:|
| `chongfan80_seg1` | 1 | 35 | 2 | 是 | 4.55 |
| `chongfan80_seg2` | 4 | 31 | 3 | 是 | 5.00 |
| `chongfan80_seg3` | 1 | 33 | 4 | 是 | 4.75 |
| `chongfan80_seg4` | 2 | 26 | 10 | 是 | 4.55 |
| `chongfan80_seg5` | 0 | 35 | 3 | 否 | 3.80 |
| `chongfan80_seg6` | 1 | 31 | 6 | 是 | 4.50 |

核心要点：`seg5` 是这轮唯一没有正样本的 query，这不是规则问题，而是它本来就没有达到正样本阈值的候选。

## 10. 当前规则的直接含义

| 结论 | 解释 |
|---|---|
| 当前正样本很少 | 说明正样本阈值比较保守，这是有意设计 |
| 当前 `fallback_positive=0` | 说明有正样本的 query 都已经有强正，不需要兜底 |
| 当前 `easy_negative` 很多 | 说明后面导 pair 时必须做采样，不能全量混入 |
| 当前 `dimension_negative + hard_negative` 有一定规模 | 已经足够支撑后面构造更有价值的训练对 |

核心要点：从训练角度看，这版标签已经能用；真正需要你下一步决定的是，后面做 pair 时，各类负样本按什么比例进入训练。
