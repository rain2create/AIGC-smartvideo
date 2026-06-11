# Retrieve Plan

## 背景

当前 `qwen_match` 这条线已经打通了：

1. `embedder.py`：本地 Qwen3-VL 文本 / 视频 embedding 封装
2. `index_builder.py`：视频侧 visual embedding 建索引
3. `retriever.py`：文案 query -> 视频 TopK 检索

当前还没有完全定死 query 组织方式。

真正需要继续验证的是：

1. 文案侧 query 到底该拼多少结构化信息
2. 多加字段之后，是不是一定更好
3. 哪种 query 版本更适合作为 V6 的主检索口径

核心要点：当前多模态链路已经能跑，但 query 还没有收敛。后面需要把“query 怎么写”单独当成一条实验线来做，而不是默认信息越多越好。

## 我建议的 query 分层

| 版本 | 内容 | 我怎么看 |
|---|---|---|
| Q1 | 只用 `script_text` | 最干净 baseline |
| Q2 | `script_text + expanded_tags` | 一个比较稳的增强版 |
| Q3 | `script_text + character_types + action_keywords + environment + expanded_tags + interaction_type` | 我最推荐先重点测的版本 |
| Q4 | `Q3 + fuzzy_tags` 全量 | 可以试，但我不建议默认用它做主版本 |

核心要点：如果问“直接拿所有结构化信息拼一起会不会更好”，答案是不一定。Q3 通常比 Q4 更像合理主线，也就是优先保留视觉化字段，先去掉抽象字段。

## 对当前代码的直接判断

| 项目 | 结论 |
|---|---|
| 现在的 `build_query_text()` | 稍微有点偏“信息贪多” |
| 最值得先收掉的部分 | `fuzzy_tags` 里的 `audience_state`、`director_intent` |
| 要不要完全删结构化信息 | 不要 |
| 要不要完全只留 `script_text` | 可以做对照组，但不建议直接当唯一方案 |

核心要点：现在代码方向没错，但 query 组织还可以更“视觉检索友好”。不是要砍掉结构化信息，而是要筛字段。

## 当前建议

| 选项 | 建议 |
|---|---|
| 主版本 | 我建议改成 Q3：`script_text + 视觉核心字段` |
| 对照实验 | 同时留一个 Q1：只用 `script_text` |
| 暂时处理 | 先把 `fuzzy_tags` 从 query 里去掉 |
| 后续验证 | 用 Judge 比较 Q1 / Q2 / Q3 / Q4 的 Top1 / Top5 |

核心要点：V6 第一版不应该默认把所有字段都塞进 query。更稳的做法是先把视觉核心字段作为主版本，再拿更干净和更激进的版本做对照。

## 当前代码口径

| 项目 | 当前值 |
|---|---|
| `retriever.py` 默认 `query_version` | `Q1` |
| 当前 baseline | 只用 `script_text` 做 query embedding |
| 后续切到其他版本 | 通过 `--query-version Q2/Q3/Q4` 切换 |

核心要点：当前代码已经先切到 Q1，保证你现在跑出来的是最干净的 baseline。后面如果要回到 Q3，不需要再改代码结构，只要切参数即可。

## 建议的实验顺序

| 顺序 | 版本 | 目的 |
|---|---|---|
| 1 | Q1 | 建立最干净 baseline |
| 2 | Q3 | 验证“视觉结构化字段”是否能稳定提升 |
| 3 | Q2 | 看 `expanded_tags` 单独贡献有多大 |
| 4 | Q4 | 验证 `fuzzy_tags` 全量是否引入噪声 |

核心要点：先测 Q1 和 Q3 最划算。Q2 和 Q4 更适合作为补充实验，而不是一开始就作为主线版本。

## 实验结果建议观察项

| 指标 | 说明 |
|---|---|
| Top1 命中率 | Judge 看 top1 是否达到可用标准 |
| Top3 / Top5 命中率 | 看 query 是否把好片段召回到前几名 |
| best_match 排名命中率 | 看系统 top1 在 Judge 里的名次 |
| 候选分数分布 | 看 query 是否整体提升候选质量 |
| query 长度与稳定性 | 看字段变多后是否出现噪声和不稳定样本 |

核心要点：后面不要只看 Top1。Q1/Q2/Q3/Q4 的比较，最好同时看 Top1、Top5 和 Judge 排名分布，这样才能知道是召回变好了，还是只是个别样本碰巧变好。
