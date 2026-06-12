# qwen_q3_rerank_q1 vs qwen_q3_rerank_q1_dpo2000 对比报告

> 生成时间：2026-06-04T15:10:56  
> Canonical Judge 来源：`judge/outputs/canonical`  
> 对比版本：`qwen_q3_rerank_q1`, `qwen_q3_rerank_q1_dpo2000`

## 1. 口径说明

Canonical Judge 只评估一次：每条文案与目标集数内所有视频片段逐一打分。不同 variant 只提供自己的 `best_match` 和 `top_matches`，本报告把这些候选放回同一份 Judge 矩阵里查排名和分数。

这样比较时，候选池和 Judge 分数固定，不会混入每次重跑 LLM-as-Judge 的随机波动。

## 2. 指标说明

### 命中率 (Hit@K)
系统选出的 `best_match` 在 Judge 排序中位于前 K 名的文案占比。
衡量「系统最看好的那一个，Judge 觉得好不好」。

### 召回率 (Recall@K)
系统返回的前 K 个候选 (`top_matches`) 与 Judge 排名前 K 个候选的交集占比：
`Recall@K = |系统 topK ∩ Judge topK| / K`。
衡量「系统推荐的范围，能覆盖多少 Judge 认为好的」。
例如：系统 top5 里有 3 个也在 Judge top5 里 → R@5 = 3/5 = 60%。
注意：当某版本某 query 候选数不足 K 时，Recall@K 标记为 -（不适用）。

### 排序质量 (NDCG@K)
衡量系统排序与 Judge 理想排序的接近程度，范围 0~1。
公式：NDCG@K = DCG@K / IDCG@K，其中 DCG = Σ(score_i / log2(i+1))，位置越靠后权重越低。
NDCG@K = 1 代表系统前 K 个排序与 Judge 理想排序完全一致。

## 3. 推荐结论

### 3a. 命中导向排名

> 选中最重要：只看系统 best_match 在 Judge 排序中的位置。

**综合最佳：`qwen_q3_rerank_q1_dpo2000`。**

排序口径：按 Hit@1 → Hit@3 → Hit@5 → Hit@10 → Hit@20 字典序排列。

胜出原因：Hit@1最高（30.6%）；Hit@3最高（53.6%）；Hit@5最高（68.3%）；Hit@10最高（84.2%）；Hit@20最高（98.4%）。

| 排名 | Variant | Hit@1 | Hit@3 | Hit@5 | Hit@10 | Hit@20 | 判断 |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | `qwen_q3_rerank_q1_dpo2000` | 30.6% | 53.6% | 68.3% | 84.2% | 98.4% | 综合最佳 |
| 2 | `qwen_q3_rerank_q1` | 27.3% | 48.6% | 63.4% | 84.2% | 98.4% | Hit@1 低 3.3pp；Hit@3 低 4.9pp；Hit@5 低 4.9pp |


### 3b. 召回导向排名

> 覆盖最重要：只看系统 topK 与 Judge topK 的交集。

**综合最佳：`qwen_q3_rerank_q1_dpo2000`。**

排序口径：按 R@1 → R@3 → R@5 → R@10 → R@20 字典序排列。

胜出原因：R@1最高（30.6%）；R@3最高（42.6%）；R@5最高（53.3%）；R@10最高（72.1%）；R@20最高（87.1%）。

| 排名 | Variant | R@1 | R@3 | R@5 | R@10 | R@20 | 判断 |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | `qwen_q3_rerank_q1_dpo2000` | 30.6% | 42.6% | 53.3% | 72.1% | 87.1% | 综合最佳 |
| 2 | `qwen_q3_rerank_q1` | 27.3% | 39.0% | 52.1% | 70.8% | 87.0% | Hit@1 低 3.3pp；Hit@3 低 4.9pp；Hit@5 低 4.9pp |


### 3c. NDCG 导向排名

> 排序质量最重要：衡量系统排序与 Judge 理想排序的接近程度。

**综合最佳：`qwen_q3_rerank_q1_dpo2000`。**

排序口径：按 NDCG@1 → NDCG@3 → NDCG@5 → NDCG@10 → NDCG@20 字典序排列。

胜出原因：NDCG@1最高（0.82）；NDCG@3最高（0.83）；NDCG@5最高（0.85）；NDCG@10最高（0.89）；NDCG@20最高（0.94）。

| 排名 | Variant | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | NDCG@20 | 判断 |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | `qwen_q3_rerank_q1_dpo2000` | 0.82 | 0.83 | 0.85 | 0.89 | 0.94 | 综合最佳 |
| 2 | `qwen_q3_rerank_q1` | 0.80 | 0.81 | 0.84 | 0.88 | 0.93 | Hit@1 低 3.3pp；Hit@3 低 4.9pp；Hit@5 低 4.9pp |



## 4. 总体指标

| Variant | 文案数 | 覆盖率 | Hit@1 | Hit@3 | Hit@5 | Hit@10 | Hit@20 | R@1 | R@3 | R@5 | R@10 | R@20 | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | NDCG@20 | 系统均分 | 平均分差 | >=4.0 | >=4.5 | 均rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 183 | 100.0% | 27.3% | 48.6% | 63.4% | 84.2% | 98.4% | 27.3% | 39.0% | 52.1% | 70.8% | 87.0% | 0.80 | 0.81 | 0.84 | 0.88 | 0.93 | 3.16 | 0.74 | 36.1% | 25.1% | 5.38 |
| `qwen_q3_rerank_q1_dpo2000` | 183 | 100.0% | 30.6% | 53.6% | 68.3% | 84.2% | 98.4% | 30.6% | 42.6% | 53.3% | 72.1% | 87.1% | 0.82 | 0.83 | 0.85 | 0.89 | 0.94 | 3.24 | 0.66 | 39.3% | 29.0% | 5.16 |

## 5. Canonical 候选池阈值命中

- 候选池总评估数：3626
- 候选池 overall：2.11（1.00 ~ 5.00）

| 阈值 | 命中数 | 命中率 |
|---|---:|---:|
| >=4.0 | 413 / 3626 | 11.4% |
| >=4.5 | 213 / 3626 | 5.9% |

## 6. 系统选中片段 Judge 均分

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 3.16 | 1.00 ~ 5.00 | 3.57 | 2.77 | 3.56 | 3.17 | 2.87 |
| `qwen_q3_rerank_q1_dpo2000` | 3.24 | 1.00 ~ 5.00 | 3.62 | 2.89 | 3.58 | 3.27 | 2.95 |

## 7. 文案级高分覆盖

说明：候选覆盖是文案级覆盖率，计算方式为“至少有 1 个候选 overall 达到阈值的文案段数 / 总文案段数”；同一条文案下多个高分候选仍只算 1 条文案命中。

| Variant | >=4.0 候选覆盖 | >=4.0 系统达标 | >=4.5 候选覆盖 | >=4.5 系统达标 |
|---|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 112 / 183 (61.2%) | 66 / 183 (36.1%) | 85 / 183 (46.4%) | 46 / 183 (25.1%) |
| `qwen_q3_rerank_q1_dpo2000` | 112 / 183 (61.2%) | 72 / 183 (39.3%) | 85 / 183 (46.4%) | 53 / 183 (29.0%) |

## 8. 分剧指标

### `ainimeiyoulengjq`

- canonical run: `judge/outputs/canonical/ainimeiyoulengjq/annotation_20260520_g1`
- candidate pool: 7 条文案，140 条评估，overall mean=1.63

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 42.9% | 28.6% | 45.7% | 57.1% | 100.0% | 42.9% | 57.1% | 57.1% | 100.0% | 5.57 | 2.00 | 2.47 | 0.45 | 28.6% | 28.6% |
| `qwen_q3_rerank_q1_dpo2000` | 100.0% | 42.9% | 38.1% | 37.1% | 60.0% | 100.0% | 42.9% | 71.4% | 85.7% | 100.0% | 3.86 | 2.00 | 2.59 | 0.34 | 28.6% | 28.6% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 2.47 | 1.00 ~ 4.70 | 3.00 | 2.14 | 2.86 | 2.29 | 2.29 |
| `qwen_q3_rerank_q1_dpo2000` | 2.59 | 1.00 ~ 4.70 | 3.14 | 2.43 | 2.86 | 2.29 | 2.29 |

### `biezaijiaowomama`

- canonical run: `judge/outputs/canonical/biezaijiaowomama/annotation_20260520_g1`
- candidate pool: 11 条文案，266 条评估，overall mean=1.78

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 0.0% | 18.2% | 38.2% | 64.5% | 73.8% | 0.0% | 0.0% | 27.3% | 100.0% | 9.18 | 9.00 | 1.95 | 1.11 | 0.0% | 0.0% |
| `qwen_q3_rerank_q1_dpo2000` | 100.0% | 0.0% | 21.2% | 36.4% | 65.5% | 75.0% | 0.0% | 18.2% | 45.5% | 100.0% | 8.64 | 8.00 | 2.01 | 1.05 | 0.0% | 0.0% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 1.95 | 1.00 ~ 2.70 | 2.82 | 1.27 | 2.73 | 1.91 | 1.18 |
| `qwen_q3_rerank_q1_dpo2000` | 2.01 | 1.00 ~ 2.80 | 3.18 | 1.27 | 2.73 | 1.91 | 1.09 |

### `due`

- canonical run: `judge/outputs/canonical/due/annotation_20260520_g1`
- candidate pool: 38 条文案，827 条评估，overall mean=2.04

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 23.7% | 36.8% | 47.9% | 68.7% | 83.5% | 23.7% | 55.3% | 68.4% | 100.0% | 5.26 | 3.00 | 3.02 | 0.76 | 28.9% | 23.7% |
| `qwen_q3_rerank_q1_dpo2000` | 100.0% | 28.9% | 42.1% | 52.6% | 68.9% | 83.5% | 28.9% | 47.4% | 60.5% | 97.4% | 6.39 | 4.00 | 3.05 | 0.73 | 34.2% | 28.9% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 3.02 | 1.00 ~ 5.00 | 3.76 | 2.58 | 3.00 | 3.11 | 2.63 |
| `qwen_q3_rerank_q1_dpo2000` | 3.05 | 1.00 ~ 5.00 | 3.63 | 2.71 | 3.03 | 3.11 | 2.76 |

### `mamazaikanwoyiy`

- canonical run: `judge/outputs/canonical/mamazaikanwoyiy/annotation_20260520_g1`
- candidate pool: 11 条文案，153 条评估，overall mean=2.11

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 27.3% | 45.5% | 69.1% | 83.6% | - | 27.3% | 54.5% | 72.7% | 100.0% | 4.09 | 3.00 | 3.08 | 0.62 | 27.3% | 27.3% |
| `qwen_q3_rerank_q1_dpo2000` | 100.0% | 18.2% | 51.5% | 63.6% | 83.6% | - | 18.2% | 63.6% | 81.8% | 100.0% | 3.73 | 2.00 | 2.97 | 0.73 | 18.2% | 18.2% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 3.08 | 1.00 ~ 5.00 | 3.36 | 2.82 | 3.45 | 3.09 | 2.73 |
| `qwen_q3_rerank_q1_dpo2000` | 2.97 | 1.20 ~ 5.00 | 3.36 | 2.73 | 3.55 | 2.82 | 2.45 |

### `maopaigjxzwlbe`

- canonical run: `judge/outputs/canonical/maopaigjxzwlbe/annotation_20260520_g1`
- candidate pool: 44 条文案，736 条评估，overall mean=2.24

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 36.4% | 44.7% | 57.3% | 75.0% | 90.0% | 36.4% | 63.6% | 72.7% | 100.0% | 4.05 | 3.00 | 3.37 | 0.65 | 43.2% | 27.3% |
| `qwen_q3_rerank_q1_dpo2000` | 100.0% | 38.6% | 46.2% | 55.9% | 74.5% | 90.0% | 38.6% | 68.2% | 77.3% | 97.7% | 4.27 | 3.00 | 3.41 | 0.61 | 45.5% | 27.3% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 3.37 | 1.00 ~ 5.00 | 3.70 | 3.02 | 3.93 | 3.27 | 3.11 |
| `qwen_q3_rerank_q1_dpo2000` | 3.41 | 1.00 ~ 5.00 | 3.52 | 3.14 | 3.95 | 3.43 | 3.11 |

### `nvwangchusyjnq6`

- canonical run: `judge/outputs/canonical/nvwangchusyjnq6/annotation_20260520_g1`
- candidate pool: 14 条文案，290 条评估，overall mean=2.83

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 28.6% | 47.6% | 51.4% | 67.9% | 92.7% | 28.6% | 57.1% | 57.1% | 100.0% | 5.36 | 3.00 | 3.92 | 0.76 | 57.1% | 35.7% |
| `qwen_q3_rerank_q1_dpo2000` | 100.0% | 35.7% | 45.2% | 55.7% | 69.3% | 92.7% | 35.7% | 57.1% | 57.1% | 100.0% | 5.21 | 2.50 | 3.95 | 0.74 | 50.0% | 42.9% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 3.92 | 2.10 ~ 5.00 | 4.21 | 3.43 | 4.57 | 3.93 | 3.86 |
| `qwen_q3_rerank_q1_dpo2000` | 3.95 | 2.25 ~ 5.00 | 4.57 | 3.36 | 4.57 | 3.86 | 3.79 |

### `shuangshenganyan`

- canonical run: `judge/outputs/canonical/shuangshenganyan/annotation_20260520_g1`
- candidate pool: 21 条文案，440 条评估，overall mean=1.78

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 19.0% | 22.2% | 41.9% | 64.8% | 81.9% | 19.0% | 28.6% | 38.1% | 95.2% | 7.95 | 6.00 | 2.54 | 0.85 | 28.6% | 14.3% |
| `qwen_q3_rerank_q1_dpo2000` | 100.0% | 14.3% | 28.6% | 48.6% | 67.1% | 81.9% | 14.3% | 23.8% | 42.9% | 95.2% | 7.52 | 6.00 | 2.62 | 0.77 | 28.6% | 14.3% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 2.54 | 1.00 ~ 4.80 | 2.57 | 2.19 | 2.76 | 2.86 | 2.43 |
| `qwen_q3_rerank_q1_dpo2000` | 2.62 | 1.00 ~ 4.80 | 2.90 | 2.24 | 2.71 | 2.86 | 2.48 |

### `tiancimengsggsj`

- canonical run: `judge/outputs/canonical/tiancimengsggsj/annotation_20260520_g1`
- candidate pool: 22 条文案，380 条评估，overall mean=2.29

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 31.8% | 48.5% | 60.9% | 77.7% | 93.6% | 31.8% | 40.9% | 77.3% | 100.0% | 3.86 | 4.00 | 3.77 | 0.63 | 59.1% | 40.9% |
| `qwen_q3_rerank_q1_dpo2000` | 100.0% | 36.4% | 50.0% | 60.0% | 80.9% | 93.6% | 36.4% | 59.1% | 77.3% | 100.0% | 3.36 | 3.00 | 3.96 | 0.44 | 63.6% | 45.5% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 3.77 | 1.70 ~ 5.00 | 3.86 | 3.45 | 4.32 | 3.82 | 3.59 |
| `qwen_q3_rerank_q1_dpo2000` | 3.96 | 1.65 ~ 5.00 | 3.95 | 3.68 | 4.41 | 4.09 | 3.82 |

### `wodeqianghlgslan`

- canonical run: `judge/outputs/canonical/wodeqianghlgslan/annotation_20260520_g1`
- candidate pool: 6 条文案，115 条评估，overall mean=1.98

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 50.0% | 61.1% | 63.3% | 71.7% | 90.0% | 50.0% | 83.3% | 83.3% | 100.0% | 2.50 | 2.00 | 3.77 | 0.43 | 33.3% | 33.3% |
| `qwen_q3_rerank_q1_dpo2000` | 100.0% | 50.0% | 66.7% | 70.0% | 75.0% | 90.0% | 50.0% | 83.3% | 100.0% | 100.0% | 2.00 | 1.50 | 3.86 | 0.34 | 50.0% | 33.3% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 3.77 | 2.00 ~ 5.00 | 4.50 | 3.50 | 4.00 | 3.50 | 3.50 |
| `qwen_q3_rerank_q1_dpo2000` | 3.86 | 2.50 ~ 5.00 | 4.50 | 3.67 | 3.67 | 3.83 | 3.50 |

### `yinshishenchuzqk`

- canonical run: `judge/outputs/canonical/yinshishenchuzqk/annotation_20260520_g1`
- candidate pool: 9 条文案，279 条评估，overall mean=2.16

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 11.1% | 33.3% | 42.2% | 63.3% | 81.2% | 11.1% | 22.2% | 55.6% | 77.8% | 8.78 | 4.00 | 3.23 | 1.27 | 22.2% | 11.1% |
| `qwen_q3_rerank_q1_dpo2000` | 100.0% | 44.4% | 40.7% | 44.4% | 68.9% | 81.2% | 44.4% | 55.6% | 88.9% | 100.0% | 3.67 | 2.00 | 3.72 | 0.78 | 55.6% | 55.6% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 3.23 | 1.85 ~ 4.80 | 3.78 | 2.78 | 4.00 | 3.11 | 2.67 |
| `qwen_q3_rerank_q1_dpo2000` | 3.72 | 1.15 ~ 4.80 | 4.00 | 3.33 | 4.11 | 3.78 | 3.56 |

## 9. 明显错排样本

| 剧名 | Variant | 文案段 | 系统片段 | Judge 排名 | 系统分 | Judge 最优片段 | Judge 最优分 | 分差 | 诊断 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `ainimeiyoulengjq` | `qwen_q3_rerank_q1` | 3 | 7 | 8 | 2.20 | 5 | 4.40 | 2.20 | 有好候选但错排 |
| `ainimeiyoulengjq` | `qwen_q3_rerank_q1` | 4 | 6 | 12 | 1.20 | 19 | 1.80 | 0.60 | 候选池不足 |
| `ainimeiyoulengjq` | `qwen_q3_rerank_q1` | 7 | 10 | 14 | 1.00 | 16 | 1.35 | 0.35 | 候选池不足 |
| `ainimeiyoulengjq` | `qwen_q3_rerank_q1_dpo2000` | 3 | 10 | 3 | 2.80 | 5 | 4.40 | 1.60 | 接近，可优化 |
| `ainimeiyoulengjq` | `qwen_q3_rerank_q1_dpo2000` | 7 | 10 | 14 | 1.00 | 16 | 1.35 | 0.35 | 候选池不足 |
| `biezaijiaowomama` | `qwen_q3_rerank_q1` | 1 | 27 | 19 | 2.65 | 4 | 4.15 | 1.50 | 有好候选但错排 |
| `biezaijiaowomama` | `qwen_q3_rerank_q1` | 2 | 6 | 11 | 1.20 | 28 | 1.75 | 0.55 | 候选池不足 |
| `biezaijiaowomama` | `qwen_q3_rerank_q1` | 3 | 13 | 8 | 1.45 | 7 | 3.20 | 1.75 | 候选池不足 |
| `biezaijiaowomama` | `qwen_q3_rerank_q1` | 4 | 30 | 9 | 2.20 | 24 | 5.00 | 2.80 | 有好候选但错排 |
| `biezaijiaowomama` | `qwen_q3_rerank_q1` | 5 | 9 | 11 | 2.65 | 1 | 4.40 | 1.75 | 有好候选但错排 |
| `biezaijiaowomama` | `qwen_q3_rerank_q1` | 6 | 11 | 8 | 1.50 | 15 | 2.45 | 0.95 | 候选池不足 |
| `biezaijiaowomama` | `qwen_q3_rerank_q1` | 10 | 11 | 9 | 2.10 | 14 | 2.90 | 0.80 | 候选池不足 |
| `biezaijiaowomama` | `qwen_q3_rerank_q1` | 11 | 6 | 13 | 1.00 | 16 | 2.00 | 1.00 | 候选池不足 |
| `biezaijiaowomama` | `qwen_q3_rerank_q1_dpo2000` | 1 | 27 | 19 | 2.65 | 4 | 4.15 | 1.50 | 有好候选但错排 |
| `biezaijiaowomama` | `qwen_q3_rerank_q1_dpo2000` | 2 | 7 | 19 | 1.00 | 28 | 1.75 | 0.75 | 候选池不足 |
| `biezaijiaowomama` | `qwen_q3_rerank_q1_dpo2000` | 3 | 13 | 8 | 1.45 | 7 | 3.20 | 1.75 | 候选池不足 |
| `biezaijiaowomama` | `qwen_q3_rerank_q1_dpo2000` | 4 | 30 | 9 | 2.20 | 24 | 5.00 | 2.80 | 有好候选但错排 |
| `biezaijiaowomama` | `qwen_q3_rerank_q1_dpo2000` | 5 | 9 | 11 | 2.65 | 1 | 4.40 | 1.75 | 有好候选但错排 |
| `biezaijiaowomama` | `qwen_q3_rerank_q1_dpo2000` | 10 | 11 | 9 | 2.10 | 14 | 2.90 | 0.80 | 候选池不足 |
| `due` | `qwen_q3_rerank_q1` | 1 | 5 | 4 | 3.35 | 4 | 4.65 | 1.30 | 有好候选但错排 |
| `due` | `qwen_q3_rerank_q1` | 2 | 7 | 9 | 2.05 | 18 | 4.30 | 2.25 | 有好候选但错排 |
| `due` | `qwen_q3_rerank_q1` | 4 | 15 | 3 | 1.95 | 34 | 3.00 | 1.05 | 候选池不足 |
| `due` | `qwen_q3_rerank_q1` | 6 | 1 | 12 | 3.40 | 3 | 5.00 | 1.60 | 有好候选但错排 |
| `due` | `qwen_q3_rerank_q1` | 12 | 9 | 12 | 1.00 | 13 | 2.60 | 1.60 | 候选池不足 |
| `due` | `qwen_q3_rerank_q1` | 13 | 12 | 8 | 3.60 | 18 | 4.70 | 1.10 | 有好候选但错排 |
| `due` | `qwen_q3_rerank_q1` | 16 | 7 | 4 | 2.95 | 11 | 4.15 | 1.20 | 有好候选但错排 |
| `due` | `qwen_q3_rerank_q1` | 18 | 4 | 8 | 2.15 | 2 | 3.15 | 1.00 | 候选池不足 |
| `due` | `qwen_q3_rerank_q1` | 19 | 14 | 16 | 2.30 | 5 | 3.55 | 1.25 | 候选池不足 |
| `due` | `qwen_q3_rerank_q1` | 20 | 20 | 4 | 3.45 | 4 | 4.55 | 1.10 | 有好候选但错排 |
| `due` | `qwen_q3_rerank_q1` | 21 | 15 | 13 | 1.20 | 13 | 2.60 | 1.40 | 候选池不足 |
| `due` | `qwen_q3_rerank_q1` | 22 | 8 | 11 | 1.75 | 11 | 3.00 | 1.25 | 候选池不足 |
| `due` | `qwen_q3_rerank_q1` | 24 | 18 | 5 | 2.80 | 19 | 4.80 | 2.00 | 有好候选但错排 |
| `due` | `qwen_q3_rerank_q1` | 27 | 13 | 11 | 2.55 | 7 | 4.85 | 2.30 | 有好候选但错排 |
| `due` | `qwen_q3_rerank_q1` | 28 | 4 | 14 | 1.00 | 1 | 1.65 | 0.65 | 候选池不足 |
| `due` | `qwen_q3_rerank_q1` | 29 | 13 | 7 | 2.05 | 11 | 4.10 | 2.05 | 有好候选但错排 |
| `due` | `qwen_q3_rerank_q1` | 33 | 16 | 5 | 3.15 | 14 | 4.15 | 1.00 | 有好候选但错排 |
| `due` | `qwen_q3_rerank_q1` | 36 | 14 | 17 | 1.35 | 16 | 2.75 | 1.40 | 候选池不足 |
| `due` | `qwen_q3_rerank_q1_dpo2000` | 2 | 7 | 9 | 2.05 | 18 | 4.30 | 2.25 | 有好候选但错排 |
| `due` | `qwen_q3_rerank_q1_dpo2000` | 4 | 24 | 34 | 1.00 | 34 | 3.00 | 2.00 | 候选池不足 |
| `due` | `qwen_q3_rerank_q1_dpo2000` | 6 | 1 | 12 | 3.40 | 3 | 5.00 | 1.60 | 有好候选但错排 |
| `due` | `qwen_q3_rerank_q1_dpo2000` | 8 | 1 | 6 | 1.20 | 4 | 2.00 | 0.80 | 候选池不足 |
| `due` | `qwen_q3_rerank_q1_dpo2000` | 12 | 10 | 6 | 1.20 | 13 | 2.60 | 1.40 | 候选池不足 |
| `due` | `qwen_q3_rerank_q1_dpo2000` | 16 | 7 | 4 | 2.95 | 11 | 4.15 | 1.20 | 有好候选但错排 |
| `due` | `qwen_q3_rerank_q1_dpo2000` | 18 | 4 | 8 | 2.15 | 2 | 3.15 | 1.00 | 候选池不足 |
| `due` | `qwen_q3_rerank_q1_dpo2000` | 19 | 14 | 16 | 2.30 | 5 | 3.55 | 1.25 | 候选池不足 |
| `due` | `qwen_q3_rerank_q1_dpo2000` | 20 | 13 | 14 | 1.95 | 4 | 4.55 | 2.60 | 有好候选但错排 |
| `due` | `qwen_q3_rerank_q1_dpo2000` | 22 | 8 | 11 | 1.75 | 11 | 3.00 | 1.25 | 候选池不足 |
| `due` | `qwen_q3_rerank_q1_dpo2000` | 27 | 13 | 11 | 2.55 | 7 | 4.85 | 2.30 | 有好候选但错排 |
| `due` | `qwen_q3_rerank_q1_dpo2000` | 28 | 8 | 9 | 1.20 | 1 | 1.65 | 0.45 | 候选池不足 |
| `due` | `qwen_q3_rerank_q1_dpo2000` | 29 | 9 | 4 | 2.75 | 11 | 4.10 | 1.35 | 有好候选但错排 |
| `due` | `qwen_q3_rerank_q1_dpo2000` | 30 | 18 | 19 | 1.00 | 8 | 1.45 | 0.45 | 候选池不足 |
| `due` | `qwen_q3_rerank_q1_dpo2000` | 32 | 18 | 15 | 1.35 | 5 | 2.25 | 0.90 | 候选池不足 |
| `due` | `qwen_q3_rerank_q1_dpo2000` | 33 | 16 | 5 | 3.15 | 14 | 4.15 | 1.00 | 有好候选但错排 |
| `due` | `qwen_q3_rerank_q1_dpo2000` | 36 | 14 | 17 | 1.35 | 16 | 2.75 | 1.40 | 候选池不足 |
| `due` | `qwen_q3_rerank_q1_dpo2000` | 38 | 19 | 6 | 1.50 | 8 | 2.40 | 0.90 | 候选池不足 |
| `mamazaikanwoyiy` | `qwen_q3_rerank_q1` | 1 | 5 | 4 | 3.95 | 4 | 5.00 | 1.05 | 有好候选但错排 |
| `mamazaikanwoyiy` | `qwen_q3_rerank_q1` | 6 | 5 | 6 | 1.20 | 6 | 1.65 | 0.45 | 候选池不足 |
| `mamazaikanwoyiy` | `qwen_q3_rerank_q1` | 8 | 12 | 12 | 1.00 | 9 | 1.90 | 0.90 | 候选池不足 |
| `mamazaikanwoyiy` | `qwen_q3_rerank_q1` | 10 | 12 | 9 | 1.60 | 6 | 4.15 | 2.55 | 有好候选但错排 |
| `mamazaikanwoyiy` | `qwen_q3_rerank_q1_dpo2000` | 1 | 5 | 4 | 3.95 | 4 | 5.00 | 1.05 | 有好候选但错排 |
| `mamazaikanwoyiy` | `qwen_q3_rerank_q1_dpo2000` | 5 | 9 | 13 | 1.35 | 3 | 4.70 | 3.35 | 有好候选但错排 |
| `mamazaikanwoyiy` | `qwen_q3_rerank_q1_dpo2000` | 6 | 5 | 6 | 1.20 | 6 | 1.65 | 0.45 | 候选池不足 |
| `maopaigjxzwlbe` | `qwen_q3_rerank_q1` | 4 | 1 | 7 | 1.15 | 9 | 2.40 | 1.25 | 候选池不足 |
| `maopaigjxzwlbe` | `qwen_q3_rerank_q1` | 6 | 2 | 2 | 2.25 | 1 | 4.45 | 2.20 | 接近，可优化 |
| `maopaigjxzwlbe` | `qwen_q3_rerank_q1` | 8 | 24 | 8 | 2.05 | 9 | 4.20 | 2.15 | 有好候选但错排 |
| `maopaigjxzwlbe` | `qwen_q3_rerank_q1` | 17 | 15 | 10 | 2.30 | 3 | 3.90 | 1.60 | 候选池不足 |
| `maopaigjxzwlbe` | `qwen_q3_rerank_q1` | 21 | 6 | 8 | 3.45 | 11 | 5.00 | 1.55 | 有好候选但错排 |
| `maopaigjxzwlbe` | `qwen_q3_rerank_q1` | 22 | 17 | 5 | 1.90 | 15 | 2.95 | 1.05 | 候选池不足 |
| `maopaigjxzwlbe` | `qwen_q3_rerank_q1` | 24 | 8 | 6 | 2.30 | 4 | 4.65 | 2.35 | 有好候选但错排 |
| `maopaigjxzwlbe` | `qwen_q3_rerank_q1` | 31 | 8 | 9 | 2.20 | 13 | 4.60 | 2.40 | 有好候选但错排 |
| `maopaigjxzwlbe` | `qwen_q3_rerank_q1` | 36 | 13 | 4 | 2.75 | 11 | 4.45 | 1.70 | 有好候选但错排 |
| `maopaigjxzwlbe` | `qwen_q3_rerank_q1` | 38 | 5 | 7 | 3.90 | 14 | 5.00 | 1.10 | 有好候选但错排 |
| `maopaigjxzwlbe` | `qwen_q3_rerank_q1` | 39 | 12 | 7 | 3.70 | 14 | 5.00 | 1.30 | 有好候选但错排 |
| `maopaigjxzwlbe` | `qwen_q3_rerank_q1` | 40 | 8 | 12 | 1.50 | 14 | 4.15 | 2.65 | 有好候选但错排 |
| `maopaigjxzwlbe` | `qwen_q3_rerank_q1` | 42 | 13 | 13 | 1.00 | 1 | 1.20 | 0.20 | 候选池不足 |
| `maopaigjxzwlbe` | `qwen_q3_rerank_q1` | 43 | 8 | 15 | 1.35 | 7 | 2.95 | 1.60 | 候选池不足 |
| `maopaigjxzwlbe` | `qwen_q3_rerank_q1` | 44 | 6 | 10 | 1.20 | 5 | 1.65 | 0.45 | 候选池不足 |
| `maopaigjxzwlbe` | `qwen_q3_rerank_q1_dpo2000` | 2 | 9 | 23 | 1.15 | 13 | 3.40 | 2.25 | 候选池不足 |
| `maopaigjxzwlbe` | `qwen_q3_rerank_q1_dpo2000` | 8 | 2 | 5 | 2.65 | 9 | 4.20 | 1.55 | 有好候选但错排 |
| `maopaigjxzwlbe` | `qwen_q3_rerank_q1_dpo2000` | 12 | 18 | 12 | 3.70 | 7 | 5.00 | 1.30 | 有好候选但错排 |
| `maopaigjxzwlbe` | `qwen_q3_rerank_q1_dpo2000` | 15 | 18 | 11 | 1.35 | 4 | 2.90 | 1.55 | 候选池不足 |
| `maopaigjxzwlbe` | `qwen_q3_rerank_q1_dpo2000` | 17 | 15 | 10 | 2.30 | 3 | 3.90 | 1.60 | 候选池不足 |
| `maopaigjxzwlbe` | `qwen_q3_rerank_q1_dpo2000` | 22 | 17 | 5 | 1.90 | 15 | 2.95 | 1.05 | 候选池不足 |
| `maopaigjxzwlbe` | `qwen_q3_rerank_q1_dpo2000` | 23 | 10 | 12 | 1.90 | 11 | 4.70 | 2.80 | 有好候选但错排 |
| `maopaigjxzwlbe` | `qwen_q3_rerank_q1_dpo2000` | 31 | 8 | 9 | 2.20 | 13 | 4.60 | 2.40 | 有好候选但错排 |
| `maopaigjxzwlbe` | `qwen_q3_rerank_q1_dpo2000` | 36 | 9 | 3 | 2.90 | 11 | 4.45 | 1.55 | 接近，可优化 |
| `maopaigjxzwlbe` | `qwen_q3_rerank_q1_dpo2000` | 38 | 5 | 7 | 3.90 | 14 | 5.00 | 1.10 | 有好候选但错排 |
| `maopaigjxzwlbe` | `qwen_q3_rerank_q1_dpo2000` | 40 | 8 | 12 | 1.50 | 14 | 4.15 | 2.65 | 有好候选但错排 |
| `maopaigjxzwlbe` | `qwen_q3_rerank_q1_dpo2000` | 42 | 9 | 11 | 1.00 | 1 | 1.20 | 0.20 | 候选池不足 |
| `maopaigjxzwlbe` | `qwen_q3_rerank_q1_dpo2000` | 43 | 9 | 9 | 1.75 | 7 | 2.95 | 1.20 | 候选池不足 |
| `nvwangchusyjnq6` | `qwen_q3_rerank_q1` | 3 | 1 | 8 | 4.15 | 11 | 5.00 | 0.85 | 排序可优化 |
| `nvwangchusyjnq6` | `qwen_q3_rerank_q1` | 5 | 12 | 3 | 3.60 | 14 | 4.75 | 1.15 | 接近，可优化 |
| `nvwangchusyjnq6` | `qwen_q3_rerank_q1` | 6 | 15 | 10 | 3.05 | 11 | 4.35 | 1.30 | 有好候选但错排 |
| `nvwangchusyjnq6` | `qwen_q3_rerank_q1` | 7 | 1 | 9 | 2.10 | 12 | 3.60 | 1.50 | 候选池不足 |
| `nvwangchusyjnq6` | `qwen_q3_rerank_q1` | 10 | 2 | 11 | 4.20 | 12 | 5.00 | 0.80 | 排序可优化 |
| `nvwangchusyjnq6` | `qwen_q3_rerank_q1` | 11 | 7 | 17 | 2.25 | 10 | 5.00 | 2.75 | 有好候选但错排 |
| `nvwangchusyjnq6` | `qwen_q3_rerank_q1` | 14 | 5 | 6 | 3.45 | 21 | 4.70 | 1.25 | 有好候选但错排 |
| `nvwangchusyjnq6` | `qwen_q3_rerank_q1_dpo2000` | 1 | 7 | 8 | 3.15 | 3 | 4.30 | 1.15 | 有好候选但错排 |
| `nvwangchusyjnq6` | `qwen_q3_rerank_q1_dpo2000` | 3 | 9 | 13 | 2.95 | 11 | 5.00 | 2.05 | 有好候选但错排 |
| `nvwangchusyjnq6` | `qwen_q3_rerank_q1_dpo2000` | 6 | 15 | 10 | 3.05 | 11 | 4.35 | 1.30 | 有好候选但错排 |
| `nvwangchusyjnq6` | `qwen_q3_rerank_q1_dpo2000` | 10 | 8 | 7 | 4.25 | 12 | 5.00 | 0.75 | 排序可优化 |
| `nvwangchusyjnq6` | `qwen_q3_rerank_q1_dpo2000` | 11 | 7 | 17 | 2.25 | 10 | 5.00 | 2.75 | 有好候选但错排 |
| `nvwangchusyjnq6` | `qwen_q3_rerank_q1_dpo2000` | 14 | 5 | 6 | 3.45 | 21 | 4.70 | 1.25 | 有好候选但错排 |
| `shuangshenganyan` | `qwen_q3_rerank_q1` | 2 | 11 | 6 | 3.75 | 14 | 4.55 | 0.80 | 排序可优化 |
| `shuangshenganyan` | `qwen_q3_rerank_q1` | 3 | 18 | 18 | 1.00 | 1 | 1.00 | 0.00 | 候选池不足 |
| `shuangshenganyan` | `qwen_q3_rerank_q1` | 4 | 11 | 11 | 1.00 | 1 | 1.00 | 0.00 | 候选池不足 |
| `shuangshenganyan` | `qwen_q3_rerank_q1` | 7 | 18 | 6 | 2.95 | 15 | 3.45 | 0.50 | 候选池不足 |
| `shuangshenganyan` | `qwen_q3_rerank_q1` | 8 | 1 | 9 | 1.95 | 17 | 2.85 | 0.90 | 候选池不足 |
| `shuangshenganyan` | `qwen_q3_rerank_q1` | 9 | 6 | 9 | 1.00 | 29 | 3.90 | 2.90 | 候选池不足 |
| `shuangshenganyan` | `qwen_q3_rerank_q1` | 10 | 22 | 23 | 1.00 | 30 | 1.40 | 0.40 | 候选池不足 |
| `shuangshenganyan` | `qwen_q3_rerank_q1` | 11 | 13 | 12 | 1.75 | 7 | 4.05 | 2.30 | 有好候选但错排 |
| `shuangshenganyan` | `qwen_q3_rerank_q1` | 12 | 11 | 6 | 2.80 | 14 | 3.55 | 0.75 | 候选池不足 |
| `shuangshenganyan` | `qwen_q3_rerank_q1` | 13 | 18 | 19 | 1.15 | 12 | 5.00 | 3.85 | 有好候选但错排 |
| `shuangshenganyan` | `qwen_q3_rerank_q1` | 14 | 18 | 8 | 1.40 | 14 | 3.65 | 2.25 | 候选池不足 |
| `shuangshenganyan` | `qwen_q3_rerank_q1` | 15 | 12 | 6 | 4.80 | 3 | 5.00 | 0.20 | 排序可优化 |
| `shuangshenganyan` | `qwen_q3_rerank_q1` | 19 | 4 | 5 | 3.35 | 8 | 5.00 | 1.65 | 有好候选但错排 |
| `shuangshenganyan` | `qwen_q3_rerank_q1` | 21 | 14 | 15 | 1.00 | 18 | 1.20 | 0.20 | 候选池不足 |
| `shuangshenganyan` | `qwen_q3_rerank_q1_dpo2000` | 2 | 11 | 6 | 3.75 | 14 | 4.55 | 0.80 | 排序可优化 |
| `shuangshenganyan` | `qwen_q3_rerank_q1_dpo2000` | 3 | 18 | 18 | 1.00 | 1 | 1.00 | 0.00 | 候选池不足 |
| `shuangshenganyan` | `qwen_q3_rerank_q1_dpo2000` | 4 | 11 | 11 | 1.00 | 1 | 1.00 | 0.00 | 候选池不足 |
| `shuangshenganyan` | `qwen_q3_rerank_q1_dpo2000` | 7 | 18 | 6 | 2.95 | 15 | 3.45 | 0.50 | 候选池不足 |
| `shuangshenganyan` | `qwen_q3_rerank_q1_dpo2000` | 8 | 18 | 8 | 2.00 | 17 | 2.85 | 0.85 | 候选池不足 |
| `shuangshenganyan` | `qwen_q3_rerank_q1_dpo2000` | 9 | 6 | 9 | 1.00 | 29 | 3.90 | 2.90 | 候选池不足 |
| `shuangshenganyan` | `qwen_q3_rerank_q1_dpo2000` | 10 | 21 | 22 | 1.00 | 30 | 1.40 | 0.40 | 候选池不足 |
| `shuangshenganyan` | `qwen_q3_rerank_q1_dpo2000` | 11 | 13 | 12 | 1.75 | 7 | 4.05 | 2.30 | 有好候选但错排 |
| `shuangshenganyan` | `qwen_q3_rerank_q1_dpo2000` | 12 | 11 | 6 | 2.80 | 14 | 3.55 | 0.75 | 候选池不足 |
| `shuangshenganyan` | `qwen_q3_rerank_q1_dpo2000` | 13 | 11 | 5 | 2.70 | 12 | 5.00 | 2.30 | 有好候选但错排 |
| `shuangshenganyan` | `qwen_q3_rerank_q1_dpo2000` | 14 | 11 | 12 | 1.35 | 14 | 3.65 | 2.30 | 候选池不足 |
| `shuangshenganyan` | `qwen_q3_rerank_q1_dpo2000` | 15 | 12 | 6 | 4.80 | 3 | 5.00 | 0.20 | 排序可优化 |
| `shuangshenganyan` | `qwen_q3_rerank_q1_dpo2000` | 19 | 4 | 5 | 3.35 | 8 | 5.00 | 1.65 | 有好候选但错排 |
| `shuangshenganyan` | `qwen_q3_rerank_q1_dpo2000` | 21 | 14 | 15 | 1.00 | 18 | 1.20 | 0.20 | 候选池不足 |
| `tiancimengsggsj` | `qwen_q3_rerank_q1` | 4 | 9 | 11 | 1.70 | 17 | 5.00 | 3.30 | 有好候选但错排 |
| `tiancimengsggsj` | `qwen_q3_rerank_q1` | 5 | 1 | 2 | 3.00 | 2 | 4.45 | 1.45 | 接近，可优化 |
| `tiancimengsggsj` | `qwen_q3_rerank_q1` | 6 | 7 | 8 | 2.95 | 13 | 4.70 | 1.75 | 有好候选但错排 |
| `tiancimengsggsj` | `qwen_q3_rerank_q1` | 9 | 9 | 8 | 1.75 | 15 | 3.30 | 1.55 | 候选池不足 |
| `tiancimengsggsj` | `qwen_q3_rerank_q1` | 20 | 7 | 6 | 2.35 | 4 | 3.85 | 1.50 | 候选池不足 |
| `tiancimengsggsj` | `qwen_q3_rerank_q1` | 22 | 12 | 6 | 2.30 | 6 | 3.15 | 0.85 | 候选池不足 |
| `tiancimengsggsj` | `qwen_q3_rerank_q1_dpo2000` | 2 | 10 | 6 | 4.45 | 5 | 5.00 | 0.55 | 排序可优化 |
| `tiancimengsggsj` | `qwen_q3_rerank_q1_dpo2000` | 5 | 1 | 2 | 3.00 | 2 | 4.45 | 1.45 | 接近，可优化 |
| `tiancimengsggsj` | `qwen_q3_rerank_q1_dpo2000` | 6 | 7 | 8 | 2.95 | 13 | 4.70 | 1.75 | 有好候选但错排 |
| `tiancimengsggsj` | `qwen_q3_rerank_q1_dpo2000` | 18 | 2 | 10 | 1.65 | 11 | 2.85 | 1.20 | 候选池不足 |
| `tiancimengsggsj` | `qwen_q3_rerank_q1_dpo2000` | 20 | 7 | 6 | 2.35 | 4 | 3.85 | 1.50 | 候选池不足 |
| `tiancimengsggsj` | `qwen_q3_rerank_q1_dpo2000` | 22 | 12 | 6 | 2.30 | 6 | 3.15 | 0.85 | 候选池不足 |
| `wodeqianghlgslan` | `qwen_q3_rerank_q1` | 2 | 16 | 6 | 2.00 | 18 | 3.10 | 1.10 | 候选池不足 |
| `wodeqianghlgslan` | `qwen_q3_rerank_q1_dpo2000` | 5 | 13 | 4 | 3.25 | 17 | 4.50 | 1.25 | 有好候选但错排 |
| `yinshishenchuzqk` | `qwen_q3_rerank_q1` | 1 | 11 | 4 | 2.55 | 3 | 4.70 | 2.15 | 有好候选但错排 |
| `yinshishenchuzqk` | `qwen_q3_rerank_q1` | 2 | 8 | 3 | 3.95 | 7 | 5.00 | 1.05 | 接近，可优化 |
| `yinshishenchuzqk` | `qwen_q3_rerank_q1` | 3 | 14 | 4 | 2.80 | 13 | 3.80 | 1.00 | 候选池不足 |
| `yinshishenchuzqk` | `qwen_q3_rerank_q1` | 4 | 7 | 21 | 3.05 | 5 | 4.70 | 1.65 | 有好候选但错排 |
| `yinshishenchuzqk` | `qwen_q3_rerank_q1` | 5 | 7 | 25 | 3.30 | 21 | 5.00 | 1.70 | 有好候选但错排 |
| `yinshishenchuzqk` | `qwen_q3_rerank_q1` | 6 | 10 | 9 | 2.70 | 1 | 4.70 | 2.00 | 有好候选但错排 |
| `yinshishenchuzqk` | `qwen_q3_rerank_q1` | 7 | 23 | 8 | 1.85 | 14 | 3.10 | 1.25 | 候选池不足 |
| `yinshishenchuzqk` | `qwen_q3_rerank_q1_dpo2000` | 1 | 11 | 4 | 2.55 | 3 | 4.70 | 2.15 | 有好候选但错排 |
| `yinshishenchuzqk` | `qwen_q3_rerank_q1_dpo2000` | 2 | 35 | 5 | 3.35 | 7 | 5.00 | 1.65 | 有好候选但错排 |
| `yinshishenchuzqk` | `qwen_q3_rerank_q1_dpo2000` | 3 | 14 | 4 | 2.80 | 13 | 3.80 | 1.00 | 候选池不足 |
| `yinshishenchuzqk` | `qwen_q3_rerank_q1_dpo2000` | 7 | 4 | 14 | 1.15 | 14 | 3.10 | 1.95 | 候选池不足 |
