# qwen_q3_rerank_q1 对比报告

> 生成时间：2026-05-25T02:47:32  
> Canonical Judge 来源：`judge/outputs/canonical`  
> 对比版本：`qwen_q3_rerank_q1`

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

**综合最佳：`qwen_q3_rerank_q1`。**

排序口径：按 Hit@1 → Hit@3 → Hit@5 → Hit@10 → Hit@20 字典序排列。

胜出原因：Hit@1最高（26.6%）；Hit@3最高（54.5%）；Hit@5最高（67.9%）；Hit@10最高（83.0%）；Hit@20最高（96.6%）。

| 排名 | Variant | Hit@1 | Hit@3 | Hit@5 | Hit@10 | Hit@20 | 判断 |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | `qwen_q3_rerank_q1` | 26.6% | 54.5% | 67.9% | 83.0% | 96.6% | 综合最佳 |


### 3b. 召回导向排名

> 覆盖最重要：只看系统 topK 与 Judge topK 的交集。

**综合最佳：`qwen_q3_rerank_q1`。**

排序口径：按 R@1 → R@3 → R@5 → R@10 → R@20 字典序排列。

胜出原因：R@1最高（26.6%）；R@3最高（40.7%）；R@5最高（51.5%）；R@10最高（66.9%）；R@20最高（81.3%）。

| 排名 | Variant | R@1 | R@3 | R@5 | R@10 | R@20 | 判断 |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | `qwen_q3_rerank_q1` | 26.6% | 40.7% | 51.5% | 66.9% | 81.3% | 综合最佳 |


### 3c. NDCG 导向排名

> 排序质量最重要：衡量系统排序与 Judge 理想排序的接近程度。

**综合最佳：`qwen_q3_rerank_q1`。**

排序口径：按 NDCG@1 → NDCG@3 → NDCG@5 → NDCG@10 → NDCG@20 字典序排列。

胜出原因：NDCG@1最高（0.79）；NDCG@3最高（0.81）；NDCG@5最高（0.83）；NDCG@10最高（0.87）；NDCG@20最高（0.92）。

| 排名 | Variant | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | NDCG@20 | 判断 |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | `qwen_q3_rerank_q1` | 0.79 | 0.81 | 0.83 | 0.87 | 0.92 | 综合最佳 |



## 4. 总体指标

| Variant | 文案数 | 覆盖率 | Hit@1 | Hit@3 | Hit@5 | Hit@10 | Hit@20 | R@1 | R@3 | R@5 | R@10 | R@20 | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | NDCG@20 | 系统均分 | 平均分差 | >=4.0 | >=4.5 | 均rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 477 | 99.6% | 26.6% | 54.5% | 67.9% | 83.0% | 96.6% | 26.6% | 40.7% | 51.5% | 66.9% | 81.3% | 0.79 | 0.81 | 0.83 | 0.87 | 0.92 | 3.22 | 0.80 | 36.9% | 25.8% | 5.52 |

## 5. Canonical 候选池阈值命中

- 候选池总评估数：50051
- 候选池 overall：2.13（1.00 ~ 5.00）

| 阈值 | 命中数 | 命中率 |
|---|---:|---:|
| >=4.0 | 5522 / 50051 | 11.0% |
| >=4.5 | 3287 / 50051 | 6.6% |

## 6. 系统选中片段 Judge 均分

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 3.22 | 1.00 ~ 5.00 | 3.74 | 2.85 | 3.47 | 3.20 | 3.00 |

## 7. 文案级高分覆盖

说明：候选覆盖是文案级覆盖率，计算方式为“至少有 1 个候选 overall 达到阈值的文案段数 / 总文案段数”；同一条文案下多个高分候选仍只算 1 条文案命中。

| Variant | >=4.0 候选覆盖 | >=4.0 系统达标 | >=4.5 候选覆盖 | >=4.5 系统达标 |
|---|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 314 / 477 (65.8%) | 176 / 477 (36.9%) | 233 / 477 (48.8%) | 123 / 477 (25.8%) |

## 8. 分剧指标

### `maopaigjxzwlbe`

- canonical run: `judge/outputs/canonical/maopaigjxzwlbe/annotation_20260520_g1`
- candidate pool: 44 条文案，736 条评估，overall mean=2.24

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 71.4% | 52.4% | 60.0% | 80.0% | 90.0% | 71.4% | 85.7% | 85.7% | 100.0% | 2.00 | 1.00 | 3.41 | 0.49 | 42.9% | 28.6% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 3.41 | 1.15 ~ 4.70 | 3.71 | 3.00 | 3.71 | 3.43 | 3.57 |

### `mamazaikanwoyiy`

- canonical run: `judge/outputs/canonical/mamazaikanwoyiy/annotation_20260520_g1`
- candidate pool: 11 条文案，153 条评估，overall mean=2.11

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 25.0% | 50.0% | 65.0% | 85.0% | - | 25.0% | 75.0% | 100.0% | 100.0% | 2.50 | 2.50 | 4.33 | 0.55 | 50.0% | 50.0% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 4.33 | 3.55 ~ 5.00 | 4.25 | 4.25 | 4.75 | 4.25 | 4.25 |

### `yinshishenchuzqk`

- canonical run: `judge/outputs/canonical/yinshishenchuzqk/annotation_20260520_g1`
- candidate pool: 9 条文案，279 条评估，overall mean=2.16

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 0.0% | 33.3% | 45.0% | 60.0% | 76.2% | 0.0% | 25.0% | 75.0% | 75.0% | 8.00 | 4.00 | 3.09 | 1.46 | 0.0% | 0.0% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 3.09 | 2.55 ~ 3.95 | 4.25 | 2.50 | 4.25 | 2.50 | 2.25 |

### `biezaijiaowomama`

- canonical run: `judge/outputs/canonical/biezaijiaowomama/annotation_20260520_g1`
- candidate pool: 11 条文案，266 条评估，overall mean=1.78

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 0.0% | 26.7% | 24.0% | 48.0% | 73.8% | 0.0% | 0.0% | 0.0% | 100.0% | 11.60 | 11.00 | 2.03 | 1.67 | 0.0% | 0.0% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 2.03 | 1.20 ~ 2.65 | 3.00 | 1.40 | 2.60 | 2.00 | 1.20 |

### `shuangshenganyan`

- canonical run: `judge/outputs/canonical/shuangshenganyan/annotation_20260520_g1`
- candidate pool: 21 条文案，440 条评估，overall mean=1.78

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 0.0% | 16.7% | 50.0% | 75.0% | 70.0% | 0.0% | 0.0% | 50.0% | 100.0% | 5.50 | 5.50 | 2.55 | 0.68 | 0.0% | 0.0% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 2.55 | 1.35 ~ 3.75 | 2.50 | 2.00 | 3.00 | 3.00 | 2.50 |

### `nvwangchusyjnq6`

- canonical run: `judge/outputs/canonical/nvwangchusyjnq6/annotation_20260520_g1`
- candidate pool: 14 条文案，290 条评估，overall mean=2.83

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 50.0% | 33.3% | 60.0% | 70.0% | 91.2% | 50.0% | 75.0% | 75.0% | 100.0% | 3.00 | 1.50 | 4.61 | 0.21 | 100.0% | 50.0% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 4.61 | 4.15 ~ 5.00 | 4.75 | 4.50 | 4.75 | 4.50 | 4.75 |

### `ainimeiyoulengjq`

- canonical run: `judge/outputs/canonical/ainimeiyoulengjq/annotation_20260520_g1`
- candidate pool: 7 条文案，140 条评估，overall mean=1.63

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 50.0% | 33.3% | 50.0% | 65.0% | 100.0% | 50.0% | 50.0% | 50.0% | 100.0% | 5.50 | 4.50 | 3.20 | 0.70 | 50.0% | 50.0% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 3.20 | 1.20 ~ 4.70 | 4.00 | 2.75 | 3.50 | 3.00 | 3.00 |

### `tiancimengsggsj`

- canonical run: `judge/outputs/canonical/tiancimengsggsj/annotation_20260520_g1`
- candidate pool: 22 条文案，380 条评估，overall mean=2.29

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 20.0% | 53.3% | 60.0% | 78.0% | 98.3% | 20.0% | 60.0% | 80.0% | 100.0% | 4.40 | 3.00 | 3.43 | 1.12 | 40.0% | 20.0% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 3.43 | 1.70 ~ 5.00 | 3.20 | 3.00 | 4.60 | 3.60 | 3.00 |

### `due`

- canonical run: `judge/outputs/canonical/due/annotation_20260520_g1`
- candidate pool: 38 条文案，827 条评估，overall mean=2.04

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 25.0% | 41.7% | 50.0% | 60.0% | 71.2% | 25.0% | 50.0% | 75.0% | 100.0% | 4.25 | 3.50 | 3.04 | 1.15 | 25.0% | 25.0% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 3.04 | 1.95 ~ 4.80 | 3.25 | 2.50 | 3.25 | 3.50 | 2.75 |

### `wodeqianghlgslan`

- canonical run: `judge/outputs/canonical/wodeqianghlgslan/annotation_20260520_g1`
- candidate pool: 6 条文案，115 条评估，overall mean=1.98

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 40.0% | 60.0% | 60.0% | 72.0% | 90.0% | 40.0% | 80.0% | 80.0% | 100.0% | 2.80 | 3.00 | 3.53 | 0.51 | 20.0% | 20.0% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 3.53 | 2.00 ~ 5.00 | 4.40 | 3.20 | 3.80 | 3.20 | 3.20 |

### `tianjiangxssjyqc`

- canonical run: `judge/outputs/canonical/tianjiangxssjyqc/annotation_20260520_g1`
- candidate pool: 20 条文案，324 条评估，overall mean=2.09

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 80.0% | 60.0% | 64.0% | 84.0% | - | 80.0% | 80.0% | 80.0% | 100.0% | 2.00 | 1.00 | 4.00 | 0.40 | 60.0% | 40.0% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 4.00 | 3.00 ~ 5.00 | 4.20 | 3.60 | 4.20 | 4.20 | 4.00 |

### `caiyuancaidjzfr`

- canonical run: `judge/outputs/canonical/caiyuancaidjzfr/annotation_20260520_g1`
- candidate pool: 19 条文案，420 条评估，overall mean=2.39

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 0.0% | 40.0% | 56.0% | 68.0% | 78.8% | 0.0% | 40.0% | 60.0% | 100.0% | 4.00 | 4.00 | 4.00 | 0.31 | 60.0% | 20.0% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 4.00 | 2.70 ~ 4.50 | 4.40 | 3.40 | 4.20 | 4.20 | 4.20 |

### `qisuidewoshisjqs`

- canonical run: `judge/outputs/canonical/qisuidewoshisjqs/annotation_20260520_g1`
- candidate pool: 32 条文案，623 条评估，overall mean=2.33

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 40.0% | 66.7% | 64.0% | 70.0% | 86.0% | 40.0% | 60.0% | 100.0% | 100.0% | 3.00 | 3.00 | 3.57 | 0.47 | 60.0% | 40.0% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 3.57 | 1.45 ~ 5.00 | 3.80 | 3.00 | 4.00 | 3.80 | 3.60 |

### `qiuhunjsjclhym0`

- canonical run: `judge/outputs/canonical/qiuhunjsjclhym0/annotation_20260520_g1`
- candidate pool: 37 条文案，830 条评估，overall mean=2.05

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 20.0% | 13.3% | 28.0% | 54.0% | 73.0% | 20.0% | 20.0% | 40.0% | 100.0% | 7.40 | 6.00 | 2.79 | 1.20 | 0.0% | 0.0% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 2.79 | 1.85 ~ 3.55 | 3.80 | 2.00 | 4.60 | 2.00 | 2.40 |

### `tianjiangxslfcr`

- canonical run: `judge/outputs/canonical/tianjiangxslfcr/annotation_20260520_g1`
- candidate pool: 35 条文案，738 条评估，overall mean=2.12

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 0.0% | 53.3% | 60.0% | 66.0% | 88.0% | 0.0% | 60.0% | 80.0% | 100.0% | 4.40 | 3.00 | 2.90 | 1.00 | 20.0% | 0.0% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 2.90 | 1.45 ~ 4.35 | 3.60 | 2.40 | 3.20 | 2.80 | 2.80 |

### `mingzhongzdwyly`

- canonical run: `judge/outputs/canonical/mingzhongzdwyly/annotation_20260520_g1`
- candidate pool: 39 条文案，682 条评估，overall mean=2.56

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 0.0% | 33.3% | 45.0% | 60.0% | 95.0% | 0.0% | 50.0% | 75.0% | 100.0% | 3.75 | 3.50 | 3.70 | 0.29 | 50.0% | 50.0% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 3.70 | 1.95 ~ 5.00 | 4.00 | 3.50 | 3.75 | 3.75 | 3.50 |

### `niepanchengfenu`

- canonical run: `judge/outputs/canonical/niepanchengfenu/annotation_20260520_g1`
- candidate pool: 7 条文案，138 条评估，overall mean=1.98

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 0.0% | 55.6% | 53.3% | 76.7% | 100.0% | 0.0% | 66.7% | 100.0% | 100.0% | 3.33 | 3.00 | 2.35 | 0.72 | 0.0% | 0.0% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 2.35 | 1.95 ~ 3.15 | 4.33 | 1.67 | 2.33 | 2.00 | 1.33 |

### `yishenrujuzbbwy`

- canonical run: `judge/outputs/canonical/yishenrujuzbbwy/annotation_20260520_g1`
- candidate pool: 27 条文案，327 条评估，overall mean=2.35

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 40.0% | 66.7% | 64.0% | 80.0% | - | 40.0% | 80.0% | 80.0% | 100.0% | 3.00 | 2.00 | 4.17 | 0.41 | 40.0% | 40.0% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 4.17 | 3.20 ~ 5.00 | 4.60 | 4.20 | 3.00 | 4.40 | 4.40 |

### `wanlaibujianxy`

- canonical run: `judge/outputs/canonical/wanlaibujianxy/annotation_20260520_g1`
- candidate pool: 30 条文案，455 条评估，overall mean=2.04

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 40.0% | 46.7% | 56.0% | 80.0% | 91.7% | 40.0% | 80.0% | 80.0% | 100.0% | 3.40 | 2.00 | 2.98 | 0.58 | 40.0% | 20.0% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 2.98 | 1.95 ~ 4.70 | 3.60 | 2.20 | 3.80 | 3.00 | 2.80 |

### `longtengsihai`

- canonical run: `judge/outputs/canonical/longtengsihai/annotation_20260520_g1`
- candidate pool: 46 条文案，812 条评估，overall mean=2.17

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 20.0% | 40.0% | 52.0% | 70.0% | 90.0% | 20.0% | 80.0% | 80.0% | 100.0% | 3.40 | 2.00 | 2.95 | 0.47 | 40.0% | 40.0% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 2.95 | 1.55 ~ 4.85 | 3.60 | 2.40 | 3.00 | 3.20 | 2.60 |

### `jiyijiaohuan`

- canonical run: `judge/outputs/canonical/jiyijiaohuan/annotation_20260520_g2`
- candidate pool: 26 条文案，456 条评估，overall mean=1.92

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 25.0% | 41.7% | 55.0% | 70.0% | - | 25.0% | 50.0% | 50.0% | 100.0% | 5.75 | 5.50 | 3.33 | 1.25 | 50.0% | 50.0% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 3.33 | 1.70 ~ 4.70 | 3.25 | 2.75 | 4.50 | 3.50 | 3.00 |

### `hekuxiangsizhuyn`

- canonical run: `judge/outputs/canonical/hekuxiangsizhuyn/annotation_20260520_g2`
- candidate pool: 33 条文案，838 条评估，overall mean=2.11

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 20.0% | 40.0% | 48.0% | 56.0% | 59.0% | 20.0% | 20.0% | 20.0% | 40.0% | 20.40 | 23.00 | 2.72 | 1.84 | 40.0% | 40.0% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 2.72 | 1.20 ~ 4.80 | 3.00 | 2.60 | 3.00 | 2.60 | 2.40 |

### `kaijushazhujiang`

- canonical run: `judge/outputs/canonical/kaijushazhujiang/annotation_20260520_g2`
- candidate pool: 26 条文案，746 条评估，overall mean=2.05

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 0.0% | 33.3% | 45.0% | 67.5% | 85.0% | 0.0% | 50.0% | 50.0% | 100.0% | 9.50 | 9.00 | 3.11 | 1.56 | 50.0% | 25.0% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 3.11 | 1.50 ~ 4.65 | 3.00 | 3.00 | 3.75 | 3.00 | 3.00 |

### `zhongshbnxccb`

- canonical run: `judge/outputs/canonical/zhongshbnxccb/annotation_20260520_g2`
- candidate pool: 20 条文案，396 条评估，overall mean=2.48

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 25.0% | 33.3% | 25.0% | 52.5% | 77.5% | 25.0% | 50.0% | 50.0% | 100.0% | 7.25 | 6.00 | 3.99 | 0.57 | 50.0% | 25.0% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 3.99 | 3.40 ~ 4.90 | 4.00 | 3.75 | 3.75 | 4.50 | 3.75 |

### `woyoushuangcs1`

- canonical run: `judge/outputs/canonical/woyoushuangcs1/annotation_20260520_g2`
- candidate pool: 28 条文案，452 条评估，overall mean=2.46

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 40.0% | 33.3% | 60.0% | 78.0% | - | 40.0% | 60.0% | 80.0% | 100.0% | 4.80 | 3.00 | 3.68 | 0.88 | 40.0% | 40.0% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 3.68 | 1.15 ~ 5.00 | 4.00 | 3.40 | 4.00 | 3.60 | 3.60 |

### `nvergjbdhwbzdm`

- canonical run: `judge/outputs/canonical/nvergjbdhwbzdm/annotation_20260520_g2`
- candidate pool: 11 条文案，193 条评估，overall mean=2.26

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 0.0% | 41.7% | 50.0% | 77.5% | 90.0% | 0.0% | 50.0% | 50.0% | 100.0% | 5.50 | 4.50 | 3.20 | 1.29 | 50.0% | 0.0% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 3.20 | 1.55 ~ 4.35 | 3.50 | 3.00 | 4.25 | 2.75 | 2.75 |

### `xindongzuobiaopiancha`

- canonical run: `judge/outputs/canonical/xindongzuobiaopiancha/annotation_20260520_g2`
- candidate pool: 16 条文案，189 条评估，overall mean=1.80

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 33.3% | 44.4% | 73.3% | 90.0% | - | 33.3% | 66.7% | 100.0% | 100.0% | 2.67 | 2.00 | 3.68 | 0.35 | 66.7% | 33.3% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 3.68 | 2.05 ~ 4.70 | 4.67 | 3.00 | 4.33 | 3.33 | 3.67 |

### `huoshimengwacnd`

- canonical run: `judge/outputs/canonical/huoshimengwacnd/annotation_20260520_g2`
- candidate pool: 22 条文案，301 条评估，overall mean=2.59

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 0.0% | 41.7% | 60.0% | 87.5% | 95.0% | 0.0% | 25.0% | 25.0% | 100.0% | 8.25 | 6.50 | 2.66 | 1.61 | 25.0% | 25.0% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 2.66 | 1.55 ~ 4.55 | 3.75 | 2.25 | 3.00 | 2.25 | 2.25 |

### `yesejiqingren`

- canonical run: `judge/outputs/canonical/yesejiqingren/annotation_20260520_g2`
- candidate pool: 66 条文案，1461 条评估，overall mean=2.06

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 30.0% | 43.3% | 54.0% | 64.0% | 72.5% | 30.0% | 70.0% | 80.0% | 100.0% | 4.20 | 2.50 | 3.82 | 0.25 | 70.0% | 50.0% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 3.82 | 1.50 ~ 4.70 | 4.40 | 3.40 | 3.90 | 3.70 | 4.10 |

### `kanwojifenxcz`

- canonical run: `judge/outputs/canonical/kanwojifenxcz/annotation_20260520_g2`
- candidate pool: 26 条文案，943 条评估，overall mean=2.25

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 50.0% | 33.3% | 50.0% | 60.0% | 70.8% | 50.0% | 66.7% | 66.7% | 100.0% | 5.00 | 2.00 | 3.86 | 0.37 | 50.0% | 33.3% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 3.86 | 1.50 ~ 5.00 | 4.17 | 3.67 | 4.00 | 3.83 | 3.67 |

### `bierejifuren`

- canonical run: `judge/outputs/canonical/bierejifuren/annotation_20260520_g2`
- candidate pool: 9 条文案，352 条评估，overall mean=1.88

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 33.3% | 22.2% | 20.0% | 36.7% | 55.0% | 33.3% | 33.3% | 33.3% | 33.3% | 21.33 | 26.00 | 1.83 | 1.70 | 0.0% | 0.0% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 1.83 | 1.20 ~ 2.15 | 3.33 | 1.33 | 2.00 | 1.33 | 1.33 |

### `naniancxhssbfxb`

- canonical run: `judge/outputs/canonical/naniancxhssbfxb/annotation_20260520_g2`
- candidate pool: 9 条文案，172 条评估，overall mean=1.71

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 25.0% | 50.0% | 65.0% | 80.0% | 75.0% | 25.0% | 25.0% | 25.0% | 100.0% | 7.75 | 9.00 | 1.89 | 1.66 | 0.0% | 0.0% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 1.89 | 1.00 ~ 3.50 | 2.00 | 1.50 | 2.00 | 2.25 | 1.75 |

### `sancunrenxinbukl`

- canonical run: `judge/outputs/canonical/sancunrenxinbukl/annotation_20260520_g2`
- candidate pool: 14 条文案，227 条评估，overall mean=1.98

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 40.0% | 46.7% | 40.0% | 68.0% | - | 40.0% | 80.0% | 100.0% | 100.0% | 2.40 | 3.00 | 2.40 | 0.64 | 0.0% | 0.0% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 2.40 | 1.55 ~ 3.10 | 3.20 | 2.20 | 2.40 | 2.00 | 2.40 |

### `shangjienvhuans`

- canonical run: `judge/outputs/canonical/shangjienvhuans/annotation_20260520_g2`
- candidate pool: 9 条文案，169 条评估，overall mean=2.29

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 50.0% | 25.0% | 40.0% | 60.0% | 92.5% | 50.0% | 50.0% | 100.0% | 100.0% | 2.75 | 2.50 | 3.31 | 0.49 | 25.0% | 0.0% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 3.31 | 2.35 ~ 4.25 | 3.75 | 2.75 | 4.75 | 3.00 | 2.75 |

### `touhuansjhqzhdn`

- canonical run: `judge/outputs/canonical/touhuansjhqzhdn/annotation_20260520_g2`
- candidate pool: 9 条文案，202 条评估，overall mean=2.51

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 33.3% | 44.4% | 46.7% | 66.7% | 96.7% | 33.3% | 33.3% | 33.3% | 100.0% | 8.00 | 11.00 | 2.75 | 1.05 | 0.0% | 0.0% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 2.75 | 1.90 ~ 3.35 | 3.33 | 2.00 | 4.67 | 2.33 | 2.00 |

### `mengxianglouxdhg`

- canonical run: `judge/outputs/canonical/mengxianglouxdhg/annotation_20260520_g2`
- candidate pool: 32 条文案，590 条评估，overall mean=2.04

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 50.0% | 72.2% | 63.3% | 73.3% | - | 50.0% | 100.0% | 100.0% | 100.0% | 1.83 | 1.50 | 3.92 | 0.53 | 66.7% | 33.3% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 3.92 | 2.20 ~ 4.85 | 4.33 | 3.83 | 3.67 | 3.83 | 4.00 |

### `zhumafeilrwzdgm`

- canonical run: `judge/outputs/canonical/zhumafeilrwzdgm/annotation_20260520_g2`
- candidate pool: 29 条文案，586 条评估，overall mean=2.34

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 25.0% | 16.7% | 25.0% | 47.5% | 81.2% | 25.0% | 50.0% | 50.0% | 100.0% | 5.75 | 4.00 | 2.88 | 0.85 | 25.0% | 25.0% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 2.88 | 1.65 ~ 4.70 | 3.50 | 2.25 | 2.75 | 3.25 | 2.75 |

### `pugsfhsgynzhm5`

- canonical run: `judge/outputs/canonical/pugsfhsgynzhm5/annotation_20260520_g2`
- candidate pool: 31 条文案，629 条评估，overall mean=2.13

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 0.0% | 46.7% | 44.0% | 66.0% | 81.2% | 0.0% | 40.0% | 60.0% | 100.0% | 5.40 | 5.00 | 3.68 | 0.88 | 40.0% | 40.0% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 3.68 | 1.75 ~ 4.70 | 3.60 | 3.60 | 3.00 | 4.20 | 3.80 |

### `yangerfanglzmmbk`

- canonical run: `judge/outputs/canonical/yangerfanglzmmbk/annotation_20260520_g2`
- candidate pool: 26 条文案，490 条评估，overall mean=2.01

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 66.7% | 0.0% | 44.4% | 66.7% | 76.7% | - | 0.0% | 33.3% | 33.3% | 66.7% | 8.50 | 8.50 | 1.98 | 0.95 | 0.0% | 0.0% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 1.98 | 1.00 ~ 2.95 | 2.00 | 2.00 | 1.50 | 2.00 | 2.50 |

### `chunfengyidujsa`

- canonical run: `judge/outputs/canonical/chunfengyidujsa/annotation_20260520_g2`
- candidate pool: 36 条文案，715 条评估，overall mean=1.88

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 60.0% | 46.7% | 72.0% | 66.0% | 85.0% | 60.0% | 80.0% | 80.0% | 100.0% | 4.00 | 1.00 | 3.76 | 0.22 | 60.0% | 40.0% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 3.76 | 1.50 ~ 4.80 | 4.20 | 3.20 | 4.00 | 4.00 | 3.60 |

### `linglongnuhuotsh`

- canonical run: `judge/outputs/canonical/linglongnuhuotsh/annotation_20260520_g3`
- candidate pool: 24 条文案，706 条评估，overall mean=2.73

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 33.3% | 33.3% | 46.7% | 70.0% | 85.0% | 33.3% | 33.3% | 66.7% | 100.0% | 5.67 | 4.00 | 3.43 | 1.13 | 33.3% | 33.3% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 3.43 | 1.75 ~ 4.70 | 4.00 | 2.67 | 3.67 | 3.67 | 3.67 |

### `huangyanlidenver`

- canonical run: `judge/outputs/canonical/huangyanlidenver/annotation_20260520_g3`
- candidate pool: 11 条文案，213 条评估，overall mean=1.82

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 0.0% | 33.3% | 46.7% | 70.0% | 95.0% | 0.0% | 33.3% | 66.7% | 100.0% | 7.33 | 5.00 | 2.17 | 1.32 | 0.0% | 0.0% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 2.17 | 1.25 ~ 3.55 | 2.33 | 1.67 | 3.00 | 2.33 | 1.67 |

### `luomishuweiheyz8`

- canonical run: `judge/outputs/canonical/luomishuweiheyz8/annotation_20260520_g3`
- candidate pool: 23 条文案，406 条评估，overall mean=2.19

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 40.0% | 33.3% | 48.0% | 62.0% | 85.0% | 40.0% | 80.0% | 100.0% | 100.0% | 2.20 | 2.00 | 3.30 | 0.25 | 40.0% | 40.0% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 3.30 | 1.55 ~ 4.70 | 4.00 | 3.20 | 2.80 | 3.20 | 3.20 |

### `mamaqingdaihg`

- canonical run: `judge/outputs/canonical/mamaqingdaihg/annotation_20260520_g3`
- candidate pool: 30 条文案，447 条评估，overall mean=2.08

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 50.0% | 33.3% | 63.3% | 76.7% | - | 50.0% | 50.0% | 66.7% | 100.0% | 5.50 | 2.50 | 2.63 | 0.72 | 0.0% | 0.0% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 2.63 | 1.00 ~ 3.90 | 2.83 | 2.83 | 2.00 | 2.67 | 2.50 |

### `gongmianyifantdd`

- canonical run: `judge/outputs/canonical/gongmianyifantdd/annotation_20260520_g3`
- candidate pool: 17 条文案，463 条评估，overall mean=1.87

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 0.0% | 27.8% | 43.3% | 68.3% | 82.0% | 0.0% | 33.3% | 50.0% | 83.3% | 11.67 | 6.50 | 2.54 | 1.52 | 16.7% | 16.7% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 2.54 | 1.15 ~ 4.60 | 3.50 | 2.33 | 2.67 | 2.17 | 2.00 |

### `shuitoulewdlqtzs`

- canonical run: `judge/outputs/canonical/shuitoulewdlqtzs/annotation_20260520_g3`
- candidate pool: 17 条文案，417 条评估，overall mean=2.21

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 0.0% | 22.2% | 40.0% | 60.0% | 80.0% | 0.0% | 33.3% | 33.3% | 100.0% | 11.33 | 12.00 | 2.05 | 1.45 | 0.0% | 0.0% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 2.05 | 1.00 ~ 3.85 | 2.00 | 2.00 | 2.33 | 2.00 | 2.00 |

### `shachangqiudianbing`

- canonical run: `judge/outputs/canonical/shachangqiudianbing/annotation_20260520_g3`
- candidate pool: 20 条文案，533 条评估，overall mean=2.62

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 20.0% | 53.3% | 56.0% | 62.0% | 95.0% | 20.0% | 100.0% | 100.0% | 100.0% | 2.20 | 2.00 | 4.63 | 0.23 | 80.0% | 80.0% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 4.63 | 3.95 ~ 5.00 | 4.60 | 4.40 | 4.60 | 4.80 | 5.00 |

### `sanshengsansxfqc`

- canonical run: `judge/outputs/canonical/sanshengsansxfqc/annotation_20260520_g3`
- candidate pool: 19 条文案，426 条评估，overall mean=2.08

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 20.0% | 33.3% | 52.0% | 68.0% | 70.0% | 20.0% | 60.0% | 60.0% | 100.0% | 5.40 | 3.00 | 3.64 | 0.78 | 60.0% | 40.0% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 3.64 | 2.10 ~ 4.70 | 4.40 | 3.00 | 4.00 | 3.60 | 3.60 |

### `biepengwodehefhq`

- canonical run: `judge/outputs/canonical/biepengwodehefhq/annotation_20260520_g3`
- candidate pool: 34 条文案，559 条评估，overall mean=2.23

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 80.0% | 53.3% | 64.0% | 82.0% | - | 80.0% | 80.0% | 100.0% | 100.0% | 1.80 | 1.00 | 4.05 | 0.11 | 80.0% | 40.0% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 4.05 | 1.50 ~ 5.00 | 4.40 | 3.80 | 4.20 | 4.00 | 4.00 |

### `kuwodexunirenl`

- canonical run: `judge/outputs/canonical/kuwodexunirenl/annotation_20260520_g3`
- candidate pool: 8 条文案，113 条评估，overall mean=2.08

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 0.0% | 33.3% | 40.0% | 80.0% | - | 0.0% | 50.0% | 50.0% | 100.0% | 5.50 | 4.50 | 2.32 | 1.43 | 0.0% | 0.0% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 2.32 | 1.00 ~ 3.30 | 2.75 | 2.00 | 2.25 | 2.25 | 2.75 |

### `zuiwen`

- canonical run: `judge/outputs/canonical/zuiwen/annotation_20260520_g3`
- candidate pool: 39 条文案，1103 条评估，overall mean=1.89

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 60.0% | 46.7% | 48.0% | 52.0% | 81.0% | 60.0% | 100.0% | 100.0% | 100.0% | 1.60 | 1.00 | 3.28 | 0.13 | 40.0% | 0.0% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 3.28 | 2.05 ~ 4.15 | 4.20 | 2.80 | 2.80 | 3.60 | 2.80 |

### `miyun`

- canonical run: `judge/outputs/canonical/miyun/annotation_20260520_g3`
- candidate pool: 6 条文案，115 条评估，overall mean=2.51

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 25.0% | 33.3% | 55.0% | 72.5% | 93.3% | 25.0% | 50.0% | 75.0% | 100.0% | 4.50 | 3.00 | 3.58 | 0.28 | 50.0% | 50.0% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 3.58 | 1.80 ~ 5.00 | 3.75 | 3.25 | 3.50 | 4.00 | 3.25 |

### `aiyiranjinqscl`

- canonical run: `judge/outputs/canonical/aiyiranjinqscl/annotation_20260520_g3`
- candidate pool: 18 条文案，481 条评估，overall mean=2.13

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 0.0% | 33.3% | 46.7% | 66.7% | 77.5% | 0.0% | 66.7% | 66.7% | 100.0% | 5.33 | 3.00 | 3.52 | 0.73 | 33.3% | 33.3% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 3.52 | 2.05 ~ 5.00 | 3.33 | 3.33 | 4.33 | 3.33 | 3.67 |

### `popozaihunexmfqv`

- canonical run: `judge/outputs/canonical/popozaihunexmfqv/annotation_20260520_g3`
- candidate pool: 7 条文案，118 条评估，overall mean=2.38

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 0.0% | 44.4% | 43.3% | 71.7% | 90.0% | 0.0% | 33.3% | 33.3% | 100.0% | 8.17 | 7.00 | 2.71 | 1.68 | 16.7% | 16.7% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 2.71 | 1.35 ~ 4.80 | 3.67 | 2.17 | 3.17 | 2.67 | 1.83 |

### `fangrennanyouqingmeizuoyongganxiaoyang`

- canonical run: `judge/outputs/canonical/fangrennanyouqingmeizuoyongganxiaoyang/annotation_20260520_g3`
- candidate pool: 30 条文案，707 条评估，overall mean=2.17

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 0.0% | 0.0% | 25.0% | 55.0% | 77.5% | 0.0% | 0.0% | 25.0% | 100.0% | 11.75 | 13.00 | 2.04 | 2.34 | 0.0% | 0.0% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 2.04 | 1.50 ~ 2.70 | 3.00 | 1.25 | 4.00 | 1.25 | 1.50 |

### `longyumingwang`

- canonical run: `judge/outputs/canonical/longyumingwang/annotation_20260520_g3`
- candidate pool: 23 条文案，233 条评估，overall mean=2.32

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 60.0% | 66.7% | 72.0% | 90.0% | - | 60.0% | 60.0% | 100.0% | 100.0% | 2.20 | 1.00 | 3.44 | 0.56 | 40.0% | 20.0% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 3.44 | 1.45 ~ 4.75 | 3.60 | 3.40 | 3.00 | 3.80 | 3.00 |

### `ruhourongguanb`

- canonical run: `judge/outputs/canonical/ruhourongguanb/annotation_20260520_g3`
- candidate pool: 25 条文案，491 条评估，overall mean=2.27

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 50.0% | 50.0% | 56.7% | 60.0% | 78.3% | 50.0% | 83.3% | 100.0% | 100.0% | 2.00 | 1.50 | 4.17 | 0.31 | 66.7% | 66.7% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 4.17 | 3.05 ~ 4.85 | 4.33 | 4.00 | 3.67 | 4.50 | 4.33 |

### `tadeaishiyicp4`

- canonical run: `judge/outputs/canonical/tadeaishiyicp4/annotation_20260520_g3`
- candidate pool: 17 条文案，431 条评估，overall mean=2.09

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 25.0% | 50.0% | 45.0% | 67.5% | 81.7% | 25.0% | 25.0% | 25.0% | 100.0% | 6.00 | 7.00 | 3.40 | 1.00 | 25.0% | 0.0% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 3.40 | 2.55 ~ 4.40 | 4.25 | 2.50 | 2.50 | 4.50 | 3.00 |

### `huakaishijiedj6`

- canonical run: `judge/outputs/canonical/huakaishijiedj6/annotation_20260520_g3`
- candidate pool: 31 条文案，541 条评估，overall mean=2.18

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 33.3% | 55.6% | 46.7% | 60.0% | 87.5% | 33.3% | 66.7% | 66.7% | 100.0% | 3.67 | 2.50 | 3.79 | 0.74 | 50.0% | 33.3% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 3.79 | 2.25 ~ 5.00 | 4.00 | 3.50 | 4.33 | 3.83 | 3.33 |

### `shaguozhouyufnx`

- canonical run: `judge/outputs/canonical/shaguozhouyufnx/annotation_20260520_g3`
- candidate pool: 10 条文案，258 条评估，overall mean=1.71

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 25.0% | 50.0% | 60.0% | 67.5% | 77.5% | 25.0% | 75.0% | 100.0% | 100.0% | 2.25 | 2.00 | 3.49 | 0.73 | 50.0% | 25.0% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 3.49 | 1.75 ~ 4.80 | 4.25 | 3.00 | 3.75 | 3.50 | 3.00 |

### `huibayumudangyuz`

- canonical run: `judge/outputs/canonical/huibayumudangyuz/annotation_20260520_g4`
- candidate pool: 27 条文案，819 条评估，overall mean=2.22

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 40.0% | 40.0% | 52.0% | 54.0% | 58.0% | 40.0% | 60.0% | 60.0% | 100.0% | 4.00 | 2.00 | 3.85 | 0.50 | 60.0% | 40.0% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 3.85 | 2.55 ~ 4.70 | 4.60 | 3.40 | 3.20 | 4.20 | 3.80 |

### `beicaiyhwfszhj`

- canonical run: `judge/outputs/canonical/beicaiyhwfszhj/annotation_20260520_g4`
- candidate pool: 17 条文案，380 条评估，overall mean=2.07

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 25.0% | 50.0% | 45.0% | 75.0% | 91.7% | 25.0% | 50.0% | 50.0% | 100.0% | 4.50 | 4.00 | 3.51 | 0.71 | 50.0% | 50.0% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 3.51 | 1.55 ~ 5.00 | 4.25 | 3.25 | 3.25 | 3.50 | 3.25 |

### `beizhahousgeztcs`

- canonical run: `judge/outputs/canonical/beizhahousgeztcs/annotation_20260520_g4`
- candidate pool: 34 条文案，593 条评估，overall mean=2.43

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 20.0% | 33.3% | 48.0% | 70.0% | - | 20.0% | 60.0% | 80.0% | 100.0% | 3.20 | 2.00 | 4.97 | 0.03 | 100.0% | 100.0% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 4.97 | 4.85 ~ 5.00 | 5.00 | 5.00 | 4.80 | 5.00 | 5.00 |

### `shuyuwutongqyml`

- canonical run: `judge/outputs/canonical/shuyuwutongqyml/annotation_20260520_g4`
- candidate pool: 51 条文案，1233 条评估，overall mean=1.98

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 30.0% | 40.0% | 42.0% | 51.0% | 67.5% | 30.0% | 50.0% | 60.0% | 90.0% | 9.90 | 3.00 | 3.27 | 1.32 | 40.0% | 40.0% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 3.27 | 1.35 ~ 5.00 | 3.90 | 2.90 | 3.80 | 3.00 | 3.00 |

### `xiaoshuziclwdj`

- canonical run: `judge/outputs/canonical/xiaoshuziclwdj/annotation_20260520_g4`
- candidate pool: 14 条文案，53 条评估，overall mean=2.69

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 50.0% | 66.7% | 100.0% | - | - | 50.0% | 100.0% | 100.0% | 100.0% | 1.75 | 1.50 | 3.98 | 0.24 | 50.0% | 50.0% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 3.98 | 3.00 ~ 5.00 | 4.00 | 3.75 | 4.75 | 3.75 | 4.00 |

### `gongsuotangli`

- canonical run: `judge/outputs/canonical/gongsuotangli/annotation_20260520_g4`
- candidate pool: 47 条文案，1036 条评估，overall mean=2.24

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 33.3% | 27.8% | 36.7% | 53.3% | 76.7% | 33.3% | 50.0% | 50.0% | 83.3% | 9.33 | 7.00 | 3.16 | 1.05 | 50.0% | 33.3% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 3.16 | 1.35 ~ 5.00 | 3.50 | 2.83 | 4.00 | 2.83 | 3.00 |

### `wokaoyizhangwoyazuiwannvefanpai`

- canonical run: `judge/outputs/canonical/wokaoyizhangwoyazuiwannvefanpai/annotation_20260520_g4`
- candidate pool: 23 条文案，424 条评估，overall mean=1.76

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 0.0% | 16.7% | 53.3% | 60.0% | 76.2% | 0.0% | 0.0% | 83.3% | 100.0% | 5.33 | 4.50 | 2.09 | 1.19 | 0.0% | 0.0% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 2.09 | 1.25 ~ 3.45 | 3.17 | 1.67 | 2.17 | 2.00 | 1.33 |

### `maquezennengtongyanfei`

- canonical run: `judge/outputs/canonical/maquezennengtongyanfei/annotation_20260520_g4`
- candidate pool: 11 条文案，99 条评估，overall mean=1.82

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 33.3% | 44.4% | 73.3% | - | - | 33.3% | 100.0% | 100.0% | 100.0% | 2.33 | 3.00 | 2.88 | 0.53 | 0.0% | 0.0% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 2.88 | 1.90 ~ 3.85 | 3.67 | 2.00 | 4.33 | 2.67 | 2.33 |

### `yinianxinkuzaij6`

- canonical run: `judge/outputs/canonical/yinianxinkuzaij6/annotation_20260520_g4`
- candidate pool: 8 条文案，199 条评估，overall mean=1.94

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 16.7% | 33.3% | 43.3% | 70.0% | 88.8% | 16.7% | 66.7% | 83.3% | 83.3% | 5.50 | 2.00 | 3.41 | 0.57 | 50.0% | 33.3% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 3.41 | 1.20 ~ 4.85 | 3.83 | 3.17 | 2.83 | 3.67 | 3.50 |

### `guoqitangshuang`

- canonical run: `judge/outputs/canonical/guoqitangshuang/annotation_20260520_g4`
- candidate pool: 32 条文案，614 条评估，overall mean=2.13

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 14.3% | 38.1% | 51.4% | 64.3% | 84.2% | 14.3% | 42.9% | 71.4% | 100.0% | 6.00 | 5.00 | 3.34 | 1.19 | 14.3% | 0.0% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 3.34 | 2.25 ~ 4.10 | 4.29 | 2.86 | 3.86 | 3.00 | 3.00 |

### `aijiufangzhiqm`

- canonical run: `judge/outputs/canonical/aijiufangzhiqm/annotation_20260520_g4`
- candidate pool: 18 条文案，235 条评估，overall mean=2.33

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 16.7% | 66.7% | 66.7% | 78.3% | - | 16.7% | 66.7% | 100.0% | 100.0% | 2.67 | 2.50 | 2.88 | 0.93 | 16.7% | 16.7% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 2.88 | 2.15 ~ 4.70 | 4.50 | 2.17 | 3.83 | 2.33 | 1.67 |

### `yujunshishiweixd`

- canonical run: `judge/outputs/canonical/yujunshishiweixd/annotation_20260520_g4`
- candidate pool: 22 条文案，982 条评估，overall mean=2.80

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 40.0% | 46.7% | 40.0% | 42.0% | 56.0% | 40.0% | 80.0% | 80.0% | 80.0% | 7.80 | 2.00 | 4.11 | 0.56 | 80.0% | 40.0% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 4.11 | 1.80 ~ 5.00 | 4.40 | 4.00 | 4.20 | 4.00 | 4.00 |

### `renqinhouclnjtk`

- canonical run: `judge/outputs/canonical/renqinhouclnjtk/annotation_20260520_g4`
- candidate pool: 24 条文案，509 条评估，overall mean=2.41

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 50.0% | 33.3% | 30.0% | 65.0% | 87.5% | 50.0% | 50.0% | 50.0% | 100.0% | 5.75 | 5.00 | 4.49 | 0.24 | 100.0% | 75.0% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 4.49 | 4.05 ~ 4.70 | 4.50 | 4.25 | 3.75 | 5.00 | 5.00 |

### `yuanliangwomam1`

- canonical run: `judge/outputs/canonical/yuanliangwomam1/annotation_20260520_g4`
- candidate pool: 27 条文案，608 条评估，overall mean=2.21

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 20.0% | 33.3% | 44.0% | 70.0% | 93.8% | 20.0% | 40.0% | 40.0% | 100.0% | 7.20 | 7.00 | 2.24 | 0.93 | 20.0% | 20.0% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 2.24 | 1.20 ~ 4.60 | 2.80 | 2.00 | 2.00 | 2.40 | 1.80 |

### `qinaidewoyaojhl`

- canonical run: `judge/outputs/canonical/qinaidewoyaojhl/annotation_20260520_g4`
- candidate pool: 11 条文案，462 条评估，overall mean=1.71

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 0.0% | 16.7% | 25.0% | 37.5% | 67.5% | 0.0% | 25.0% | 25.0% | 75.0% | 15.00 | 13.00 | 2.46 | 0.65 | 25.0% | 0.0% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 2.46 | 1.55 ~ 4.40 | 4.50 | 1.75 | 2.25 | 2.00 | 2.00 |

### `yingyingwuzhongm`

- canonical run: `judge/outputs/canonical/yingyingwuzhongm/annotation_20260520_g4`
- candidate pool: 28 条文案，728 条评估，overall mean=1.29

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 0.0% | 22.2% | 26.7% | 30.0% | 73.3% | 0.0% | 0.0% | 0.0% | 66.7% | 14.67 | 15.00 | 1.00 | 0.28 | 0.0% | 0.0% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 1.00 | 1.00 ~ 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

### `hunlidangtgjcl`

- canonical run: `judge/outputs/canonical/hunlidangtgjcl/annotation_20260520_g4`
- candidate pool: 6 条文案，90 条评估，overall mean=2.41

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 33.3% | 55.6% | 66.7% | 50.0% | 87.5% | 33.3% | 66.7% | 100.0% | 100.0% | 2.67 | 3.00 | 3.37 | 0.50 | 33.3% | 33.3% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 3.37 | 2.45 ~ 5.00 | 4.00 | 3.33 | 3.67 | 3.00 | 2.67 |

### `tianjfbzcddcgl`

- canonical run: `judge/outputs/canonical/tianjfbzcddcgl/annotation_20260520_g4`
- candidate pool: 58 条文案，825 条评估，overall mean=1.99

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 33.3% | 37.0% | 62.2% | 76.7% | - | 33.3% | 55.6% | 55.6% | 100.0% | 5.22 | 3.00 | 2.74 | 1.27 | 44.4% | 11.1% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 2.74 | 1.15 ~ 4.55 | 3.00 | 2.33 | 3.00 | 2.89 | 2.67 |

### `xingheandanweijn`

- canonical run: `judge/outputs/canonical/xingheandanweijn/annotation_20260520_g4`
- candidate pool: 28 条文案，588 条评估，overall mean=1.98

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 20.0% | 53.3% | 52.0% | 76.0% | 96.2% | 20.0% | 80.0% | 80.0% | 100.0% | 3.40 | 3.00 | 3.12 | 0.20 | 40.0% | 40.0% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 3.12 | 1.55 ~ 5.00 | 3.80 | 2.80 | 3.40 | 3.00 | 2.60 |

### `wanliheshanwobzq`

- canonical run: `judge/outputs/canonical/wanliheshanwobzq/annotation_20260520_g4`
- candidate pool: 42 条文案，914 条评估，overall mean=2.24

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 83.3% | 50.0% | 44.4% | 50.0% | 61.7% | 91.2% | 50.0% | 66.7% | 66.7% | 83.3% | 3.20 | 1.00 | 4.19 | 0.48 | 50.0% | 50.0% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 4.19 | 2.60 ~ 5.00 | 4.60 | 4.00 | 3.80 | 4.40 | 4.00 |

### `shenmeshizunjsnd`

- canonical run: `judge/outputs/canonical/shenmeshizunjsnd/annotation_20260520_g5`
- candidate pool: 43 条文案，1305 条评估，overall mean=1.71

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 33.3% | 27.8% | 56.7% | 70.0% | 88.3% | 33.3% | 33.3% | 83.3% | 100.0% | 4.67 | 4.00 | 3.07 | 0.90 | 33.3% | 16.7% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 3.07 | 1.15 ~ 4.70 | 3.33 | 2.67 | 3.67 | 3.00 | 3.00 |

### `jiushijiuciyuanl`

- canonical run: `judge/outputs/canonical/jiushijiuciyuanl/annotation_20260520_g5`
- candidate pool: 20 条文案，408 条评估，overall mean=2.11

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 40.0% | 40.0% | 44.0% | 62.0% | 87.5% | 40.0% | 80.0% | 80.0% | 100.0% | 4.20 | 2.00 | 3.52 | 0.31 | 60.0% | 60.0% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 3.52 | 1.20 ~ 4.85 | 3.60 | 3.40 | 2.80 | 4.00 | 3.60 |

### `jiaofuzhifu`

- canonical run: `judge/outputs/canonical/jiaofuzhifu/annotation_20260520_g5`
- candidate pool: 11 条文案，167 条评估，overall mean=1.89

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 0.0% | 50.0% | 40.0% | 72.5% | - | 0.0% | 50.0% | 50.0% | 100.0% | 6.00 | 5.50 | 2.04 | 2.09 | 0.0% | 0.0% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 2.04 | 1.55 ~ 2.60 | 2.75 | 1.75 | 2.75 | 1.50 | 1.75 |

### `yiaizhimingshizs`

- canonical run: `judge/outputs/canonical/yiaizhimingshizs/annotation_20260520_g5`
- candidate pool: 35 条文案，675 条评估，overall mean=1.94

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 20.0% | 20.0% | 44.0% | 70.0% | 87.0% | 20.0% | 60.0% | 100.0% | 100.0% | 2.80 | 2.00 | 3.41 | 0.30 | 40.0% | 0.0% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 3.41 | 2.50 ~ 4.30 | 3.60 | 2.80 | 3.80 | 3.60 | 3.80 |

### `zhuajinshoudggsk`

- canonical run: `judge/outputs/canonical/zhuajinshoudggsk/annotation_20260520_g5`
- candidate pool: 31 条文案，439 条评估，overall mean=2.25

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 50.0% | 66.7% | 70.0% | 85.0% | - | 50.0% | 100.0% | 100.0% | 100.0% | 1.50 | 1.50 | 3.21 | 0.30 | 50.0% | 0.0% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 3.21 | 1.85 ~ 4.45 | 4.00 | 3.00 | 2.75 | 3.00 | 3.50 |

### `baobeidengwf`

- canonical run: `judge/outputs/canonical/baobeidengwf/annotation_20260520_g5`
- candidate pool: 13 条文案，231 条评估，overall mean=2.14

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 25.0% | 41.7% | 75.0% | 80.0% | 97.5% | 25.0% | 25.0% | 75.0% | 100.0% | 5.75 | 4.50 | 2.83 | 0.62 | 25.0% | 25.0% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 2.83 | 1.60 ~ 4.70 | 4.00 | 2.00 | 3.00 | 3.00 | 2.25 |

### `qingmeiyibzmni`

- canonical run: `judge/outputs/canonical/qingmeiyibzmni/annotation_20260520_g5`
- candidate pool: 27 条文案，445 条评估，overall mean=2.40

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 50.0% | 33.3% | 60.0% | 80.0% | 81.2% | 50.0% | 50.0% | 100.0% | 100.0% | 2.75 | 2.50 | 4.54 | 0.25 | 100.0% | 50.0% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 4.54 | 4.15 ~ 5.00 | 4.50 | 4.25 | 4.50 | 4.75 | 5.00 |

### `wangshicangni`

- canonical run: `judge/outputs/canonical/wangshicangni/annotation_20260520_g5`
- candidate pool: 47 条文案，1292 条评估，overall mean=2.10

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 33.3% | 33.3% | 43.3% | 56.7% | 64.2% | 33.3% | 50.0% | 50.0% | 83.3% | 9.17 | 8.00 | 3.32 | 0.73 | 33.3% | 33.3% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 3.32 | 1.40 ~ 5.00 | 4.00 | 2.50 | 3.33 | 3.67 | 3.50 |

### `changmenyuan`

- canonical run: `judge/outputs/canonical/changmenyuan/annotation_20260520_g5`
- candidate pool: 35 条文案，860 条评估，overall mean=2.21

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 40.0% | 26.7% | 52.0% | 52.0% | 83.0% | 40.0% | 40.0% | 40.0% | 100.0% | 7.00 | 7.00 | 3.34 | 0.83 | 40.0% | 40.0% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 3.34 | 1.35 ~ 4.70 | 3.40 | 3.40 | 3.20 | 3.20 | 3.60 |

### `tadeyouxiwodeqit`

- canonical run: `judge/outputs/canonical/tadeyouxiwodeqit/annotation_20260520_g5`
- candidate pool: 7 条文案，109 条评估，overall mean=2.28

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 50.0% | 58.3% | 65.0% | 70.0% | - | 50.0% | 100.0% | 100.0% | 100.0% | 1.75 | 1.50 | 3.42 | 0.36 | 50.0% | 50.0% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 3.42 | 2.00 ~ 4.60 | 4.25 | 3.00 | 3.50 | 3.50 | 2.75 |

### `yaonienaibazaidb`

- canonical run: `judge/outputs/canonical/yaonienaibazaidb/annotation_20260520_g5`
- candidate pool: 27 条文案，615 条评估，overall mean=2.20

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 0.0% | 11.1% | 33.3% | 56.7% | 84.0% | 0.0% | 0.0% | 33.3% | 100.0% | 9.33 | 7.50 | 2.24 | 0.70 | 0.0% | 0.0% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 2.24 | 1.00 ~ 3.90 | 2.83 | 1.83 | 2.83 | 2.00 | 2.00 |

### `chuyuhoumamibcsc`

- canonical run: `judge/outputs/canonical/chuyuhoumamibcsc/annotation_20260520_g5`
- candidate pool: 5 条文案，104 条评估，overall mean=1.91

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 50.0% | 58.3% | 55.0% | 77.5% | 80.0% | 50.0% | 50.0% | 50.0% | 100.0% | 6.25 | 4.00 | 2.71 | 0.47 | 25.0% | 0.0% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 2.71 | 1.45 ~ 4.35 | 3.00 | 2.50 | 3.00 | 2.75 | 2.25 |

### `fenghuoqianjinm`

- canonical run: `judge/outputs/canonical/fenghuoqianjinm/annotation_20260520_g5`
- candidate pool: 80 条文案，1855 条评估，overall mean=2.08

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 10.0% | 36.7% | 50.0% | 68.0% | 95.0% | 10.0% | 60.0% | 70.0% | 100.0% | 5.70 | 3.00 | 2.19 | 1.07 | 10.0% | 0.0% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 2.19 | 1.15 ~ 4.35 | 3.40 | 1.50 | 3.00 | 1.90 | 1.40 |

### `jiehunwunnsanl`

- canonical run: `judge/outputs/canonical/jiehunwunnsanl/annotation_20260520_g5`
- candidate pool: 14 条文案，298 条评估，overall mean=2.21

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 0.0% | 50.0% | 70.0% | 72.5% | 82.5% | 0.0% | 25.0% | 100.0% | 100.0% | 3.75 | 4.00 | 3.42 | 0.94 | 25.0% | 25.0% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 3.42 | 2.55 ~ 4.70 | 4.75 | 3.25 | 4.00 | 2.50 | 2.75 |

### `jushafengba8`

- canonical run: `judge/outputs/canonical/jushafengba8/annotation_20260520_g5`
- candidate pool: 12 条文案，224 条评估，overall mean=1.91

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 0.0% | 50.0% | 65.0% | 67.5% | 91.7% | 0.0% | 75.0% | 75.0% | 100.0% | 5.25 | 2.50 | 3.21 | 0.72 | 25.0% | 0.0% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 3.21 | 2.65 ~ 4.45 | 3.00 | 3.00 | 3.75 | 3.50 | 2.75 |

### `lianshangnidec`

- canonical run: `judge/outputs/canonical/lianshangnidec/annotation_20260520_g5`
- candidate pool: 28 条文案，302 条评估，overall mean=2.52

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 25.0% | 50.0% | 80.0% | 92.5% | - | 25.0% | 50.0% | 75.0% | 100.0% | 3.25 | 3.00 | 3.26 | 1.40 | 50.0% | 25.0% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 3.26 | 1.40 ~ 5.00 | 3.50 | 2.75 | 3.75 | 3.50 | 3.00 |

### `yuhoubufengjux`

- canonical run: `judge/outputs/canonical/yuhoubufengjux/annotation_20260520_g5`
- candidate pool: 22 条文案，434 条评估，overall mean=2.30

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 0.0% | 33.3% | 52.0% | 74.0% | 83.3% | 0.0% | 20.0% | 60.0% | 100.0% | 6.00 | 5.00 | 3.49 | 1.02 | 40.0% | 40.0% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 3.49 | 2.05 ~ 4.75 | 4.00 | 3.00 | 4.80 | 3.00 | 3.20 |

### `baifengshanmiwu`

- canonical run: `judge/outputs/canonical/baifengshanmiwu/annotation_20260520_g5`
- candidate pool: 25 条文案，391 条评估，overall mean=2.00

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 0.0% | 25.0% | 30.0% | 60.0% | 93.3% | 0.0% | 25.0% | 25.0% | 100.0% | 8.00 | 6.00 | 2.17 | 1.09 | 0.0% | 0.0% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 2.17 | 1.35 ~ 3.50 | 3.25 | 1.75 | 1.50 | 2.50 | 1.50 |

### `dashandegongxun`

- canonical run: `judge/outputs/canonical/dashandegongxun/annotation_20260520_g5`
- candidate pool: 19 条文案，473 条评估，overall mean=2.14

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 20.0% | 20.0% | 36.0% | 68.0% | 81.0% | 20.0% | 40.0% | 40.0% | 100.0% | 5.40 | 6.00 | 3.44 | 0.57 | 40.0% | 40.0% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 3.44 | 1.90 ~ 5.00 | 3.40 | 3.40 | 4.00 | 3.20 | 3.40 |

### `xiangyanhuonayang-wuBGM`

- canonical run: `judge/outputs/canonical/xiangyanhuonayang-wuBGM/annotation_20260520_g5`
- candidate pool: 10 条文案，190 条评估，overall mean=1.81

| Variant | 覆盖率 | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@1 | Hit@3 | Hit@5 | Hit@20 | 均 rank | 中位 rank | 系统均分 | 平均分差 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 100.0% | 0.0% | 50.0% | 55.0% | 65.0% | - | 0.0% | 25.0% | 50.0% | 100.0% | 8.00 | 5.00 | 2.16 | 0.98 | 0.0% | 0.0% |

| Variant | overall 均值 | overall 范围 | entity | event | scene | tone | editing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen_q3_rerank_q1` | 2.16 | 1.00 ~ 3.10 | 2.25 | 1.50 | 3.00 | 2.75 | 1.25 |

## 9. 明显错排样本

| 剧名 | Variant | 文案段 | 系统片段 | Judge 排名 | 系统分 | Judge 最优片段 | Judge 最优分 | 分差 | 诊断 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `maopaigjxzwlbe` | `qwen_q3_rerank_q1` | 4 | 1 | 7 | 1.15 | 9 | 2.40 | 1.25 | 候选池不足 |
| `maopaigjxzwlbe` | `qwen_q3_rerank_q1` | 6 | 2 | 2 | 2.25 | 1 | 4.45 | 2.20 | 接近，可优化 |
| `mamazaikanwoyiy` | `qwen_q3_rerank_q1` | 1 | 5 | 4 | 3.95 | 4 | 5.00 | 1.05 | 有好候选但错排 |
| `yinshishenchuzqk` | `qwen_q3_rerank_q1` | 1 | 11 | 4 | 2.55 | 3 | 4.70 | 2.15 | 有好候选但错排 |
| `yinshishenchuzqk` | `qwen_q3_rerank_q1` | 2 | 8 | 3 | 3.95 | 7 | 5.00 | 1.05 | 接近，可优化 |
| `yinshishenchuzqk` | `qwen_q3_rerank_q1` | 3 | 14 | 4 | 2.80 | 13 | 3.80 | 1.00 | 候选池不足 |
| `yinshishenchuzqk` | `qwen_q3_rerank_q1` | 4 | 7 | 21 | 3.05 | 5 | 4.70 | 1.65 | 有好候选但错排 |
| `biezaijiaowomama` | `qwen_q3_rerank_q1` | 1 | 27 | 19 | 2.65 | 4 | 4.15 | 1.50 | 有好候选但错排 |
| `biezaijiaowomama` | `qwen_q3_rerank_q1` | 2 | 6 | 11 | 1.20 | 28 | 1.75 | 0.55 | 候选池不足 |
| `biezaijiaowomama` | `qwen_q3_rerank_q1` | 3 | 13 | 8 | 1.45 | 7 | 3.20 | 1.75 | 候选池不足 |
| `biezaijiaowomama` | `qwen_q3_rerank_q1` | 4 | 30 | 9 | 2.20 | 24 | 5.00 | 2.80 | 有好候选但错排 |
| `biezaijiaowomama` | `qwen_q3_rerank_q1` | 5 | 9 | 11 | 2.65 | 1 | 4.40 | 1.75 | 有好候选但错排 |
| `shuangshenganyan` | `qwen_q3_rerank_q1` | 2 | 11 | 6 | 3.75 | 14 | 4.55 | 0.80 | 排序可优化 |
| `nvwangchusyjnq6` | `qwen_q3_rerank_q1` | 3 | 1 | 8 | 4.15 | 11 | 5.00 | 0.85 | 排序可优化 |
| `ainimeiyoulengjq` | `qwen_q3_rerank_q1` | 3 | 7 | 8 | 2.20 | 5 | 4.40 | 2.20 | 有好候选但错排 |
| `ainimeiyoulengjq` | `qwen_q3_rerank_q1` | 4 | 6 | 12 | 1.20 | 19 | 1.80 | 0.60 | 候选池不足 |
| `tiancimengsggsj` | `qwen_q3_rerank_q1` | 4 | 9 | 11 | 1.70 | 17 | 5.00 | 3.30 | 有好候选但错排 |
| `tiancimengsggsj` | `qwen_q3_rerank_q1` | 5 | 1 | 2 | 3.00 | 2 | 4.45 | 1.45 | 接近，可优化 |
| `due` | `qwen_q3_rerank_q1` | 1 | 5 | 4 | 3.35 | 4 | 4.65 | 1.30 | 有好候选但错排 |
| `due` | `qwen_q3_rerank_q1` | 2 | 7 | 9 | 2.05 | 18 | 4.30 | 2.25 | 有好候选但错排 |
| `due` | `qwen_q3_rerank_q1` | 4 | 15 | 3 | 1.95 | 34 | 3.00 | 1.05 | 候选池不足 |
| `wodeqianghlgslan` | `qwen_q3_rerank_q1` | 2 | 16 | 6 | 2.00 | 18 | 3.10 | 1.10 | 候选池不足 |
| `tianjiangxssjyqc` | `qwen_q3_rerank_q1` | 4 | 14 | 6 | 3.00 | 16 | 5.00 | 2.00 | 有好候选但错排 |
| `caiyuancaidjzfr` | `qwen_q3_rerank_q1` | 2 | 2 | 6 | 4.50 | 3 | 5.00 | 0.50 | 排序可优化 |
| `caiyuancaidjzfr` | `qwen_q3_rerank_q1` | 4 | 22 | 6 | 4.45 | 16 | 4.75 | 0.30 | 排序可优化 |
| `qisuidewoshisjqs` | `qwen_q3_rerank_q1` | 3 | 26 | 5 | 2.45 | 11 | 4.20 | 1.75 | 有好候选但错排 |
| `qiuhunjsjclhym0` | `qwen_q3_rerank_q1` | 2 | 5 | 4 | 2.80 | 18 | 4.55 | 1.75 | 有好候选但错排 |
| `qiuhunjsjclhym0` | `qwen_q3_rerank_q1` | 3 | 14 | 17 | 1.85 | 26 | 3.05 | 1.20 | 候选池不足 |
| `qiuhunjsjclhym0` | `qwen_q3_rerank_q1` | 4 | 26 | 6 | 3.10 | 16 | 4.70 | 1.60 | 有好候选但错排 |
| `qiuhunjsjclhym0` | `qwen_q3_rerank_q1` | 5 | 18 | 9 | 3.55 | 26 | 5.00 | 1.45 | 有好候选但错排 |
| `tianjiangxslfcr` | `qwen_q3_rerank_q1` | 1 | 11 | 10 | 1.45 | 16 | 2.05 | 0.60 | 候选池不足 |
| `tianjiangxslfcr` | `qwen_q3_rerank_q1` | 2 | 31 | 2 | 3.45 | 33 | 4.55 | 1.10 | 接近，可优化 |
| `tianjiangxslfcr` | `qwen_q3_rerank_q1` | 4 | 3 | 5 | 2.00 | 32 | 4.20 | 2.20 | 有好候选但错排 |
| `mingzhongzdwyly` | `qwen_q3_rerank_q1` | 1 | 9 | 6 | 3.05 | 8 | 3.90 | 0.85 | 候选池不足 |
| `niepanchengfenu` | `qwen_q3_rerank_q1` | 1 | 3 | 4 | 1.95 | 2 | 3.25 | 1.30 | 候选池不足 |
| `yishenrujuzbbwy` | `qwen_q3_rerank_q1` | 5 | 4 | 9 | 3.20 | 12 | 4.70 | 1.50 | 有好候选但错排 |
| `wanlaibujianxy` | `qwen_q3_rerank_q1` | 2 | 7 | 10 | 2.00 | 22 | 4.50 | 2.50 | 有好候选但错排 |
| `longtengsihai` | `qwen_q3_rerank_q1` | 4 | 11 | 9 | 1.55 | 20 | 3.35 | 1.80 | 候选池不足 |
| `jiyijiaohuan` | `qwen_q3_rerank_q1` | 2 | 17 | 8 | 2.40 | 10 | 4.80 | 2.40 | 有好候选但错排 |
| `jiyijiaohuan` | `qwen_q3_rerank_q1` | 4 | 6 | 11 | 1.70 | 18 | 4.30 | 2.60 | 有好候选但错排 |
| `hekuxiangsizhuyn` | `qwen_q3_rerank_q1` | 1 | 11 | 37 | 1.20 | 22 | 4.20 | 3.00 | 有好候选但错排 |
| `hekuxiangsizhuyn` | `qwen_q3_rerank_q1` | 3 | 8 | 35 | 1.20 | 11 | 4.05 | 2.85 | 有好候选但错排 |
| `hekuxiangsizhuyn` | `qwen_q3_rerank_q1` | 4 | 52 | 23 | 1.65 | 22 | 4.80 | 3.15 | 有好候选但错排 |
| `hekuxiangsizhuyn` | `qwen_q3_rerank_q1` | 5 | 37 | 6 | 4.80 | 24 | 5.00 | 0.20 | 排序可优化 |
| `kaijushazhujiang` | `qwen_q3_rerank_q1` | 2 | 28 | 17 | 2.05 | 2 | 4.35 | 2.30 | 有好候选但错排 |
| `kaijushazhujiang` | `qwen_q3_rerank_q1` | 4 | 8 | 15 | 1.50 | 11 | 4.85 | 3.35 | 有好候选但错排 |
| `zhongshbnxccb` | `qwen_q3_rerank_q1` | 1 | 26 | 16 | 3.40 | 15 | 4.65 | 1.25 | 有好候选但错排 |
| `zhongshbnxccb` | `qwen_q3_rerank_q1` | 2 | 22 | 10 | 4.90 | 5 | 5.00 | 0.10 | 排序可优化 |
| `woyoushuangcs1` | `qwen_q3_rerank_q1` | 1 | 10 | 3 | 3.35 | 1 | 4.70 | 1.35 | 接近，可优化 |
| `woyoushuangcs1` | `qwen_q3_rerank_q1` | 2 | 13 | 14 | 1.15 | 5 | 4.20 | 3.05 | 有好候选但错排 |
| `nvergjbdhwbzdm` | `qwen_q3_rerank_q1` | 2 | 4 | 11 | 1.55 | 9 | 3.60 | 2.05 | 候选池不足 |
| `nvergjbdhwbzdm` | `qwen_q3_rerank_q1` | 4 | 10 | 6 | 2.65 | 6 | 4.70 | 2.05 | 有好候选但错排 |
| `huoshimengwacnd` | `qwen_q3_rerank_q1` | 1 | 1 | 6 | 1.70 | 4 | 3.35 | 1.65 | 候选池不足 |
| `huoshimengwacnd` | `qwen_q3_rerank_q1` | 3 | 2 | 18 | 1.55 | 23 | 4.30 | 2.75 | 有好候选但错排 |
| `huoshimengwacnd` | `qwen_q3_rerank_q1` | 4 | 4 | 7 | 2.85 | 5 | 4.90 | 2.05 | 有好候选但错排 |
| `yesejiqingren` | `qwen_q3_rerank_q1` | 1 | 8 | 6 | 4.45 | 14 | 5.00 | 0.55 | 排序可优化 |
| `yesejiqingren` | `qwen_q3_rerank_q1` | 7 | 13 | 18 | 1.50 | 23 | 2.35 | 0.85 | 候选池不足 |
| `kanwojifenxcz` | `qwen_q3_rerank_q1` | 3 | 18 | 13 | 3.90 | 23 | 4.55 | 0.65 | 排序可优化 |
| `kanwojifenxcz` | `qwen_q3_rerank_q1` | 5 | 14 | 11 | 1.50 | 10 | 3.05 | 1.55 | 候选池不足 |
| `bierejifuren` | `qwen_q3_rerank_q1` | 1 | 2 | 26 | 1.20 | 15 | 3.45 | 2.25 | 候选池不足 |
| `bierejifuren` | `qwen_q3_rerank_q1` | 3 | 5 | 37 | 2.15 | 21 | 5.00 | 2.85 | 有好候选但错排 |
| `naniancxhssbfxb` | `qwen_q3_rerank_q1` | 1 | 8 | 7 | 1.90 | 15 | 3.45 | 1.55 | 候选池不足 |
| `naniancxhssbfxb` | `qwen_q3_rerank_q1` | 2 | 1 | 12 | 1.00 | 10 | 2.45 | 1.45 | 候选池不足 |
| `naniancxhssbfxb` | `qwen_q3_rerank_q1` | 4 | 8 | 11 | 1.15 | 10 | 4.80 | 3.65 | 有好候选但错排 |
| `sancunrenxinbukl` | `qwen_q3_rerank_q1` | 5 | 2 | 3 | 3.10 | 3 | 4.55 | 1.45 | 接近，可优化 |
| `shangjienvhuans` | `qwen_q3_rerank_q1` | 4 | 9 | 5 | 3.65 | 7 | 4.70 | 1.05 | 有好候选但错排 |
| `touhuansjhqzhdn` | `qwen_q3_rerank_q1` | 1 | 5 | 12 | 3.00 | 9 | 4.70 | 1.70 | 有好候选但错排 |
| `touhuansjhqzhdn` | `qwen_q3_rerank_q1` | 2 | 1 | 11 | 1.90 | 3 | 3.35 | 1.45 | 候选池不足 |
| `mengxianglouxdhg` | `qwen_q3_rerank_q1` | 4 | 1 | 3 | 2.20 | 3 | 4.05 | 1.85 | 接近，可优化 |
| `zhumafeilrwzdgm` | `qwen_q3_rerank_q1` | 2 | 20 | 14 | 1.75 | 8 | 4.35 | 2.60 | 有好候选但错排 |
| `zhumafeilrwzdgm` | `qwen_q3_rerank_q1` | 4 | 14 | 6 | 1.65 | 6 | 2.35 | 0.70 | 候选池不足 |
| `pugsfhsgynzhm5` | `qwen_q3_rerank_q1` | 2 | 12 | 10 | 1.75 | 8 | 4.70 | 2.95 | 有好候选但错排 |
| `pugsfhsgynzhm5` | `qwen_q3_rerank_q1` | 3 | 10 | 6 | 4.65 | 7 | 5.00 | 0.35 | 排序可优化 |
| `yangerfanglzmmbk` | `qwen_q3_rerank_q1` | 1 | 14 | 14 | 1.00 | 3 | 1.70 | 0.70 | 候选池不足 |
| `yangerfanglzmmbk` | `qwen_q3_rerank_q1` | 3 | 8 | 3 | 2.95 | 2 | 4.15 | 1.20 | 接近，可优化 |
| `chunfengyidujsa` | `qwen_q3_rerank_q1` | 4 | 2 | 14 | 1.50 | 16 | 2.25 | 0.75 | 候选池不足 |
| `linglongnuhuotsh` | `qwen_q3_rerank_q1` | 3 | 6 | 12 | 1.75 | 22 | 4.85 | 3.10 | 有好候选但错排 |
| `huangyanlidenver` | `qwen_q3_rerank_q1` | 2 | 2 | 14 | 1.70 | 5 | 4.65 | 2.95 | 有好候选但错排 |
| `mamaqingdaihg` | `qwen_q3_rerank_q1` | 2 | 6 | 12 | 1.60 | 13 | 3.00 | 1.40 | 候选池不足 |
| `mamaqingdaihg` | `qwen_q3_rerank_q1` | 4 | 10 | 4 | 2.10 | 11 | 4.45 | 2.35 | 有好候选但错排 |
| `mamaqingdaihg` | `qwen_q3_rerank_q1` | 6 | 14 | 14 | 1.00 | 13 | 1.60 | 0.60 | 候选池不足 |
| `gongmianyifantdd` | `qwen_q3_rerank_q1` | 1 | 7 | 8 | 1.90 | 6 | 4.45 | 2.55 | 有好候选但错排 |
| `gongmianyifantdd` | `qwen_q3_rerank_q1` | 2 | 2 | 16 | 1.90 | 9 | 4.75 | 2.85 | 有好候选但错排 |
| `gongmianyifantdd` | `qwen_q3_rerank_q1` | 5 | 16 | 36 | 1.15 | 9 | 3.65 | 2.50 | 候选池不足 |
| `shuitoulewdlqtzs` | `qwen_q3_rerank_q1` | 1 | 31 | 12 | 1.30 | 17 | 4.20 | 2.90 | 有好候选但错排 |
| `shuitoulewdlqtzs` | `qwen_q3_rerank_q1` | 2 | 17 | 19 | 1.00 | 25 | 1.65 | 0.65 | 候选池不足 |
| `sanshengsansxfqc` | `qwen_q3_rerank_q1` | 3 | 19 | 15 | 2.45 | 18 | 4.35 | 1.90 | 有好候选但错排 |
| `sanshengsansxfqc` | `qwen_q3_rerank_q1` | 5 | 1 | 6 | 2.10 | 10 | 3.55 | 1.45 | 候选池不足 |
| `kuwodexunirenl` | `qwen_q3_rerank_q1` | 1 | 1 | 6 | 3.05 | 13 | 4.40 | 1.35 | 有好候选但错排 |
| `kuwodexunirenl` | `qwen_q3_rerank_q1` | 2 | 1 | 3 | 3.30 | 5 | 4.50 | 1.20 | 接近，可优化 |
| `kuwodexunirenl` | `qwen_q3_rerank_q1` | 3 | 3 | 3 | 1.95 | 1 | 3.05 | 1.10 | 候选池不足 |
| `kuwodexunirenl` | `qwen_q3_rerank_q1` | 4 | 9 | 10 | 1.00 | 5 | 3.05 | 2.05 | 候选池不足 |
| `miyun` | `qwen_q3_rerank_q1` | 4 | 5 | 11 | 1.80 | 11 | 2.30 | 0.50 | 候选池不足 |
| `aiyiranjinqscl` | `qwen_q3_rerank_q1` | 1 | 1 | 3 | 2.05 | 2 | 3.05 | 1.00 | 候选池不足 |
| `aiyiranjinqscl` | `qwen_q3_rerank_q1` | 2 | 4 | 10 | 3.50 | 29 | 4.70 | 1.20 | 有好候选但错排 |
| `popozaihunexmfqv` | `qwen_q3_rerank_q1` | 2 | 3 | 16 | 3.30 | 2 | 4.70 | 1.40 | 有好候选但错排 |
| `popozaihunexmfqv` | `qwen_q3_rerank_q1` | 3 | 24 | 7 | 1.80 | 21 | 4.15 | 2.35 | 有好候选但错排 |
| `popozaihunexmfqv` | `qwen_q3_rerank_q1` | 4 | 9 | 15 | 1.35 | 16 | 4.85 | 3.50 | 有好候选但错排 |
| `popozaihunexmfqv` | `qwen_q3_rerank_q1` | 5 | 19 | 2 | 2.85 | 20 | 4.55 | 1.70 | 接近，可优化 |
| `popozaihunexmfqv` | `qwen_q3_rerank_q1` | 6 | 3 | 7 | 2.15 | 6 | 3.05 | 0.90 | 候选池不足 |
| `fangrennanyouqingmeizuoyongganxiaoyang` | `qwen_q3_rerank_q1` | 2 | 32 | 16 | 2.30 | 25 | 4.70 | 2.40 | 有好候选但错排 |
| `fangrennanyouqingmeizuoyongganxiaoyang` | `qwen_q3_rerank_q1` | 3 | 3 | 14 | 1.65 | 7 | 4.70 | 3.05 | 有好候选但错排 |
| `fangrennanyouqingmeizuoyongganxiaoyang` | `qwen_q3_rerank_q1` | 4 | 3 | 12 | 1.50 | 24 | 4.50 | 3.00 | 有好候选但错排 |
| `longyumingwang` | `qwen_q3_rerank_q1` | 3 | 3 | 4 | 1.45 | 1 | 3.45 | 2.00 | 候选池不足 |
| `ruhourongguanb` | `qwen_q3_rerank_q1` | 2 | 10 | 4 | 3.10 | 9 | 4.50 | 1.40 | 有好候选但错排 |
| `tadeaishiyicp4` | `qwen_q3_rerank_q1` | 1 | 9 | 8 | 2.55 | 17 | 4.40 | 1.85 | 有好候选但错排 |
| `tadeaishiyicp4` | `qwen_q3_rerank_q1` | 3 | 23 | 9 | 3.20 | 17 | 4.25 | 1.05 | 有好候选但错排 |
| `tadeaishiyicp4` | `qwen_q3_rerank_q1` | 4 | 12 | 6 | 3.45 | 13 | 4.55 | 1.10 | 有好候选但错排 |
| `huakaishijiedj6` | `qwen_q3_rerank_q1` | 1 | 12 | 6 | 3.25 | 11 | 4.70 | 1.45 | 有好候选但错排 |
| `huakaishijiedj6` | `qwen_q3_rerank_q1` | 2 | 7 | 9 | 4.15 | 8 | 5.00 | 0.85 | 排序可优化 |
| `huakaishijiedj6` | `qwen_q3_rerank_q1` | 5 | 21 | 2 | 3.25 | 23 | 4.25 | 1.00 | 接近，可优化 |
| `huakaishijiedj6` | `qwen_q3_rerank_q1` | 6 | 2 | 3 | 2.25 | 1 | 3.40 | 1.15 | 候选池不足 |
| `shaguozhouyufnx` | `qwen_q3_rerank_q1` | 3 | 8 | 4 | 1.75 | 26 | 3.95 | 2.20 | 候选池不足 |
| `huibayumudangyuz` | `qwen_q3_rerank_q1` | 1 | 51 | 8 | 2.55 | 61 | 3.90 | 1.35 | 候选池不足 |
| `huibayumudangyuz` | `qwen_q3_rerank_q1` | 5 | 30 | 8 | 4.15 | 34 | 5.00 | 0.85 | 排序可优化 |
| `beicaiyhwfszhj` | `qwen_q3_rerank_q1` | 2 | 10 | 6 | 2.85 | 7 | 4.80 | 1.95 | 有好候选但错排 |
| `beicaiyhwfszhj` | `qwen_q3_rerank_q1` | 4 | 6 | 9 | 1.55 | 7 | 2.40 | 0.85 | 候选池不足 |
| `beizhahousgeztcs` | `qwen_q3_rerank_q1` | 5 | 17 | 6 | 5.00 | 4 | 5.00 | 0.00 | 排序可优化 |
| `shuyuwutongqyml` | `qwen_q3_rerank_q1` | 3 | 1 | 12 | 2.15 | 2 | 4.65 | 2.50 | 有好候选但错排 |
| `shuyuwutongqyml` | `qwen_q3_rerank_q1` | 4 | 21 | 4 | 3.10 | 20 | 4.55 | 1.45 | 有好候选但错排 |
| `shuyuwutongqyml` | `qwen_q3_rerank_q1` | 5 | 15 | 48 | 1.35 | 12 | 4.85 | 3.50 | 有好候选但错排 |
| `shuyuwutongqyml` | `qwen_q3_rerank_q1` | 6 | 2 | 20 | 2.15 | 54 | 4.70 | 2.55 | 有好候选但错排 |
| `shuyuwutongqyml` | `qwen_q3_rerank_q1` | 10 | 29 | 8 | 2.20 | 45 | 5.00 | 2.80 | 有好候选但错排 |
| `gongsuotangli` | `qwen_q3_rerank_q1` | 1 | 9 | 22 | 1.95 | 27 | 4.10 | 2.15 | 有好候选但错排 |
| `gongsuotangli` | `qwen_q3_rerank_q1` | 5 | 8 | 18 | 1.35 | 1 | 2.00 | 0.65 | 候选池不足 |
| `gongsuotangli` | `qwen_q3_rerank_q1` | 6 | 13 | 11 | 1.65 | 1 | 4.85 | 3.20 | 有好候选但错排 |
| `wokaoyizhangwoyazuiwannvefanpai` | `qwen_q3_rerank_q1` | 2 | 12 | 5 | 1.75 | 18 | 4.00 | 2.25 | 有好候选但错排 |
| `wokaoyizhangwoyazuiwannvefanpai` | `qwen_q3_rerank_q1` | 4 | 5 | 4 | 2.50 | 22 | 4.45 | 1.95 | 有好候选但错排 |
| `wokaoyizhangwoyazuiwannvefanpai` | `qwen_q3_rerank_q1` | 5 | 16 | 10 | 1.35 | 1 | 2.35 | 1.00 | 候选池不足 |
| `wokaoyizhangwoyazuiwannvefanpai` | `qwen_q3_rerank_q1` | 6 | 7 | 5 | 1.25 | 11 | 2.30 | 1.05 | 候选池不足 |
| `maquezennengtongyanfei` | `qwen_q3_rerank_q1` | 2 | 5 | 3 | 2.90 | 6 | 4.25 | 1.35 | 接近，可优化 |
| `yinianxinkuzaij6` | `qwen_q3_rerank_q1` | 3 | 16 | 21 | 1.20 | 3 | 2.85 | 1.65 | 候选池不足 |
| `guoqitangshuang` | `qwen_q3_rerank_q1` | 1 | 2 | 5 | 3.75 | 9 | 5.00 | 1.25 | 有好候选但错排 |
| `guoqitangshuang` | `qwen_q3_rerank_q1` | 3 | 4 | 7 | 3.30 | 16 | 4.70 | 1.40 | 有好候选但错排 |
| `guoqitangshuang` | `qwen_q3_rerank_q1` | 4 | 13 | 5 | 2.40 | 10 | 4.45 | 2.05 | 有好候选但错排 |
| `guoqitangshuang` | `qwen_q3_rerank_q1` | 6 | 3 | 18 | 2.25 | 14 | 4.85 | 2.60 | 有好候选但错排 |
| `aijiufangzhiqm` | `qwen_q3_rerank_q1` | 1 | 6 | 4 | 2.80 | 8 | 4.35 | 1.55 | 有好候选但错排 |
| `aijiufangzhiqm` | `qwen_q3_rerank_q1` | 2 | 5 | 4 | 2.15 | 8 | 4.00 | 1.85 | 有好候选但错排 |
| `aijiufangzhiqm` | `qwen_q3_rerank_q1` | 5 | 10 | 3 | 2.80 | 12 | 4.50 | 1.70 | 接近，可优化 |
| `yujunshishiweixd` | `qwen_q3_rerank_q1` | 1 | 2 | 33 | 1.80 | 18 | 4.25 | 2.45 | 有好候选但错排 |
| `renqinhouclnjtk` | `qwen_q3_rerank_q1` | 1 | 12 | 12 | 4.55 | 6 | 5.00 | 0.45 | 排序可优化 |
| `renqinhouclnjtk` | `qwen_q3_rerank_q1` | 4 | 17 | 9 | 4.05 | 10 | 4.55 | 0.50 | 排序可优化 |
| `yuanliangwomam1` | `qwen_q3_rerank_q1` | 1 | 20 | 16 | 1.85 | 1 | 4.10 | 2.25 | 有好候选但错排 |
| `yuanliangwomam1` | `qwen_q3_rerank_q1` | 3 | 5 | 9 | 1.20 | 13 | 2.20 | 1.00 | 候选池不足 |
| `yuanliangwomam1` | `qwen_q3_rerank_q1` | 5 | 9 | 7 | 1.40 | 6 | 2.55 | 1.15 | 候选池不足 |
| `qinaidewoyaojhl` | `qwen_q3_rerank_q1` | 2 | 3 | 32 | 1.55 | 27 | 3.70 | 2.15 | 候选池不足 |
| `qinaidewoyaojhl` | `qwen_q3_rerank_q1` | 3 | 28 | 13 | 1.95 | 12 | 2.10 | 0.15 | 候选池不足 |
| `qinaidewoyaojhl` | `qwen_q3_rerank_q1` | 4 | 33 | 13 | 1.95 | 32 | 2.25 | 0.30 | 候选池不足 |
| `yingyingwuzhongm` | `qwen_q3_rerank_q1` | 1 | 8 | 8 | 1.00 | 1 | 1.00 | 0.00 | 候选池不足 |
| `yingyingwuzhongm` | `qwen_q3_rerank_q1` | 2 | 14 | 15 | 1.00 | 23 | 1.45 | 0.45 | 候选池不足 |
| `yingyingwuzhongm` | `qwen_q3_rerank_q1` | 3 | 18 | 21 | 1.00 | 20 | 1.40 | 0.40 | 候选池不足 |
| `hunlidangtgjcl` | `qwen_q3_rerank_q1` | 2 | 16 | 3 | 2.45 | 26 | 3.95 | 1.50 | 候选池不足 |
| `tianjfbzcddcgl` | `qwen_q3_rerank_q1` | 2 | 3 | 3 | 1.95 | 10 | 4.00 | 2.05 | 接近，可优化 |
| `tianjfbzcddcgl` | `qwen_q3_rerank_q1` | 4 | 5 | 9 | 1.25 | 11 | 4.40 | 3.15 | 有好候选但错排 |
| `tianjfbzcddcgl` | `qwen_q3_rerank_q1` | 5 | 4 | 14 | 1.15 | 17 | 2.45 | 1.30 | 候选池不足 |
| `tianjfbzcddcgl` | `qwen_q3_rerank_q1` | 7 | 8 | 7 | 1.35 | 10 | 4.60 | 3.25 | 有好候选但错排 |
| `tianjfbzcddcgl` | `qwen_q3_rerank_q1` | 8 | 2 | 8 | 1.50 | 11 | 3.00 | 1.50 | 候选池不足 |
| `xingheandanweijn` | `qwen_q3_rerank_q1` | 1 | 20 | 8 | 1.80 | 15 | 2.25 | 0.45 | 候选池不足 |
| `wanliheshanwobzq` | `qwen_q3_rerank_q1` | 5 | 15 | 10 | 2.60 | 10 | 4.55 | 1.95 | 有好候选但错排 |
| `shenmeshizunjsnd` | `qwen_q3_rerank_q1` | 2 | 13 | 13 | 1.15 | 16 | 2.55 | 1.40 | 候选池不足 |
| `shenmeshizunjsnd` | `qwen_q3_rerank_q1` | 3 | 24 | 5 | 1.85 | 23 | 4.30 | 2.45 | 有好候选但错排 |
| `jiushijiuciyuanl` | `qwen_q3_rerank_q1` | 5 | 6 | 15 | 1.20 | 23 | 2.00 | 0.80 | 候选池不足 |
| `jiaofuzhifu` | `qwen_q3_rerank_q1` | 1 | 13 | 8 | 1.55 | 5 | 3.70 | 2.15 | 候选池不足 |
| `jiaofuzhifu` | `qwen_q3_rerank_q1` | 2 | 5 | 3 | 2.60 | 16 | 3.80 | 1.20 | 候选池不足 |
| `jiaofuzhifu` | `qwen_q3_rerank_q1` | 3 | 15 | 10 | 1.65 | 7 | 4.50 | 2.85 | 有好候选但错排 |
| `jiaofuzhifu` | `qwen_q3_rerank_q1` | 4 | 13 | 3 | 2.35 | 12 | 4.50 | 2.15 | 接近，可优化 |
| `baobeidengwf` | `qwen_q3_rerank_q1` | 3 | 16 | 13 | 1.60 | 10 | 2.05 | 0.45 | 候选池不足 |
| `baobeidengwf` | `qwen_q3_rerank_q1` | 4 | 4 | 5 | 2.35 | 20 | 4.10 | 1.75 | 有好候选但错排 |
| `wangshicangni` | `qwen_q3_rerank_q1` | 2 | 5 | 15 | 3.35 | 12 | 4.85 | 1.50 | 有好候选但错排 |
| `wangshicangni` | `qwen_q3_rerank_q1` | 4 | 31 | 14 | 2.45 | 29 | 4.65 | 2.20 | 有好候选但错排 |
| `wangshicangni` | `qwen_q3_rerank_q1` | 6 | 50 | 22 | 1.40 | 20 | 1.80 | 0.40 | 候选池不足 |
| `changmenyuan` | `qwen_q3_rerank_q1` | 2 | 23 | 13 | 1.35 | 14 | 2.15 | 0.80 | 候选池不足 |
| `changmenyuan` | `qwen_q3_rerank_q1` | 3 | 2 | 7 | 3.65 | 5 | 4.70 | 1.05 | 有好候选但错排 |
| `changmenyuan` | `qwen_q3_rerank_q1` | 5 | 27 | 13 | 2.35 | 26 | 4.65 | 2.30 | 有好候选但错排 |
| `tadeyouxiwodeqit` | `qwen_q3_rerank_q1` | 3 | 9 | 2 | 2.00 | 7 | 3.05 | 1.05 | 候选池不足 |
| `yaonienaibazaidb` | `qwen_q3_rerank_q1` | 1 | 19 | 6 | 3.90 | 20 | 5.00 | 1.10 | 有好候选但错排 |
| `yaonienaibazaidb` | `qwen_q3_rerank_q1` | 2 | 15 | 19 | 1.35 | 4 | 2.60 | 1.25 | 候选池不足 |
| `yaonienaibazaidb` | `qwen_q3_rerank_q1` | 3 | 9 | 9 | 3.70 | 24 | 4.65 | 0.95 | 排序可优化 |
| `yaonienaibazaidb` | `qwen_q3_rerank_q1` | 4 | 13 | 13 | 1.00 | 1 | 1.00 | 0.00 | 候选池不足 |
| `chuyuhoumamibcsc` | `qwen_q3_rerank_q1` | 2 | 18 | 16 | 1.45 | 17 | 2.70 | 1.25 | 候选池不足 |
| `chuyuhoumamibcsc` | `qwen_q3_rerank_q1` | 4 | 10 | 7 | 4.35 | 11 | 5.00 | 0.65 | 排序可优化 |
| `fenghuoqianjinm` | `qwen_q3_rerank_q1` | 2 | 1 | 5 | 1.80 | 17 | 4.45 | 2.65 | 有好候选但错排 |
| `fenghuoqianjinm` | `qwen_q3_rerank_q1` | 3 | 1 | 12 | 1.90 | 10 | 4.90 | 3.00 | 有好候选但错排 |
| `fenghuoqianjinm` | `qwen_q3_rerank_q1` | 6 | 1 | 2 | 2.60 | 8 | 4.10 | 1.50 | 接近，可优化 |
| `fenghuoqianjinm` | `qwen_q3_rerank_q1` | 7 | 1 | 16 | 1.15 | 18 | 2.90 | 1.75 | 候选池不足 |
| `fenghuoqianjinm` | `qwen_q3_rerank_q1` | 10 | 21 | 11 | 1.60 | 10 | 2.20 | 0.60 | 候选池不足 |
| `jiehunwunnsanl` | `qwen_q3_rerank_q1` | 3 | 6 | 5 | 2.85 | 5 | 4.60 | 1.75 | 有好候选但错排 |
| `jushafengba8` | `qwen_q3_rerank_q1` | 2 | 23 | 14 | 2.70 | 4 | 4.70 | 2.00 | 有好候选但错排 |
| `lianshangnidec` | `qwen_q3_rerank_q1` | 2 | 5 | 6 | 1.40 | 10 | 4.30 | 2.90 | 有好候选但错排 |
| `lianshangnidec` | `qwen_q3_rerank_q1` | 4 | 6 | 4 | 2.65 | 4 | 5.00 | 2.35 | 有好候选但错排 |
| `yuhoubufengjux` | `qwen_q3_rerank_q1` | 2 | 4 | 5 | 2.30 | 5 | 4.15 | 1.85 | 有好候选但错排 |
| `yuhoubufengjux` | `qwen_q3_rerank_q1` | 4 | 4 | 12 | 2.05 | 2 | 4.40 | 2.35 | 有好候选但错排 |
| `yuhoubufengjux` | `qwen_q3_rerank_q1` | 5 | 2 | 6 | 4.70 | 6 | 5.00 | 0.30 | 排序可优化 |
| `baifengshanmiwu` | `qwen_q3_rerank_q1` | 1 | 2 | 6 | 1.35 | 3 | 1.75 | 0.40 | 候选池不足 |
| `baifengshanmiwu` | `qwen_q3_rerank_q1` | 3 | 14 | 6 | 3.50 | 8 | 4.30 | 0.80 | 排序可优化 |
| `baifengshanmiwu` | `qwen_q3_rerank_q1` | 4 | 9 | 18 | 1.75 | 2 | 4.70 | 2.95 | 有好候选但错排 |
| `dashandegongxun` | `qwen_q3_rerank_q1` | 1 | 7 | 10 | 1.90 | 2 | 2.85 | 0.95 | 候选池不足 |
| `dashandegongxun` | `qwen_q3_rerank_q1` | 2 | 21 | 6 | 2.10 | 15 | 2.65 | 0.55 | 候选池不足 |
| `dashandegongxun` | `qwen_q3_rerank_q1` | 5 | 18 | 8 | 3.50 | 28 | 4.55 | 1.05 | 有好候选但错排 |
| `xiangyanhuonayang-wuBGM` | `qwen_q3_rerank_q1` | 1 | 8 | 6 | 2.00 | 11 | 3.10 | 1.10 | 候选池不足 |
| `xiangyanhuonayang-wuBGM` | `qwen_q3_rerank_q1` | 2 | 15 | 3 | 3.10 | 14 | 4.20 | 1.10 | 接近，可优化 |
| `xiangyanhuonayang-wuBGM` | `qwen_q3_rerank_q1` | 4 | 15 | 19 | 1.00 | 19 | 2.00 | 1.00 | 候选池不足 |
