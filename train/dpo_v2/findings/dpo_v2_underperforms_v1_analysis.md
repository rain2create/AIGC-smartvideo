# DPO v2 不如 DPO v1 的原因复盘

> 记录时间：2026-06-05  
> 关联文档：`train/dpo_v2/findings/dpo_v2_hard_mining_design.md`  
> 对比报告：`judge/outputs/compare_dpov2_train.md`、`judge/outputs/compare_dpov2_val.md`、`judge/outputs/compare_dpov2_test.md`

## 1. 结论

DPO v2 的 hard-mining 方向不是完全错误，但当前版本不适合作为主线替代 DPO v1。

实测结果显示：

- DPO v2 明显优于 base，说明 rank-based hard pair 确实有纠错价值。
- DPO v2 在 train / val / test 上都没有超过 DPO2000，说明问题不是泛化波动，而是数据构造和训练目标本身没有 v1 对齐最终排序指标。
- 当前最稳主线仍然是 `Base -> DPO2000 -> AWR`。
- DPO v2 更适合作为 DPO2000 后的小步 hard-case patch，而不是从 base 单独训练成最终模型。

## 2. 核心实验结果

### Train split

train 集 2218 条文案，样本量足够判断方向。

| Variant | Hit@1 | Hit@3 | Hit@5 | R@5 | NDCG@5 | 系统均分 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base `qwen_q3_rerank_q1` | 26.9% | 54.5% | 68.3% | 53.1% | 0.84 | 3.19 | 36.4% | 25.8% |
| DPO v1 `qwen_q3_rerank_q1_dpo2000` | 30.9% | 59.7% | 73.2% | 56.3% | 0.87 | 3.32 | 42.2% | 30.3% |
| DPO v2 `qwen_q3_rerank_q1_dpo_v21200` | 29.8% | 57.1% | 70.7% | 53.9% | 0.85 | 3.26 | 41.1% | 29.3% |

v2 相比 base 是正收益，但相比 v1 全面偏弱：

- Hit@1 低 1.1pp
- Hit@3 低 2.7pp
- Hit@5 低 2.5pp
- R@5 低 2.4pp
- NDCG@5 低 0.02
- 系统均分低 0.06
- >=4.0 少 23 条
- >=4.5 少 23 条

### Test split

test 集 183 条文案，结论与 train 一致。

| Variant | Hit@1 | Hit@3 | Hit@5 | R@5 | NDCG@5 | 系统均分 | >=4.0 | >=4.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | 27.3% | 48.6% | 63.4% | 52.1% | 0.835 | 3.16 | 36.1% | 25.1% |
| DPO v1 | 30.6% | 53.6% | 68.3% | 53.3% | 0.846 | 3.24 | 39.3% | 29.0% |
| DPO v2-1200 | 31.1% | 48.6% | 63.9% | 49.6% | 0.824 | 3.14 | 37.7% | 26.2% |

v2-1200 的 Hit@1 比 v1 多 1 条（57/183 vs 56/183），但这不是实质胜出。它同时损失了：

- Hit@3 少 9 条
- Hit@5 少 8 条
- >=4.0 少 3 条
- >=4.5 少 5 条
- 系统均分低 0.096
- NDCG@5 低 0.022

这说明 v2 更激进地换 top1，但整体 top-K 和 best_match 质量不如 v1 稳。

## 3. 数据差异

### DPO v1 数据

`judge/training_data/v3/splits/train_pairs.jsonl`

| 指标 | 数值 |
|---|---:|
| pairs | 21,596 |
| unique queries | 1,506 |
| pairs/query 均值 | 14.34 |
| unique chosen candidates | 3,978 |
| unique rejected candidates | 8,011 |

pair type:

| pair type | 数量 |
|---|---:|
| strong_vs_hard | 10,701 |
| strong_vs_easy | 7,706 |
| fallback_vs_hard | 3,189 |

### DPO v2 数据

`train/dpo_v2/train_pairs.jsonl`

| 指标 | 数值 |
|---|---:|
| pairs | 6,797 |
| unique queries | 1,241 |
| pairs/query 均值 | 5.48 |
| unique chosen candidates | 2,721 |
| unique rejected candidates | 3,815 |

pair type:

| pair type | 数量 |
|---|---:|
| strong_vs_hard | 5,523 |
| fallback_vs_hard | 1,274 |

pair subtype:

| subtype | 数量 |
|---|---:|
| inversion | 4,236 |
| semi_hard | 2,561 |

v2 的数据更精、更硬，但覆盖面明显更窄，并且完全移除了 `strong_vs_easy`。

## 4. 主要原因分析

### 4.1 hard-mining 把“废信号”砍得太狠

v2 的初衷是过滤 v1 中大量模型已经排对的 pair。这个判断本身没错：v1 中确实有很多 pair 的 chosen rank 已经高于 rejected rank。

但实测说明，这些 pair 不全是无用噪声。它们至少承担了三个作用：

1. 维持全局正负分数间隔。
2. 稳定模型在普通 query 上的排序。
3. 作为隐式正则，防止 LoRA 只围绕少数 hard case 过度移动。

DPO 不是局部修改某个 query 的排序，而是更新全局 LoRA 参数。只训练模型当前排错的 hard pair，会让参数朝这些局部错误强烈偏移，可能伤到原本已经排对的大量普通样本。

### 4.2 去掉 easy_negative 后，全局校准变弱

v1 中有 7,706 条 `strong_vs_easy`。这些 pair 单看很简单，88% 左右模型本来就排对，所以在 hard-mining 视角下像是低价值样本。

但它们对最终指标有实际价值：

- 保证明显差的候选持续低分。
- 拉开 strong positive 与 easy negative 的整体间隔。
- 提升系统均分、>=4.0、>=4.5 这类质量指标。

v2 完全不用 easy_negative，只用 hard_negative。这样训练更聚焦边界样本，但低分区和高分区的整体结构变弱，导致 top-K 和 best_match 质量不如 v1 稳。

### 4.3 v2 数据带有强选择偏置

v2 只保留 base reranker 排错或近错的 pair：

```text
inversion: rank_gap <= 0
semi-hard: 0 < rank_gap <= 3 且 rejected_rank <= 10
```

这些样本通常更难、更边界、更容易有 judge 不确定性。它们当然值得训练，但如果只训练这些样本，训练分布就不再代表真实推理分布。

真实推理时，模型需要处理所有 query 的 top20，而不是只处理 base 排错的那一小部分。v2 的选择偏置让模型更像是在修补 hard case，而不是学习完整排序函数。

### 4.4 覆盖 query 更少，训练目标更局部

train samples 有 2,218 个 query。

v1 pairs 覆盖 1,506 个 query，v2 pairs 覆盖 1,241 个 query。v2 少覆盖约 265 个有 pair 训练信号的 query。

另外，v2 unique chosen / rejected candidates 都明显少于 v1。这意味着模型见到的正负候选形态更窄，泛化到完整候选池时更容易不稳。

### 4.5 小数据多 epoch，容易过度修正

v2 约 6,797 pairs，1200 step 约 2.8 epoch。

v1 约 21,596 pairs，2000 step 约 1.5 epoch。

v2 数据更窄、重复更多，同时又全是 hard case，因此更容易把 hard pair 的方向放大。它能提升 Hit@1，但会牺牲 Hit@3/5、NDCG、系统均分。

### 4.6 v2 更适合 patch，不适合替代 v1

从结果看，v2 的定位应该调整：

```text
不适合：Base -> DPO v2
更适合：Base -> DPO v1 -> DPO v2 hard-case patch
```

v1 学到的是 broad preference / global calibration。  
v2 学到的是 hard-case correction。

当前实验直接用 v2 替代 v1，相当于用局部纠错信号承担完整排序训练职责，所以效果不如 v1。

## 5. 为什么简单混合不一定有意义

如果只是把 v1 pairs 和 v2 pairs 直接合并，本质上和 v1 差别不大，只是对 hard case 稍微增加了采样权重。

真正有意义的混合应该是分层采样或阶段式训练：

```text
每个 batch:
  50% v1 strong_vs_hard / fallback_vs_hard
  20% v1 strong_vs_easy
  30% v2 inversion / semi-hard
```

或者更推荐：

```text
DPO2000 checkpoint
  -> v2 hard pairs
  -> 低 lr
  -> 短训 200-400 step
```

这样 v1 负责全局校准，v2 只负责补 hard case，职责更清晰。

## 6. 后续建议

### 主线建议

继续使用：

```text
Base -> DPO2000 -> AWR
```

理由：

- DPO2000 在 train / val / test 上都是最稳的 checkpoint。
- AWR 的目标是改善 listwise / top-K / NDCG，而 DPO2000 的 top-K 和 NDCG 已经比 v2 更好。
- v2 在 top-K 和系统均分上偏弱，不适合作为 AWR 起点。

### v2 后续可选实验

如果还想验证 v2 的价值，建议只做小成本 ablation：

1. `DPO2000 -> v2 hard pairs`，lr 降到 `1e-6` 或 `2e-6`，训练 200-400 step。
2. v2b：加入 `easy_negative inversion only`，验证 easy negative 排到前面时是否值得纠正。
3. v2c：只保留 inversion，去掉 semi-hard，降低边界样本噪声。
4. 分层混合采样，而不是简单合并 v1/v2 全量数据。

### 不建议

不建议继续投入：

```text
Base -> DPO v2 -> AWR
```

当前 train / val / test 已经共同说明，v2 不是比 v1 更好的主干 checkpoint。

## 7. 最终判断

DPO v2 的失败不是因为 hard-mining 思路完全错，而是因为当前版本把训练信号从“广覆盖偏好学习”收缩成了“局部 hard-case 纠错”。这提高了部分错误样本的修复强度，但损失了 v1 的全局排序校准能力。

因此：

- v1 是主干训练。
- v2 是可能的补丁训练。
- AWR 应该接在 DPO2000 后，而不是接在当前 v2 后。
