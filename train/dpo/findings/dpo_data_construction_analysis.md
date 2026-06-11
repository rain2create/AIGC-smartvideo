# DPO 数据构造分析与反思

> 记录时间：2026-06-02
> 数据来源：judge/training_data/v3/splits/train_pairs.jsonl（21596 pairs）

## 1. 正负样本定义（Bucket）

数据来自 train split（90部剧，46425 条 samples），按 judge 评分分为 5 个 bucket：

┌───────────────────┬─────────────────────────────────────────────┬────────┐
│ Bucket            │ 规则                                        │ 数量   │
├───────────────────┼─────────────────────────────────────────────┼────────┤
│ strong_positive   │ overall>=4.5, would_use=True, conf>=0.8    │ 3074   │
│ fallback_positive │ overall>=3.5, would_use=True, non-strong   │ 3417   │
│ ambiguous_positive│ overall>=4.0, would_use=False              │ 14     │
│ hard_negative     │ 2.0<=overall<=3.5                          │ 12770  │
│ easy_negative     │ overall<2.0                                │ 27148  │
└───────────────────┴─────────────────────────────────────────────┴────────┘

DPO 中 chosen = positive（strong/fallback），rejected = negative（hard/easy）。ambiguous 不参与。

## 2. DPO Pair 构造规则

┌───────────────────┬─────────────────────────────────────────────┬────────┐
│ Pair Type         │ 规则                                        │ 数量   │
├───────────────────┼─────────────────────────────────────────────┼────────┤
│ strong_vs_hard    │ 每个 strong 配 4 个 hard（overall 降序取）  │ 10701  │
│ strong_vs_easy    │ 每个 strong 配 3 个 easy                    │ 7706   │
│ fallback_vs_hard  │ 仅无 strong 的 query，每 fallback 配 4 hard │ 3189   │
├───────────────────┼─────────────────────────────────────────────┼────────┤
│ 总计              │                                             │ 21596  │
└───────────────────┴─────────────────────────────────────────────┴────────┘

### 构造初衷

本质是**穷举所有可用的正负配对**，按 bucket 自然分层，不做 hard pair mining：

1. **strong_vs_hard 是主力**（49.6%）：strong 和 hard 的 overall 差距适中（均值 1.84），是最有信息量的 pair——差距太大模型学不到细粒度区分，太小信号模糊
2. **strong_vs_easy 是兜底**（35.7%）：确保模型至少能把明显差的排到后面，信号最强但最"简单"
3. **fallback_vs_hard 是补充**（14.8%）：有些 query 没有 strong，fallback 是唯一的正样本，不加这些 pair 这些 query 就完全没训练信号

没有按模型当前排序来筛选 pair，导致大量 pair 模型本来就排对了（见 §4 分析）。

## 3. 模型学到什么

DPO loss 教模型：chosen 的 score 应该比 rejected 高。

```
loss = -logsigmoid(β * ((s_chosen - s_chosen_ref) - (s_rejected - s_rejected_ref)))
```

模型学到的是按 judge 评分的 **bucket 级别偏好**：
- strong（4.5+）应该比 hard（2.0-3.5）分高
- strong（4.5+）应该比 easy（<2.0）分高
- fallback（3.5-4.5）应该比 hard（2.0-3.5）分高

### 各 pair type 的 overall 分差

┌───────────────────┬──────────────┬───────────────┬──────────────────────────┐
│ Pair Type         │ chosen 均值  │ rejected 均值 │ 分差 (chosen-rejected)   │
├───────────────────┼──────────────┼───────────────┼──────────────────────────┤
│ strong_vs_hard    │ 4.76         │ 2.92          │ mean=1.84, range=0.55~3.0│
│ strong_vs_easy    │ 4.75         │ 1.70          │ mean=3.05, range=2.55~4.0│
│ fallback_vs_hard  │ 3.98         │ 2.84          │ mean=1.13, range=-0.25~2.45│
└───────────────────┴──────────────┴───────────────┴──────────────────────────┘

- strong_vs_easy 分差最大（均值 3.05），信号最"简单"
- fallback_vs_hard 分差最小（均值 1.13），且存在 gap<0 的 pair（40.3% gap<1.0）

## 4. 问题：大量 pair 是"正确废话"

将 pair 中 chosen/rejected 的 base model rank 进行比对：

┌───────────────────┬────────┬─────────────────────┬─────────────────────┐
│ Pair Type         │ 有rank │ 模型已排对          │ 模型排错了          │
├───────────────────┼────────┼─────────────────────┼─────────────────────┤
│ strong_vs_hard    │ 8951   │ 6713 (75.0%)        │ 2238 (25.0%)        │
│ strong_vs_easy    │ 5768   │ 5078 (88.0%)        │ 690 (12.0%)         │
│ fallback_vs_hard  │ 2855   │ 1831 (64.1%)        │ 1024 (35.9%)        │
├───────────────────┼────────┼─────────────────────┼─────────────────────┤
│ 总计              │ 17574  │ 13622 (77.5%)       │ 3952 (22.5%)        │
└───────────────────┴────────┴─────────────────────┴─────────────────────┘

**77.5% 的 pair 模型本来就排对了**——chosen rank < rejected rank，模型在这些 pair 上接收的是它已经知道的信号，梯度贡献有限。

问题分解：
1. **strong_vs_easy（88% 已排对）**：strong 和 easy 差距巨大，模型基本不会搞混。这 7706 条 pair 里只有 690 条有纠错价值。
2. **strong_vs_hard（75% 已排对）**：有 25% 的纠错空间，是三类里最有价值的。
3. **fallback_vs_hard（64% 已排对）**：纠错比例最高（35.9%），但 fallback 和 hard 的 overall 分差小（均值仅 1.13），信号本身就模糊。

### 这解释了 DPO2000 提升有限的原因

- Hit@1 在 train 上只涨了 +4.8pp，test 上 +6.8pp
- 因为 77.5% 的训练 pair 是冗余信号，模型在大部分 pair 上学到的是它已经会的东西
- 真正的纠错信号（22.5%）被稀释在大量冗余 pair 中

### "模型已排对"不是错误信号，是废信号

需要区分三种情况：

**废信号（77.5%）—— 模型已排对，梯度小：**

```
Query: "男主向女主告白"
  chosen:  片段A（overall=4.8，reranker rank=2）
  rejected: 片段B（overall=2.5，reranker rank=15）
```

模型本来就把 A 排在 B 前面。DPO loss 要求 score(A) > score(B)，模型已经满足，
loss 很小，梯度很小，几乎不学东西。方向是对的，但没有纠错价值。

**有效信号（22.5%）—— 模型排错了，梯度大：**

```
Query: "男主向女主告白"
  chosen:  片段C（overall=4.6，reranker rank=12）
  rejected: 片段D（overall=2.8，reranker rank=3）
```

模型把 D 排在 C 前面，但 judge 说 C 好得多。这个 pair 的 loss 大，梯度强，
模型真正从这里学到了新东西。

**结论：不是 77.5% 的 pair 有害，是它们浪费了训练步数。**
有用的 22.5% 被稀释在大量冗余 pair 中，训练效率低。

## 5. 可能的改进方向

### 方案：Rank-Based Hard Pair Mining

只保留模型排错的 pair（chosen rank > rejected rank），过滤掉模型已经排对的：

```
当前: 21596 pairs (77.5% 冗余)
筛后: ~3952 pairs (100% 纠错信号) + 4022 pairs (无rank，保留)
```

优点：
- 每条 pair 都是有效梯度信号
- 减少训练时间（数据量缩小）
- 模型不会在已知偏好上过度强化

风险：
- 数据量大幅减少，可能不够 LoRA 收敛
- 如果只学"模型错的"，可能引入选择偏差

这是后续优化方向，当前先跑 SFT → DPO（原始数据），看 SFT 纠错层能带来多少增量。
