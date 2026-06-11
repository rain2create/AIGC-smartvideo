# Score-based DPO 目标函数设计分析

> 记录时间：2026-06-01
> 关联论文：Qwen3-VL-Embedding and Qwen3-VL-Reranker (2026-01-21)

## 1. 背景：标准 DPO vs Reranker DPO

### 标准 DPO（生成式）

同一个 prompt，两段不同的 response（chosen/rejected），比较生成概率：

```
log π(y|x) = Σ_t log p(y_t | y_{<t}, x)    # 每个位置取 1 个 label token 的 log-probability
delta = log π_θ(y|x) - log π_ref(y|x)       # policy 与 reference 的概率比值
loss = -logsigmoid(β * (delta_w - delta_r))
```

每个位置只取 **1 个 token** 的 logit（经过 softmax），多个位置累加。

### Reranker DPO（本项目）

两个不同的输入 (query+chosen_doc vs query+rejected_doc)，模型不生成回答，而是打分：

```
s = logit_yes - logit_no                     # 1 个位置取 2 个 token 的 logit 做差
delta = s_θ - s_ref
loss = -logsigmoid(β * (delta_w - delta_r))
```

本质区别：标准 DPO 优化的是**生成概率**，Reranker DPO 优化的是**打分函数**。

---

## 2. 为什么一个位置取两个 token 的 logit 做差

### 标准生成式 DPO

```
位置1: log p(y_1|x)         ← 取 y_1 这个 token 的 logit
位置2: log p(y_2|y_1,x)     ← 取 y_2 这个 token 的 logit
...
每个位置取 1 个 label token
```

### Reranker 场景

模型只有 1 个预测位置（assistant 回答的第一个 token），不是在"生成"，而是在做二分类判断：相关（yes）还是不相关（no）。

如果只取 `logit_yes`，那 `logit_no` 里包含的"模型认为不相关"的信号就丢了。做差才是完整的二分类决策：

```
s = logit_yes - logit_no = log(p_binary(yes) / p_binary(no))    # log-odds ratio
```

这与论文公式 (5) 一致：

> s = sigmoid(logit(yes) − logit(no))

`logit_yes - logit_no` 是二分类的 log-odds，是 reranker 打分函数的自然定义。

---

## 3. 为什么训练时用 raw score 而不是 sigmoid(score)

论文推理时用 `sigmoid(s)` 把分数压到 [0,1]，但训练时我们直接用 raw score `s = logit_yes - logit_no`。

原因：

1. **sigmoid 是单调函数**：`sigmoid(a) > sigmoid(b)` 当且仅当 `a > b`，排序等价，不影响 DPO 的偏好学习
2. **梯度通畅**：sigmoid 在极端值（|s| > 5）饱和，梯度趋近于 0，LoRA 参数更新会被卡死
3. **避免双重压缩**：DPO loss 本身已经有 logsigmoid，如果 score 里再套一层 sigmoid，等于双重 squash，信号被过度压缩

因此训练用 raw score，推理用 sigmoid，两者互不影响。

---

## 4. 为什么 preference loss 用 logsigmoid

DPO loss 中的 `logsigmoid` 与论文推理的 `sigmoid` **完全是两个东西**，来源不同：

### 来源：Bradley-Terry 偏好模型

假设人类偏好符合 Bradley-Terry 模型：

```
P(chosen 优于 rejected) = sigmoid(reward_w - reward_r)
```

训练目标 = 最大化这个概率 = 最小化负对数似然：

```
loss = -log P(chosen 优于 rejected)
     = -log sigmoid(β * advantage)
     = -logsigmoid(β * advantage)
```

其中 `advantage = delta_w - delta_r = (s_w - s_w_ref) - (s_r - s_r_ref)`

### 两个 sigmoid 的区别

| | 论文推理的 sigmoid | DPO loss 的 logsigmoid |
|---|---|---|
| 输入 | 单个样本的 score | chosen 与 rejected 的优势差 |
| 含义 | 这个 doc 相关的概率 | chosen 比 rejected 好的概率 |
| 用途 | 推理打分，映射到 [0,1] | 偏好学习的损失函数 |
| 来源 | Reranker 二分类设计 | Bradley-Terry 偏好模型 |

---

## 5. 完整计算链路

```
输入: (query, chosen_doc), (query, rejected_doc)
                    ↓
前向打分（4 次）:
    s_w     = logit_yes - logit_no   (policy, chosen)
    s_w_ref = logit_yes - logit_no   (reference, chosen)
    s_r     = logit_yes - logit_no   (policy, rejected)
    s_r_ref = logit_yes - logit_no   (reference, rejected)
                    ↓
DPO advantage:
    delta_w = s_w - s_w_ref          (chosen 的 score 偏移)
    delta_r = s_r - s_r_ref          (rejected 的 score 偏移)
    adv = delta_w - delta_r          (优势：chosen 偏移 > rejected 偏移)
                    ↓
损失:
    preference_loss = -logsigmoid(β * adv)       (Bradley-Terry)
    anchor_loss = β_anchor * (delta_w² + delta_r²) (防 score explosion)
    total_loss = preference_loss + anchor_loss
```

---

## 6. 关于 raw score 与 logsigmoid(score) 的理论细节

严格的单 token DPO 应该用 `logsigmoid(s)` 作为 log-probability：

```
log p_binary(yes|x) = logsigmoid(s) = log sigmoid(logit_yes - logit_no)
```

我们直接用 raw `s` 近似了 `logsigmoid(s)`：

| | raw s | logsigmoid(s) |
|---|---|---|
| s 在 [-3,3] | 近似线性，差异不大 | 差异不大 |
| s 很大 (>10) | 无界，可能爆炸 | 饱和趋近于 0，自然抑制 |
| 需要 anchor | 是，防止 score explosion | 不需要，自带饱和 |

当前方案用 raw score + anchor loss 兜底，实测可行。如果后续想去掉 anchor loss，可以考虑切换到 `logsigmoid(s)` 作为 score 函数。

---

## 7. 结论

1. **取两个 token 做差**：合理。Reranker 是二分类，`logit_yes - logit_no` 是 log-odds，与论文推理公式一致
2. **用 raw score**：合理。避免 sigmoid 饱和导致梯度消失，且排序等价
3. **preference loss 用 logsigmoid**：标准 Bradley-Terry 偏好学习公式，与推理的 sigmoid 是不同层面的东西
4. **整体设计**：不是严格的标准 DPO（标准 DPO 优化生成概率），而是用 DPO 框架优化 reranker 打分函数，理论和实践均可行
