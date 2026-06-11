# Score Explosion in Score-Based DPO

## 背景：为什么用 logit diff 作为 score

### Qwen3-VL-Reranker 的结构

这个模型不是传统 cross-encoder reranker（BERT → 线性层 → 标量）。
它是 decoder-only 生成模型，被训练成在 prompt 末尾预测"yes"或"no"来判断相关性。

推理时不真的生成 token，而是直接读最后一个位置的 logit：

```python
score = logits[:, -1, yes_id] - logits[:, -1, no_id]
```

这是这类模型的标准用法（RankGPT 等 LLM-as-reranker 论文均采用此方式）。

### 这里的 logits 是 raw logit（softmax 之前）

`outputs.logits` 是模型输出的原始分数，未经 softmax 归一化，范围无界，可以是任意实数。

---

## 标准 DPO 的做法及与本项目的对比

### 标准 DPO 用序列级 log probability

```python
log_prob = F.log_softmax(logits, dim=-1)       # softmax + log
seq_log_prob = log_prob[response_tokens].sum()  # 整个序列求和
```

**为什么是 log_softmax 而不是直接用概率**：序列概率 = 每个 token 概率连乘，对长序列会数值下溢（0.5^100 ≈ 10^-30，在 float32 里直接变 0）。取 log 后乘法变加法，数值稳定。

**log probability 是大负数，为什么没问题**：DPO 用的是差值——两个大负数相减，差值是有意义的有限数（如 +0.3）。直接用概率则因下溢都变成 0，差值也是 0，信息完全丢失。

### 关键对比

| | 本项目 | 标准 DPO |
|---|---|---|
| 取的值 | raw logit diff（单 token，pre-softmax） | log probability（整个序列，含 log_softmax） |
| 有没有上界 | 无上界 | 无下界，但受 softmax 归一化约束 |
| 能否"悄悄退化" | 能（见下文） | 不能（见下文） |

---

## 为什么标准 DPO 没有这个问题

### 标准 DPO 也没有显式 KL 项

标准 DPO 的 loss 里没有一个显式的 KL 约束项，KL 只是推导时的动机。
真正起约束作用的是 **softmax 归一化**：每个位置的 token 概率加起来必须 = 1。
想推高 yes 的概率，就必须压低其他 token 的概率，这是结构性约束。

### 为什么不能"同时把 chosen 和 rejected 都推到很负"

一个合理的反驳是：标准 DPO 的 log prob 没有下界，如果 chosen 和 rejected 的 log prob 同时变得很负，差值不也一样吗？

答：做不到。softmax 在每个位置永远输出合法的概率分布（概率和 = 1，永远 > 0）。
"同时压低 chosen 和 rejected 的 log prob"，必然意味着把概率质量**转移**到其他 token 上，模型的生成行为会整体崩坏。这是另一种明显的 failure，不是"悄悄退化而 loss 不变"的路径。

### 本项目的退化路径为什么更隐蔽

raw logit diff 没有任何归一化约束：yes-logit 可以**独立地无限增大**，完全不影响其他 token，不影响模型任何其他行为，loss 不变，梯度也感知不到。这是一条完全隐蔽的退化路径，在标准 DPO 里结构上不存在。

---

## 现象

训练约 step=197 时（LR 刚到峰值 ~2e-6），两张卡的 s_w 和 s_r 同时触顶 clamp 上界（10.0）且不再下来：

```
step=196: s_w=-1.75, s_r=-1.50, margin=0.0
step=197: s_w=10.0,  s_r=10.0,  margin=0.0  ← 爆炸点
step=198: s_w=10.0,  s_r=10.0,  margin=0.0
```

loss 全程停在 0.693 ≈ log(2)，模型完全没有学到任何区分能力。

---

## 根本原因

### LoRA 找到了"全局正偏置"这条捷径

DPO 梯度明确要求：对 chosen 输入提高 s_w，对 rejected 输入降低 s_r。

但由于每个 pair 是**同一个 query + 不同视频**，两类输入经过 vision encoder 后特征相近，LoRA 在参数空间里难以找到能同时做到"chosen 涨、rejected 降"的方向。

代价最小的路径是对**所有输入加一个全局正偏置**（统一推高 yes-logit）：

- s_w 涨 ✓（符合 DPO 对 chosen 的要求）
- s_r 也涨 ✗（违背 DPO 对 rejected 的要求）

但 DPO 优势项 = (s_w - s_w_ref) - (s_r - s_r_ref) = 12 - 12 = 0，loss 不变，梯度阻力为零，模型一直沿这个方向走直到触顶。

### Clamp 不能反推

`clamp(-10, 10)` 只截断数值，不产生反向梯度。一旦分数超过 10，该方向完全没有梯度，模型无法被推回来。

---

## 修复方案

在 DPO loss 中加入 **anchor 正则项**，显式惩罚 policy 分数偏离 reference 太远：

```python
anchor = beta_anchor * (
    (s_w - s_w_ref).pow(2) + (s_r - s_r_ref).pow(2)
).mean()
loss = preference_loss + anchor
```

**为什么有效**：当模型试图推高所有 yes-logit 时，`(s_w - s_w_ref)²` 迅速增大，产生强烈的反推梯度，使"全局偏置"这条路有代价，迫使模型只能通过真正区分 chosen/rejected 来降低 loss。

**参数**：`beta_anchor = 0.01`。DPO 梯度强度约 0.05，anchor 在偏离 ~2.5 个单位时产生相当的反推力，允许适量区分性学习，防止超出合理范围。

---

## 关联文件

- `dpo_train.py` — `dpo_loss()` 函数，anchor 在此添加
- `dpo_config.yaml` — `beta_anchor: 0.01`
