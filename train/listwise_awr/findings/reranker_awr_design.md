# Reranker Listwise AWR 目标函数设计分析

> 记录时间：2026-06-01
> 关联：train/dpo/findings/score_based_dpo_design.md（DPO 版本的同类分析）

## 1. 标准 GRPO vs Reranker Listwise AWR

### 标准 GRPO（生成式 LLM）

同一个 prompt，模型生成 G 条不同 response，每条 response 有 reward。

```
1. output.logits  →  [batch, seq_len, vocab_size]
2. log_softmax(logits, dim=-1)  →  log_probs       # 经过 softmax 变成真概率
3. 每个位置取 label token 的 log_prob：log_p_t = log_probs[t, y_t]
4. 整条 response 累加：log π(y|x) = Σ_t log_p_t
5. 比值：ratio_i = log π_θ(y_i|x) - log π_ref(y_i|x)
6. advantage：A_i = (r_i - mean(r)) / std(r)        # z-score
7. loss = -mean(A_i * ratio_i)
```

关键：每条 response 有**多个 token**，每个 token 都经过 **softmax** 归一化后取对应 label 的 log probability，最后累加得到整条 response 的对数概率。

### 我们的 Reranker Listwise AWR

同一个 query，有 G 个不同的候选文档（视频片段），每个候选有预计算的 reward（来自 judge）。

```
1. output.logits  →  [batch, seq_len, vocab_size]
2. 取最后一个位置的 logits（模型只预测一个 token：yes 或 no）
3. 直接取 yes/no 两个 logit 做差：s = logit_yes - logit_no    # 没有 softmax
4. 比值：delta_i = s_θ(q, d_i) - s_ref(q, d_i)
5. advantage：A_i = (r_i - mean(r)) / std(r)                   # z-score
6. loss = -mean(A_i * delta_i) + β_anchor * mean(delta_i²)
```

### 核心区别

| | 标准 GRPO | Reranker Listwise AWR |
|---|---|---|
| 输出 | 多 token response | 单位置二分类（yes/no） |
| 概率计算 | softmax 归一化 → log probability | raw logit 做差（无 softmax） |
| 比值含义 | 整条 response 的对数概率比 | 单个 score 的差 |
| 正则化 | KL penalty（概率本身有界） | anchor loss（raw score 无界需兜底） |
| Reward | 在线计算（reward model 或 verifiable） | 离线预计算（judge 评分） |

---

## 2. 为什么不经过 softmax

### 严格做法

Reranker 预测一个 token，如果要拿到"真概率"，应该对全词表（~15 万 token）做 softmax：

```
p(yes|x) = exp(logit_yes) / Σ_v exp(logit_v)        # 全词表 softmax
log p(yes|x) = logit_yes - logsumexp(所有 logit)
```

然后比值应该是：

```
ratio = log p_θ(yes|x) - log p_ref(yes|x)
      = [logit_yes_θ - logsumexp_θ] - [logit_yes_ref - logsumexp_ref]
```

### 二分类近似

因为 Reranker 预训练时被训练成"第一个 token 只回答 yes 或 no"，其他 token 的概率极低，所以：

```
p(yes) + p(no) ≈ 1
```

此时全词表 softmax 退化为二分类 sigmoid：

```
p(yes|x) ≈ exp(logit_yes) / (exp(logit_yes) + exp(logit_no))
          = sigmoid(logit_yes - logit_no)
          = sigmoid(s)
```

**恒等式推导**：设 a = logit_yes, b = logit_no, s = a - b

```
exp(a) / (exp(a) + exp(b))
= exp(a) / (exp(a) + exp(b))       分子分母同除 exp(b)
= exp(a-b) / (exp(a-b) + 1)
= exp(s) / (exp(s) + 1)
= 1 / (1 + exp(-s))
= sigmoid(s)                        ∎
```

这与论文推理公式一致：`score = sigmoid(logit_yes - logit_no)`

对应的 log probability：

```
log p(yes|x) ≈ logsigmoid(s)
```

比值：

```
ratio = logsigmoid(s_θ) - logsigmoid(s_ref)
```

### 我们的进一步近似

我们直接用 raw score 差代替 logsigmoid 差：

```
ratio ≈ s_θ - s_ref      （而不是 logsigmoid(s_θ) - logsigmoid(s_ref)）
```

原因：
1. **梯度通畅**：logsigmoid 在 |s| > 5 时饱和，梯度趋近于 0，LoRA 参数更新被卡死（详见下方推导）
2. **排序等价**：sigmoid 是单调函数，排序不受影响
3. **anchor 兜底**：`β_anchor * delta²` 惩罚 score 偏离 reference 太远，防止 raw score 爆炸

#### logsigmoid 梯度饱和推导

```
logsigmoid(s) = -log(1 + exp(-s))

对 s 求导：
d/ds logsigmoid(s) = exp(-s) / (1 + exp(-s)) = sigmoid(-s)
```

代入具体数值：

┌───────┬─────────────────┬──────────────────────────────┐
│ s     │ logsigmoid(s)   │ 梯度 sigmoid(-s)             │
├───────┼─────────────────┼──────────────────────────────┤
│ 0     │ -0.693          │ 0.500                        │
│ 1     │ -0.313          │ 0.269                        │
│ 3     │ -0.049          │ 0.047                        │
│ 5     │ -0.0067         │ 0.0067（仅剩 s=0 时的 1.3%） │
│ 10    │ -0.000045       │ 0.000045（仅剩 0.009%）      │
└───────┴─────────────────┴──────────────────────────────┘

s=5 时梯度只剩 s=0 时的 1.3%，s=10 时只剩 0.009%。

**对训练的影响**：如果用 logsigmoid(s) 做 loss，对 s_θ 的梯度为 `-A_i * sigmoid(-s_θ)`。当模型已经给了高分（s_θ=10）时，sigmoid(-10) ≈ 0.000045，梯度几乎为零——模型无法继续学习区分"比较好"和"非常好"的候选。

用 raw score 的话，梯度恒定为 `-A_i`，不受 score 大小影响，梯度流畅。

---

## 3. 近似链路总结

从严格到我们的实现，经过了两层近似：

```
全词表 softmax 概率
    ↓ 近似①：其他 token 概率 ≈ 0，退化为 yes/no 二分类
sigmoid(s) = sigmoid(logit_yes - logit_no)
    ↓ 取 log
logsigmoid(s)        ← "严格的"对数概率
    ↓ 近似②：logsigmoid(s) ≈ s（中等范围内近似线性）
raw score s          ← 我们实际用的
```

两层近似的代价：
- 近似①（二分类）：对预训练好的 Reranker 几乎无损
- 近似②（去 logsigmoid）：需要 anchor loss 防止 score 到极端值后近似崩坏

---

## 4. 为什么 Listwise AWR 比 DPO 更适合改善 top-K 排序

### DPO 的结构性局限

DPO 每次只看一对（chosen vs rejected），优化"A 比 B 好"：

```
loss = -logsigmoid(β * (delta_chosen - delta_rejected))
```

- 只有二元信号：好 vs 坏
- 对 2-5 名之间的排序没有信号
- top-1 改善但 top-3/5 可能变差

### GRPO/AWR 的 listwise 优势

AWR 一次看整个 group（~15 个候选），用 z-score advantage 提供连续梯度信号：

```
A_i = (r_i - mean(r)) / std(r)

例如 group 内：
  候选 A: reward=4.7  →  A = +1.8（强正推力）
  候选 B: reward=3.8  →  A = +0.9（中等正推力）
  候选 C: reward=2.5  →  A = -0.4（轻微负推力）
  候选 D: reward=1.0  →  A = -1.6（强负推力）
```

- **细粒度区分**：4.7 和 3.8 获得不同大小的推力，DPO 里它们都是"chosen"
- **完整排序信号**：一个 step 看到所有候选的相对关系
- **天然 listwise learning-to-rank**：更适合优化 NDCG@K

---

## 5. Loss 可用性分析

### 梯度方向验证

对 s_i(θ) 求梯度：

```
∂loss/∂s_i(θ) = -A_i + 2·β_anchor·δ_i
```

- 好候选（A_i > 0）：梯度为负 → 推高 s_i(θ) → 给高分 ✓
- 差候选（A_i < 0）：梯度为正 → 压低 s_i(θ) → 给低分 ✓
- 推力大小和 |A_i| 成正比 ✓

### 均衡点分析

令梯度为零：`δ_i* = A_i / (2·β_anchor)`

收敛时每个候选的 score 偏移和 advantage 成正比，有唯一最优解，loss 是凸的（二次项保证）。

### 理论定位：AWR 而非严格 GRPO

**与标准 GRPO 的关键差异**：标准 GRPO 要求模型随机采样生成不同 response（策略是随机的），通过"探索"发现哪种 response 更好。我们的 reranker 是确定性的——同一输入永远得到同一 score，候选也是固定给定的而非模型生成的：

```
标准 GRPO：同一个 prompt → 模型随机采样生成 G 条不同 response → 各自拿 reward
我们：     同一个 query  → G 个候选是固定给定的（不是模型生成的）→ reward 预算好的
```

因此我们的 loss 本质是 **Advantage-Weighted Regression（AWR）**——一种成熟的 offline RL 方法：

```
标准 GRPO:   -mean(A_i · log(π_θ/π_ref))     ← 概率比值，需要在线采样
我们:         -mean(A_i · (s_θ - s_ref))       ← score 差值 + anchor，离线数据即可
```

AWR 不要求策略随机性，不需要在线探索，直接用已有的 (候选, reward) 数据做 advantage 加权回归。对确定性打分函数完全适用。

**不需要探索反而是优势**：
- 不需要在线 rollout（省掉 GRPO 最重的环节）
- 不需要 reward model 在线推理
- 数据利用效率更高（每个候选都有明确的 reward 信号）

### 风险与对策

| 关注点 | 风险 | 对策 |
|---|---|---|
| raw score 无界 | score 可能爆炸到极端值 | anchor loss 兜底，均衡点有界 |
| 无 PPO clipping | 单步更新可能过大 | anchor 起类似 trust region 效果 |
| 近似了概率 | 不是严格的概率空间操作 | ranking 任务只需排序正确，不需校准概率 |
| 确定性策略 | 不满足标准 GRPO 随机性假设 | AWR 理论覆盖，不需要随机性 |

---

## 6. 完整 AWR 计算链路

```
输入: query q, 候选 {d_1, ..., d_G}, 预计算 rewards {r_1, ..., r_G}
                    ↓
前向打分（2G 次）:
    对每个候选 d_i:
        s_i     = logit_yes - logit_no   (policy model)
        s_i_ref = logit_yes - logit_no   (reference model, LoRA disabled)
                    ↓
比值:
    delta_i = s_i - s_i_ref              (每个候选的 score 偏移)
                    ↓
Advantage:
    A_i = (r_i - mean(r)) / std(r)       (group 内 z-score 归一化)
                    ↓
损失:
    pg_loss     = -mean(A_i * delta_i)           (policy gradient)
    anchor_loss = β_anchor * mean(delta_i²)      (防 score explosion)
    total_loss  = pg_loss + anchor_loss
```
