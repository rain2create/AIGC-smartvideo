# Reranker 微调：从 SFT 到 Listwise AWR 的四阶段探索

> 对 Qwen3-VL-Reranker-8B 做 LoRA 微调，目标是改善文案-视频匹配的排序质量。
> 训练数据底座：Canonical Judge 产出的 50,900 samples → 23,741 DPO pairs → 1,669 AWR groups。
> 全部实验以 `qwen_q3_rerank_q1`（未微调零样本 Reranker）为 base 基线做比较。

---

## 0. 先看结论：哪条路走通了，哪条没有

| 阶段 | 路线 | vs Base (test) | 结论 |
|:---:|---|---|:---:|
| SFT | Pointwise BCE | Hit@1 **-2.3pp**, Hit@5 **-11.4pp** | 负优化，方向性错误 |
| **DPO v1** | Pairwise 偏好学习 | Hit@1 **+6.8pp**, NDCG@5 **+0.02** | **最优，泛化良好，主线** |
| DPO v2 | Rank-based hard-mining | Hit@1 +3.8pp 但整体弱于 v1 | 优于 base 但不如 v1，定位为补丁 |
| AWR v2 | Listwise 从 DPO v1 继续 | Hit@5 **+2.7pp**, NDCG 持平 | 有微量改善但未超越 DPO v1 |

**当前最优模型：DPO v1 checkpoint-2000**。后续实验应在此 checkpoint 基础上继续。

---

## 1. 总览：四条路径的接力逻辑

```
                    ┌─────────────┐
                    │ Base (零样本) │  Hit@1=31.8% test
                    └──────┬──────┘
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
     ┌──────────┐   ┌────────────┐   ┌────────────┐
     │   SFT    │   │  DPO v1    │   │  DPO v2    │
     │ pointwise│   │ pairwise   │   │ hard-mining│
     │  失败 ✗  │   │  成功 ✓    │   │  半成功 △  │
     └──────────┘   └─────┬──────┘   └────────────┘
                          │
                          ▼
                   ┌────────────┐
                   │  AWR v2    │
                   │  listwise  │
                   │  微弱改善  │
                   └────────────┘
```

四个阶段不是预先设计的蓝图，而是逐阶段根据实验结果调整出来的。每一次失败或不如预期都产生了明确的方向信号。

---

## 2. 共同的工程基础

### 2.1 模型架构与打分方式

所有训练阶段共享同一个模型和打分函数：

```
模型: Qwen3-VL-Reranker-8B (Qwen3VLForConditionalGeneration)
微调: LoRA, rank=64, alpha=128
      仅文本 transformer 36 层的 attention+MLP
      vision encoder 完全冻结
GPU:  2 × L20 46GB, DDP (torchrun --nproc_per_node=2)

打分: s = logit_yes - logit_no (raw logit diff, 不 sigmoid)
     训练用 raw s (保留梯度), 推理用 sigmoid(s) (映射到 [0,1])
```

### 2.2 视频帧缓存

所有阶段的训练数据都使用同一次预抽帧结果：

```
fps=1.0, max_frames=8
存储: judge/training_data/frame_cache/{video_stem}/frame_0001.jpg ...
总大小: ~17GB
```

训练时 DataLoader 读 JPEG → PIL → processor，与推理侧完全一致。

### 2.3 对话格式

训练、推理、Judge 评估共用同一套对话模板：

```python
system: "Judge whether the Document meets the requirements..."
user:
  <Instruct>: "Retrieve the most relevant video clip for the given query."
  <Query>: {query_text}
  <Document>: {video_frames}
# 模型在最后一个 token 位置输出 yes/no logit
```

---

## 3. SFT：Pointwise BCE 的失败

### 3.1 动机

训练的第一步直觉是"先教模型分辨好坏"：用 pointwise BCE 让 reranker 学会给正样本打高分、负样本打低分。

### 3.2 样本筛选

从 46,425 条 train_samples 中选取 SFT 训练样本。核心原则：**SFT 只做"纠粗错"，不碰"排序微调"——后者是 DPO 的职责。**

#### 为什么用 rank 而不是 similarity 绝对值

初版方案试图用 similarity 阈值（正样本 sim<0.30 / 负样本 sim>0.50）定义"模型判断错了"。但实际数据显示这个方法行不通：

```
strong_positive 在 base reranker 下的 sim 分布:
  sim 均值: 0.44
  P25:      0.38
  P75:      0.50
```

模型的分数整体压缩在 0.2~0.5 的窄区间内，sim<0.30 不代表"模型认为它不好"，而是模型整体输出范围偏低。**只有 query-local 的 rank（在同一候选池的相对位置）才能判断是否真错排。**

#### 数据分布：base reranker 各 bucket 的 rank 分布

```
strong_positive (3,074 条):
  rank ≤ 1:    573 条 (18.6%)   ← 排对了
  rank ≤ 5:  1,777 条 (57.8%)   ← 半数以上已在 top5
  rank ≤ 10: 2,458 条 (80.0%)   ← 八成已在 top10
  rank > 10:   485 条 (15.8%)   ← 模型真漏了
  not_in_top20: 131 条 (4.3%)   ← embedding 召回失败，非 reranker 之过

fallback_positive (3,417 条):
  rank ≤ 1:    381 条 (11.2%)
  rank ≤ 10: 2,445 条 (71.6%)
  rank > 10:   756 条 (22.1%)

hard_negative (12,770 条):
  rank = 1:     716 条 (5.6%)    ← 模型最看好的候选是坏的
  rank ≤ 3:   2,292 条 (17.9%)

easy_negative (27,148 条):
  rank = 1:     542 条 (2.0%)
```

#### 正样本为什么选 rank > 10

rank 10 不是凭空取的。面临的选择是：

| 方案 | 正样本量 | 判断 |
|---|---|---|
| rank > 1 | ~10,000+ | 大量 rank 2-5 的样本"模型已经差不多对了"，信号稀释严重 |
| rank > 5 | 3,192 | rank 6-10 的模型给了中等排名，是"排序不够精"而非"判断错了" |
| **rank > 10** | **~1,241 → 过滤后 765** | 模型明显把好片段丢到后半段，是最干净的纠错信号 |

核心判断：**rank 6-10 的正样本，模型知道它是好的（排在 top10 已经是不错的信号），只是排序精度不够。这属于 DPO 的精细排序范畴，不是 SFT 的纠粗错范畴。**

此外还过滤了两类样本：
- **not_in_top20 (347 条)**：embedding 召回失败，不是 reranker 误判。推理时 reranker 只重排 top20，这些样本永远不会出现在候选池里，训练无意义
- **overall < 4.0 的 fallback (338 条)**：BCE 硬标签要求正负之间有明确 gap。一个 overall=3.6 的候选硬推 target=1 会污染信号

#### 负样本为什么只取 rank = 1

| 方案 | 负样本量 | 正负比 | 判断 |
|---|---|---|---|
| rank ≤ 3 | 4,365 | 1:2.8 | 失衡，rank 2-3 的模型没给最佳位置，惩罚优先级低 |
| **rank = 1** | **1,223** | **1:1.6** | "模型选为第一的那个是负样本"——最严重、最干净的误判 |

rank=1 还带来一个便利：每 query 天然最多 1 条负样本（best_match 只有一个），自动避免了负样本集中的问题。

同样过滤了边界样本：overall ≥ 3.5 的 hard_negative 约 35 条被移除，这些边界样本（quality 接近正样本）在 BCE 硬标签下会产生与正样本矛盾的梯度，留给 DPO 用 pairwise 信号处理更合适。

#### 最终数据与正负 gap

```
正样本 (target=1): 765 条
  strong_positive  rank>10, overall≥4.0  →  424 条
  fallback_positive rank>10, overall≥4.0  →  341 条
  per-query ≤ 3 (strong 优先, overall 降序)

负样本 (target=0): 1,223 条
  hard_negative  rank=1, overall<3.5  →  681 条
  easy_negative  rank=1               →  542 条

总计: 1,988 条, 正负比 1:1.6, 覆盖 1,538 个 query

正负 overall 分布: 正 ≥ 4.0, 负 < 3.5, 中间 0.5 gap 无重叠
```

#### SFT 与 DPO 的分工

| | SFT | DPO |
|---|---|---|
| 信号类型 | pointwise（"这个好/不好"） | pairwise（"A 比 B 好"） |
| 解决的问题 | **纠粗错**：模型把坏的排第1、把好的漏到最后 | **精细排序**：在正确的候选池里排对顺序 |
| 样本特点 | rank=1 的坏 × rank>10 的好，量少质高 | 全 bucket 穷举配对，覆盖面广 |
| 适合的 rank 区间 | 两端的严重错误 (rank=1, rank=11-20) | 中段的排序微调 (rank=1-10) |

### 3.3 目标函数

```
score = logit_yes - logit_no          # raw logit diff [batch]
loss  = BCEWithLogitsLoss(score, target, pos_weight=1.6)
        正样本 target=1 → 推高 score
        负样本 target=0 → 压低 score
```

### 3.4 超参

| 参数 | 值 | 说明 |
|---|---|---|
| lr | 5e-6 | 标准 LoRA 学习率 |
| max_steps | 375 | ~1.5 epoch (1,988 samples) |
| effective batch | 8 | 2 GPU × 1 × grad_accum=4 |
| warmup_steps | 20 | cosine decay to 0.1× |
| pos_weight | 1.6 | 正负比 765:1223 |

### 3.5 效果：全面负优化

| 指标 | Base | SFT375 | 增量 vs Base |
|---|---|---|---|
| **test Hit@1** | 31.8% | 29.5% | **-2.3pp** |
| **test Hit@5** | 68.2% | 56.8% | **-11.4pp** |
| **test R@5** | 52.7% | 50.5% | **-2.2pp** |
| **test NDCG@5** | 0.83 | 0.81 | **-0.02** |
| **test 系统均分** | 3.34 | 3.14 | **-0.20** |
| train Hit@5 | 67.9% | 63.7% | -4.2pp |
| val Hit@5 | 78.6% | 71.4% | -7.2pp |

三个 split 全部退化。不是过拟合——过拟合至少 train 上会涨。

### 3.6 根因：目标函数与任务不对齐

SFT 失败不是因为数据少或训练不充分，而是 **pointwise BCE 从根本上不适合排序任务**。

**问题 1：Pointwise 不优化排序**

BCE 教的是"这个视频相关/不相关"的绝对判断。但 reranker 需要的是"同一 query 的 20 个候选里谁排前面"的相对排序。SFT 可以把正样本分推高、负样本分压低——训练时 accuracy 确实从 50% 涨到了 70%——但这不能保证排序变好。极端的例子：正样本 score 从 2 涨到 4，但同时原本排第 1 的正确候选也从 5 被压到了 4.5，排序反而乱了。

**问题 2：共享权重的副作用**

LoRA 参数被 1,988 条 hard examples 的梯度调整后，影响传导到所有 50,900 条样本的打分。SFT 修好了约 2,000 条的 yes/no 判断，但打乱了更多本来排对的样本的相对顺序。

**问题 3：训练信号与评估指标断裂**

训练 loss 下降、accuracy 上升、score gap 拉开到 0.66——训练目标全部达标。但评估指标（Hit@K / NDCG）看的是全量样本的排序质量，而非这 2,000 条的 yes/no 准确率。**BCE 优化的是一个与最终目标无关的代理指标**。

### 3.7 结论

**Pointwise BCE 不适合 Reranker 微调。必须用 pairwise/listwise loss 直接优化排序。**

SFT 路线废弃。后续不再考虑 SFT 纠错层，DPO 成为主线起点。

---

## 4. DPO v1：Pairwise 偏好学习

### 4.1 动机

SFT 失败后，转向 pairwise 方法。DPO (Direct Preference Optimization) 直接比较同一 query 下的两个候选，学习"A 比 B 好"的相对偏好。目标函数与排序任务天然对齐：每条 pair 的梯度就是在改善相对排序。

### 4.2 样本筛选

从 rl_data_builder.py v3 产出的 train_pairs.jsonl 加载，不额外过滤：

```
DPO v1 数据 (从 samples 穷举配对):
  strong_vs_hard:   10,701 pairs  (strong配overall最高的4个hard)
  strong_vs_easy:    7,706 pairs  (strong配3个easy)
  fallback_vs_hard:  3,189 pairs  (无strong的query，fallback配4个hard)
  ─────────────────────────────
  总计:             21,596 pairs
  覆盖:              1,506 / 2,218 queries
```

配对逻辑回顾（详见 judge_system.md §7.3）：
- 每个 strong 配 overall 最高的 4 个 hard + 3 个 easy
- 每个 fallback 配 4 个 hard（仅无 strong 的 query）
- 不按 rank_gap 过滤，不做 hard-mining

### 4.3 Score-based DPO 设计

标准 DPO 优化的是生成概率（序列级 log probability），本项目的 Reranker 不是生成模型，不能直接用标准 DPO 公式。需要一套"Score-based DPO"：

#### 打分函数的改造

```
标准 DPO:        log π(y|x) = Σ_t log p(y_t | context)    # 生成概率
Score-based DPO: s = logit_yes - logit_no                  # 二分类 log-odds
```

训练时用 raw s（不 sigmoid），原因：
- sigmoid 在 |s| > 5 时饱和，梯度趋近 0，LoRA 更新被卡死
- sigmoid 是单调函数，排序等价，不影响偏好学习
- DPO loss 本身已有 logsigmoid，再套一层 sigmoid 等于双重压缩

#### Reference Model：零额外显存

```python
# policy forward（有梯度）
s_w, s_r = forward(model, chosen_inputs), forward(model, rejected_inputs)

# reference forward（无梯度，LoRA 失效）
model.disable_adapter_layers()
with torch.no_grad():
    s_w_ref, s_r_ref = forward(model, chosen_inputs), forward(model, rejected_inputs)
model.enable_adapter_layers()
```

`disable_adapter_layers()` 让 LoRA 权重归零，退回到原始 base 行为，不需要额外存一份 reference model。

#### Anchor Loss：防 Score Explosion

训练早期发现，在 step≈197 时两张 GPU 的 s_w 和 s_r 同时爆炸到 clamp 上限（10.0）：

```
step=196: s_w=-1.75, s_r=-1.50  ← 正常
step=197: s_w=10.0,  s_r=10.0   ← 爆炸
step=198: s_w=10.0,  s_r=10.0   ← 持续触顶
```

**根因**：LoRA 找到了"全局正偏置"这条捷径。由于每个 pair 是同一个 query + 不同视频，两类输入经过 vision encoder 后特征相近。LoRA 在参数空间里难以同时做到"chosen 涨、rejected 降"，代价最小的路径是对**所有输入加全局正偏置**：

```
s_w 涨 ✓（符合 DPO 对 chosen 的要求）
s_r 也涨 ✗（违背 DPO 对 rejected 的要求）
但 advantage = (s_w - s_w_ref) - (s_r - s_r_ref) = 12 - 12 = 0
→ loss 不变，梯度阻力为零，模型一直沿这个方向走直到触顶
```

**修复**：在 DPO loss 中加入 anchor 正则项，显式惩罚 policy 分数偏离 reference 太远：

```python
delta_w = s_w - s_w_ref
delta_r = s_r - s_r_ref
anchor_loss = β_anchor * (delta_w² + delta_r²).mean()
total_loss = preference_loss + anchor_loss
```

当模型试图推高所有 yes-logit 时，(δ²) 迅速增大，产生强烈的反推梯度。"全局偏置"这条路从此有代价了。

#### 完整损失函数

```
s_w, s_r     = policy  forward → chosen/rejected score
s_w_ref, s_r_ref = reference forward (LoRA disabled, no_grad)

delta_w = s_w - s_w_ref       # chosen 的 score 偏移
delta_r = s_r - s_r_ref       # rejected 的 score 偏移
adv     = delta_w - delta_r   # 优势

preference_loss = -logsigmoid(β · adv)       # Bradley-Terry 偏好模型
anchor_loss     = β_anchor · (δ_w² + δ_r²)   # 防 score explosion
total_loss      = preference_loss + anchor_loss
```

两个 sigmoid 的区别：
- 推理时的 `sigmoid(score)`：这个文档相关的概率 → [0,1]
- DPO loss 的 `logsigmoid`：chosen 比 rejected 好的概率 → Bradley-Terry

同是 sigmoid 但输入和含义完全不同。

### 4.4 超参

| 参数 | 值 | 说明 |
|---|---|---|
| lr | 5e-6 | 标准 LoRA |
| max_steps | 2,000 | optimizer steps (~1.5 epoch) |
| effective batch | 16 | 2 GPU × 1 × grad_accum=8 |
| beta | 0.1 | DPO 温度，控制偏好强度 |
| beta_anchor | 0.01 | anchor 正则系数 |
| warmup_steps | 20 | cosine decay to 0.1× |

### 4.5 效果：三组一致提升，泛化良好

#### Base vs DPO2000 核心指标（test 为 44 段文案口径，train/val 同下）

| 指标 | split | Base | DPO2000 | 增量 |
|---|---|---|---|---|
| Hit@1 | train | 26.1% | 30.9% | **+4.8pp** |
| | val | 39.3% | 42.9% | **+3.6pp** |
| | test | 31.8% | 38.6% | **+6.8pp** |
| Hit@5 | train | 67.9% | 72.5% | **+4.6pp** |
| | val | 78.6% | 82.1% | **+3.5pp** |
| | test | 68.2% | 72.7% | **+4.5pp** |
| R@5 | train | 51.4% | 54.9% | **+3.5pp** |
| | test | 52.7% | 54.5% | **+1.8pp** |
| NDCG@5 | train | 0.83 | 0.86 | **+0.03** |
| | test | 0.83 | 0.85 | **+0.02** |
| 系统均分 | train | 3.21 | 3.36 | **+0.15** |
| | test | 3.34 | 3.45 | **+0.11** |
| >=4.0 占比 | train | 36.4% | 42.2% | **+5.8pp** |
| | test | 36.1% | 39.3% | **+3.2pp** |

#### 过拟合检验

| 检验项 | 证据 |
|---|---|
| train 增幅 > test? | **否**。Hit@1 +4.8pp (train) < +6.8pp (test) |
| 三组提升一致? | **是**。NDCG@5 全部 +0.02~0.03，系统均分全部 +0.11~0.15 |
| R@20 受影响? | **否**。三组均持平，DPO 只改善头部，不影响尾部覆盖 |

**结论：DPO2000 无过拟合，具备泛化能力，可作为可靠主线。**

### 4.6 与 SFT 的核心区别

| 维度 | SFT | DPO v1 | 为什么 DPO 对了 |
|---|---|---|---|
| 训练目标 | 绝对判断 | 相对偏好 | 排序要的是"A比B好"而非"A好不好" |
| 优化的是 | pointwise accuracy | pairwise ranking | DPO 的每条梯度直接改善一对候选的相对顺序 |
| 训练信号 | hard examples 的 yes/no | 所有 pair 的排序关系 | 覆盖面更广，不只聚焦错误样本 |
| 权重更新 | 1,988 条的梯度影响全量 | 21,596 条的梯度均匀分布 | 更广的数据面 → 更稳的参数更新 |

---

## 5. DPO v2：Hard-Mining 的尝试

### 5.1 动机

DPO v1 虽然有效，但有一个明显的问题：

```
DPO v1 21,596 pairs 中:
  ~77.5% 的 pair，base reranker 已经排对了 (chosen rank < rejected rank)
  → 这些 pair 产生的梯度信号接近于零（advantage ≈ 0 → preference_loss ≈ log(2) ≈ 0.693）
  → 约 16,737 个 pair 是"废信号"
```

思路很自然：**只在模型排错或排得不稳的 pair 上训练，提高数据效率。**

### 5.2 样本筛选

从 train_samples 全量重构，不直接用 v1 pairs。核心是 `rank_gap = rejected_rank - chosen_rank`：

```
Chosen:
  strong_positive 优先, 无 strong 时用 fallback_positive
  in_top20, system_rank 非空, overall >= 4.0

Rejected:
  hard_negative only (不含 easy_negative)
  in_top20, system_rank 非空, overall <= 3.5

Hard 保留规则:
  inversion:  rank_gap <= 0              (负样本排在正样本之上或并列)
  semi-hard:  0 < rank_gap <= 3 且 rejected_rank <= 10

跳过: rank_gap > 3  (模型已排对且有余量)

限额:
  per-query    ≤ 20
  per-chosen   ≤ 6
  per-rejected ≤ 8
```

排序优先级：inversion > semi-hard → rejected_rank 小 → rank_gap 小 → judge_gap 小 → chosen_rank 大。

### 5.3 数据量对比

| | DPO v1 | DPO v2 |
|---|---|---|
| pairs | 21,596 | 6,797 |
| queries | 1,506 | 1,241 |
| chosen candidates | 3,978 | 2,721 |
| rejected candidates | 8,011 | 3,815 |
| pair type | strong_vs_hard + strong_vs_easy + fallback_vs_hard | strong_vs_hard + fallback_vs_hard |
| pair subtype | - | inversion 4,236 + semi_hard 2,561 |
| 负样本类型 | hard + easy | hard only |

训练复用了 DPO v1 的 `dpo_train.py` + `dpo_model.py`，仅数据不同。1200 steps ≈ 2.8 epoch (6,797 pairs / batch=16)。

### 5.4 效果

| 指标 | Base | DPO v1 | DPO v2-1200 |
|---|---|---|---|
| **test Hit@1** | 27.3% | **30.6%** | 31.1% |
| **test Hit@3** | 48.6% | **53.6%** | 48.6% |
| **test Hit@5** | 63.4% | **68.3%** | 63.9% |
| **test R@5** | 52.1% | **53.3%** | 49.6% |
| **test NDCG@5** | 0.835 | **0.846** | 0.824 |
| **test 系统均分** | 3.16 | **3.24** | 3.14 |
| train Hit@1 | 26.9% | **30.9%** | 29.8% |
| train NDCG@5 | 0.84 | **0.87** | 0.85 |

DPO v2 明显优于 base，但**在所有核心指标上都不如 DPO v1**。

### 5.5 根因分析

DPO v2 的方向不是完全错的——它确实优于 base，说明 rank-based hard pair 有纠错价值。但它不如 v1 的原因有三层：

**问题 1：过度聚焦 hard case，丢了全局校准**

DPO 不是局部修改某个 query 的排序，而是更新全局 LoRA 参数。只训练模型当前排错的 hard pair，参数会朝这些局部错误强烈偏移，可能伤到原本已排对的大量普通样本。v1 的"冗余" pair 实际上承担了三个功能：
1. 维持全局正负分数间隔
2. 稳定模型在普通 query 上的排序
3. 作为隐式正则，防止 LoRA 只围绕少数 hard case 移动

**问题 2：去掉 easy_negative 后全局校准变弱**

v1 中有 7,706 条 `strong_vs_easy`。它们在 hard-mining 视角下像低价值样本（88% 已排对），但其实承担着关键作用：
- 保证明显差的候选持续低分
- 拉大 strong positive 与 easy negative 的整体间隔
- 支撑系统均分、>=4.0、>=4.5 等质量指标

v2 完全不用 easy_negative，低分区和高分区的整体结构变弱。

**问题 3：训练分布 ≠ 推理分布**

v2 的选择偏置让训练数据不再代表真实推理分布。推理时模型需要处理所有 query 的 top20，而 v2 只训练 base 排错的那一小部分。模型更像是在修补 hard case，而不是学习完整排序函数。

**问题 4：小数据多 epoch，容易过度修正**

v2 约 6,797 pairs，1200 step 约 2.8 epoch。v1 约 21,596 pairs，2000 step 约 1.5 epoch。更窄的数据 × 更多重复 × 全是 hard case = 更容易把 hard pair 的方向放大。

### 5.6 结论

**DPO v2 不是比 v1 更好的主干 checkpoint。它的定位应该是补丁，不是替代。**

```text
不适合: Base → DPO v2                     (hard-mining 顶替 broad training)
更适合: Base → DPO v1 → DPO v2 patch      (v1 做全局校准, v2 只做 hard-case 修补)
```

后续 AWR 训练从 DPO v1 继续，不从 v2 继续。

---

## 6. AWR v2：Listwise 排序优化

### 6.1 动机

DPO v1 优化的是 pairwise 偏好（"A 比 B 好"），但 reranker 推理时面对的是 **一个 query + 20 个候选的整体排序**。pairwise 视角天然缺失 listwise 信息——它不知道"在 20 个候选中，谁应该排第一，谁应该排第二"。

AWR (Advantage Weighted Regression) 的目标是直接在 group 级改善排序：一个 group = 一个 query 的 8-12 个候选，模型对所有候选打分，按 bucket-aware reward 排序学习。

### 6.2 模型初始化：Merge DPO v1 → 重挂 LoRA

从 DPO v1 checkpoint 继续训练不是一个简单的 `load_adapter()`：

```
问题:
  直接加载 DPO v1 LoRA → 训练 AWR
  reference = base model（裸模型）
  delta 初始就很大 → anchor_loss 巨大 → 阻碍训练

方案: Merge + 重挂 LoRA
  1. 加载 base Qwen3-VL-Reranker-8B
  2. 加载 DPO v1 LoRA (checkpoint-2000) → merge_and_unload()
  3. 得到 "DPO v1 merged model" 作为新 base
  4. 挂载全新 LoRA（权重全零初始化）→ 开始 AWR 训练

效果:
  disable_adapter_layers() 回退到 DPO v1（正确的 reference）
  delta 从 0 开始，anchor_loss 初始为 0
  AWR 只学习"在 DPO v1 基础上的增量改进"
```

推理部署时必须两阶段加载：

```python
# 正确:
base → load DPO2000 LoRA → merge → load AWR v2 LoRA → 推理

# 错误:
base → load AWR v2 LoRA → 推理  (reference 不对)
```

### 6.3 样本筛选：DPO2000-Aware 加权 Group

不直接从旧的 groups.jsonl（bucket 机械抽）训练，而是用 DPO v1 merged model 对候选重新打分后，构造 DPO2000-aware weighted groups：

```
每组候选构成:
  正样本 (3-5个):  strong(2-3) + fallback(1) + ambiguous(1)
  DPO v1 易错负样本 (3-5个): v1 排名靠前的 hard_negative / easy_negative
  校准负样本 (4-6个):  hard(2-3, overall高的优先) + easy(2-3)
  总计: 8-12 候选

加权规则 (group-local):
  dpo_top_in_group = DPO v1 对组内候选打分最高者
  judge_best_in_group = 组内 reward 最高者

  if dpo_top == judge_best:   weight = 1.0   (v1已排对)
  elif judge_rank <= 3:        weight = 2.0   (v1错但接近)
  else:                        weight = 3.0   (v1严重错排)
```

关键设计决策：**anchor loss 不加权，只对 pg_loss 加权**。anchor 是 trust region 约束，不应因 hard group 权重大就同步放大，否则学习信号会被加强的 anchor 抵消。

### 6.4 目标函数：Linear AWR

放弃了 PPO-style clip loss（`min(surr1, surr2)`），改用更稳定的 Linear AWR：

```python
delta     = scores - scores_ref              # policy 相对 reference 的偏移 [G]
advantage = (rewards - mean) / (std + ε)     # per-group z-score

pg_loss     = -awr_coef * mean(advantage * delta)
anchor_loss =  β_anchor * mean(delta²)
total_loss  =  norm_weight * pg_loss + anchor_loss   # anchor 不加权!
```

为什么选 Linear 不选 Clip：
- 去掉 `exp(delta)`，delta 对 loss 的贡献是线性的，梯度更稳定
- `awr_coef` 直接控制 advantage 信号强度，比 clip_eps 更好调
- clip 版本的 `delta ∈ [-0.223, 0.182]` 在 raw score 空间太窄，易卡死

其中 `advantage` 的 z-score 归一化消除了不同 query 之间 reward 分布的差异，跨 query 混合训练可比。

### 6.5 超参

| 参数 | DPO v1 | AWR v2 (from DPO v1) | 调整理由 |
|---|---|---|---|
| lr | 5e-6 | **2e-6** | 从更强模型出发，小步确认方向 |
| max_steps | 2,000 | **300** | 细调而非重学 |
| warmup_steps | 20 | **10** | lr 更小，warmup 可缩短 |
| effective batch | 16 | **8** | 2 GPU × 1 × grad_accum=4 |
| awr_coef | - | **0.02** | 新增，控制 advantage 信号强度 |
| save_steps | 500 | **50** | 更密 checkpoint，精确选点 |

### 6.6 效果：与 DPO v1 基本持平

AWR v2 在 test 上的表现与 DPO v1 非常接近：

| 指标 (test 183 条文案) | DPO v1 | AWR v2-200 | 差异 |
|---|---|---|---|
| Hit@1 | **30.6%** | 30.1% | -0.5pp |
| Hit@5 | 68.3% | **71.0%** | +2.7pp |
| R@5 | 53.3% | **53.4%** | +0.1pp |
| NDCG@5 | 0.846 | **0.849** | +0.003 |
| 系统均分 | 3.24 | **3.25** | +0.01 |
| >=4.0 | 39.3% | 38.8% | -0.5pp |
| >=4.5 | 29.0% | 29.0% | 0 |

在 NDCG 导向上 AWR v2 微弱胜出，在 Hit 导向上 DPO v1 微弱胜出。实质上是平局。

### 6.7 根因分析

AWR v2 没有明显超越 DPO v1 的原因：

**Reward 信号与排序目标的张力**：AWR v2 的 reward 是 bucket-aware 的（正样本=overall，负样本=overall-4.0），这确实在正负之间制造了约 3.5 分的断层。但同一 bucket 内的 overall 差异（如 hard_negative 的 2.0 vs 3.5）反映的不一定是"哪个排前面更好"——也可能是 Judge 自身的评分噪声。Linear AWR 的 `advantage * delta` 项会放大这些噪声。

**DPO v1 的排序已经足够好**：零样本 base 在 test Hit@1 约 32%，DPO v1 已提到 39%。剩余的提升空间可能受限于 embedding 召回质量（好的候选根本不在 top20 里）或 Judge 标注噪声，而非 reranker 排序能力。

**Group 构造的近似性**：group 内只有 8-12 个采样候选，全局 judge_best 可能根本不在 group 内。这意味着 AWR 学到的"group 内最优排序"和"全局最优排序"之间存在 gap。

### 6.8 结论

**AWR v2 验证了从 DPO checkpoint 继续 listwise 训练在技术上是可行的**（merge+重挂LoRA 的初始化方案正确、Linear AWR loss 梯度稳定），但当前版本的增量收益不显著。DPO v1 仍是最优模型。

---

## 7. 全阶段效果总览

### 7.1 Test split (10 部剧, 183 条文案, 模型从未见过)

| 模型 | Hit@1 | Hit@5 | R@5 | NDCG@5 | 系统均分 | >=4.0 | vs Base |
|---|---|---|---|---|---|---|---|
| **Base** (零样本) | 27.3% | 63.4% | 52.1% | 0.835 | 3.16 | 36.1% | - |
| **DPO v1** | **30.6%** | **68.3%** | **53.3%** | **0.846** | **3.24** | **39.3%** | **+3.3pp Hit@1** |
| DPO v2 | 31.1% | 63.9% | 49.6% | 0.824 | 3.14 | 37.7% | Hit@1 略高但整体差 |
| AWR v2 | 30.1% | 71.0% | 53.4% | 0.849 | 3.25 | 38.8% | 与DPOv1持平 |

> **注**：SFT375 未在 183 文案口径下评估。在 44 文案口径（早期 test）下 SFT375 全面退化（Hit@1=29.5% vs Base=31.8%，Hit@5=-11.4pp，详见 §3.5）。

### 7.2 Train split (90 部剧, 2,218 条文案, 训练集)

| 模型 | Hit@1 | Hit@5 | R@5 | NDCG@5 | 系统均分 | >=4.0 | vs Base |
|---|---|---|---|---|---|---|---|
| Base (零样本) | 26.9% | 68.3% | 53.1% | 0.84 | 3.19 | 36.4% | - |
| SFT375 | 24.0% | 63.7% | 51.1% | 0.83 | 3.15 | - | -2.9pp Hit@1 |
| **DPO v1** | **30.9%** | **73.2%** | **56.3%** | **0.87** | **3.32** | **42.2%** | **+4.0pp Hit@1** |
| DPO v2 | 29.8% | 70.7% | 53.9% | 0.85 | 3.26 | 41.1% | +2.9pp Hit@1 |
| AWR v2 | - | - | - | - | - | - | 未全量评估 |

### 7.3 Val split (6 部旧剧, 28 条文案)

| 模型 | Hit@1 | Hit@5 | R@5 | NDCG@5 | 系统均分 | >=4.5 |
|---|---|---|---|---|---|---|
| Base (零样本) | 39.3% | 78.6% | 49.3% | 0.84 | 3.96 | 42.9% |
| SFT375 | 39.3% | 71.4% | 49.3% | 0.83 | 3.82 | - |
| **DPO v1** | **42.9%** | **82.1%** | **55.7%** | **0.87** | **4.11** | **46.4%** |

> **注**：SFT375 在 val 上 Hit@1 持平 base，但 Hit@5 -7.2pp、系统均分 -0.14，说明它没有改善头部分选，反而破坏了中段排序。三组（train/val/test）一致退化是 SFT 被判定为"方向性失败"的核心证据。

### 7.4 谁是最终最优

**DPO v1 checkpoint-2000** 在 train/val/test 三组上全面超越 base，泛化良好，无过拟合。Hit@1 +3.3~4.0pp，系统均分 +0.11~0.15，>=4.0 占比 +3.2~5.8pp。是当前可部署的最优模型。

---

## 8. 关键设计决策回顾

| 决策 | 原因 | 效果 |
|---|---|---|
| 必须用 pairwise/listwise，不能用 pointwise | BCE 优化绝对值，排序才是目标 | SFT 的失败是方向性证据 |
| 用 raw logit diff 而非 sigmoid(score) | 防止 sigmoid 饱和导致梯度消失 | 训练全程梯度通畅 |
| 加 anchor loss 防 score explosion | raw score 无上界，LoRA 会走"全局正偏置"捷径 | DPO 训练稳定，不再触 clamp 上限 |
| LoRA only on text transformer | vision encoder 的通用视觉理解不应被微调破坏 | 36 层文本层足以学偏好 |
| DPO v1 不按 rank_gap 过滤 | 模型已排对的 pair 承担全局校准作用 | DPO v2 的失败证实了这一点 |
| Merge DPO v1 → 重挂 LoRA 再训 AWR | reference = DPO v1（非裸 base），delta 从 0 开始 | 技术上可行，anchor_loss 初始为 0 |
| 按剧切分 train/val/test | 同剧 query 分开放会导致数据泄漏 | 评估结论保守可靠 |

---

## 9. 后续方向

| 方向 | 说明 |
|---|---|
| **DPO v1 + AWR 进一步调参** | 当前 AWR 的 reward 噪声可能是主要瓶颈，可探索直接用 `overall` 而非 bucket-aware reward |
| **DPO v1 → DPO v2 patch** | v2 的 hard-mining 思路不全是错的，它应该作为 v1 之后的小步修正（低 lr, 200-400 step）而非替代 |
| **更大规模 pretrain** | 50K samples 的正样本仅 7,193 条，更多数据可能突破当前天花板 |
| **Listwise loss 的其他形式** | LambdaRank / ListNet 等专门为排序设计的损失函数，可能比 AWR 更适合 reranker 场景 |
