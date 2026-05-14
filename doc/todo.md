# TODO

> 当前优先：基于 Canonical Judge 标注构造 reranker 微调数据集

## 背景

当前 B1 是最优 embedding 配置，B1+reranker Q1 是最优整体配置。但这个结论是贪心得到的：先固定 emb 找最优 query，再固定 emb 找最优 reranker。可能遗漏全局最优——简单 emb 配 reranker 反而更好。

## Step 1：Recall@20 天花板分析 ✅

**目的**：看各 emb 变体的 top20 有没有覆盖到 Judge 优质候选。Recall@20 是 reranker 能发挥的上限。

- [x] 跑 6 剧全量 judge，输出各 emb 变体（V4, Q1, Q2, Q3, B1, B2）的 Recall@20
- [x] 分析结论：如果 Recall@20 差异不大 → 简 emb 就够了；如果差异大 → emb 复杂度有价值

**数据**：已有，直接跑 judge 即可。

### 结论

**天花板基本一致，emb 复杂度差异在排序能力而非候选池质量。**

| Variant | Oracle 均分 | >=4.0 总数 | >=4.5 总数 | 实际均分 |
|---------|------------|-----------|-----------|---------|
| B1      | 4.391      | 102       | 70        | 4.03    |
| Q3      | 4.391      | 102       | 71        | 3.84    |
| Q1      | 4.388      | 94        | 66        | 3.77    |
| Q2      | 4.388      | 98        | 69        | 3.73    |
| V4      | 4.388      | 95        | 66        | 3.14    |
| B2      | 4.286      | 87        | 61        | 3.03    |

- Oracle 均分（完美 reranker 能从 top-20 挑出的最好候选的 Judge 均分）极差仅 0.105，几乎无差异
- >=4.0/>=4.5 候选覆盖数，B1 vs Q1 每条 query 仅差 0.28/0.14 个，差距很小
- 但实际均分极差 1.00（3.03~4.03），差距在"把好候选排到第一位"的能力
- Q1 交集均分（2.806）甚至高于 B1（2.619），说明 Q1 的 top-20 里跟 Judge 重叠的部分质量不差
- **核心启示**：简单 emb + reranker 可能追平复杂 emb，因为 reranker 弥补的正是排序能力——而简单 emb 的 Oracle 上限并不低

## Step 2：全因子实验（6 剧）✅

**目的**：扫 emb query × reranker 全因子网格，找全局最优组合。

- [x] 跑 B1+Q1、B1+Q2 S4 检索（6 剧）
- [x] 跑 V1+Q1/Q2/Q3+reranker Q1、B1+Q1/Q2+reranker Q1（6 剧）
- [x] Judge 对比 12 组合全量（含已有变体共 15 变体）

### 结论

**B1+Q3（无 reranker）仍是综合最佳，reranker 帮弱 query 不帮强 query。**

#### Reranker 对弱 query（Q1/Q2）提升显著

| 变体 | 无 reranker Hit@1 | +reranker Q1 Hit@1 | 提升 |
|------|:-:|:-:|:-:|
| V1+Q1 | 28.6% | 35.7% | +7.1pp |
| V1+Q2 | 32.1% | 39.3% | +7.1pp |
| B1+Q1 | 25.0% | 35.7% | +10.7pp |
| B1+Q2 | 35.7% | 39.3% | +3.6pp |

#### Reranker 对强 query（Q3）无效

| 变体 | 无 reranker Hit@1 | +reranker Q1 Hit@1 |
|------|:-:|:-:|
| V1+Q3 | 39.3% | 39.3%（均分 3.84→3.96） |
| B1+Q3 | 39.3% | 39.3%（均分 4.03→3.96 ⚠️倒退） |

#### 综合排名（Top 5）

| 排名 | Variant | Hit@1 | 均分 | R@10 | R@20 | >=4.5 |
|:---:|---|:-:|:-:|:-:|:-:|:-:|
| 1 | `qwen_b1_q3`（无reranker） | 39.3% | 4.03 | 68.2% | 80.0% | 42.9% |
| 2 | `qwen_q3_rerank_q1` | 39.3% | 3.96 | 57.9% | 75.2% | 42.9% |
| 3 | `qwen_b1_q3_rerank_q1` | 39.3% | 3.96 | 59.6% | 78.5% | 42.9% |
| 4 | `qwen_q3`（无reranker） | 39.3% | 3.84 | 62.5% | 75.4% | 32.1% |
| 5 | `qwen_b1_q2_rerank_q1` | 39.3% | 3.79 | 58.9% | 77.9% | 35.7% |

#### 决策：选定 V1+Q3 为 reranker 微调底座

零样本 reranker 下，V1+Q3+reranker 的 Hit@1 已追平 B1+Q3（39.3%），均分仅差 0.07（3.96 vs 4.03）。微调 reranker 预计能超越 B1+Q3：
- Cross-encoder 表达能力天然强于 bi-encoder
- 零样本差距仅 0.07 均分，微调稍微学一点就能跨过去
- 微调后 reranker 直接从视频帧学 Judge 偏好，不再依赖 B1 sidecar 文本
- B1+Q3（无 reranker）作为 fallback 保留

## Step 3：构造 Reranker 微调数据集

**目的**：基于 Canonical Judge 标注，构造 (query, positive_video, negative_video) 三元组微调 Qwen3-VL-Reranker。

- [ ] 确定微调数据格式，对齐 Qwen3-VL-Reranker 训练接口
- [ ] 从 Canonical Judge 矩阵提取正负样本
- [ ] 评估数据质量（正负样本可区分性、hard negative 比例）
- [ ] 划分训练/验证集
- [ ] 微调实验

## 后续

- [ ] Qwen3-VL-Reranker 微调实验
- [ ] 生产化部署方案
