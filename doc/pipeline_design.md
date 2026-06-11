# 微调数据集构造 — 开发规划

> 目标：基于 Canonical Judge 标注，构造 DPO / PPO / Listwise AWR 系列 RL 微调数据集
> 前置：Step 1（Recall@20 天花板）✅ + Step 2（全因子实验，定 V1+Q3 为微调底座）✅

---

## 1. 现状盘点

### 1.1 数据量（6 剧已完成）

| 剧名 | Canonical 样本 | Query 数 | 候选/query |
|------|------:|------:|------:|
| buxi5 | 138 | 5 | ~28 |
| chongfan80 | 228 | 6 | 38 |
| duibuqinidexiaohaishishebao | 79 | 4 | ~20 |
| luxiansheng | 220 | 5 | ~44 |
| mamadechangshou5 | 77 | 4 | ~19 |
| weinishiguit4 | 107 | 4 | ~27 |
| **合计** | **849** | **28** | - |

### 1.2 代码资产（Phase B 更新后）

| 文件 | 状态 | 版本 |
|------|------|------|
| `judge/rl_data_builder.py` | 待重构 | samples.v2，无批量/pairs/groups |
| `judge/doc/samples_jsonl_schema.md` | ✅ v3 定稿 | 最小增量升级，详见文档 |
| `judge/doc/data_bulider.md` | ✅ 已更新 | 对齐 v3 schema |
| `judge/doc/sample.md` | 待更新 | 需随 Phase C 新规则更新 |

### 1.3 Phase A 数据快照（6 剧 samples.v2 口径）

| bucket | 6 剧合计 | 占比 |
|------|-----:|-----:|
| strong_positive | 78 | 9.2% |
| fallback_positive | 11 | 1.3% |
| hard_negative | 50 | 5.9% |
| dimension_negative | 47 | 5.5% |
| easy_negative | 554 | 65.3% |
| unusable | 109 | 12.8% |
| **正样本合计** | **89** | **10.5%** |
| **负样本合计** | **651** | **76.7%** |


关键问题：正样本极少、easy_negative 压倒性多数、7/28 query 无正样本、70% 样本在 tie_group。

---

## 2. 开发总览

```
Phase A: 数据摸底       →  ✅ 6 剧全量 stats
Phase B: Schema 设计    →  ✅ samples.jsonl v3 定稿（最小增量）
Phase X: 扩展数据覆盖   →  ✅ 100 剧 V1+Q3+Reranker Q1 Pipeline + Canonical Judge + Builder stats
Phase C: 正负样本策略   →  ✅ bucket 定义已定，⏳ 采样配比讨论中
Phase D: Builder 重构   →  重写 rl_data_builder.py → v3，支持批量
Phase E: 导出与切分     →  pairs / groups 导出 + train/val/test
Phase F: 质量验证       →  stats 检查 + 人工抽样
```

> **Phase X 完成**：100 部新剧已跑完 V1+Q3+Reranker Q1 Pipeline + Canonical Judge（50,051 次评估）+ Builder stats（50,051 samples）。全量分布见 Phase C。

---

## 3. Phase A：数据摸底 ✅

用现有 builder 跑 6 剧 samples.v2，拿到统一口径的全局分布。

- [x] A1. 批量跑 6 部剧，输出 `judge/training_data/v3/{dj}/`
- [x] A2. 汇总全局 stats：每剧/每 query 的正负分布、tie 比例、overall 分布
- [x] A3. 人工抽样验证 judge 打分和 bucket 分类

产出路径：
```
judge/training_data/v3/
├── buxi5/annotation_20260508_032425/samples.jsonl + stats.json
├── chongfan80/annotation_20260508_081413/samples.jsonl + stats.json
├── duibuqinidexiaohaishishebao/annotation_20260508_055352/...
├── luxiansheng/annotation_20260508_063407/...
├── mamadechangshou5/annotation_20260508_051542/...
└── weinishiguit4/annotation_20260508_042728/...
```

---

## 4. Phase B：Schema 设计 ✅

最终产出：`judge/doc/samples_jsonl_schema.md` v3 定稿。

设计原则：
1. **单底座派生**：`samples.jsonl` 是唯一真相源，pairs/groups 都从它派生
2. **字段可追溯**：每条样本能回溯到原始 judge 结果、系统检索排名
3. **RL 方法无关**：底座不绑定 DPO 或 Listwise AWR，训练格式在外层适配
4. **按剧隔离**：source 字段保留 dj/run_id，方便按剧切分训练集
5. **最小增量**：从 v2 只做必要调整（labels 新增 2 字段 + retrieval 精简），不裁剪已有内容

### 三层数据模型

```
┌─────────────────────────────────────────┐
│ Layer 1: samples.jsonl (底座)            │
│ 一行 = 一个 (query, candidate)          │
│ 完整 judge + retrieval + labels         │
├─────────────────────────────────────────┤
│ Layer 2: pairs.jsonl (DPO 偏好对)       │
│ 从 samples 按规则筛采样生成              │
├─────────────────────────────────────────┤
│ Layer 3: groups.jsonl (Listwise AWR/PPO 候选组) │
│ 从 samples 按 query 聚合生成             │
└─────────────────────────────────────────┘
```

v3 相对 v2 的核心变化：
- labels：新增 `preference_eligible`、`sample_weight`；`is_positive_candidate` → `is_positive`
- retrieval：`system_best_match` 对象拆为标量字段，`system_top_matches` 条目精简
- query/candidate：删除训练无关字段（audio、duration 等）
- judge / source：不动

---

## 5. Phase X：扩展数据覆盖 ✅

### 5.1 目标

6 剧数据量不足以稳健定义正负样本策略。在 100 部新剧上跑 **V1+Q3+Reranker Q1** Pipeline → Canonical Judge → Builder v3，扩展数据面后再做 Phase C。

### 5.2 选定变体

基于 `judge/outputs/compare_all.md` 全因子对比结论：**`qwen_q3_rerank_q1`** 是微调 reranker 的最优底座。

| 变体 | Hit@1 | 均分 | >=4.5 | 选择理由 |
|------|:---:|:---:|:---:|------|
| `qwen_b1_q3` | 39.3% | 4.03 | 42.9% | 综合最佳，但依赖 B1 sidecar 文本 |
| **`qwen_q3_rerank_q1`** | **39.3%** | **3.96** | **42.9%** | 零样本 reranker 已追平 Hit@1，微调后可超越 |
| `qwen_b1_q3_rerank_q1` | 39.3% | 3.96 | 42.9% | 与上相同，但多了 B1 依赖 |

### 5.3 流程

```
新剧 (100部)
  │
  ├─ V1+Q3+Reranker Q1 Pipeline (S1→S6)
  │     S1 → structured_description.json (视频结构化)
  │     S3 → *_structured.json (文案结构化 + visual_query)
  │     S2 → embeddings_vl.pkl (Qwen3-VL 视觉 embedding)
  │     S4 → retriever V1+Q3 匹配 (top_matches)
  │     S4.5 → reranker Q1 重排序 (top_matches 重排)
  │     S5 → 视频合成 (mp4)
  │     S6 → final.json
  │     变体名: qwen_q3_rerank_q1
  │
  ├─ Canonical Judge 标注
  │     → judge/outputs/canonical/{dj}/{run_id}/results.jsonl
  │
  └─ Builder v3 构造 samples.jsonl
        → judge/training_data/v3/{dj}/
```

### 5.4 任务

- [x] X1. 选定新剧列表（tasks_rerank2.txt，100 部）
- [x] X2. 跑 V1+Q3+Reranker Q1 Pipeline（⚠️ 第一轮误配成 V1+Q3 无 reranker，见 5.6）
- [x] X3. Canonical Judge 标注（100 剧，50,051 次评估，Doubao-seed-2-0-lite，24 条丢失）
- [x] X4. Builder v2 批量统计（106 剧，50,900 samples，见 Phase C 全量分布）

### 5.5 产出

```
text_video_match/cache/{dj}/
  *_structured_qwen_q3_rerank_q1.json   ← S4+S4.5 匹配结果（reranker 重排后）

finish/qwen_q3_rerank_q1/
  {dj}_q3_rerank_q1_final.mp4           ← S5 最终视频

judge/training_data/v3/
  {dj}/annotation_xxx/
    samples.jsonl + stats.json + dropped.jsonl
```

### 5.6 ⚠️ 第一轮误跑记录 (2026-05-13)

**问题**：`tasks_rerank2.txt` 文件中 `rerank_version` 列为空，实际跑成了 `qwen_q3`（V1+Q3，无 reranker），而非预期的 `qwen_q3_rerank_q1`。

**原因**：文件命名为 "rerank2" 但配置列第 9 列（rerank_version）未填写。

**影响**：41 部剧已完成 S1→S6，产物在 `finish/qwen_q3/` 和 `cache/{dj}/*_structured_qwen_q3.json`，这些剧重跑时需要补 S4.5（reranker）+ S5 + S6，S1/S3/S2 可跳过。

**修复**：2026-05-14 已修正 tasks_rerank2.txt，100 行全部配置为 `V1,Q1,`（rerank_version=Q1）。

### 5.7 失败记录 (7/100)

| 剧名 | 失败步骤 | 根因 | 类型 |
|------|------|------|:--:|
| caiyuancaidjzfr | 文案准备 | LLM 输出缺 `## SOP 5`，5次重试均失败 | LLM格式 |
| luomishuweiheyz8 | 文案准备 | 同上 `missing_section:SOP 5` | LLM格式 |
| maquezennengtongyanfei | 文案准备 | 同上 `missing_section:SOP 5` | LLM格式 |
| jushafengba8 | 文案准备 | LLM 输出过短(892字符)，被判无效正文 | LLM格式 |
| mingzhongzdwyly | 文案准备 | 同上 `too_short:45` | LLM格式 |
| jintongyunvlaisf | S1视频索引 | `_4.mp4` 隔行扫描，scenedetect 切出0场景 | 视频编码 |
| luozongnifurcdl | S1视频索引 | `_10.mp4` 文件损坏 (`moov atom not found`) | 视频损坏 |

**建议**：
- LLM格式类(5部)：重试（模型随机性可能通过），或切换到 flash 模型可改善格式遵循
- 视频编码类(1部)：`ffmpeg -i ... -vf yadif ...` 反隔行扫描后重试
- 视频损坏类(1部)：确认源文件是否存在/可修复

---

## 6. Phase C：正负样本策略 ✅ 完成

> 决策日期：2026-05-25
> 数据基础：106 剧全量 stats（50,900 samples，builder v2 口径）

### 6.1 全量数据分布（v2 临时规则）

```
总样本: 50,900（100 新剧 50,051 + 6 旧剧 849）

  strong_positive     3,365 ( 6.6%)
  fallback_positive     539 ( 1.1%)
  hard_negative       2,885 ( 5.7%)
  dimension_negative  5,726 (11.2%)
  easy_negative      30,586 (60.1%)  ← 严重过采样
  unusable            7,799 (15.3%)

  正样本合计: 3,904 (7.7%)
  负样本合计: 39,197 (77.0%)
```

关键发现：
- easy_negative 占 60%，DPO/PPO 必须强制采样
- 正样本分布极不均匀（最好剧 29%，最差剧 1%）
- 933 个 query 完全没有 strong_positive
- unusable 中 58% 是 2.0-2.5，19% 是 >=4.0 但 would_use=False

### 6.2 最终 bucket 定义（Phase C 定稿）

#### strong_positive（强正样本）
```
规则: overall >= 4.5  AND  would_use = True  AND  confidence >= 0.8
数量: 3,365 (6.6%)
用途: DPO chosen / Listwise AWR 高分组
```
> 决策：保持 >=4.5，不降到 4.0。DPO 需要明确偏好信号，正样本宁缺毋滥。

#### fallback_positive（回退正样本）
```
规则: 该 query 下无 strong_positive 时，overall >= 4.0  AND  would_use = True
数量: 539 (1.1%)
用途: 覆盖无强正 query 的正样本
```
> 待定：是否去掉 would_use 限制以扩充回退覆盖。

#### ambiguous_positive（弱正样本）★ 新增
```
规则: overall >= 4.0  AND  would_use = False
来源: 原 unusable 中 4.0+ 区间
数量: ~1,500 (3.0% est.)
用途: Listwise AWR 中等分组 / DPO "弱偏好"对
```
> 决策：Judge 给了高分但认为不可实操，训练时可作为次优正样本。

#### hard_negative（难负样本）★ 扩大
```
规则: 2.0 <= overall <= 3.5
     （不再要求有高维 >= 4.0，不区分 dimension_negative）
来源: 原 hard_negative + dimension_negative + unusable 中 2.0-3.5 区间
数量: ~13,000 (25.5% est.)
用途: DPO rejected 主力 / Listwise AWR 低分组
```
> 决策：合并 dimension_negative，扩大区间从 2.5-3.5 → 2.0-3.5。维度矛盾本质就是难负，单独分类对训练无意义。

#### easy_negative（简单负样本）
```
规则: overall <= 2.0
数量: 30,586 (60.1%)
用途: 训练时强制采样，不能全量进
```
> 决策：定义不变，但 DPO 每正样本只配 2-3 个 easy，Listwise AWR 每 query 只保留 10-15 个 easy。多余丢弃。

#### unusable ★ 取消
```
原定义: 以上都不满足
数量: 0（已全部拆解分配）
拆解去向:
  - 2.0-2.5 (~4,500) → hard_negative
  - 3.5-4.0 (~1,800) → would_use=True 进 fallback_positive, False 进 hard_negative
  - 4.0+ (~1,500)     → ambiguous_positive (新增)
```

### 6.3 调整后实际分布（v3 builder 产出）

```
bucket                 数量       占比
strong_positive       3,365      6.6%   强正
fallback_positive     3,810      7.5%   回退正（含原 unusable 3.5-4.0 would_use=True）
ambiguous_positive       18      0.0%   弱正（极少）
hard_negative        13,841     27.2%   难负（含原 dimension_neg + unusable 2.0-4.0 would_use=False）
easy_negative        29,866     58.7%   简单负（按配比采样）
-------------------------------------------
unusable                  0      0.0%   已全部归桶
```

### 6.4 采样配比（DPO pairs）✅ 已定稿，已产出

| 配对 | 策略 | chosen | rejected | 实际 pairs |
|------|------|:--:|:--:|:--:|
| strong_positive vs hard_negative | 全量，每个强正配 ~4 个 hard | strong | hard | 11,706 |
| strong_positive vs easy_negative | 每个强正配 3 个 easy | strong | easy | 8,496 |
| fallback_positive vs hard_negative | 全量，每个配 ~4 个 hard | fallback | hard | 3,539 |

- ambiguous_positive 不参与 DPO：would_use=False 作为 chosen 信号矛盾，仅用于 Listwise AWR
- **总计: 23,741 DPO pairs**

### 6.5 采样配比（Listwise AWR groups）✅ 已定稿，已产出

#### 分组原则

每 query 一组，bucket 保梯度，bucket-aware reward 保正负：

```
每组候选构成:
  strong_positive      1-3 个
  ambiguous_positive   0-1 个
  fallback_positive    0-1 个
  hard_negative        6-8 个    ← 增加数量，弥补小梯度
  easy_negative        4-6 个    ← 大幅压缩，避免拉低组内均值
  总计：~15-19 候选
```

#### Reward 设计：Bucket-aware 分段

```python
# 正样本：保留原始分（4.0~5.0，正数）
if bucket in (strong_positive, fallback_positive, ambiguous_positive):
    reward = overall

# 负样本：整体下移 4.0（hard → -2.0~-0.5，easy → <-2.0，负数）
else:
    reward = overall - 4.0
```

设计动机：
- 正负两区间之间制造 ~3.5 分的断层，确保负样本 reward 为负
- 保留组内连续细粒度（hard 3.5 比 hard 2.0 惩罚小，语义合理）
- hard 惩罚幅度小但数量多（6-8 个），累积梯度足够
- 不需要 gap 过滤，不丢信息

#### Reward 归一化（Per-group）

每组内对 reward 做 z-score 归一化后再传给 Listwise AWR：

```python
advantage_i = (reward_i - mean(group_rewards)) / (std(group_rewards) + 1e-8)
```

- 消除不同剧之间打分分布差异（有的剧普遍高分、有的剧普遍低分）
- 每组 advantage 统一在同一量纲，跨剧混合训练可比
- 由训练框架在 forward 阶段完成，groups.jsonl 只存原始 reward

#### 边界处理

| 情况 | 处理 |
|------|------|
| query 候选 < 5 | 跳过，对比度不足 |
| query 无任何 overall >= 4.0 候选 | 跳过，全是低分没信息 |
| query 候选 > 25 | 截断到 25，easy 优先截 |
| 每组上限 | 25 候选 |

> 正样本定义：strong_positive 或 fallback_positive 均算；ambiguous（best_overall>=4.0 但 would_use=False）极少（全量仅 3 个 query），可忽略不计。

#### 预估产出（基于 106 剧实测）

- 有效 query: 1,669
- 总 groups: 1,669
- 每组: 5-18 候选，平均 12.7
- 总候选: 21,134

---

## 7. Phase D：Builder 重构 ✅ 完成

`judge/rl_data_builder.py` v3 已完成（2026-05-25）：

- ✅ context-independent 5-bucket 分类（fallback 阈值 3.5，hard 上界 < 4.0）
- ✅ v3 schema：新增 query_text，精简 retrieval，重命名 is_positive/is_negative
- ✅ 批量跑多剧（`--all`）
- ✅ pairs.jsonl 导出（DPO，`--export-pairs`）
- ✅ groups.jsonl 导出（Listwise AWR，`--export-groups`）
- ✅ `--export-all` 一键导出

**产出（106 剧，v3）：**
```
samples:  50,900  （unusable = 0）
pairs:    23,741  DPO
groups:    1,669  Listwise AWR（总候选 21,134，均值 12.7/组）
```

输出路径：`judge/training_data/v3/{dj}/{run_id}/`

---

## 8. Phase E：训练/验证/测试划分 ✅ 完成

按剧切分，避免同剧 query 跨 split 泄漏。

- val：固定 6 部旧剧（`judge/tasks.txt`）
- 其余 100 部按 seed=42 随机切 90/10 → train/test

```
split   剧数    pairs    groups   samples
train     90   21,596    1,521    46,425
val        6      578       25       849
test      10    1,567      123     3,626
```

输出路径：`judge/training_data/v3/splits/`（含 `split_manifest.json`）

## 9. Phase F：质量验证 ✅ 跳过

人工抽查由训练阶段反馈驱动，不单独执行。

