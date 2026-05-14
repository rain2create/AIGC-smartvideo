# 微调数据集构造 — 开发规划

> 目标：基于 Canonical Judge 标注，构造 DPO / PPO / GRPO 系列 RL 微调数据集
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
Phase X: 扩展数据覆盖   →  用 V1+Q3 跑新剧 + Canonical Judge 标注
Phase C: 正负样本策略   →  基于全量数据重审 bucket 定义 + 采样配比
Phase D: Builder 重构   →  重写 rl_data_builder.py → v3，支持批量
Phase E: 导出与切分     →  pairs / groups 导出 + train/val/test
Phase F: 质量验证       →  stats 检查 + 人工抽样
```

> **Phase X 插入说明**：6 剧样本量偏小（849 条，28 query），基于此定正负策略可能过窄。先用 V1+Q3 跑更多剧、Canonical Judge 标注、Builder v3 统一跑 samples，再基于全量数据做 Phase C。

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
3. **RL 方法无关**：底座不绑定 DPO 或 GRPO，训练格式在外层适配
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
│ Layer 3: groups.jsonl (GRPO/PPO 候选组) │
│ 从 samples 按 query 聚合生成             │
└─────────────────────────────────────────┘
```

v3 相对 v2 的核心变化：
- labels：新增 `preference_eligible`、`sample_weight`；`is_positive_candidate` → `is_positive`
- retrieval：`system_best_match` 对象拆为标量字段，`system_top_matches` 条目精简
- query/candidate：删除训练无关字段（audio、duration 等）
- judge / source：不动

---

## 5. Phase X：扩展数据覆盖 ⏳ 进行中

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
- [ ] X3. Canonical Judge 标注
- [ ] X4. Builder v3 构造 samples.jsonl + stats.json

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

## 6. Phase C：正负样本策略（待 Phase X 完成后）

### 6.1 当前 bucket 规则回顾（samples.v2）

| bucket | 规则 | 问题 |
|------|------|------|
| strong_positive | overall>=4.5, would_use=true, conf>=0.8 | 阈值是否过严？ |
| fallback_positive | query 无强正时，overall>=4.0 且 would_use=true | 当前触发率低 |
| hard_negative | 2.5<=overall<=3.5，且至少一维>=4 或来自系统topk | 2.5~3.5 区间可能过窄 |
| easy_negative | overall<=2.0 | 太多了（65%），需要采样 |
| dimension_negative | 存在低维+高维共存 | 规则复杂，实际命中少 |
| unusable | 以上都不满足 | 12.8%，是否细分？ |

### 6.2 待讨论调整项

| 议题 | 选项 A | 选项 B | 影响 |
|------|------|------|------|
| strong_positive 阈值 | 保持 >=4.5 | 降到 >=4.0 + conf>=0.8 | 正样本量可能翻倍 |
| hard_negative 区间 | 保持 2.5~3.5 | 扩大到 2.0~3.5 | 覆盖更多边界难例 |
| dimension_negative 保留？ | 保留，单独类型 | 合并进 hard_negative | 简化分类 |
| unusable 处理 | 不进训练 | 划为 extra_easy_negative | 扩充负样本 |

### 6.3 采样配比（DPO pairs）

- strong_positive vs hard_negative: 全部构造（高价值）
- strong_positive vs easy_negative: 每正样本采样 2-3 个
- 最终比例目标：hard/easy ≈ 1:1 到 1:2

### 6.4 采样配比（GRPO groups）

- 每条 query 保留全部 positive + hard_negative + dimension_negative
- easy_negative 采样至每组不超过 ~15-20 个
- unusable 不进组

---

## 7. Phase D：Builder 重构（待 Phase C 完成后）

`judge/rl_data_builder.py` 升级（v2 → v3）：

- 支持批量跑多剧
- 输出 v3 schema 的 samples.jsonl / dropped.jsonl / stats.json
- 新增 pairs.jsonl / groups.jsonl 导出
- 参数化 bucket 阈值和采样比例

```bash
python judge/rl_data_builder.py --all --export-all --dataset-version v3
```

---

## 8. Phase E：训练/验证/测试划分

按剧切分（推荐），避免同剧 query 跨 split 泄漏。

---

## 9. Phase F：质量验证

自动检查 + 人工抽查。

---

## 10. 待确认的关键决策

1. **新剧列表**：哪些剧？跑几部？
2. **strong_positive 阈值**（Phase C）：保持 4.5 还是降到 4.0？
3. **pairs 构造方式**（Phase C）：strong vs hard 全量 + easy 采样？
4. **数据切分**（Phase E）：按剧切还是按 query？
