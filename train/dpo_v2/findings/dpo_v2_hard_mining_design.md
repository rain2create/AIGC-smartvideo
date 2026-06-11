# DPO v2 Hard-Mining 设计文档

> 记录时间：2026-06-03
> 背景：DPO v1 废信号 77.5%，SFT pointwise BCE 路线失败，需要更精准的 pairwise 训练数据

## 1. 为什么要做 DPO v2

### DPO v1 的问题
DPO v1 穷举所有 bucket 内 (正, 负) 配对，生成 21,596 pairs。但经分析：

┌──────────────────┬────────┬────────────────────────┐
│ Pair Type        │ 数量   │ Reranker 已排对（废信号）│
├──────────────────┼────────┼────────────────────────┤
│ strong_vs_hard   │ 10,701 │ 75.0%                  │
│ strong_vs_easy   │ 7,706  │ 88.0%                  │
│ fallback_vs_hard │ 3,189  │ 64.1%                  │
├──────────────────┼────────┼────────────────────────┤
│ 合计             │ 21,596 │ 77.5%（~16,737 废梯度） │
└──────────────────┴────────┴────────────────────────┘

DPO2000 只提升 Hit@1 +4.8pp（train）/ +6.8pp（test），与 77.5% 信号浪费直接相关。

### 为什么不从 v1 pairs 过滤

v1 pairs 按 overall 分数抽负样本（每个 strong 配 overall 最高的 4 个 hard），会**系统性漏掉一类关键样本**：judge 分不一定最高，但 base reranker 排得很靠前的负样本。DPO v2 的核心是 rank-based hard-mining，必须从 samples 全量重构才能抓住这类样本。

### SFT 路线失败的教训
- Pointwise BCE 与排序任务不对齐 → 头部排序被破坏（test Hit@5 -11.4pp）
- 结论：**必须用 pairwise loss**（DPO），但数据质量要提升

## 2. Hard-Mining 核心思路

**只在 reranker 排错 / 排得不稳的 pair 上训练。**

### 设计原则

┌──────────────┬─────────────────────────────────────────────────┐
│ 信息源       │ 用途                                            │
├──────────────┼─────────────────────────────────────────────────┤
│ judge overall│ 定义 preference 是否可信（chosen/rejected 身份） │
│ bucket       │ 定义正负样本类型                                │
│ system_rank  │ 判断 hard 程度（query-local，跨 query 可比）     │
└──────────────┴─────────────────────────────────────────────────┘

- **rank 判断 hard 程度**：`system_rank` 是 query-local 的排序位，天然比 raw score / similarity 更稳定
- **分数定义 preference**：chosen/rejected 靠 judge 分数和 bucket 定义
- **不用 reranker_score/sim 绝对值**：分数尺度压缩且跨 query 不可比

### system_rank 来源
`system_rank` = reranker 的最终排序位（非 embedder），代码链路：
```
reranker.py:381-413 → 按 reranker_score 重排 top_matches
rl_data_builder.py:256-272 → find_system_rank() 从重排后的 top_matches 取位置
```

### Hard Pair 定义

```
rank_gap = rank(rejected) - rank(chosen)
```

┌─────────────────┬────────────────────┬─────────────────────────────────────┬──────────┐
│ 类型            │ 条件               │ 含义                                │ 信号强度 │
├─────────────────┼────────────────────┼─────────────────────────────────────┼──────────┤
│ Type A: 反转    │ gap ≤ 0            │ 负样本排在正样本之上或并列           │ 最高     │
│ Type B: 近距    │ 0<gap≤3            │ 负样本在正样本下方但仅差 1-3 位      │ 中等     │
│                 │ 且 rejected_rank≤10│                                     │          │
│ （跳过）        │ gap > 3            │ 模型已排对且有余量                   │ 无       │
└─────────────────┴────────────────────┴─────────────────────────────────────┴──────────┘

## 3. 数据构造规则

### 数据源
`judge/training_data/v3/splits/train_samples.jsonl`（46,425 samples, 90 dramas）

从 samples 按 query 分组，重新构造 pairs。

### Chosen（正样本）筛选
- bucket = `strong_positive`；若该 query 无 strong，才用 `fallback_positive`
- `in_system_top_matches = True`（在 top-20 内）
- `system_rank` 不为空
- `overall >= 4.0`

### Rejected（负样本）筛选
- bucket = `hard_negative`（第一版不含 easy_negative，见 §7 ablation）
- `in_system_top_matches = True`
- `system_rank` 不为空
- `overall <= 3.5`

### Hard 规则
```
rank_gap = rejected_rank - chosen_rank

保留条件：
- inversion:  rank_gap <= 0（任何 rejected_rank）
- semi-hard:  0 < rank_gap <= 3 且 rejected_rank <= 10
```

### 限额控制
- 每个 query 最多 **20 pairs**（防止长尾 query 主导）
- 每个 chosen 最多配 **6 个 rejected**（防止单正样本过拟合）
- 每个 rejected 最多配 **8 个 chosen**（防止同一负样本过度重复）

### 排序优先级（per-query 限额裁剪时）
```
1. inversion 优先于 semi-hard
2. rejected_rank 越小越优先（rank-1 的坏负样本最致命）
3. rank_gap 越小越优先（反转越严重越优先）
4. judge_gap = chosen_overall - rejected_overall 越小越优先（质量越接近越难区分，越值得训练）
5. chosen_rank 越大越优先（被模型低估越多的正样本越值得纠正）
```

### 预估数据量

┌───────────────────────────────────────────┬────────┐
│ 描述                                      │ pairs  │
├───────────────────────────────────────────┼────────┤
│ 从 train_samples 重构                     │        │
│ strong 优先 + fallback 补位               │        │
│ rejected = hard_negative only             │        │
│ boundary: chosen≥4.0, rejected≤3.5        │        │
│ gap≤0 (inversion) + gap≤3 & rank≤10      │        │
│ 预估总量                                  │ ~~6.8k  │
│ 覆盖 queries                              │ ~1,241 │
├───────────────────────────────────────────┼────────┤
│ vs DPO v1                                │ 21,596 │
│ 压缩比                                    │ ~68%   │
│ 预估有效信号占比                           │ ~80%+  │
└───────────────────────────────────────────┴────────┘

## 4. 实现方案

### 文件结构
```
train/dpo_v2/
├── export_dpo_v2_pairs.py   # 从 samples 重构 hard-mined pairs
├── dpo_v2_config.yaml       # 训练超参
├── run_dpo_v2.sh            # 训练入口
└── findings/                # 本文档所在
```

### 复用（不新写训练代码）
- `train/dpo/dpo_model.py` — `load_model_and_processor()`, `compute_score()`, `save_lora()`
- `train/dpo/dpo_dataset.py` — `DPODataset`, `DPODataCollator`, `build_conversation()`
- `train/dpo/dpo_train.py` — 训练主循环（DPO loss, DDP, logging, checkpoint）

### export_dpo_v2_pairs.py 逻辑

输入：`judge/training_data/v3/splits/train_samples.jsonl`

```python
# 伪代码
for query_id, samples in grouped_by_query.items():
    # 1. 筛选 chosen
    strongs = [s for s in samples
               if s.bucket == "strong_positive"
               and s.in_top20 and s.system_rank is not None
               and s.overall >= 4.0]
    if strongs:
        chosen_pool = strongs
    else:
        chosen_pool = [s for s in samples
                       if s.bucket == "fallback_positive"
                       and s.in_top20 and s.system_rank is not None
                       and s.overall >= 4.0]

    # 2. 筛选 rejected（仅 hard_negative）
    rejected_pool = [s for s in samples
                     if s.bucket == "hard_negative"
                     and s.in_top20 and s.system_rank is not None
                     and s.overall <= 3.5]

    # 3. 枚举 pairs，按 hard 规则过滤
    pairs = []
    chosen_neg_count = defaultdict(int)   # per-chosen limit
    rejected_pos_count = defaultdict(int)  # per-rejected limit
    for pos in chosen_pool:
        for neg in rejected_pool:
            rank_gap = neg.system_rank - pos.system_rank
            if rank_gap <= 0:
                pair_subtype = "inversion"
            elif rank_gap <= 3 and neg.system_rank <= 10:
                pair_subtype = "semi_hard"
            else:
                continue
            pairs.append((pos, neg, pair_subtype, rank_gap))

    # 4. 排序 + 限额裁剪
    pairs.sort(key=lambda p: (
        0 if p[2] == "inversion" else 1,  # inversion 优先
        p[1].system_rank,                  # rejected_rank 小优先
        p[3],                              # rank_gap 小优先
        (p[0].overall - p[1].overall),     # judge_gap 小优先（越接近越hard）
        -p[0].system_rank,                 # chosen_rank 大优先
    ))

    # 5. 应用 per-query / per-chosen / per-rejected 限额
    final_pairs = []
    for pos, neg, subtype, gap in pairs:
        if len(final_pairs) >= MAX_PAIRS_PER_QUERY:  # 20
            break
        if chosen_neg_count[pos.sample_id] >= MAX_NEG_PER_CHOSEN:  # 6
            continue
        if rejected_pos_count[neg.sample_id] >= MAX_POS_PER_REJECTED:  # 8
            continue
        final_pairs.append(...)
        chosen_neg_count[pos.sample_id] += 1
        rejected_pos_count[neg.sample_id] += 1
```

### 输出 schema
```json
{
  "pair_id": "<query_id>_c<chosen_cand>_r<rejected_cand>",
  "query_id": "...",
  "pair_type": "strong_vs_hard",
  "pair_subtype": "inversion",
  "rank_gap": -3,
  "judge_gap": 1.8,
  "query": { "script_text": "...", "query_text": "..." },
  "chosen": {
    "sample_id": "...", "candidate_id": "...",
    "bucket": "strong_positive", "overall": 4.6,
    "video_segment_path": "...", "system_rank": 8
  },
  "rejected": {
    "sample_id": "...", "candidate_id": "...",
    "bucket": "hard_negative", "overall": 2.8,
    "video_segment_path": "...", "system_rank": 3
  },
  "source": { "dj": "..." }
}
```

导出时输出统计摘要：pair 总数、inversion/semi-hard 分布、query 覆盖率、rank_gap 直方图。

### 训练超参

┌──────────────────────┬──────────┬──────────┬────────────────────────────┐
│ 参数                 │ DPO v1   │ DPO v2   │ 调整理由                   │
├──────────────────────┼──────────┼──────────┼────────────────────────────┤
│ 目标函数             │ DPO loss │ DPO loss │ 不变                       │
│ pairs 数量           │ 21,596   │ ~6,849   │ hard-mining 重构           │
│ max_steps            │ 2,000    │ 1,500    │ ~3.5 epoch，多 ckpt 评估   │
│ effective batch      │ 16       │ 16       │ 不变                       │
│ save_steps           │ 500      │ 300      │ 每 ~0.7 epoch 存一次，5 ckpt│
│ lr                   │ 5e-6     │ 5e-6     │ 不变                       │
│ beta                 │ 0.1      │ 0.1      │ 不变                       │
│ beta_anchor          │ 0.01     │ 0.01     │ 不变                       │
│ LoRA rank/alpha      │ 64/128   │ 64/128   │ 不变                       │
└──────────────────────┴──────────┴──────────┴────────────────────────────┘

**注意**：~6,849 pairs / effective_batch 16 = ~428 steps/epoch。max_steps=1500 ≈ 3.5 epoch。
必须评估 300/600/900/1200/1500 全部 checkpoint，不要默认 final 最好。

## 5. 与 DPO v1 的核心区别

┌──────────────┬──────────────────────┬──────────────────────────┐
│ 维度         │ DPO v1               │ DPO v2                   │
├──────────────┼──────────────────────┼──────────────────────────┤
│ 数据源       │ samples → 穷举配对    │ samples → rank-based 重构│
│ 负样本类型   │ hard + easy          │ hard only（v2a 主实验）   │
│ 配对依据     │ overall 分数排序取 N  │ rank gap 过滤            │
│ 数据量       │ 21,596 pairs         │ ~6,849 pairs             │
│ 有效信号     │ 22.5%                │ ~80%+                    │
│ Boundary     │ 无 overall 过滤       │ chosen≥4.0, rejected≤3.5 │
│ 头部聚焦     │ 无                    │ semi-hard: rejected≤10   │
│ 目标函数     │ DPO loss             │ DPO loss（不变）          │
└──────────────┴──────────────────────┴──────────────────────────┘

## 6. 验证计划
1. 运行 `export_dpo_v2_pairs.py`，确认 ~6,849 pairs + 分布合理
2. 检查 frame_cache 覆盖（in_top20 样本帧应已缓存）
3. 训练，save_steps=300，评估全部 5 个 checkpoint
4. Val 6 部剧推理 → `compare_variants` 对比 Base / DPO v1(2000) / DPO v2(best ckpt)

## 7. 后续 Ablation（非第一版）

| 实验 | 改动 | 目的 |
|------|------|------|
| v2b | + easy_negative inversion only | 验证 easy neg 排到前面是否值得纠正 |
| v2c | MARGIN=0（纯 inversion） | 验证 semi-hard 是否有增量收益 |
| v2d | rejected_rank ≤ 5 | 更激进的头部聚焦 |
