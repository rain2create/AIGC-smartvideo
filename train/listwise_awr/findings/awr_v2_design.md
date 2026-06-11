# AWR v2 Design：从 DPO v1 继续 Listwise AWR 训练

> 记录时间：2026-06-05
> 关联：reranker_awr_design.md（AWR v1 原始设计），train/dpo_v2/findings/（DPO v2 经验教训）

## 0. 背景与动机

DPO v1 (checkpoint-2000) 目前是所有微调版本中最优的 reranker：

| 版本 | train Hit@1 | test Hit@1 | train 均分 | test 均分 |
|---|---|---|---|---|
| base | 26.9% | 27.3% | 3.19 | 3.16 |
| DPO v1 | **30.9%** | **30.6%** | **3.32** | **3.24** |
| DPO v2 @1200 | 29.8% | 31.1% | 3.26 | 3.14 |
| SFT final | 25.3% | - | 3.15 | - |

DPO v2 尝试了 pairwise hard-mining 但**只挖 hard_negative，没有 easy_negative**，结果反而不如 v1。教训明确：**过度聚焦 hard case 不如全难度覆盖**。

**AWR v2 策略**：
1. 从 DPO v1 继续训练（merge LoRA → 新 base）
2. 复用 canonical samples/rewards，用 DPO v1 merged model 给候选打分，重新构造 DPO2000-aware weighted groups
3. group 内保留全难度候选（高分正样本 + v1 易错负样本 + 普通 hard + easy），避免 DPO v2 的失败模式
4. 两种 loss 做 ablation：PPO ratio clip vs linear AWR
5. 离线 group eval 快速选 checkpoint，再跑完整 compare

---

## 1. 模型初始化：Merge DPO v1 → 新 Base

### 问题

当前 AWR 代码通过 `disable_adapter_layers()` 切换到 reference model。如果直接加载 DPO v1 的 LoRA 继续训练：
- reference = base model（裸模型），不是 DPO v1
- delta 初始就很大 → anchor_loss 巨大 → 阻碍训练

### 方案：Merge + 重挂 LoRA

```
1. 加载 base Qwen3-VL-Reranker-8B
2. 加载 DPO v1 LoRA (checkpoint-2000) → PeftModel
3. model.merge_and_unload() → 得到 "DPO v1 merged model"
4. 在 merged model 上挂载全新 LoRA（权重全零初始化）
5. 开始 AWR 训练
```

**效果**：
- `disable_adapter_layers()` 回退到 DPO v1 merged model（正确的 reference）
- delta 从 0 开始，anchor loss 初始为 0
- AWR 只学习"在 DPO v1 基础上的增量改进"

### 代码改动

`train/dpo/dpo_model.py` 新增：

```python
def load_model_with_merged_lora(cfg):
    """加载 base → merge init_lora → 挂新 LoRA"""
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(cfg.model_path, ...)
    from peft import PeftModel as PeftModelCls
    model = PeftModelCls.from_pretrained(base_model, cfg.init_lora_path)
    model = model.merge_and_unload()
    # 挂新 LoRA
    lora_cfg = LoraConfig(r=cfg.lora_rank, ...)
    model = get_peft_model(model, lora_cfg)
    # 冻结 vision encoder
    ...
    return model, processor, yes_id, no_id
```

`train/listwise_awr/awr_train.py`：根据 config 有无 `init_lora_path` 选择加载方式。

### 部署/推理加载链

AWR v2 的 LoRA 是相对 **DPO v1 merged model** 的 adapter，不是相对原始 base。推理时必须：

```python
# 正确：
base → load DPO2000 LoRA → merge → load AWR v2 LoRA → 推理

# 错误：
base → load AWR v2 LoRA → 推理  （reference 不对，效果完全不同）
```

`run_mul_qwen.py` 需要支持 `--init-lora-path` 参数，或在 config 中指定两阶段加载。
训练完成后也可导出 fully-merged model（merge 两次：DPO → merge → AWR → merge）简化部署。

---

## 2. 训练数据策略：DPO2000-Aware Group 重采样 + 加权

### 核心思路

**不重跑 judge，不重新标注。复用 canonical samples/rewards + DPO v1 score，重新构造 groups。**

旧 train_groups.jsonl 的问题：
- 按 bucket 机械抽取（strong 前几个 + hard 前几个 + easy 前几个），与 DPO v1 排序状态无关
- group 内 89% 候选在 base top20 内，但未考虑 DPO v1 的错排情况
- 1,499 / 2,218 = 68% 覆盖率，有 31 个有正样本的 query 没有 group

旧 groups 可以低成本先跑 sanity check，但主实验应使用 DPO2000-aware 重采样版本。

### 候选选择规则

每个 query 仍然一组，保留全量 query 中有正样本的部分。group size 控制在 8-12。

**正样本**（来自 canonical samples，不变）：
- strong_positive：最多 2-3 个
- fallback_positive：最多 1 个
- ambiguous_positive：最多 1 个（可选）

**DPO v1 易错负样本**（新增，用 DPO v1 merged model 离线打分后选取）：
- DPO v1 score 排名靠前的 hard_negative / easy_negative，取 3-5 个
- **特别保留 DPO v1 排到 top1/top3 的负样本**（这些是 v1 最需要纠正的）

**普通校准负样本**（保持全局校准）：
- hard_negative：按 overall 高的取 2-3 个
- easy_negative：取 2-3 个

**关键**：group 内同时包含——高分正样本 + 当前模型易错负样本 + 普通 hard + easy。这才是吸取 DPO v2 教训的正确构造。

### 加权方案

全量 group 保留，按 group 内 DPO v1 的排序质量给不同权重：

```python
# group-local 定义
judge_best_in_group = group candidates 中 reward 最高的候选
dpo_top_in_group    = DPO v1 对 group candidates 打分最高的候选

# 加权规则
if dpo_top_in_group == judge_best_in_group:
    weight = 1.0   # v1 已排对，保留校准信号
elif judge_rank_in_group(dpo_top_in_group) <= 3:
    weight = 2.0   # v1 top-1 错但在前 3，中等难度
else:  # judge_rank > 3 or score_gap 大
    weight = 3.0   # v1 排序严重错误，强化学习
```

> **关键**：judge_best 和 dpo_top 都是 **group 内局部** 定义。
> train_groups 里的 group 只有 8-12 个采样候选，全局 judge_best 可能不在 group 内。

### 实现

新增脚本 `train/listwise_awr/build_awr_v2_groups.py`：

```python
输入：
  - judge/training_data/v3/splits/train_samples.jsonl（复用 canonical rewards）
  - DPO v1 merged model（离线对候选打分）
输出：
  - train_groups_dpo2000_weighted.jsonl
  - val_groups_dpo2000_weighted.jsonl
  - test_groups_dpo2000_weighted.jsonl
  - group_build_stats.json（各 weight 级别数量、候选来源分布、group size 分布）
```

---

## 3. 目标函数：两种 Loss 做 Ablation

### 3a. AWR-v2a：PPO Ratio Clip（现有方案）

```python
delta = s_policy - s_ref                                    # [G]
advantage = (rewards - rewards.mean()) / (rewards.std())    # z-score
ratio = exp(delta)
surr1 = ratio * advantage
surr2 = clamp(ratio, 1-ε, 1+ε) * advantage
pg_loss = -min(surr1, surr2).mean()
anchor_loss = β_anchor * mean(δ²)
total_loss = norm_weight * pg_loss + anchor_loss    # anchor 不加权！
```

**隐患**：delta 是 raw score 差而非真正 log probability ratio。clip_eps=0.2 等价于只允许 delta ∈ [-0.223, 0.182]，窗口很窄。可能训练一开始就大量 clip 导致学不动，或 exp(delta) 对 raw score 过于敏感导致梯度不稳。

### 3b. AWR-v2b：Linear AWR（推荐先跑）

```python
delta = s_policy - s_ref                                    # [G]
advantage = (rewards - rewards.mean()) / (rewards.std())    # z-score
pg_loss = -awr_coef * mean(advantage * delta)
anchor_loss = beta_anchor * mean(δ²)
total_loss = norm_weight * pg_loss + anchor_loss            # anchor 不加权！
```

**优点**：
- 去掉 exp()，delta 对 loss 的贡献是线性的，梯度更稳定
- awr_coef 直接控制 advantage 信号强度，更容易调
- anchor loss 仍然兜底防 score explosion
- 目标 delta 量级更可控

**初始超参**：
- `awr_coef`: 0.02
- `beta_anchor`: 0.01

### 对比实验计划

两个版本可以用同一个 `awr_train.py`，通过 config 中 `loss_type: "clip"` / `"linear"` 切换：

| 实验 | loss_type | 特有参数 |
|---|---|---|
| AWR-v2a | clip | clip_eps=0.2 |
| AWR-v2b | linear | awr_coef=0.02 |

建议先跑 **v2b（linear）**，更稳定；如果效果不够再跑 v2a。

### Weight 与 Loss 的关系

**anchor 不加权，只对 pg_loss 加权**。weight 均值 normalize 到 1.0：

```python
norm_weight = weight / mean_weight_across_dataset
total_loss = norm_weight * pg_loss + anchor_loss
```

理由：anchor 是 trust region 约束，不应因为 hard group 权重大就同步放大——否则 hard group 的学习信号会被加强的 anchor 抵消。

---

## 4. 超参

| 参数 | AWR v1（from base） | AWR v2（from DPO v1） | 调整理由 |
|---|---|---|---|
| lr | 5.0e-6 | **2.0e-6** | 从更强模型出发，小步确认方向 |
| warmup_steps | 20 | **10** | lr 更小，warmup 可缩短 |
| max_steps | 500 | **300** | 细调而非重学，宁愿先少步观察 |
| grad_accum | 4 | 4 | 不变 |
| beta_anchor | 0.01 | 0.01 | 不变，merge 后 delta 从 0 开始 |
| clip_eps (v2a) | 0.2 | 0.2 | PPO 标准值 |
| awr_coef (v2b) | - | **0.02** | 新参数，控制 advantage 信号强度 |
| save_steps | 100 | **50** | 更密 checkpoint，精确选点 |
| lora_rank | 64 | 64 | 不变 |
| lora_alpha | 128 | 128 | 不变 |

---

## 5. 离线 Group Eval（快速选 Checkpoint）

### 动机

完整 pipeline（推理 → compare）太慢，而且只看最终 end-to-end 结果难以定位问题。
需要一个轻量 group 级评估：对每个 checkpoint 直接在 groups.jsonl 上算指标。

### 新增脚本

`train/listwise_awr/eval_awr_groups.py`

输入：checkpoint 路径 + val_groups.jsonl / test_groups.jsonl
输出：每个 checkpoint 的 group 级指标

### 评估指标

| 指标 | 含义 |
|---|---|
| group NDCG@3/5/10 | group 内按 policy score 排序 vs reward 排序的 NDCG |
| reward-top1 hit | policy top-1 == group 内 reward 最高候选的比例 |
| positive mean rank | strong/fallback 候选在 policy 排序中的平均位置 |
| pairwise accuracy | group 内所有 (pos, neg) pair 中 policy score(pos) > score(neg) 的比例 |
| Spearman(score, reward) | policy score 与 reward 的 Spearman 相关系数 |
| bucket 平均 score | strong/fallback/hard/easy 各 bucket 的 policy score 均值，观察是否拉开 |

### 使用流程

```bash
# 每个 checkpoint 快速评估
python3 train/listwise_awr/eval_awr_groups.py \
    --checkpoint train/checkpoints/qwen_awr_v2/checkpoint-50 \
    --groups train/listwise_awr/val_groups_dpo2000_weighted.jsonl \
    --init-lora-path train/checkpoints/qwen_dpo/checkpoint-2000

# 对比多个 checkpoint
python3 train/listwise_awr/eval_awr_groups.py \
    --checkpoints train/checkpoints/qwen_awr_v2/checkpoint-{50,100,150,200,250,300} \
    --groups train/listwise_awr/test_groups_dpo2000_weighted.jsonl
```

先用 group eval 选出最佳 checkpoint，再只对该 checkpoint 跑完整 compare pipeline。

### 离线 Group Eval 成功标准

| 指标 | 目标 | 说明 |
|---|---|---|
| val group NDCG@5 | > DPO v1 reference（未训练时的值） | AWR 必须超过起点 |
| pairwise accuracy | 不下降 | 全局校准不能退化 |
| bucket 平均 score 顺序 | strong > fallback > hard > easy | 基本区分度必须保持 |
| Spearman(score, reward) | > DPO v1 reference | score 与 reward 相关性提高 |

---

## 6. 成功标准

### 主指标（AWR 的核心目标：listwise/top-K 改善）

| 指标 | 目标 | DPO v1 基线 |
|---|---|---|
| NDCG@5 | > DPO v1 | train 0.87, test 0.85 |
| R@5 | > DPO v1 | train 56.3%, test 53.3% |
| NDCG@10 | >= DPO v1 | train 0.90, test 0.89 |

### 保护指标（不能大幅退化）

| 指标 | 底线 | 说明 |
|---|---|---|
| Hit@1 | 不低于 DPO v1 超过 1pp | 允许 train 29.9%+, test 29.6%+ |
| 系统均分 | 不低于 DPO v1 超过 0.05 | 允许 train 3.27+, test 3.19+ |
| >=4.0 / >=4.5 覆盖 | 不明显下降 | 相对 DPO v1 不降超 2pp |

### 训练安全指标

| 指标 | 健康范围 | 异常信号 |
|---|---|---|
| delta_mean | 缓慢增长 0→0.5 | > 2.0：score explosion |
| clip_frac (v2a) | 0.05~0.30 | < 0.01：lr 太小；> 0.5：lr 太大 |
| anchor_loss | < 0.1 | > 1.0：score 漂移 |
| score_raw_max | < 15 | > 20：爆分风险 |

---

## 7. 完整配置

```yaml
# awr_v2_config.yaml
model_path: /home/jovyan/data03/search_public/xby/models/Qwen3-VL-Reranker-8B
init_lora_path: train/checkpoints/qwen_dpo/checkpoint-2000
variant_name: qwen_awr_v2

lora_rank: 64
lora_alpha: 128
lora_dropout: 0.05
lora_target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]

# Loss
loss_type: linear           # "linear" (v2b, 推荐先跑) 或 "clip" (v2a)
awr_coef: 0.02              # linear 版本的 advantage 系数
beta_anchor: 0.01
clip_eps: 0.2               # clip 版本的 PPO clipping 范围
instruction: "Retrieve the most relevant video clip for the given query."

# 视频帧
video_root: /home/jovyan/data03
frame_cache_dir: judge/training_data/frame_cache
total_pixels: 10485760

# 数据
groups_jsonl: train/listwise_awr/train_groups_dpo2000_weighted.jsonl
splits_dir: judge/training_data/v3/splits

# 训练
lr: 2.0e-6
warmup_steps: 10
max_steps: 300
batch_size_per_gpu: 1
grad_accum: 4
max_seq_len: 4096
gradient_checkpointing: true
bf16: true
save_steps: 50

output_dir: train/checkpoints/qwen_awr_v2
log_dir: train/logs/qwen_awr_v2
```

---

## 8. 实施步骤

1. **build_awr_v2_groups.py**：新建，复用 canonical samples + DPO v1 打分，重新构造 DPO2000-aware weighted groups（train/val/test 三套）
2. **dpo_model.py**：新增 `load_model_with_merged_lora()` 函数
3. **awr_train.py**：
   - 根据 `init_lora_path` 选择加载方式
   - 支持 `groups_jsonl` 配置（不再硬编码 splits_dir）
   - 新增 `loss_type` 支持 linear/clip 切换
   - weight 只作用于 pg_loss，不作用于 anchor_loss
4. **eval_awr_groups.py**：新建，离线 group 级评估（NDCG、pairwise acc、Spearman、bucket score）
5. **awr_v2_config.yaml**：新建
6. **run_awr_v2.sh**：新建
7. 执行顺序：build_groups → 训练（先 linear） → group eval 选 checkpoint → 推理 → compare
