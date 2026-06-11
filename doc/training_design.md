# Todo2：Reranker 模型微调

> 基于 todo.md 产出的 v3 训练数据，对 Qwen3-VL-Reranker 做 DPO + Listwise AWR 微调。
> 数据路径：`judge/training_data/v3/splits/`

---

## 0. 输出隔离约定（大前提）

**原则：base / qwen_dpo / qwen_awr 三个变体的所有输出路径完全隔离，任何时候不互相覆盖。**

### 目录约定

```
train/
  checkpoints/
    qwen_dpo/           # DPO LoRA adapter（peft 格式）
    qwen_awr/          # Listwise AWR LoRA adapter（peft 格式）
  logs/
    qwen_dpo/           # 训练日志、loss 曲线
    qwen_awr/

judge/training_data/
  frame_cache/          # 视频帧 JPEG，共享（与模型无关）

judge/outputs/
  canonical/            # 已完成，固定基准，不再重跑
  qwen_dpo/
    {dj}/{run_id}/      # DPO-ft eval 输出
  qwen_awr/
    {dj}/{run_id}/      # Listwise AWR-ft eval 输出
```

### run_id 命名规范

- 训练 config 中统一用 `variant_name` 字段（如 `qwen_dpo`）控制所有输出路径
- eval 脚本读取 `variant_name` 自动拼接路径，禁止硬编码

### 各输出类型隔离方式

| 输出类型 | 隔离 | 路径规则 |
|---|:---:|---|
| 训练 checkpoint | ✅ | `train/checkpoints/{variant_name}/` |
| 训练日志 | ✅ | `train/logs/{variant_name}/` |
| Judge eval 结果 | ✅ | `judge/outputs/{variant_name}/{dj}/{run_id}/` |
| 视频帧缓存（JPEG） | 共享 | `judge/training_data/frame_cache/`（内容不变，无需隔离）|
| Reranker 中间分数 | 按需 | 若落盘，路径带 `{variant_name}/` 前缀 |

---

## 1. 前置确认 ✅ 全部完成

| 问题 | 结论 |
|------|------|
| 权重结构 | 标准 transformers（Qwen3VLForConditionalGeneration），打分 = sigmoid(logit_yes - logit_no)，可直接挂 LoRA |
| 训练模式 | LoRA，rank=64，仅文本 transformer 36 层（vision encoder 冻住）|
| GPU | 2 × L20 46GB，单卡可装（~28-30GB），DDP 跑 |
| DPO/Listwise AWR 顺序 | DPO 先，Listwise AWR 后；DPO 不用 VeRL |
| 视频帧缓存 | 预抽帧存 JPEG（fps=1.0, max_frames=8），仅 reranker 侧，EMB 不需要，~17GB |
| max_seq_len | 4096 |
| 评估 | 跑 val 集 Judge 打分，训练过程不做额外监控 |

---

## 2. Phase A：环境准备

- [ ] 验证 `Qwen3VLForConditionalGeneration` + `peft` LoRA 挂载可正常 forward（梯度回传到文本层）
- [ ] 预抽帧脚本：遍历 `train_pairs.jsonl` + `val_pairs.jsonl` 中所有 video_segment_path，抽帧存 JPEG
- [ ] 确认 `transformers==4.57.0`（config 中写的版本）与 `peft`、`flash-attn` 兼容

---

## 3. Phase B：数据格式适配 ✅ 已定稿

### B1. DPO 输入格式
```
train_pairs.jsonl 每行：
  query.query_text          → prompt（直接用，Q3 格式化字符串）
  chosen.video_segment_path → 预抽帧目录 → 读 JPEG 序列
  rejected.video_segment_path → 同上
```

### B2. Listwise AWR 输入格式（Phase D 再细化）
```
train_groups.jsonl 每行：
  query.query_text              → prompt
  candidates[].video_segment_path → 候选视频帧
  candidates[].reward           → offline reward，直接用，不跑 reward model
```

### B3. 视频帧预处理规格
- fps=1.0, max_frames=8，与推理侧一致
- 存储格式：`{frame_cache_dir}/{video_stem}/frame_{i:04d}.jpg`
- 训练时 DataLoader 读 JPEG → PIL → processor 处理

---

## 4. Phase C：DPO 训练

### C1. 方案 ✅ 已定稿

| 项目 | 决策 |
|------|------|
| 框架 | 自实现（TRL DPOTrainer 不直接支持 score-based 多模态）|
| Reference model | 同一模型 `disable_adapter_layers()` 切换，零额外显存 |
| Loss | 标准 DPO：`-log σ( β · [(s_w − s_w_ref) − (s_r − s_r_ref)] )`<br>s = logit_yes − logit_no（raw logit，不 sigmoid） |
| LoRA target | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`（文本层，36 层）|
| 多卡 | 2 × L20，DDP（`torchrun --nproc_per_node=2`） |

### C2. 超参
```yaml
model_path: models/Qwen3-VL-Reranker-8B
lora_rank: 64
lora_alpha: 128
lora_dropout: 0.05
lora_target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
beta: 0.1
lr: 5e-6
warmup_steps: 100
max_steps: 2000          # ~1350 steps/epoch × 1.5 epoch
batch_size_per_gpu: 1
grad_accum: 8            # effective batch = 16 pairs
max_seq_len: 4096
gradient_checkpointing: true
save_steps: 500
frame_cache_dir: judge/training_data/frame_cache
```

### C3. 开发任务
- [x] `train/fix_paths.py`：JSONL 路径改为相对路径（已完成）
- [x] `train/extract_frames.py`：预抽帧脚本，遍历 pairs/groups 所有视频（已完成）
- [x] `train/dpo/dpo_config.yaml`：超参配置文件
- [x] `train/dpo/dpo_model.py`：LoRA 挂载 + 可微分前向，输出 `logit_yes - logit_no`
- [x] `train/dpo/dpo_dataset.py`：Dataset + DataCollator，返回 `(input_ids, pixel_values, attention_mask)` × chosen/rejected
- [x] `train/dpo/dpo_train.py`：DPO loss + DDP 主循环 + checkpoint 保存

### C4. 关键实现细节

**可微分前向**（区别于推理侧的 `@torch.no_grad`）：
```python
# 推理侧（不可用于训练）
hidden = model(**inputs).last_hidden_state[:, -1]
score = sigmoid(score_linear(hidden))  # detached

# 训练侧（需要梯度）
outputs = model(**inputs, output_hidden_states=True)
hidden = outputs.last_hidden_state[:, -1]          # 保留计算图
logit = lm_head(hidden)                             # 完整 lm_head
s = logit[:, token_yes_id] - logit[:, token_no_id] # raw score，可微分
```

**Reference model 切换**：
```python
# policy forward
logits_policy = forward(model, inputs)
# reference forward（no_grad，LoRA 权重失效）
with model.disable_adapter_layers(), torch.no_grad():
    logits_ref = forward(model, inputs)
```

---

## 5. Phase D：Listwise AWR 训练（VeRL）

> DPO 完成后再细化，当前仅记录待确认项。

### 待确认
- [ ] VeRL 是否支持多模态（video+text）输入，或需要自行适配 dataloader
- [ ] Listwise AWR policy 方案：score-based 直接优化 vs 包装成 yes/no 生成任务
- [ ] Offline reward 接入方式（直接从 groups.jsonl 读 reward，不跑在线 reward model）

### 架构草图
```
Actor（Qwen3-VL-Reranker + LoRA，从 DPO checkpoint 继续）
  ↓ 对每组 15-18 candidates 各输出 score
  ↓ reward = groups.jsonl candidates[].reward（offline，bucket-aware）
  ↓ advantage = per-group z-score（训练框架完成）
  ↓ Listwise AWR policy update
```

---

## 6. Phase E：评估

- 微调完成后，在 val 6 部剧上重跑完整链路（emb 召回 → rerank → Judge 打分）
- 对比 base model vs DPO-ft vs Listwise AWR-ft 的 Judge overall 均值

| 指标 | base | DPO-ft | Listwise AWR-ft |
|------|:----:|:------:|:-------:|
| Judge overall avg (val 6 剧) | - | - | - |

---

## 7. Phase F：推理更新

- [ ] LoRA adapter 挂载到现有 `reranker.py`（`peft.PeftModel.from_pretrained`）
- [ ] 验证推理输出格式与分数范围不变
- [ ] 端到端回归测试（val 6 剧）

---

## 当前待决策项

| # | 问题 | 阶段 |
|---|------|------|
| 1 | Listwise AWR policy：score-based vs 生成 token | Phase D |
| 2 | VeRL 多模态适配方案 | Phase D |
| 3 | DPO → Listwise AWR 是否继续在 DPO checkpoint 上训练 | Phase D |
