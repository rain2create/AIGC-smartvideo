# smart_videoV6 — 短剧解说视频多模态检索与Reranker微调

> 面向中文短剧解说场景，完成"文案生成 → 视频片段检索 → Judge质量评估 → Reranker微调"的全链路。

## 架构总览

```
┌──────────────────────────────────────────────────────────────────────┐
│  1. 解说文案生成 (commentary/)                                        │
│     LLM 对短剧生成病毒式解说文案 + 每句话标注目标集数                   │
├──────────────────────────────────────────────────────────────────────┤
│  2. 文案-视频匹配 (text_video_match/)                                  │
│     Qwen3-VL Embedding 双路召回 → Qwen3-VL Reranker 精排              │
├──────────────────────────────────────────────────────────────────────┤
│  3. LLM-as-Judge 评估 (judge/)                                       │
│     5维度评分(entity/event/scene/tone/editing) → 训练数据构造          │
├──────────────────────────────────────────────────────────────────────┤
│  4. Reranker 微调 (train/)                                           │
│     SFT → DPO → DPOv2 → Listwise AWR 渐进式微调                      │
└──────────────────────────────────────────────────────────────────────┘
```

## 目录结构

```
smart_videoV6/
├── README.md                   # 本文件
├── CLAUDE.md                   # AI 协作规范
├── run_mul_qwen.py             # V6 主流程编排（Qwen3-VL 检索+Rerank）
├── run_mul.py                  # V4 遗留流程（4阶段匹配+外挂embedding）
├── delete.py                   # 缓存清理工具
│
├── commentary/                 # 解说文案生成
│   ├── skill_video_commentary_text.py   # 主流程：LLM生成解说文案
│   ├── text0204.py                      # RAG增强版文案生成
│   ├── suoying.py                       # 文案句子→集数映射
│   ├── extract_final_script_to_text.py  # Markdown→纯文本提取
│   ├── model_client.py                  # LLM API客户端
│   ├── prompts/                         # 系统提示词模板
│   └── results/{dj}/                    # 每剧文案产出
│
├── text_video_match/           # 文案-视频匹配
│   ├── skill_video_episode_indexer.py   # S1: 视频集数结构化索引
│   ├── skill_struct_info_emb.py         # S2: 结构化信息向量化（V4）
│   ├── skill_text_structured_description.py # S3: 文案结构化拆分
│   ├── skill_match_and_rank.py          # S4: V4四阶段匹配
│   ├── skill_postprocess.py             # S5: 视频合成输出
│   ├── qwen/                            # V6: Qwen3-VL 检索模块
│   │   ├── embedder.py                  # Qwen3-VL-Embedding 封装
│   │   ├── index_builder.py             # 视频embedding索引构建
│   │   ├── retriever.py                 # V1: 纯视觉检索
│   │   ├── retriever_b2.py              # B2: 视觉+文本双路RRF融合
│   │   ├── retriever_b3.py              # B3: 视觉+文本加权融合
│   │   └── reranker.py                  # Qwen3-VL-Reranker 精排
│   └── tools/                           # 视频分割、TTS、分析工具
│
├── judge/                      # LLM-as-Judge 评估系统
│   ├── match_judge.py                  # 单条评估核心（5维度打分）
│   ├── batch_judge.py                  # 批量评估编排
│   ├── rl_data_builder.py              # 训练数据构造（v3, samples/pairs/groups）
│   ├── compare_variants_with_canonical.py # 变体效果对比
│   ├── split_dataset.py                # Phase E: 数据集切分
│   ├── batch_build_stats.py            # 全量统计
│   ├── doc/                            # schema设计文档
│   ├── outputs/                        # Judge评估结果
│   └── training_data/                  # 训练数据产出 (v3/)
│
├── train/                      # Reranker 微调训练
│   ├── extract_frames.py               # 视频帧预抽取 (JPEG缓存)
│   ├── fix_paths.py                    # 路径修正工具
│   ├── sft/                            # SFT: 监督微调
│   ├── dpo/                            # DPO: 直接偏好优化
│   ├── dpo_v2/                         # DPOv2: 硬负样本挖掘
│   ├── listwise_awr/                   # AWR: 优势加权回归
│   ├── checkpoints/                    # LoRA适配器权重
│   └── logs/                           # 训练日志/曲线
│
├── doc/                        # 规划设计文档
│   ├── pipeline_design.md              # Pipeline+RL数据管道设计（含决策记录）
│   ├── training_design.md              # Reranker微调方案设计
│   └── matching_changes.md             # 文案匹配改动日志
│
├── tasks/                      # 批量任务文件
│   ├── canonical_judge/                # Canonical Judge 100剧任务
│   └── inference/                      # 各版本推理任务（SFT/DPO/AWR）
│
└── scripts/                    # Shell脚本
    ├── run_qwen.sh                     # V6 Pipeline启动
    ├── run_v4.sh                       # V4 Pipeline启动
    ├── batch_judge.sh                  # V4 Judge批量评估
    └── batch_judge_qwen.sh             # Qwen变体Judge评估
```

## Pipeline 版本

| Pipeline | 匹配方式 | 状态 |
|----------|---------|:----:|
| **V6** | Qwen3-VL Embedding 多路召回 + Qwen3-VL Reranker 精排 | 当前主线 |
| V4 | 4阶段匹配（语义→视觉→情感→LLM）+ 外挂embedding API | 遗留 |

### V6 检索变体

| 变体 | Query版本 | Index版本 | Reranker | 说明 |
|------|:--------:|:--------:|:--------:|------|
| qwen_q3_rerank_q1 | Q3 | V1 | Q1 | **微调底座**（综合最佳） |
| qwen_b1_q3 | Q3 | B1 | - | 视觉+文本sidecar联合embedding |
| qwen_b2_q3 | Q3 | B2 | - | 双路独立检索 + RRF融合 |
| qwen_b3_q3 | Q3 | B3 | - | 双路检索 + 加权融合 |

## 训练链路

### 数据管道

```
Canonical Judge (100剧, 50K样本)
  → rl_data_builder.py (5 bucket分类)
    → samples.jsonl (50,900条)
      ├── pairs.jsonl (23,741 DPO pairs)
      └── groups.jsonl (1,669 AWR groups)
        → split_dataset.py → train/val/test
```

### 5 Bucket 分类

| Bucket | 规则 | 用途 |
|--------|------|------|
| strong_positive | overall>=4.5, would_use=True, confidence>=0.8 | DPO chosen / AWR高分组 |
| fallback_positive | overall>=3.5, would_use=True (无strong时才用) | 覆盖无强正query |
| ambiguous_positive | overall>=4.0, would_use=False | AWR中分组 |
| hard_negative | 2.0<=overall<4.0 | DPO rejected主力 |
| easy_negative | overall<2.0 | 强制采样，不全会入 |

### 训练路径

```
SFT (375 steps, pos_weight=1.6)
  → DPO (2000 steps, beta=0.1)
    → DPOv2 (1200 steps, 硬负样本挖掘)
      → AWR (300 steps, awr_coef=0.02, 从DPO checkpoint继续)
```

## 快速开始

### 1. 单剧Pipeline

```bash
# V6: Qwen3-VL 检索 + Reranker 精排
bash scripts/run_qwen.sh --foreground chongfan80 1 10 output.mp4 5 Q3 0 --rerank-version Q1

# V4: 经典4阶段匹配
bash scripts/run_v4.sh chongfan80 1 10
```

### 2. 批量Pipeline

```bash
# 用任务文件批量跑
bash scripts/run_qwen.sh --batch tasks/inference/sft375_all.txt
```

### 3. Judge评估

```bash
# V4变体评估
bash scripts/batch_judge.sh tasks/canonical_judge/all.txt

# Qwen变体评估
bash scripts/batch_judge_qwen.sh --dj chongfan80 --query-version Q3
```

### 4. 训练数据构造

```bash
# 全量构造 samples + pairs + groups
python judge/rl_data_builder.py --all --export-all --dataset-version v3
```

### 5. 训练

```bash
# SFT
bash train/sft/run_sft.sh

# DPO
bash train/dpo/run_dpo.sh

# AWR（需先完成DPO训练）
bash train/listwise_awr/run_awr_v2.sh
```

### 6. 微调模型推理

```bash
# checkpoint_path 指向 train/checkpoints/ 下的LoRA adapter
python run_mul_qwen.py chongfan80 1 10 output.mp4 5 Q3 0 V1 Q1 --checkpoint train/checkpoints/qwen_dpo/checkpoint-2000
```

## 产物路径速查

| 阶段 | 产物路径 |
|------|------|
| 解说文案 | `commentary/results/{dj}/` |
| 视频索引 | `text_video_match/cache/{dj}/structured_description.json` |
| V6检索结果 | `text_video_match/cache/{dj}/*_structured_qwen_*.json` |
| 视频帧缓存 | `judge/training_data/frame_cache/` |
| Judge评估 | `judge/outputs/{variant}/{dj}/{run_id}/` |
| 训练数据 | `judge/training_data/v3/splits/` |
| 模型权重 | `train/checkpoints/{variant}/` |
| 最终视频 | `finish/{variant}/` |
