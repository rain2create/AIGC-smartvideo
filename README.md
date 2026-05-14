# 影视解说-画面匹配系统 V4

## 完整 Pipeline（6个阶段）

```
输入：剧名(dj) + 集数范围(start~end)
  │
  ├─ 准备：获取解说文案（txt）
  │    先找现成 txt → 没有则从 md 提取 → 再没有则用 LLM 生成(text0204.py)
  │
  ├─ S1: 视频集数索引化 (skill_video_episode_indexer.py)
  │    每集视频 → VideoSegmenter(场景检测+切割) → VideoEvaluator(多模态LLM理解每个片段)
  │    输出: cache/{dj}/structured_description.json（含episode字段）
  │
  ├─ S2: 结构化信息索引化 (skill_struct_info_emb.py)
  │    把视频结构化描述拼接成文本 → 调embedding API → 保存向量
  │    输出: cache/{dj}/embeddings.pkl
  │
  ├─ S3: 解说文案拆分 (skill_text_structured_description.py)
  │    ① suoying模块：用LLM把每句话映射到对应集数
  │    ② 每句话单独做结构化描述（visual_query: characters, actions, environment, fuzzy_tags...）
  │    ③ TTS生成音频
  │    输出: cache/{dj}/{txt}_structured.json
  │
  ├─ S4: 四阶段匹配与精排 (skill_match_and_rank.py) ← 核心模块
  │    Stage1: 语义召回 → TOP 20（文案embedding vs 视频embedding余弦相似度）
  │    Stage2: 视觉精排 → TOP 10（characters/actions/environment/dialogue加权相似度）
  │    Stage3: 情感深排 → TOP 5（interaction/emotion/intent加权）
  │    Stage4: LLM精排 → TOP 1（当前已禁用，直接用Stage3 TOP1）
  │    权重：硬编码，根据文案类型(emotion_action/balanced等)动态选择
  │
  ├─ S5: 视频合成 (skill_postprocess.py)
  │    按匹配结果裁剪视频片段 → 合并音频 → 拼接输出
  │    只处理前20秒的文案
  │
  └─ S6: 生成 final.json（每段保留top1匹配结果）
```

## 目录结构

```
smart_videoV4/
├── run.py                          # Pipeline 主入口（单剧）
├── run_mul.py                      # Pipeline 主入口（批量并行）
├── run.sh                          # Shell 版 Pipeline（支持批量+日志）
├── tasks.txt                       # 批量任务配置
├── plan.md                         # 升级改造方案（Phase 1-4）
│
├── text_video_match/               # 核心匹配模块
│   ├── skill_video_episode_indexer.py   # S1: 视频集数索引化
│   ├── skill_struct_info_emb.py         # S2: 结构化信息索引化
│   ├── skill_text_structured_description.py  # S3: 解说文案拆分
│   ├── skill_match_and_rank.py          # S4: 四阶段匹配与精排
│   ├── skill_postprocess.py             # S5: 视频合成
│   ├── base_skill.py                    # Skill 基类
│   ├── tools/
│   │   ├── video_segmenter.py           # 视频场景检测+切割（scenedetect）
│   │   ├── video_evaluator.py           # 视频理解（多模态LLM）
│   │   └── video_tts.py                 # TTS 语音合成（edge-tts）
│   ├── cache/{dj}/                      # 每部剧的中间数据
│   └── README.md
│
├── video_commentary_text/          # 文案生成模块
│   ├── text0204.py                      # LLM 生成解说文案
│   ├── extract_final_script_to_text.py  # 从 md 提取纯文本文案
│   ├── model_client.py                  # LLM API 客户端
│   ├── suoying.py                       # 文案→集数映射（基于ASR）
│   ├── rag_retrieval_labeling.py        # RAG 检索标注
│   └── results/{dj}/                    # 每部剧的文案结果
│
├── judge/                          # LLM-as-Judge 评估模块（开发中）
│   └── judge.md                         # Judge 设计文档
│
├── finish/                         # 最终合成视频输出
└── logs/                           # 运行日志
```

## 外部依赖

| 服务 | 地址 | 用途 |
|------|------|------|
| Embedding API | `10.163.166.76:17998/emb/` | 文本向量化（BGM/BERT） |
| LLM API | `api.360.cn` | 文案生成、结构化描述、视频理解（Gemini-3-Pro/Flash） |
| edge-tts | 本地 | TTS 语音合成 |
| ffmpeg | 本地 | 视频切割/合成 |
| scenedetect | 本地 | 视频场景检测 |

## 用法

```bash
# 单剧
python run.py <dj> [start] [end] [output_filename] [top_k]
python run.py buxi5 1 15

# 单剧（Shell版，带日志）
bash run.sh buxi5 1 15

# 批量并行
python run_mul.py --batch tasks.txt [top_k]
bash run.sh --batch tasks.txt [top_k]
```
