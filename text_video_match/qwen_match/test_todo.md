## 测试 Todo

### 当前基线

| 项目 | 路径 |
|---|---|
| V4 历史 Judge 基线 | `judge/outputs/v4/chongfan80/annotation_20260423_104822/` |
| 当前 Qwen query 版本 | `Q1` |
| 当前 Qwen variant | `qwen_q1` |

核心要点：历史 V4 Judge 结果已经迁到标准版本目录下，后面新跑的 Qwen Judge 结果会进 `judge/outputs/qwen_q1/...`，可以直接横向对比。

### 1. 基本的多模态 emb 检索召回跑通

| 项目 | 命令 |
|---|---|
| 1A. 视频 visual emb 建索引 | `python text_video_match/qwen_match/index_builder.py chongfan80 --max-segments 5 --overwrite --fps 3 --max-frames 30 --max-length 32768` |
| 1B. Q1 检索 smoke test | `python text_video_match/qwen_match/retriever.py chongfan80 text_video_match/cache/chongfan80/1_10_20260423_155935_structured.json --top-k 5 --max-segments 3 --query-version Q1 --output-path text_video_match/cache/chongfan80/1_10_20260423_155935_structured_qwen_q1.json --overwrite` |

核心要点：这一步只验证“视频 emb 建库 + 文案到视频的 TopK 检索”是否打通，不检查最终合成。

| 检查点 | 预期产物 |
|---|---|
| 索引文件 | `text_video_match/cache/chongfan80/embeddings_vl.pkl` |
| 索引清单 | `text_video_match/cache/chongfan80/embeddings_vl_manifest.json` |
| 检索结果 | `text_video_match/cache/chongfan80/1_10_20260423_155935_structured_qwen_q1.json` |

核心要点：如果这一步通过，说明多模态检索链路本身是通的，后面才能继续测完整流程和 Judge。

### 2. run_mul_qwen 跑通

| 项目 | 命令 |
|---|---|
| Q1 完整流程 | `bash run_qwen.sh chongfan80 1 10 chongfan80_q1_final.mp4 5 Q1 0` |

核心要点：这一步会串完整的 `S1 -> S3 -> Qwen-S2 -> Qwen-S4 -> S5 -> S6`，主要验证中间文件和最终产物路径不会跟旧链路混掉。

| 检查点 | 预期产物 |
|---|---|
| Qwen 检索结果 JSON | `text_video_match/cache/chongfan80/1_10_20260423_155935_structured_qwen_q1.json` |
| Qwen final.json | `text_video_match/cache/chongfan80/chongfan80_final_qwen_q1.json` |
| Qwen 最终视频 | `finish/qwen_q1/chongfan80_q1_final.mp4` |

核心要点：这一步如果通过，说明 V6 Qwen 版已经和 `S5/S6` 接上了，而且产物路径按 `qwen_q1` 隔离存放。

### 3. Judge -- qwen 跑通

| 项目 | 命令 |
|---|---|
| Q1 Judge 前台测试 | `bash run_batch_judge_qwen.sh --dj chongfan80 --query-version Q1 --foreground` |

核心要点：这一步会自动找 `*_structured_qwen_q1.json`，并把结果存到版本化目录，不会再和 V4 Judge 结果混在一起。

| 检查点 | 预期产物 |
|---|---|
| Qwen Judge 输出根目录 | `judge/outputs/qwen_q1/chongfan80/` |
| 本次 Judge 批次目录 | `judge/outputs/qwen_q1/chongfan80/qwen_q1_时间戳/` |
| 对照基线目录 | `judge/outputs/v4/chongfan80/annotation_20260423_104822/` |

核心要点：这一步通过后，就可以开始做同一部剧 `V4 vs V6(Q1)` 的 Judge 报告对比。

### 当前建议的测试顺序

| 顺序 | 先做什么 | 原因 |
|---|---|---|
| 1 | 先跑“基本的多模态 emb 检索召回” | 先确认检索链本身通了 |
| 2 | 再跑 `run_qwen.sh` 完整流程 | 再确认 `S5/S6` 和路径隔离没问题 |
| 3 | 最后跑 `run_batch_judge_qwen.sh` | 这时再做 V4 vs V6 Judge 对比最稳 |

核心要点：现在最稳的顺序就是“先检索、再完整流程、最后 Judge”。这样出了问题最好定位。 
