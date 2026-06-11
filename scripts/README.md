# Shell 脚本说明

| 脚本 | 用途 |
|------|------|
| `run_qwen.sh` | V6 Pipeline：Qwen3-VL 检索+Reranker，支持单剧/批量/GPU队列管理 |
| `run_v4.sh` | V4 Pipeline：4阶段匹配+外挂embedding，支持单剧/批量 |
| `batch_judge.sh` | V4 变体批量 Judge 评估（Gemini模型） |
| `batch_judge_qwen.sh` | Qwen 变体 Judge 评估（自动定位 structured json） |

> `kill_*.sh` 是自动生成的进程终止脚本，已 .gitignore。
