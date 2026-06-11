# 任务文件说明

## 格式

### Judge 任务格式
每行一个剧名（纯文本）：
```
chongfan80
buxi5
luxiansheng
```

### 推理/Rerank 任务格式
每行 CSV（7-10 列）：
```
dj,start,end,output,top_k,query_version,max_segments,video_index_version,rerank_version,extra_args,checkpoint_path
```
各列含义见 `run_mul_qwen.py` 的 batch 模式文档。

## 目录

```
tasks/
├── canonical_judge/   # 100剧 Canonical Judge 任务
│   ├── all.txt        # 全量100部
│   └── g1.txt~g5.txt  # 分5组，每组~20部
└── inference/         # 各版本推理任务
    ├── val6.txt       # 6部验证剧
    ├── sft375_*.txt   # SFT 375步 checkpoint 推理
    ├── dpo2000_*.txt  # DPO 2000步 checkpoint 推理
    ├── dpov2_*.txt    # DPO v2 checkpoint 推理
    ├── awrv2_*.txt    # AWR v2 checkpoint 推理
    └── early_rerank/  # 早期实验任务（V1/V2/V3/V4）
```
