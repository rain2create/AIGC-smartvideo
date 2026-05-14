# Canonical Judge Workflow

## 目标

Judge 只跑一次，固定一份“文案 x 目标集数内所有视频片段”的评分矩阵。

后续 `v4`、`qwen_q1`、`qwen_q2`、`qwen_q3` 只提供各自的 `best_match`，评估报告统一从同一份 canonical Judge 矩阵里查分数和排名。

这样可以避免每个 variant 重跑 Judge 带来的分数波动，也能节省评估成本。

## 1. 跑 canonical Judge

使用基础 `*_structured.json` 作为 Judge 输入，不绑定任何匹配方法。

```bash
cd /nlp_liuhuan_data03/search_public/xby/smart_videoV6

JUDGE_VARIANT=canonical \
MODEL=google/gemini-3-pro-preview \
bash run_batch_judge.sh task.txt
```

输出位置：

```text
judge/outputs/canonical/<dj>/<run_id>/
```

每个 run 里应有：

```text
results.jsonl
report.json
summary.md
config.json
```

如果中断，使用同一个 `run_id` 断点续跑：

```bash
python judge/batch_judge.py --dj chongfan80 --variant canonical --run-id <run_id> --model google/gemini-3-pro-preview
```

## 2. 用 canonical 结果比较多个 variant

当各 variant 的匹配结果已经生成后，直接跑：

```bash
cd /nlp_liuhuan_data03/search_public/xby/smart_videoV6

RUN_ID=compare_v4_vs_qwen_q1_vs_qwen_q2_vs_qwen_q3_$(date +%Y%m%d_%H%M%S)

python judge/compare_variants_with_canonical.py task.txt \
  --canonical-variant canonical \
  --variants v4,qwen_q1,qwen_q2,qwen_q3 \
  --out "judge/outputs/${RUN_ID}.md" \
  --json-out "judge/outputs/${RUN_ID}.json" \
  --detail-run-id "$RUN_ID"
```

如果不传 `--out` / `--json-out`，脚本也会默认使用类似
`compare_v4_vs_qwen_q1_vs_qwen_q2_vs_qwen_q3_20260508_183000` 的版本对比命名。

脚本默认会为每部剧自动选择最新完成的 canonical run。

总汇总报告放在：

```text
judge/outputs/<RUN_ID>.md
judge/outputs/<RUN_ID>.json
```

其中 `.md` 面向人工阅读，`.json` 是同一份报告的机器可读版本，保留了每个 variant、每部剧、每条文案段的完整指标，方便后续做二次统计、画图或自动抽取问题样本。

每个匹配版本、每部剧的详细效果报告放在：

```text
judge/outputs/<variant>/<dj>/<RUN_ID>/summary.md
judge/outputs/<variant>/<dj>/<RUN_ID>/summary.txt
judge/outputs/<variant>/<dj>/<RUN_ID>/report.json
judge/outputs/<variant>/<dj>/<RUN_ID>/config.json
```

## 3. 只比较已有版本

例如当前只想比较 `v4` 和 `qwen_q1`：

```bash
python judge/compare_variants_with_canonical.py task.txt \
  --variants v4,qwen_q1 \
  --out judge/outputs/compare_v4_vs_qwen_q1.md
```

## 4. 调试用：复用已有 V4 Judge 结果当 canonical

如果还没有真正跑 `canonical`，可以先用已有 `v4` Judge 输出测试脚本逻辑：

```bash
python judge/compare_variants_with_canonical.py task.txt \
  --canonical-variant v4 \
  --variants v4,qwen_q1 \
  --out /tmp/compare_v4_vs_qwen_q1_smoke.md
```

这只用于脚本验证。正式 benchmark 仍建议使用 `--canonical-variant canonical`。

## 5. 口径约束

适用前提：

- 文案拆分固定
- `episode_target` 固定
- 目标集数内视频片段固定

如果后续某个方案改变了文案拆分或目标集数，就不能直接复用同一份 canonical Judge 矩阵。

## 6. 推荐关注指标

优先看：

- Top1 / Top3 / Top5 / Top20
- 系统 best_match 平均排名
- 系统选中片段平均 Judge 分
- 平均距离 Judge 最优分差
- 系统选中片段 `>=4.0` / `>=4.5` 占比
- 明显错排样本

候选池整体均值只作为题目难度背景，不作为 variant 排序能力的主要指标。
