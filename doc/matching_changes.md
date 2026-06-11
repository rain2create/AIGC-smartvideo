# 全量文案匹配 — 改动记录

> 目的：Phase X 跑 100 部新剧时，不再限制前 20s 文案，所有文案段都走匹配 + Judge，最大化训练数据量。

## 改动点

### 1. S4 检索：`text_video_match/qwen_match/retriever.py:358`

```
# 原代码
target_items = self._select_items_within_20_seconds(script_items)

# 改为
# target_items = self._select_items_within_20_seconds(script_items)
target_items = list(script_items)
```

影响：S4 匹配阶段对所有文案段做检索，不再按 audio_duration 累加到 20s 截断。

### 2. Judge 评估：`judge/batch_judge.py:807`

```
# 原代码
segments = _select_segments_within_20_seconds(load_structured_segments(structured_json))

# 改为
# segments = _select_segments_within_20_seconds(load_structured_segments(structured_json))
segments = load_structured_segments(structured_json)
```

影响：Judge 评估所有文案段，不再截断前 20s。

### 3. Judge 报告：`judge/batch_judge.py:385`

同上逻辑，报告统计也覆盖全量段。

## 不做改动的

| 位置 | 说明 |
|------|------|
| S3 文案结构化 | 从未限制，一直全量 |
| S5 视频合成 (`skill_postprocess.py:44`) | 保持 20s 截断，只影响最终 mp4 输出，不影响训练数据 |
| `_select_items_within_20_seconds` 函数本身 | 保留不删（`#` 注释），后续如需恢复可反注 |

## 效果

```
之前（前 20s 截断）: 每剧约 4~6 段文案参与匹配和 Judge
之后（全量）:       每剧全部文案段（通常 15~40 段）

预估：100 部新剧 + 全量 = 数千条 query → 数万条训练样本
```
