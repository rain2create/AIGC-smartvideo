# video_commentary_text Workflow

## 1. 输入
- 参数：`dj`、`start`、`end`
- 数据：
  - 多集 ASR 文本（`audio/videos/<dj>/<dj>_<ep>_asr.json`）
  - 方法论文档（`prompts/skill_video_commentary_text_formatted.md`）

## 2. 生成前处理（text0204.py）
- 获取短剧简介（可能为空）
- 加载 ASR 文本并拼接
- 高级打标（主标签/次标签）
- 生成 `PlotGraph`（剧情事实图）
- 生成 `BeatSheet`（节奏约束）
- 读取并增强 system prompt，保存：
  - `<ts>_enhanced_system_prompt.md`

## 3. 文案生成（无评估模块模式）
- 调用模型生成 `raw_response`
- 先做固定格式校验（硬约束）：
  - 必须按顺序包含：`## SOP 1` → `## SOP 2` → `## SOP 3` → `## SOP 4` → `## SOP 5` → `## 最终解说文案`
- 再做正文可用性检测：
  - 若命中 SOP/模板体/编号提纲等无效内容，判定失败并重试
- 重试上限 3 次：
  - 仍不满足格式或正文要求，直接 `RuntimeError` 退出流程（fail-fast，不继续下游）
- 产出一个合格 `raw_response`

## 4. 去幻觉检查与修补
- 从 `raw_response` 尝试提取“最终解说文案”段
- 若提取不到，直接失败退出（不再回退到完整 `raw_response`）
- 基于 `PlotGraph` 检查冲突（事实矛盾/过度确定）
- 若有冲突，执行定点修补，得到“去幻觉后的解说文案”

## 5. 结果落盘
- `<ts>.json`
  - 包含 `raw_response`、`conversation_history`、冲突检查结果等
- `<ts>.md`
  - 结构通常为：
    - `## 原始生成的解说文案`
    - `## 冲突检查报告`
    - `## 去幻觉后的解说文案`
- 由 `extract_final_script_to_text.py` 提取为 `<ts>.txt`，供下游匹配/TTS使用

## 6. 提取脚本规则（extract_final_script_to_text.py）
- 优先提取 `## 去幻觉后的解说文案`
- 回退提取 `## 完整文案输出` + `【爆款短视频解说文案】`
- 再回退 `## 原始生成的解说文案`
- 过滤表格、SOP标题、方法说明、编号提纲等噪声行

## 7. 当前策略说明
- 主 system prompt 已增加固定输出结构硬约束
- 无评估分支的 retry prompt 已与固定结构对齐
- `extract_final_commentary` 已支持 `## 最终解说文案` 标题提取
- 目标：减少“SOP教程体”并稳定产出“最终解说文案”段

## 8. 本次更新（2026-04-22）
- 去掉“提取不到最终解说文案时回退完整 `raw_response`”的策略
- 改为：提取不到即失败退出（fail-fast），避免 SOP/流程文本污染去幻与下游
