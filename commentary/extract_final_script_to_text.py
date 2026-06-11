#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从影视解说 md 中提取「去幻觉后的解说文案」或「爆款短视频解说文案」段落 → 纯文本
用法：
  python extract_final_script_to_text.py path/to/file.md
  python extract_final_script_to_text.py path/to/file.md --out path/to/file.txt
"""
import argparse
import re
from pathlib import Path


# 匹配 "## 去幻觉后的解说文案" 标题 - 多种变体
FINAL_SECTION_PATTERNS = [
    r"^##\s*最终解说文案\s*$",
    r"^##\s*最终解说文案",
    r"^#+\s*最终解说文案",
    r"最终解说文案",
    r"^##\s*去幻觉后的解说文案\s*$",
    r"^##\s*去幻觉后的解说文案",
    r"^#+\s*去幻觉后的解说文案",
    r"去幻觉后的解说文案",
]

# 匹配 "## 完整文案输出" 区块下的 **【爆款短视频解说文案】**
COMPLETE_OUTPUT_PATTERN = r"^##\s*完整文案输出\s*$"
POPULAR_TAG_PATTERN = r"^\*{0,2}【爆款短视频解说文案】\*{0,2}$"
FULL_SCRIPT_TAG_PATTERN = r"^\*{0,2}【完整解说文案】\*{0,2}$"

NOISE_LINE_PATTERNS = [
    r"^\|.*\|$",                           # markdown table rows
    r"^[-:| ]+$",                         # separators
    r"^\*{0,2}【SOP\s*\d+.*】\*{0,2}$",     # SOP tags
    r"^#+\s*SOP\s*\d+",
    r"^SOP\s*\d+",
    r"^\*{0,2}开头钩子\*{0,2}$",
    r"^\*{0,2}铺垫推进\*{0,2}$",
    r"^\*{0,2}冲突升级\*{0,2}$",
    r"^\*{0,2}反转/反差\*{0,2}$",
    r"^\*{0,2}高潮爆点\*{0,2}$",
    r"^\*{0,2}结尾悬念\*{0,2}$",
    r"^##\s*冲突检查报告",
    r"^##\s*说明",
    r"^##\s*原始生成的解说文案",
    r"^###\s*核心剧情梳理",
    r"^###\s*核心冲突标记",
    r"^###\s*关键字提炼表",
    r"^###\s*开头设计",
    r"^###\s*主体文案",
    r"^###\s*结尾设计",
    r"^主冲突\s*[:：]",
    r"^关键转折\s*[:：]",
    r"^情感线\s*[:：]",
    r"^最终选择\s*[:：]?",
    r"^检查结果\s*[:：]",
    r"^解说文案与 PlotGraph",
    r"^请注意，以上内容",
    r"^如果需要更准确",
    r"^- 已检查",
    r"^- 解说文案与 PlotGraph",
    r"^- 允许解说中有 PlotGraph",
]

META_LINE_PATTERNS = [
    r"^根据提供的.*(指令|信息).*(逐步处理|生成)",
    r"^首先，我们需要",
    r"^以下是根据.*整理的大纲",
    r"^接下来，我们将",
    r"^这些关键字包括",
    r"^一个好的开头",
    r"^在正文部分，我们需要",
    r"^最后，我们需要设计",
    r"^整体输出风格控制参考",
    r"^参考案例",
]


def _clean(line: str) -> str:
    """去掉括号备注、小标题、bold、italic"""
    text = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", line)          # **bold** / *italic*
    text = re.sub(r"（[^）]*）", "", text)                          # （...）
    text = re.sub(r"\([^)]*\)", "", text)                           # (...)
    text = re.sub(r"【[^】]*】", "", text)                          # 【...】
    text = re.sub(r"\[[^\]]*\]", "", text)                          # [...]
    return text.strip()


def _is_noise_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    for pattern in NOISE_LINE_PATTERNS:
        if re.search(pattern, stripped, re.IGNORECASE):
            return True
    return False


def _is_meta_line(line: str) -> bool:
    stripped = line.strip()
    for pattern in META_LINE_PATTERNS:
        if re.search(pattern, stripped, re.IGNORECASE):
            return True
    # 列表编号开头常见于 SOP/提纲，默认过滤
    if re.match(r"^\d+\s*[.、]\s*", stripped):
        return True
    return False


def _collect_body(lines: list, start_idx: int) -> str:
    """收集正文，自动跳过表格、SOP、小标题和说明性噪音。"""
    body = []
    i = start_idx
    n = len(lines)

    while i < n:
        stripped = lines[i].strip()
        if re.match(r"^##\s+", stripped):
            break
        if re.match(r"^\*{0,2}【.*】\*{0,2}$", stripped):
            break
        if _is_noise_line(stripped) or _is_meta_line(stripped):
            i += 1
            continue

        cleaned = _clean(lines[i].lstrip("- ").strip())
        if cleaned and not _is_noise_line(cleaned) and not _is_meta_line(cleaned):
            body.append(cleaned)
        i += 1

    if not body:
        raise ValueError("正文段落为空")

    return "\n".join(body) + "\n"


def _extract_by_inner_tag(lines: list, start_idx: int, tag_pattern: str) -> str:
    """在某个大区块内继续找更具体的正文标记。"""
    i = start_idx + 1
    n = len(lines)
    while i < n:
        stripped = lines[i].strip()
        if re.match(tag_pattern, stripped):
            return _collect_body(lines, i + 1)
        if re.match(r"^##\s+", stripped):
            break
        i += 1
    raise ValueError(f"未找到内部标记: {tag_pattern}")


def extract_popular_script(lines: list, start_idx: int) -> str:
    """
    从 "## 完整文案输出" 区块中提取 **【爆款短视频解说文案】** 下的内容
    """
    i = start_idx + 1
    n = len(lines)
    
    # 找到 **【爆款短视频解说文案】** 标记
    popular_start = None
    while i < n:
        stripped = lines[i].strip()
        if re.match(POPULAR_TAG_PATTERN, stripped):
            popular_start = i
            break
        # 如果遇到其他 ## 标题，说明没找到
        if re.match(r"^##\s+", stripped):
            break
        i += 1
    
    if popular_start is None:
        raise ValueError("未找到【爆款短视频解说文案】标记")
    
    # 跳过标记行和后面的空行/分隔符
    i = popular_start + 1
    while i < n:
        stripped = lines[i].strip()
        if stripped == "" or stripped == "---":
            i += 1
            continue
        # 如果遇到其他标记如【...】，跳过
        if re.match(r"^\*{0,2}【.*】\*{0,2}$", stripped):
            i += 1
            continue
        break
    
    return _collect_body(lines, i)


def _extract_fallback_from_full_text(lines: list) -> str:
    """无标题兜底：从全文中尽可能提取可直接配音的正文。"""
    body = []
    for line in lines:
        stripped = line.strip()
        if _is_noise_line(stripped) or _is_meta_line(stripped):
            continue
        cleaned = _clean(stripped.lstrip("- ").strip())
        if not cleaned:
            continue
        if _is_noise_line(cleaned) or _is_meta_line(cleaned):
            continue
        # 需要包含中文并且不是纯符号片段
        if not re.search(r"[\u4e00-\u9fff]", cleaned):
            continue
        body.append(cleaned)
    if not body:
        raise ValueError("兜底提取失败：未找到可用正文")
    return "\n".join(body) + "\n"


def extract_final_script(lines: list, start_idx: int) -> str:
    """
    从 "## 去幻觉后的解说文案" 区块中提取内容
    """
    i = start_idx + 1
    n = len(lines)
    
    # 优先在区块内部找更明确的正文标记
    for inner_pattern in [FULL_SCRIPT_TAG_PATTERN, POPULAR_TAG_PATTERN]:
        try:
            return _extract_by_inner_tag(lines, start_idx, inner_pattern)
        except ValueError:
            pass

    # 跳过标题行本身，以及后面的空行/分隔符/标记行
    while i < n:
        stripped = lines[i].strip()
        # 跳过空行
        if stripped == "":
            i += 1
            continue
        # 跳过 --- 分隔线
        if stripped == "---":
            i += 1
            continue
        # 遇到其他 ## 标题，说明内容为空（但跳过【xxx】标记）
        if re.match(r"^##\s+", stripped) and not re.match(r"^\*{0,2}【.*】\*{0,2}$", stripped):
            raise ValueError("去幻觉后的解说文案段落为空（遇到下一个标题）")
        # 如果遇到 【xxx】 标记，也跳过
        if re.match(r"^\*{0,2}【.*】\*{0,2}$", stripped):
            i += 1
            continue
        # 否则就是正文开始
        break
    
    return _collect_body(lines, i)


def extract(md_text: str) -> str:
    lines = md_text.splitlines()
    
    # 先搜索所有可能的标题行
    # ========== 第一步：优先匹配 "## 去幻觉后的解说文案" ==========
    final_start = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        for pattern in FINAL_SECTION_PATTERNS:
            if re.search(pattern, stripped):
                final_start = i
                break
        if final_start is not None:
            break
    
    # 如果找到去幻觉标题，尝试提取
    if final_start is not None:
        try:
            return extract_final_script(lines, final_start)
        except ValueError as e:
            pass
    
    # ========== 第二步：回退匹配 "## 完整文案输出" + "【爆款短视频解说文案】" ==========
    for i, line in enumerate(lines):
        stripped = line.strip()
        if re.search(COMPLETE_OUTPUT_PATTERN, stripped):
            try:
                return extract_popular_script(lines, i)
            except ValueError as e:
                continue
    
    # ========== 第三步：fallback - 用 "## 原始生成的解说文案" ==========
    ORIGINAL_PATTERN = r"^##\s*原始生成的解说文案\s*$"
    for i, line in enumerate(lines):
        stripped = line.strip()
        if re.search(ORIGINAL_PATTERN, stripped):
            try:
                return extract_final_script(lines, i)
            except ValueError as e:
                break

    # ========== 第四步：无标题兜底 ==========
    return _extract_fallback_from_full_text(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="从 md 提取纯文本解说文案（优先去幻觉，其次爆款短视频）")
    ap.add_argument("md", help="markdown 文件")
    ap.add_argument("--out", help="输出 txt（默认同名替换后缀）")
    args = ap.parse_args()

    md_path = Path(args.md)
    out_path = Path(args.out) if args.out else md_path.with_suffix(".txt")

    try:
        out_path.write_text(extract(md_path.read_text(encoding="utf-8")), encoding="utf-8")
        print(f"[OK] {md_path} -> {out_path}")
        return 0
    except Exception as e:
        print(f"[FAIL] {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
