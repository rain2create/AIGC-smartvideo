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
    r"^##\s*去幻觉后的解说文案\s*$",
    r"^##\s*去幻觉后的解说文案",
    r"^#+\s*去幻觉后的解说文案",
    r"去幻觉后的解说文案",
]

# 匹配 "## 完整文案输出" 区块下的 **【爆款短视频解说文案】**
COMPLETE_OUTPUT_PATTERN = r"^##\s*完整文案输出\s*$"
POPULAR_TAG_PATTERN = r"^\*{0,2}【爆款短视频解说文案】\*{0,2}$"


def _clean(line: str) -> str:
    """去掉括号备注、小标题、bold、italic"""
    text = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", line)          # **bold** / *italic*
    text = re.sub(r"（[^）]*）", "", text)                          # （...）
    text = re.sub(r"\([^)]*\)", "", text)                           # (...)
    text = re.sub(r"【[^】]*】", "", text)                          # 【...】
    text = re.sub(r"\[[^\]]*\]", "", text)                          # [...]
    return text.strip()


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
            print(f"[DEBUG] 找到【爆款短视频解说文案】在第 {i} 行")
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
    
    # 收集正文，直到遇到下一个 ## 标题或【...】标记
    body = []
    while i < n:
        stripped = lines[i].strip()
        # 遇到下一个 ## 标题停止
        if re.match(r"^##\s+", stripped):
            break
        # 遇到其他【...】标记停止（如【专业影视解说文案】）
        if re.match(r"^\*{0,2}【.*】\*{0,2}$", stripped):
            break
        # 跳过分隔线
        if stripped == "---":
            i += 1
            continue
        cleaned = _clean(lines[i])
        if cleaned:
            body.append(cleaned)
        i += 1
    
    if not body:
        raise ValueError("【爆款短视频解说文案】段落为空")
    
    return "\n".join(body) + "\n"


def extract_final_script(lines: list, start_idx: int) -> str:
    """
    从 "## 去幻觉后的解说文案" 区块中提取内容
    """
    i = start_idx + 1
    n = len(lines)
    
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
    
    # 收集正文
    body = []
    while i < n:
        stripped = lines[i].strip()
        # 遇到下一个 ## 标题就停止
        if re.match(r"^##\s+", stripped):
            break
        # 跳过分隔线
        if stripped == "---":
            i += 1
            continue
        cleaned = _clean(lines[i])
        if cleaned:
            body.append(cleaned)
        i += 1
    
    if not body:
        raise ValueError("去幻觉后的解说文案段落为空")
    
    return "\n".join(body) + "\n"


def extract(md_text: str) -> str:
    lines = md_text.splitlines()
    
    # 先搜索所有可能的标题行
    print(f"[DEBUG] 总行数: {len(lines)}")
    for i, line in enumerate(lines):
        if "去幻觉" in line or "解说文案" in line or "完整文案" in line:
            print(f"[DEBUG] 第{i}行: {repr(line)}")
    
    # ========== 第一步：优先匹配 "## 去幻觉后的解说文案" ==========
    final_start = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        for pattern in FINAL_SECTION_PATTERNS:
            if re.search(pattern, stripped):
                final_start = i
                print(f"[DEBUG] 匹配到'去幻觉后的解说文案'在第 {i} 行")
                break
        if final_start is not None:
            break
    
    # 如果找到去幻觉标题，尝试提取
    if final_start is not None:
        try:
            return extract_final_script(lines, final_start)
        except ValueError as e:
            print(f"[DEBUG] 去幻觉段落提取失败: {e}，尝试匹配爆款短视频解说文案")
    
    # ========== 第二步：回退匹配 "## 完整文案输出" + "【爆款短视频解说文案】" ==========
    for i, line in enumerate(lines):
        stripped = line.strip()
        if re.search(COMPLETE_OUTPUT_PATTERN, stripped):
            print(f"[DEBUG] 匹配到'## 完整文案输出'在第 {i} 行")
            try:
                return extract_popular_script(lines, i)
            except ValueError as e:
                print(f"[DEBUG] 从完整文案输出提取失败: {e}")
                continue
    
    raise ValueError("未找到 '## 去幻觉后的解说文案' 或 '## 完整文案输出' 标题")


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