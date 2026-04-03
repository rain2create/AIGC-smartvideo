#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JSONL 转 JSON 脚本
将每行一个 JSON 对象的 JSONL 文件转换为标准的 JSON 数组格式
"""

import json
import argparse
import sys
from pathlib import Path


def jsonl_to_json(jsonl_file, json_file=None, indent=2):
    """
    将 JSONL 文件转换为 JSON 文件
    
    Args:
        jsonl_file: 输入的 JSONL 文件路径
        json_file: 输出的 JSON 文件路径，如果为 None 则自动生成
        indent: JSON 输出的缩进，默认为 2
    """
    jsonl_path = Path(jsonl_file)
    
    if not jsonl_path.exists():
        print(f"错误: 文件 '{jsonl_file}' 不存在", file=sys.stderr)
        sys.exit(1)
    
    # 如果没有指定输出文件，自动生成
    if json_file is None:
        json_path = jsonl_path.with_suffix('.json')
    else:
        json_path = Path(json_file)
    
    # 读取 JSONL 文件并转换为列表
    data = []
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # 跳过空行
                    continue
                try:
                    obj = json.loads(line)
                    data.append(obj)
                except json.JSONDecodeError as e:
                    print(f"警告: 第 {line_num} 行 JSON 解析失败: {e}", file=sys.stderr)
                    continue
        
        # 写入 JSON 文件
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
        
        print(f"成功转换: {len(data)} 条记录")
        print(f"输入文件: {jsonl_path}")
        print(f"输出文件: {json_path}")
        
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='将 JSONL 文件转换为 JSON 文件',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s input.jsonl                    # 输出到 input.json
  %(prog)s input.jsonl -o output.json     # 指定输出文件
  %(prog)s input.jsonl -o output.json -i 4 # 使用 4 空格缩进
        """
    )
    
    parser.add_argument(
        'input',
        help='输入的 JSONL 文件路径'
    )
    
    parser.add_argument(
        '-o', '--output',
        dest='output',
        help='输出的 JSON 文件路径（默认：输入文件名.json）'
    )
    
    parser.add_argument(
        '-i', '--indent',
        type=int,
        default=2,
        help='JSON 输出的缩进空格数（默认：2）'
    )
    
    args = parser.parse_args()
    
    jsonl_to_json(args.input, args.output, args.indent)


if __name__ == '__main__':
    main()

