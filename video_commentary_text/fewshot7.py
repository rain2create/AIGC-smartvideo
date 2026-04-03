#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
正向SOP生成器（对比实验用）
从黄金文案出发，严格按照SOP文档逐步生成SOP1-5
"""

import json
import logging
import argparse
import os
from pathlib import Path
from typing import List, Dict, Optional
import random

from model_client import ModelClient

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_rag_jsonl(rag_path: str) -> List[Dict]:
    """加载 rag.jsonl 文件"""
    entries = []
    with open(rag_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    logger.info(f"从 {rag_path} 加载了 {len(entries)} 条数据")
    return entries


def retrieve_by_label(entries: List[Dict], label: str, k: int = 2) -> List[Dict]:
    """根据 label 检索案例"""
    matched = [e for e in entries if e.get('label') == label]
    if not matched:
        logger.warning(f"未找到 label={label} 的案例")
        return []
    
    # 随机选择 k 个
    selected = random.sample(matched, min(k, len(matched)))
    logger.info(f"从 {len(matched)} 个 label={label} 案例中随机选择了 {len(selected)} 个")
    return selected


def load_sop_doc(doc_path: str) -> str:
    """加载 SOP 文档"""
    with open(doc_path, 'r', encoding='utf-8') as f:
        content = f.read()
    logger.info(f"加载 SOP 文档: {doc_path}, 长度: {len(content)} 字符")
    return content


def post_process_remove_analysis_tone(sop_text: str) -> str:
    """
    后处理：移除分析口吻的表述（保险机制）
    
    如果阶段2的 LLM 还是生成了"原版"、"原文"等字眼，用正则替换
    """
    import re
    
    original = sop_text
    
    # 替换规则
    replacements = [
        # "原版开头采用了" -> "采用了"
        (r'原版开头采用了', '采用了'),
        (r'原文开头采用了', '采用了'),
        (r'黄金文案的开头采用了', '采用了'),
        
        # "原版使用了" -> "使用了"
        (r'原版使用了', '使用了'),
        (r'原文使用了', '使用了'),
        
        # "原版的" -> "这个"
        (r'原版的', '这个'),
        (r'原文的', '这个'),
        
        # "理由：原版..." -> "理由：这个..."
        (r'（理由：原版', '（理由：这个'),
        (r'理由：原版', '理由：这个'),
        
        # "原版采用" -> "采用"
        (r'原版采用', '采用'),
        (r'原文采用', '采用'),
        
        # "根据原版" -> "根据设计"
        (r'根据原版', '根据设计'),
        (r'根据原文', '根据设计'),
        
        # "保持原版" -> "保持"
        (r'保持原版', '保持'),
        (r'保持原文', '保持'),
    ]
    
    for pattern, replacement in replacements:
        sop_text = re.sub(pattern, replacement, sop_text)
    
    # 检查是否有修改
    if sop_text != original:
        logger.info("  🔧 后处理：移除了部分分析口吻表述")
    
    return sop_text


def build_forward_sop_prompt(sop_doc: str, golden_content: str) -> str:
    """
    构建正向生成 SOP 的 Prompt
    
    核心思路：
    1. 给定黄金文案
    2. 严格按照 SOP 文档规范
    3. 逐步生成 SOP1 -> SOP2 -> SOP3 -> SOP4 -> SOP5
    """
    
    prompt = f"""你是一位专业的短视频解说文案分析专家。

# 任务说明

现在给你一篇**黄金解说文案**（已经过精心打磨的优质文案），以及一份完整的**SOP创作规范文档**。

你的任务是：**严格按照 SOP 文档的规范，逐步分析并生成该文案的 SOP1-5 创作过程**。

---

## SOP 创作规范文档

{sop_doc}

---

## 黄金解说文案（输入素材）

```
{golden_content}
```

---

## 输出要求

**【重要】**：只输出 SOP1-5 的具体内容，不要输出文档的头部（如"角色定义"、"主要任务"等）。

**【格式要求】**：
1. 直接从 `### SOP 1: 列出大纲` 开始输出
2. 每个 SOP 章节之间必须用空行分隔
3. 表格前后必须有空行
4. 每个段落之间必须有空行
5. SOP4 的每句文案后面必须换行

---

### SOP 1: 列出大纲（剧情分析阶段）

#### 核心剧情梳理（按时序+六要素）

| 时序 | 人 | 地 | 事 | 因 | 情 | 细/爆点标记 |
|------|-----|-----|-----|-----|-----|-------------|
| （按照文案内容填写，时序只写阶段名称如"开篇"、"发展"，不要加时间戳） | ... | ... | ... | ... | ... | ... |

#### 核心冲突标记
- （列出3-5个核心冲突点）

---

### SOP 2: 提取核心剧情和关键字

#### 关键字提炼表

**【重要逻辑】**：这个表格展示"删除冗余"的过程：从原始啰嗦版 → 提取关键字 → 黄金文案实际使用的精简版

| 原始大纲剧情（大白话） | 提取的核心关键字 | 最终凝练输出（文案架构基础） |
|----------------------|----------------|---------------------------|
| （构造一个带冗余修饰的版本，比最终文案更啰嗦） | （提取的核心关键字） | （黄金文案中实际使用的句子，从黄金文案中提取） |

**【关键要求】**：
1. **最终凝练输出**：必须是黄金文案中实际使用的原句，逐一对应黄金文案中的句子
2. **原始大纲剧情**：基于最终文案，反向构造一个更啰嗦、有冗余修饰的版本（如加"谁能想到"、"令人震惊的是"、"大家都知道"等冗余表达）
3. **核心关键字**：从原始剧情中提取的关键要素

**示例**：
- 原始：一个不起眼的中年男人，他表面上看起来就是个颓废的老狗，但谁能想到，他实际上却是个退休多年的顶级杀手。
- 关键字：颓废老狗，顶级杀手
- 最终：他表面上是个颓废的中年老狗，实际上是个退休多年的顶级杀手。（黄金文案原句）

---

### SOP 3: 构思爆款开头（虎头）

#### 开头设计（应用反差+夸张+细节）

**【重要】**：请先识别黄金文案的实际开头（第一句话），然后分析可能的4种设计方案，最后在"最终选择"中正确标注实际使用的版本。

**Version 1（技巧说明）**：
"（开头文案）"
（设计思路：...）

**Version 2（技巧说明）**：
"（开头文案）"
（设计思路：...）

**Version 3（技巧说明）**：
"（开头文案）"
（设计思路：...）

**Version 4（技巧说明）**：
"（开头文案）"
（设计思路：...）

**最终选择（必须与黄金文案的实际开头一致）**：
"（黄金文案的实际开头，必须完全一致）"
（理由：说明为什么这个开头最有效，具体采用了哪些技巧）

**【关键要求】**："最终选择"中的开头文案**必须**与黄金文案的实际第一句话完全一致，并正确标注是 Version 几。

---

### SOP 4: 串联核心剧情（豹身）

#### 主体文案（应用细节+钩子+解释+连接+递进）

第一句文案内容。（技巧标注：如"开头：...；反差：..."）

第二句文案内容。（技巧标注：如"连接：却；递进：..."）

第三句文案内容。（技巧标注：...）

...（继续按此格式，每句一行，句末标注技巧）

**重要**：文案内容必须与黄金文案完全一致，每句话单独一行，在句末括号内标注使用的技巧。

---

### SOP 5: 悬念结尾（凤尾）

#### 结尾设计（爆点一分为二+定格悬念）

（分析该文案的结尾设计思路，说明使用了哪些技巧，可在文案句子后标注）

---

---

## 重要提醒

1. **输出格式**：只输出 SOP1-5 的具体内容，从 `### SOP 1` 开始，不要输出文档头部
2. **SOP1**: 时序字段只写阶段名称（如"开篇"、"发展"、"高潮"），不要加时间戳
3. **SOP2**:
   - **【关键】"最终凝练输出"必须是黄金文案中实际使用的原句**，逐一从黄金文案中提取
   - "原始大纲剧情"是反向构造的啰嗦版（加冗余修饰词）
   - 逻辑：啰嗦版 → 提取关键字 → 实际文案（黄金文案原句）
4. **SOP3**: 
   - 必须设计4个不同版本的开头，每个版本应用不同技巧
   - **【关键】"最终选择"必须是黄金文案的实际第一句话，不能选择其他版本**
   - 先读取黄金文案的第一句，然后在Version 1-4中包含这句话，最后在"最终选择"中正确标注
5. **SOP4**: 文案内容必须与黄金文案**完全一致**，每句话单独一行，在句末括号内标注使用的技巧
6. **SOP5**: 输出结尾文案，可在句末标注使用的技巧
7. **格式**: 严格保持段落之间的空行，表格前后必须有空行
8. **严格遵循 SOP 文档规范**：每个步骤的分析必须符合文档中的方法论

---

请开始生成（直接从 `### SOP 1: 列出大纲` 开始输出，不要包含文档头部）：
"""
    
    return prompt


def generate_forward_sop(
    model_client: ModelClient,
    sop_doc: str,
    golden_content: str,
    max_retries: int = 2
) -> str:
    """
    调用 LLM 生成正向 SOP（两阶段法，避免"原版"字眼）
    
    阶段1：从黄金文案反推 SOP（允许分析口吻）
    阶段2：用阶段1的 SOP 重新生成说明（创作口吻，不给黄金文案）
    
    Args:
        model_client: 模型客户端
        sop_doc: SOP 文档内容
        golden_content: 黄金文案内容
        max_retries: 最大重试次数
        
    Returns:
        生成的 SOP Markdown 内容
    """
    
    # ========== 阶段1：逆向分析 ==========
    logger.info("🔍 阶段1：从黄金文案反推 SOP 结构...")
    
    prompt_stage1 = build_forward_sop_prompt(sop_doc, golden_content)
    
    raw_stage1 = None
    for attempt in range(max_retries + 1):
        try:
            logger.info(f"  正在生成 SOP 结构... (尝试 {attempt + 1}/{max_retries + 1})")
            
            raw_stage1, _ = model_client.call(
                system_prompt="你是一位专业的短视频解说文案分析专家，擅长拆解和分析爆款文案的创作流程。你必须严格遵循给定的 SOP 文档规范，确保每个步骤都符合文档的方法论和格式要求。",
                user_prompt=prompt_stage1,
                temperature=0.7,
                top_p=0.6
            )
            
            if raw_stage1:
                raw_stage1 = raw_stage1.replace('\\n', '\n')
                logger.info(f"  ✅ 阶段1 完成，SOP 结构长度: {len(raw_stage1)} 字符")
                break
            else:
                logger.warning(f"  尝试 {attempt + 1} 返回空响应")
                
        except Exception as e:
            logger.error(f"  尝试 {attempt + 1} 发生错误: {e}")
            if attempt < max_retries:
                logger.info(f"  等待 3 秒后重试...")
                import time
                time.sleep(3)
    
    if not raw_stage1:
        raise RuntimeError(f"阶段1 生成 SOP 结构失败，已重试 {max_retries} 次")
    
    # ========== 阶段2：正向表述重写 ==========
    logger.info("✨ 阶段2：用创作口吻重新表述 SOP...")
    
    prompt_stage2 = f"""你是一位专业的短视频解说文案创作者。现在给你一套已经分析好的 SOP 创作流程（但其中包含了一些"分析"的表述）。

# 任务说明

请你**用"创作"的口吻**重新表述这套 SOP，就像这是你自己从头设计的创作流程一样。

**关键要求**：必须严格遵循 SOP 文档规范，每个步骤都要符合文档中的方法论和格式要求。

---

## SOP 创作规范文档（必须严格遵循）

{sop_doc}

---

## 原始 SOP（带分析口吻，需要重写）

{raw_stage1}

---

## 重写要求

1. **口吻转换**：
   - ❌ 不要说"原版开头采用了..."
   - ❌ 不要说"原文使用了..."
   - ❌ 不要说"黄金文案的..."
   - ✅ 改为"我选择了..."
   - ✅ 改为"采用了..."
   - ✅ 改为"这个开头使用了..."

2. **特别关注 SOP2（关键字提炼）**：
   - 确保"最终凝练输出"列是黄金文案的原句，不要改动
   - "原始大纲剧情"列可以保持啰嗦版（带冗余修饰）
   - 逻辑必须是：啰嗦版 → 关键字 → 实际文案

3. **特别关注 SOP3（开头设计）**：
   - 在"最终选择"部分，不要提"原版"、"原文"
   - 用第一人称创作者的口吻："我选择 Version X，因为..."
   - 理由部分用"这个开头..."而不是"原版开头..."
   - **【关键】确保"最终选择"中的开头文案与实际文案的第一句话完全一致，不要改动内容**

4. **特别关注 SOP4 和 SOP5**：
   - 文案内容必须与黄金文案完全一致
   - 每句话单独一行
   - 在句末括号内标注使用的技巧（用于教学）

5. **严格遵循 SOP 文档规范**：
   - 每个 SOP 步骤的格式必须符合上述文档
   - 表格结构、字段名称要与文档一致
   - 方法论要符合文档的要求
   - SOP1-5 的具体内容、表格、数据都保持不变

6. **格式保持**：
   - 保持原有的 Markdown 格式
   - 保持段落、空行、表格结构
   - 每个 SOP 章节之间用空行分隔

7. **输出格式**：
   - 只输出 SOP1-5 的具体内容（从 `### SOP 1` 开始）
   - 不要输出文档头部（如"角色定义"、"主要任务"等）

---

请直接输出重写后的完整 SOP1-5（从 `### SOP 1: 列出大纲` 开始，Markdown 格式，严格遵循 SOP 文档规范）："""

    for attempt in range(max_retries + 1):
        try:
            logger.info(f"  正在重写 SOP 表述... (尝试 {attempt + 1}/{max_retries + 1})")
            
            raw_stage2, _ = model_client.call(
                system_prompt="你是一位专业的短视频解说文案创作者，用创作者的口吻表达。你必须严格遵循给定的 SOP 文档规范，确保重写后的内容仍然符合文档的所有要求。",
                user_prompt=prompt_stage2,
                temperature=0.5,  # 较低温度，保持内容一致性
                top_p=0.6
            )
            
            if raw_stage2:
                raw_stage2 = raw_stage2.replace('\\n', '\n')
                # 后处理：移除可能残留的分析口吻
                raw_stage2 = post_process_remove_analysis_tone(raw_stage2)
                logger.info(f"  ✅ 阶段2 完成，最终 SOP 长度: {len(raw_stage2)} 字符")
                return raw_stage2
            else:
                logger.warning(f"  尝试 {attempt + 1} 返回空响应")
                
        except Exception as e:
            logger.error(f"  尝试 {attempt + 1} 发生错误: {e}")
            if attempt < max_retries:
                logger.info(f"  等待 3 秒后重试...")
                import time
                time.sleep(3)
    
    # 如果阶段2失败，返回阶段1的结果（降级方案）
    logger.warning("⚠️  阶段2 失败，使用阶段1的结果并应用后处理")
    raw_stage1 = post_process_remove_analysis_tone(raw_stage1)
    return raw_stage1


def save_output(output_content: str, output_path: str) -> None:
    """保存输出到文件"""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(output_content)
    
    logger.info(f"输出已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='正向SOP生成器（对比实验）')
    parser.add_argument(
        '--label',
        type=str,
        default="温情叙事",
        help='标签名称，如"走心催泪"、"悬疑解谜"等'
    )
    
    parser.add_argument(
        '--rag-jsonl',
        type=str,
        default='rag-jsonl/rag1230.jsonl',
        help='rag.jsonl 文件路径'
    )
    
    parser.add_argument(
        '--sop-doc',
        type=str,
        default='prompts/skill_video_commentary_text_1230.md',
        help='SOP 文档路径'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='output2/forward_sop_{label}.md',
        help='输出文件路径（可使用 {label} 作为占位符）'
    )
    
    parser.add_argument(
        '--num-samples',
        type=int,
        default=2,
        help='从 rag.jsonl 中随机抽取的案例数量'
    )
    
    parser.add_argument(
        '--api-key',
        type=str,
        default=os.environ.get("API_KEY", ""),
        help='API Key'
    )
    
    args = parser.parse_args()
    
    # 初始化模型客户端
    model_client = ModelClient(api_key=args.api_key)
    
    # 1. 加载 rag.jsonl
    rag_entries = load_rag_jsonl(args.rag_jsonl)
    
    # 2. 根据 label 检索案例
    selected_cases = retrieve_by_label(rag_entries, args.label, k=args.num_samples)
    
    if not selected_cases:
        logger.error(f"未找到 label={args.label} 的案例，退出")
        return 1
    
    # 3. 加载 SOP 文档
    sop_doc = load_sop_doc(args.sop_doc)
    
    # 4. 为每个案例生成 SOP
    all_outputs = []
    
    for idx, case in enumerate(selected_cases, 1):
        case_id = case.get('id', f'case_{idx}')
        golden_content = case.get('content', '')
        
        logger.info(f"\n{'='*60}")
        logger.info(f"处理案例 {idx}/{len(selected_cases)}: {case_id}")
        logger.info(f"{'='*60}\n")
        
        try:
            # 生成 SOP
            sop_output = generate_forward_sop(
                model_client=model_client,
                sop_doc=sop_doc,
                golden_content=golden_content
            )
            
            # 构建完整输出
            case_output = f"""## 案例 {idx}（id={case_id} | label={args.label}）

{sop_output}

---

## 完整文案输出

**【黄金解说文案｜原文】**

{golden_content}

---

"""
            
            all_outputs.append(case_output)
            
        except Exception as e:
            logger.error(f"案例 {case_id} 生成失败: {e}")
            continue
    
    # 5. 合并所有输出
    if all_outputs:
        header = f"""# 正向SOP生成结果（对比实验）

> **说明**：本文档展示从黄金文案出发，严格按照 SOP 文档规范逐步生成的 SOP1-5 过程。
> 
> - **Label**: {args.label}
> - **生成方式**: 正向生成（Forward Generation）
> - **案例数量**: {len(all_outputs)}

---

"""
        
        final_output = header + '\n'.join(all_outputs)
        
        # 6. 保存输出
        output_path = args.output.format(label=args.label)
        save_output(final_output, output_path)
        
        logger.info(f"\n✅ 所有案例处理完成！共生成 {len(all_outputs)} 个案例")
        logger.info(f"📄 输出文件: {output_path}")
        
        return 0
    else:
        logger.error("所有案例生成失败")
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())

