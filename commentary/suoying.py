#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
解说文案 -> ASR 集数映射工具

使用大模型将解说文案的每一句话映射到对应的 ASR 集数。
"""

import os
import json
import re
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any


from .model_client import ModelClient

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# 辅助函数
# ============================================================================

def _strip_code_fences(s: str) -> str:
    """去除代码块标记"""
    s = (s or "").strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()


def _extract_first_json_object(s: str) -> Optional[str]:
    """从模型输出中提取第一个 JSON 对象"""
    if not s:
        return None
    s = _strip_code_fences(s)
    # 去掉思考标签
    s = re.sub(r"<think>[\s\S]*?</think>", "", s, flags=re.IGNORECASE)
    s = re.sub(r"<analysis>[\s\S]*?</analysis>", "", s, flags=re.IGNORECASE)
    s = s.strip()

    # 尝试整体解析
    try:
        json.loads(s)
        return s
    except Exception:
        pass

    # 括号平衡提取第一个 {...} 对象（处理嵌套 JSON）
    start = s.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return s[start:i + 1].strip()
    return None


# ============================================================================
# ASR 文件加载
# ============================================================================

def load_asr_files(dj: str, start: int, end: int, asr_base_path: str) -> Dict[int, Dict]:
    """
    加载指定集数范围的 ASR 文件
    
    Args:
        dj: 短剧名
        start: 开始集数
        end: 结束集数
        asr_base_path: ASR 文件基础路径（默认: /mnt_liuhuan/search_public/aigc/smart_video_v2/audio/videos）
    
    Returns:
        Dict[int, Dict]: {集数: ASR数据}
    """
    asr_data = {}
    # 修正：ASR文件在 {asr_base_path}/{dj}/{dj}_{episode}_asr.json
    base_path = Path(asr_base_path) / dj
    # 触发 autofs 挂载
    import subprocess, time
    for _ in range(3):
        subprocess.run(["ls", str(base_path)], capture_output=True)
        if base_path.exists():
            break
        time.sleep(1)
    
    for episode in range(start, end + 1):
        asr_file = base_path / f"{dj}_{episode}_asr.json"
        if asr_file.exists():
            try:
                with open(asr_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    episode_idx = int(data.get("episode_index", episode))
                    asr_data[episode_idx] = data
                    logger.info(f"✅ 加载第 {episode_idx} 集 ASR 数据: {asr_file}")
            except Exception as e:
                logger.warning(f"⚠️  加载 {asr_file} 失败: {e}")
        else:
            logger.warning(f"⚠️  未找到 ASR 文件: {asr_file}")
    
    return asr_data


def extract_episode_summary(asr_data: Dict[int, Dict], max_utterances: int = 20) -> Dict[int, str]:
    """
    从 ASR 数据中提取每集的摘要（用于给模型参考）
    
    Args:
        asr_data: ASR 数据字典
        max_utterances: 每集最多使用的 utterances 数量（避免太长）
    
    Returns:
        Dict[int, str]: {集数: 摘要文本}
    """
    episode_summaries = {}
    
    for episode, data in asr_data.items():
        utterances = data.get("utterances", [])
        # 取前 N 个 utterances 作为摘要
        selected = utterances[:max_utterances]
        # 拼接文本
        texts = [u.get("text", "") for u in selected]
        summary = " ".join(texts)
        episode_summaries[episode] = summary
        logger.info(f"第 {episode} 集摘要长度: {len(summary)} 字符")
    
    return episode_summaries


# ============================================================================
# 解说文案处理
# ============================================================================

def split_commentary_sentences(commentary: str) -> List[str]:
    """
    将解说文案按"。"分割成句子
    
    Args:
        commentary: 解说文案
    
    Returns:
        List[str]: 句子列表
    """
    # 按"。"、"！"、"？"分割
    sentences = re.split(r'[。！？]', commentary)
    # 过滤空字符串并去除首尾空格
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


# ============================================================================
# 大模型映射
# ============================================================================

EPISODE_MAPPING_SYSTEM_PROMPT = """你是【解说文案与ASR集数匹配器】。
你的任务：根据解说文案的每一句话，判断它最可能来自哪一集的ASR内容。

硬规则：
1) 只输出严格 JSON，不要 Markdown，不要多余文字
2) 每句话必须分配一个集数（如果无法确定，选择最可能的集数）
3) 输出格式：{"mappings": [{"sentence_idx": 1, "sentence": "句子内容", "episode": 9, "confidence": 0.8, "reason": "一句话理由"}]}
4) confidence 范围 0-1，表示匹配的置信度
5) reason 用一句话说明为什么匹配到这一集（例如：提到了"祖师爷"、"300年前"等关键词）
"""


def map_commentary_to_episodes_with_llm(
    model_client: ModelClient,
    commentary: str,
    asr_data: Dict[int, Dict],
    temperature: float = 0.1,
    max_utterances_per_episode: int = 20,
) -> Dict[int, List[Tuple[int, str, float, str]]]:
    """
    使用大模型将解说文案映射到 ASR 集数
    
    Args:
        model_client: 模型客户端
        commentary: 解说文案
        asr_data: ASR 数据字典 {集数: ASR数据}
        temperature: 采样温度
        max_utterances_per_episode: 每集最多使用的 utterances 数量
    
    Returns:
        Dict[int, List[Tuple[int, str, float, str]]]: 
            {集数: [(句子索引, 句子内容, 置信度, 理由), ...]}
    """
    # 1. 分割解说文案
    sentences = split_commentary_sentences(commentary)
    if not sentences:
        logger.warning("解说文案为空或无法分割")
        return {}
    
    logger.info(f"解说文案分割为 {len(sentences)} 句话")
    
    # 2. 提取每集的摘要
    episode_summaries = extract_episode_summary(asr_data, max_utterances=max_utterances_per_episode)
    
    if not episode_summaries:
        logger.warning("未找到任何 ASR 摘要")
        return {}
    
    # 3. 构建 user prompt
    episodes_info = []
    for episode in sorted(episode_summaries.keys()):
        summary = episode_summaries[episode]
        # 限制摘要长度，避免 prompt 过长
        summary_short = summary[:2000] if len(summary) > 2000 else summary
        episodes_info.append(f"第 {episode} 集：{summary_short}")
    
    sentences_text = "\n".join([f"{idx}. {s}" for idx, s in enumerate(sentences, 1)])
    
    user_prompt = f"""请将下面解说文案的每一句话匹配到对应的 ASR 集数。

【ASR 集数内容摘要】
{chr(10).join(episodes_info)}

【解说文案（按"。"分割）】
{sentences_text}

请为每句话分配最可能的集数，输出 JSON 格式：
{{
  "mappings": [
    {{
      "sentence_idx": 1,
      "sentence": "句子内容",
      "episode": 9,
      "confidence": 0.8,
      "reason": "一句话理由（说明为什么匹配到这一集）"
    }}
  ]
}}

注意：
- sentence_idx 从 1 开始，必须与上面列出的句子序号对应
- episode 必须是上面列出的集数之一
- confidence 表示匹配置信度（0-1），越接近 1 表示越确定
- reason 说明匹配依据（如关键词、情节、人物等）
"""
    
    # 4. 调用模型（最多 5 次重试）
    max_tries = 5
    mappings: List[Dict] = []
    for attempt in range(1, max_tries + 1):
        logger.info("调用大模型进行映射（第 %s/%s 次）...", attempt, max_tries)
        try:
            raw, _ = model_client.call(
                system_prompt=EPISODE_MAPPING_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=temperature,
                top_p=0.4,
            )
        except Exception as e:
            logger.warning("模型调用失败（尝试 %s/%s）: %s", attempt, max_tries, e)
            if attempt < max_tries:
                continue
            logger.error("模型调用失败，已重试 %s 次", max_tries)
            return {}

        # 5. 解析 + 校验结果
        js = _extract_first_json_object(raw or "")
        if not js:
            logger.warning("无法从模型输出中提取 JSON（尝试 %s/%s）", attempt, max_tries)
            if attempt < max_tries:
                continue
            return {}

        try:
            parsed = json.loads(js)
            mappings = parsed.get("mappings", [])
        except Exception as e:
            logger.warning("JSON 解析失败（尝试 %s/%s）: %s", attempt, max_tries, e)
            if attempt < max_tries:
                continue
            return {}

        if not mappings:
            logger.warning("模型返回的 mappings 为空（尝试 %s/%s）", attempt, max_tries)
            if attempt < max_tries:
                continue
            return {}

        # 校验每个 mapping 必需字段
        ok = True
        for m in mappings:
            if not isinstance(m, dict) or 'sentence_idx' not in m or 'episode' not in m:
                ok = False
                break
        if not ok:
            logger.warning("mapping 格式校验失败，缺少 sentence_idx/episode（尝试 %s/%s）", attempt, max_tries)
            mappings = []
            if attempt < max_tries:
                continue
            return {}

        break  # 成功
    
    # 6. 组织结果
    episode_mapping = {}
    for mapping in mappings:
        episode = mapping.get("episode")
        sentence_idx = mapping.get("sentence_idx")
        sentence = mapping.get("sentence", "")
        confidence = mapping.get("confidence", 0.0)
        reason = mapping.get("reason", "")
        
        if episode is not None and sentence_idx is not None:
            if episode not in episode_mapping:
                episode_mapping[episode] = []
            episode_mapping[episode].append((sentence_idx, sentence, confidence, reason))
    
    # 按句子索引排序
    for episode in episode_mapping:
        episode_mapping[episode].sort(key=lambda x: x[0])
    
    logger.info(f"✅ 映射完成，共 {len(mappings)} 句话，分布在 {len(episode_mapping)} 个集数中")
    
    return episode_mapping


# ============================================================================
# 输出格式化
# ============================================================================

def format_mapping_output(
    episode_mapping: Dict[int, List[Tuple[int, str, float, str]]],
    commentary: str
) -> str:
    """
    格式化输出映射结果
    
    Args:
        episode_mapping: 映射结果
        commentary: 原始解说文案
    
    Returns:
        str: 格式化的输出文本
    """
    output_parts = []
    output_parts.append("## 解说文案 -> ASR 集数映射（大模型匹配）\n\n")
    
    # 按集数排序
    sorted_episodes = sorted([e for e in episode_mapping.keys() if e is not None])
    
    for episode in sorted_episodes:
        sentences = episode_mapping[episode]
        output_parts.append(f"### 第 {episode} 集\n\n")
        
        for idx, sentence, confidence, reason in sentences:
            output_parts.append(f"- **句子 {idx}** (置信度: {confidence:.3f}): {sentence}。\n")
            output_parts.append(f"  - 匹配理由: {reason}\n")
        
        output_parts.append("\n")
    
    # 生成索引对应表
    output_parts.append("---\n\n")
    output_parts.append("## 索引对应表\n\n")
    output_parts.append("| 句子索引 | 句子内容 | ASR 集数 | 置信度 | 匹配理由 |\n")
    output_parts.append("|---------|---------|---------|--------|----------|\n")
    
    all_sentences = []
    for episode, sentences in episode_mapping.items():
        for idx, sentence, confidence, reason in sentences:
            all_sentences.append((idx, sentence, episode, confidence, reason))
    
    all_sentences.sort(key=lambda x: x[0])
    
    for idx, sentence, episode, confidence, reason in all_sentences:
        sentence_short = sentence[:40] + "..." if len(sentence) > 40 else sentence
        reason_short = reason[:30] + "..." if len(reason) > 30 else reason
        output_parts.append(f"| {idx} | {sentence_short} | {episode} | {confidence:.3f} | {reason_short} |\n")
    
    return "".join(output_parts)


# ============================================================================
# 主函数
# ============================================================================

def map_commentary_to_episodes(
    model_client: ModelClient,
    commentary: str,
    dj: str,
    start: int,
    end: int,
    asr_base_path: str = "/home/humiao/image/ad-materials-aigc/",
    temperature: float = 0.1,
    max_utterances_per_episode: int = 20,
) -> Dict:
    """
    主函数：将解说文案映射到 ASR 集数（使用大模型）
    
    Args:
        model_client: 模型客户端
        commentary: 解说文案
        dj: 短剧名
        start: 开始集数
        end: 结束集数
        asr_base_path: ASR 文件基础路径
        temperature: 采样温度
        max_utterances_per_episode: 每集最多使用的 utterances 数量
    
    Returns:
        Dict: 包含映射结果的字典
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"开始映射解说文案到 ASR 集数")
    logger.info(f"短剧: {dj}, 集数范围: {start}-{end}")
    logger.info(f"{'='*60}\n")
    
    # 1. 加载 ASR 数据
    logger.info("步骤 1: 加载 ASR 数据...")
    asr_data = load_asr_files(dj, start, end, asr_base_path)
    if not asr_data:
        error_msg = f"未找到 {dj} 的 ASR 数据（集数 {start}-{end}）"
        logger.error(error_msg)
        return {"error": "no_asr_data", "message": error_msg}
    
    logger.info(f"✅ 成功加载 {len(asr_data)} 集的 ASR 数据\n")
    
    # 2. 使用大模型进行映射
    logger.info("步骤 2: 使用大模型进行映射...")
    episode_mapping = map_commentary_to_episodes_with_llm(
        model_client=model_client,
        commentary=commentary,
        asr_data=asr_data,
        temperature=temperature,
        max_utterances_per_episode=max_utterances_per_episode,
    )
    
    if not episode_mapping:
        error_msg = "映射失败，未得到任何结果"
        logger.error(error_msg)
        return {"error": "mapping_failed", "message": error_msg}
    
    # 3. 格式化输出
    formatted_output = format_mapping_output(episode_mapping, commentary)
    
    # 4. 返回结果
    result = {
        "commentary": commentary,
        "episode_range": {"start": start, "end": end},
        "dj": dj,
        "mapping": {
            str(episode): [
                {
                    "sentence_idx": idx,
                    "sentence": sentence,
                    "confidence": confidence,
                    "reason": reason
                }
                for idx, sentence, confidence, reason in sentences
            ]
            for episode, sentences in episode_mapping.items()
        },
        "formatted_output": formatted_output,
    }
    
    logger.info(f"\n✅ 映射完成！\n")
    
    return result


# ============================================================================
# 命令行接口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='解说文案 -> ASR 集数映射工具（使用大模型）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法：
  python suoying.py \\
    --commentary "男人自称是祖师徒弟。三百年前的合照惊现。" \\
    --dj buxi5 \\
    --start 9 \\
    --end 9 \\
    --asr-base-path /home/humiao/image/ad-materials-aigc/
        """
    )
    
    # 输入参数
    parser.add_argument(
        '--commentary',
        type=str,
        default="豪门掌权人准备了四千亿来赎罪，可家里的二叔却甩出一千万，把那位祖宗当叫花子打发了。事情还得从几天前说起，昆仑山施工队意外挖出了一个沉睡三千年的怪人。这老头名叫陈北玄，被好心的曾孙陈小凡收留后，竟掏出一张泛黄的房契。他信誓旦旦地说，这是当年末代皇帝溥仪送给他的四合院。可旁边的校花拿过来一看，瞬间无语，这哪里是四合院，分明是如今的亲王府公园。眼看发财梦碎，老头又不死心，带着曾孙来到了汇丰银行取古董。没想到冤家路窄，竟然撞上了当地首富家的恶少宋世杰。这恶少仗着家里有钱，不仅当众羞辱校花收了他家三千万聘礼，更指挥手下要打断陈小凡的腿。就在这千钧一发之际，银行高层二叔匆匆赶到。本以为是来主持公道的，谁知这二叔反手就是一巴掌打在恶少脸上，接着却掏出一张支票。他轻蔑地把一千万甩在老头身上，像打发乞丐一样让他们滚蛋。老头没说什么，拿着钱转身就走。可二叔做梦都想不到，他这一举动，直接捅破了天。当家里的老爷子宋全福得知恩主被羞辱后，气得差点当场暴毙。老爷子咆哮着下令，立刻变卖家族一半资产，凑齐四千亿，只为求老祖宗原谅。而另一边，陈北玄住进了校花的别墅，闲来无事在公园打起了太极。这一练不要紧，周围围观的退休老头们全都吓傻了。这些老头可不简单，不是退休的封疆大吏，就是退隐的武学宗师。他们惊恐地发现，这老头练的竟是失传三百年的太极母拳。一张照片传到京城，整个高层彻底震动。经过比对，这老头竟然是三百年前太极祖师的师父，更是活了无数岁月的长生者。国家最高部门立刻下令，封锁一切消息，不惜一切代价接触这位活神仙。可就在各方大佬疯狂赶来朝圣的时候，那个不知死活的恶少宋世杰，竟然带着打手闯进了别墅。他狞笑着逼向校花，扬言要当着陈小凡的面生米煮成熟饭。陈小凡拼死阻拦，却被恶少的手下打得吐血倒地。看着重伤的曾孙，一直沉默的陈北玄，终于怒了。他缓缓抬起头，眼神中闪过一丝凌厉的杀气。恶少请来的八卦掌宗师还想逞凶，嘲笑老头拳头太软。可下一秒，陈北玄只用了一招。仅仅一招，那位不可一世的宗师便筋断骨折，像死狗一样飞了出去。恶少彻底傻眼了，可他不知道，真正的绝望才刚刚开始。此时此刻，天空之上，军部的战斗机正在清空航道。地面之上，宋家老爷子带着四千亿资产正在疯狂飙车。更有无数退隐的大佬正朝着这里极速集结。当这位活了三千年的祖宗彻底展露身份时，整个世界都将为之颤抖。",
        help='解说文案（必填）'
    )
    parser.add_argument(
        '--commentary-file',
        type=str,
        default=None,
        help='解说文案文件路径（如果提供，将覆盖 --commentary）'
    )
    parser.add_argument(
        '--dj',
        type=str,
        default="buxi5",
        help='短剧名（必填）'
    )
    parser.add_argument(
        '--start',
        type=int,
        default=1,
        help='开始集数（必填）'
    )
    parser.add_argument(
        '--end',
        type=int,
        default=15,
        help='结束集数（必填）'
    )
    parser.add_argument(
        '--asr-base-path',
        type=str,
        default='/mnt_liuhuan/search_public/aigc/smart_video_v2/audio/videos',
        help='ASR 文件基础路径'
    )
    
    # 模型参数
    parser.add_argument(
        '--api-url',
        type=str,
        default='https://api.360.cn/v1/chat/completions',
        help='API 地址'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        default='fk1379989049.aEmK59LGEcoiwxat0pGE18JPvtSBkrybaf481994',
        help='API Key'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='google/gemini-3-pro-preview',
        help='模型名称'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.1,
        help='采样温度（默认 0.1，越低越确定）'
    )
    
    # 其他参数
    parser.add_argument(
        '--max-utterances',
        type=int,
        default=20,
        help='每集最多使用的 utterances 数量（默认 20，避免 prompt 过长）'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./mapping_results',
        help='输出目录（默认 ./mapping_results）'
    )
    parser.add_argument(
        '--output-format',
        type=str,
        choices=['json', 'markdown', 'both'],
        default='both',
        help='输出格式（默认 both）'
    )
    
    args = parser.parse_args()
    
    # 读取解说文案
    if args.commentary_file:
        try:
            with open(args.commentary_file, 'r', encoding='utf-8') as f:
                commentary = f.read()
            logger.info(f"从文件读取解说文案: {args.commentary_file}")
        except Exception as e:
            logger.error(f"读取解说文案文件失败: {e}")
            return 1
    else:
        commentary = args.commentary
    
    if not commentary or not commentary.strip():
        logger.error("解说文案为空")
        return 1
    
    logger.info(f"解说文案长度: {len(commentary)} 字符")
    
    # 初始化模型客户端
    logger.info("初始化模型客户端...")
    try:
        model_client = ModelClient(
            api_url=args.api_url,
            api_key=args.api_key,
            model=args.model,
            timeout=300
        )
    except Exception as e:
        logger.error(f"初始化模型客户端失败: {e}")
        return 1
    
    # 执行映射
    try:
        result = map_commentary_to_episodes(
            model_client=model_client,
            commentary=commentary,
            dj=args.dj,
            start=args.start,
            end=args.end,
            asr_base_path=args.asr_base_path,
            temperature=args.temperature,
            max_utterances_per_episode=args.max_utterances,
        )
        
        if "error" in result:
            logger.error(f"映射失败: {result.get('message')}")
            return 1
        
        # 保存结果
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{args.dj}_{args.start}_{args.end}_{timestamp}"
        
        if args.output_format in ['json', 'both']:
            json_path = output_dir / f"{base_filename}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            logger.info(f"✅ JSON 结果已保存: {json_path}")
        
        if args.output_format in ['markdown', 'both']:
            md_path = output_dir / f"{base_filename}.md"
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(result["formatted_output"])
            logger.info(f"✅ Markdown 结果已保存: {md_path}")
        
        # 打印摘要
        logger.info(f"""
{'='*60}
🎉 映射完成！
{'='*60}

结果摘要:
  - 短剧: {args.dj}
  - 集数范围: {args.start}-{args.end}
  - 解说文案: {len(commentary)} 字符
  - 映射到的集数: {', '.join(map(str, sorted([int(k) for k in result['mapping'].keys()])))}
  - 输出目录: {output_dir}
        """)
        
        return 0
        
    except Exception as e:
        logger.error(f"执行失败: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())

