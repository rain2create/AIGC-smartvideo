#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
短剧解说文案质量评估技能
基于 BaseSkill 实现，用于评估爆款短视频解说文案的质量
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from base_skill import BaseSkill
from model_client import ModelClient


class SkillVideoCommentaryQualityEval(BaseSkill):
    """
    短剧解说文案质量评估技能类
    
    输入格式：
        input_data = {
            "commentary_text": "待评估的解说文案文本"
        }
        或者直接传入字符串：
        input_data = "待评估的解说文案文本"
    """
    
    def __init__(
        self,
        api_url: str = "https://api.360.cn/v1/chat/completions",
        api_key: str = "fk1379989049.aEmK59LGEcoiwxat0pGE18JPvtSBkrybaf481994",
        model: str = "google/gemini-3-pro-preview",
        # model: str = "openai/gpt-5.2",
        # model: str = "xai/grok-4",
        # model: str = "anthropic/claude-sonnet-4-5",
        timeout: int = 300
    ):
        """
        初始化技能类
        
        Args:
            api_url: API 地址
            api_key: API 密钥
            model: 模型名称
            timeout: 请求超时时间（秒）
        """
        # 初始化模型客户端
        self.model_client = ModelClient(
            api_url=api_url,
            api_key=api_key,
            model=model,
            timeout=timeout
        )
        
        # 获取当前文件所在目录
        current_dir = Path(__file__).parent
        # 构建 markdown 文件路径
        self.generation_methodology_path = current_dir / "prompts" / "skill_video_commentary_text_formatted.md"
        self.eval_methodology_path = current_dir / "prompts" / "skill_video_commentary_quality_eval.md"
        # 构建 results 目录路径（相对于 video_commentary_text 目录）
        self.results_base_dir = current_dir / "results"
        
        # 设置日志
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        self.logger.info(f"初始化 SkillVideoCommentaryQualityEval: api_url={api_url}, model={model}")
    
    def _get_description(self) -> str:
        """获取 Skill 描述"""
        return "短剧解说文案质量评估技能：根据爆款文案评估方法论，对解说文案进行质量评估"
    
    def get_role_definition(self) -> str:
        """定义角色"""
        return "爆款文案解说质量评估员"
    
    def get_principles(self) -> str:
        """总原则"""
        return "理解文案生产的方法论，以及文案质量评估的方法论，对一段文案进行质量评估"
    
    def get_output_style(self) -> str:
        """输出风格"""
        return "严格按照爆款文案评估方法论进行评估，输出结构化的质量评估报告"
    
    def get_sop(self) -> str:
        """标准操作流程说明"""
        return """
        文案质量评估遵循以下流程：
        1. 理解爆款文案生成方法论（学习文案的写作手法，辅助进行质量评估）
        2. 理解爆款文案评估方法论（评估维度和标准，是主要依赖的方法论）
        3. 对输入的解说文案进行质量评估
        4. 输出结构化的评估报告（包括总分、分项评分、问题清单、优化建议等）
        """
    
    def get_notes(self) -> str:
        """注意事项"""
        return """
        1. 目标是根据评估方法论进行评估，不是生成新的解说文案
        2. 爆款文案生成方法论可以帮助理解解说文案生成过程，辅助评估
        3. 严格按照爆款文案评估方法论中的评估维度和标准进行评估
        """
    
    def get_output_format(self) -> str:
        """输出格式"""
        return """
        按照评估方法论中的输出模板输出：
        1. 一句话总评
        2. 评分总表（总分100）
        3. 分项诊断（虎头/豹身/凤尾）
        4. 硬伤项（若触发则直接判不及格）
        5. 可执行优化建议（按优先级）
        """
    
    def get_output_example(self) -> str:
        """输出示例"""
        return """
        一句话总评：这篇文案开头爆点前置到位，但中段存在流水账问题，结尾钩子力度不足。

        评分总表：
        | 模块 | 得分 | 关键结论 |
        |---|---|---|
        | 虎头（20） | 16/20 | 爆点前置到位，要素控制良好 |
        | 豹身（60） | 42/60 | 存在流水账问题，钩子密度不足 |
        | 凤尾（20） | 12/20 | 有钩子但力度一般 |
        | **总分（100）** | **70/100** | **等级：合格** |

        [后续包含详细的分项诊断、问题清单和优化建议]
        """
    
    def _load_markdown_file(self, file_path: Path) -> str:
        """
        加载 markdown 文件内容
        
        Args:
            file_path: markdown 文件路径
            
        Returns:
            str: 文件内容
        """
        try:
            self.logger.debug(f"读取文件: {file_path}")
            if not file_path.exists():
                raise FileNotFoundError(f"文件不存在: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            self.logger.info(f"成功读取文件，长度: {len(content)} 字符")
            return content
        except Exception as e:
            self.logger.error(f"读取文件失败: {str(e)}")
            raise Exception(f"读取文件失败: {str(e)}")
    
    def get_system_prompt(self) -> str:
        """
        获取系统提示词：按照规则构造
        
        Returns:
            str: 系统提示词
        """
        try:
            # 读取爆款文案生成方法论
            generation_methodology = self._load_markdown_file(self.generation_methodology_path)
            
            # 读取爆款文案评估方法论
            eval_methodology = self._load_markdown_file(self.eval_methodology_path)
            
            # 按照规则构造 system_prompt
            system_prompt = """你是一个爆款文案解说质量评估员，请理解文案生产的方法论，以及文案质量评估的方法论，对一段文案进行质量评估

爆款文案生成方法论（学习文案的写作手法，辅助你进行质量评估）

{generation_methodology}

爆款文案评估方法论（评估维度和标准，是你主要依赖的方法论）

{eval_methodology}

""".format(
                generation_methodology=generation_methodology,
                eval_methodology=eval_methodology
            )
            
            self.logger.info(f"系统提示词构造完成，长度: {len(system_prompt)} 字符")
            return system_prompt
        except Exception as e:
            self.logger.error(f"构造系统提示词失败: {str(e)}")
            raise Exception(f"构造系统提示词失败: {str(e)}")
    
    def get_user_prompt(self, input_data: Any, additional_context: Optional[str] = None) -> str:
        """
        获取用户提示词
        
        Args:
            input_data: 输入数据，可以是字符串或包含 "commentary_text" 的字典
            additional_context: 额外的上下文信息（可选）
            
        Returns:
            str: 用户提示词
        """
        self.logger.info(f"生成用户提示词: input_data类型={type(input_data)}")
        
        # 解析输入数据
        if isinstance(input_data, str):
            commentary_text = input_data
        elif isinstance(input_data, dict):
            commentary_text = input_data.get("commentary_text", "")
            if not commentary_text:
                self.logger.error("input_data 字典中必须包含 'commentary_text' 字段")
                raise ValueError("input_data 字典中必须包含 'commentary_text' 字段")
        else:
            self.logger.error(f"input_data 必须是字符串或包含 'commentary_text' 的字典，当前类型: {type(input_data)}")
            raise ValueError("input_data 必须是字符串或包含 'commentary_text' 的字典")
        
        if not commentary_text or not commentary_text.strip():
            self.logger.error("commentary_text 不能为空")
            raise ValueError("commentary_text 不能为空")
        
        # 构建用户提示词
        user_prompt = f"""
            输入解说文案结果：

            {commentary_text}

            给定一段文案写作过程和结果，请严格按照爆款文案评估方法论，对最终生成的文案进行质量评估

            注意：
            1. 目标是根据评估方法论进行评估，不是生成新的解说文案
            2. 爆款文案生成方法论可以帮助你进行理解解说文案生成过程，帮助评估,忽略爆款文案生成方法论中的参考案例，以上面输入的解说文案为准
            3, 最后的一句话总评必须包含 FAIL-不合格/NORMAL-合格/SUPER-优秀,其中一项
        """
        # 如果有额外的上下文信息，添加到提示词中
        if additional_context:
            user_prompt += f"\n\n额外上下文信息：\n{additional_context}"
        
        self.logger.info(f"用户提示词生成完成，长度: {len(user_prompt)} 字符")
        return user_prompt
    
    def call_model(self, input_data: Any, user_prompt: str) -> Tuple[str, Dict[str, Any]]:
        """
        模型调用函数
        
        Args:
            input_data: 输入数据（可以是文件路径、文本、字典等任意类型）
            user_prompt: 用户提示词
            
        Returns:
            tuple: (模型返回的原始响应, 请求 payload)
        """
        # 获取系统提示词
        system_prompt = self.get_system_prompt()
        
        # 使用 ModelClient 调用模型
        raw_response, payload = self.model_client.call(system_prompt, user_prompt)
        
        return raw_response, payload
    
    def post_process(self, raw_response: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        后处理函数：解析模型返回的结果
        
        Args:
            raw_response: 模型返回的原始响应
            payload: 完整的请求 payload（包含所有提示词信息）
            
        Returns:
            Dict[str, Any]: 解析后的结构化结果
        """
        self.logger.info("开始后处理模型响应")
        result = {
            "raw_response": raw_response,
            "evaluation_report": raw_response.strip(),
            "status": "success",
            "model_input": {
                "payload": payload
            }
        }
        self.logger.info(f"后处理完成，结果状态: {result['status']}")
        return result
    
    def sop(self, input_data: Any, additional_context: Optional[str] = None) -> Dict[str, Any]:
        """
        标准操作流程 (SOP) 函数
        
        Args:
            input_data: 输入数据，可以是字符串或包含 "commentary_text" 的字典
            additional_context: 额外的上下文信息（可选）
            
        Returns:
            Dict[str, Any]: 完整的分析结果
        """
        self.logger.info(f"开始执行 SOP 流程: input_data类型={type(input_data)}")
        try:
            # 1. 获取用户提示词
            self.logger.info("步骤 1: 获取用户提示词")
            user_prompt = self.get_user_prompt(input_data, additional_context)
            
            # 2. 调用模型
            self.logger.info("步骤 2: 调用模型")
            raw_response, payload = self.call_model(input_data, user_prompt)
            
            # 3. 后处理
            self.logger.info("步骤 3: 后处理")
            result = self.post_process(raw_response, payload)
            
            # 4. 添加元信息
            self.logger.info("步骤 4: 添加元信息")
            result["input_data"] = input_data
            result["additional_context"] = additional_context
            
            self.logger.info(f"SOP 流程执行成功，结果状态: {result.get('status')}")
            return result
        
        except Exception as e:
            self.logger.error(f"SOP 流程执行失败: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "error_message": str(e),
                "input_data": input_data,
                "additional_context": additional_context
            }
    
    def _save_result(self, result: Dict[str, Any], commentary_text: str, dj: Optional[str] = None) -> str:
        """
        保存结果到文件
        
        Args:
            result: 要保存的结果字典
            commentary_text: 被评估的文案文本（用于生成文件名）
            dj: 短剧名（可选），如果提供则保存到 results/{dj}/ 目录
            
        Returns:
            str: 保存的 JSON 文件路径
        """
        # 根据是否提供 dj 参数决定保存目录
        if dj:
            # 保存到 results/{dj}/ 目录
            eval_dir = self.results_base_dir / dj
        else:
            # 保存到 results/eval 目录（向后兼容）
            eval_dir = self.results_base_dir / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成文件名：eval_{date}
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"eval_{date_str}"
        
        # 保存 JSON 文件
        json_file_path = eval_dir / f"{base_filename}.json"
        try:
            with open(json_file_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            self.logger.info(f"结果已保存到: {json_file_path}")
        except Exception as e:
            self.logger.error(f"保存 JSON 结果失败: {str(e)}")
            raise Exception(f"保存 JSON 结果失败: {str(e)}")
        
        # 保存评估报告为 markdown 文件
        if "evaluation_report" in result:
            md_file_path = eval_dir / f"{base_filename}.md"
            try:
                with open(md_file_path, 'w', encoding='utf-8') as f:
                    f.write(result["evaluation_report"])
                self.logger.info(f"评估报告已保存到: {md_file_path}")
            except Exception as e:
                self.logger.warning(f"保存 markdown 文件失败: {str(e)}")
                # markdown 文件保存失败不影响主流程
        
        return str(json_file_path)
    
    def execute(self, input_data: Any, additional_context: Optional[str] = None, dj: Optional[str] = None) -> Dict[str, Any]:
        """
        组合所有信息，触发执行过程（主入口函数）
        
        Args:
            input_data: 输入数据，可以是字符串或包含 "commentary_text" 的字典
            additional_context: 额外的上下文信息（可选）
            dj: 短剧名（可选），如果提供则保存到 results/{dj}/ 目录
            
        Returns:
            Dict[str, Any]: 完整的分析结果
        """
        self.logger.info(f"执行入口函数被调用: input_data类型={type(input_data)}, dj={dj}")
        result = self.sop(input_data, additional_context=additional_context)
        self.logger.info(f"执行完成，最终状态: {result.get('status')}")
        
        # 保存结果到文件
        try:
            # 提取文案文本用于生成文件名
            if isinstance(input_data, str):
                commentary_text = input_data
            elif isinstance(input_data, dict):
                commentary_text = input_data.get("commentary_text", "")
            else:
                commentary_text = ""
            
            if commentary_text:
                saved_path = self._save_result(result, commentary_text, dj=dj)
                result["saved_path"] = saved_path
                self.logger.info(f"结果已保存: {saved_path}")
        except Exception as e:
            self.logger.warning(f"保存结果时出现异常（不影响返回结果）: {str(e)}")
        
        return result


if __name__ == "__main__":
    # 示例用法
    skill = SkillVideoCommentaryQualityEval()
    
    # 方式1：直接传入文案文本
#     input_data = """富家千金为了所谓的真爱，不惜跟亲爹断绝关系也要下嫁，结果大喜的日子直接被这穷小子一家当猴耍。
# 本来女人还傻乎乎地护着老公，生怕伴娘闹得太凶，可这男人倒好，还没进门就开始跟兄弟炫耀，
# 说反正老婆已经怀了孕，根本没必要给彩礼。"""
    input_data = """
根据您提供的短剧信息和对白内容，我将严格按照《爆款解说文案创作技能文档》的SOP流程进行处理，为您生成一份高质量的解说文案。

---

### SOP 1: 结构化分析 (大纲)

**核心剧情梳理（按时序+六要素）**

| 时序 | 人 | 地 | 事 | 因 | 情 | 细/爆点标记 |
|:---|:---|:---|:---|:---|:---|:---|
| 1 | 云如烟、季明川 | 卧室/家中 | 云如烟甩出1000万分手费，拒绝续约 | 五年替身合约到期 | 决绝 vs 占有欲 | 甩钱反差 ※ |
| 2 | 季明川 | 家中 | 发现云如烟已搬走，下令全城搜捕 | 无法接受“玩物”脱离掌控 | 暴怒 | 挖地三尺 ※ |
| 3 | 云如烟、蒋少 | KTV | 云如烟庆祝自由，偶遇恶少蒋少被羞辱 | 蒋少觊觎季明川的女人 | 厌恶、恐惧 | “双倍价钱”羞辱 ※ |
| 4 | 蒋少、众手下 | KTV | 蒋少欲强行霸占云如烟，悬赏手下 | 云如烟拒绝“脏钱”激怒蒋少 | 危机、绝望 | 悬赏房地产项目 ※ |
| 5 | 季明川 | KTV | 季明川及时赶到，救下云如烟并封杀蒋家 | 护短，宣示主权 | 霸气、愤怒 | 全面封杀 ※ |
| 6 | 云如烟、季明川 | 医院外/车旁 | 季明川以安全为由强迫续约，云如烟反讽 | 蒋少势力大，云如烟需庇护 | 胁迫 vs 挑衅 | “爱上我了吧” ※ |
| 7 | 苏文文 | 现场 | “白月光”苏文文突然回国，打断两人 | 留学归来 | 错愕、修罗场 | 正主归来 ※ |
| 8 | 三人 | 现场 | 云如烟撇清关系离开，苏文文试探季明川 | 云如烟不想纠缠 | 尴尬、猜忌 | “我不熟” ※ |

---

### SOP 2: 内容精简 (提炼)

**关键字提炼表**

| 原始大纲剧情（大白话） | 提取的核心关键字 | 最终凝练输出（文案架构基础） |
|:---|:---|:---|
| 云如烟给季明川1000万要分手，拒绝续约。 | 1000万、分手、替身 | 女人甩出1000万支票要结束替身关系 |
| 蒋少在KTV羞辱云如烟，说给双倍价钱。 | 恶少、双倍、玩物 | 恶少出双倍价钱羞辱她是玩物 |
| 季明川赶到，说要封杀蒋家所有产业。 | 封杀、救人 | 总裁赶到，当场宣布封杀恶少家族 |
| 云如烟问季明川是不是爱上自己了。 | 假戏真做、爱上 | 女人反问总裁是否假戏真做 |
| 苏文文回来了，云如烟说跟季明川不熟。 | 白月光、正主、不熟 | 正主回国，替身立刻撇清关系 |

---

### SOP 3: 强力开头 (虎头)

**设计思路：**
采用**“反差前置 + 爆点”**的原则。通常剧情是霸总给女人钱，这里是替身给霸总钱，这是一个极强的反差爆点。

**开头文案：**
**女人甩出一张一千万的支票。**（要素控制：2名词+1动词，极简）
**她做了总裁五年的替身，今天合约到期，她只想拿钱走人。**（解释背景，制造悬念）

---

### SOP 4: 主体串联 (豹身)

**设计思路：**
遵循**“细节 + 钩子 + 解释 + 连接 + 剧情”**公式。
1.  **连接**：从分手直接过渡到KTV危机。
2.  **强化**：描写蒋少的恶心和季明川的霸气。
3.  **去流水账**：使用“没想到”、“更过分的是”等连接词。

**主体文案：**
男人却红了眼，不仅撕毁了支票，更扬言要强行续约。（递进：红了眼、强行）
女人冷笑一声，趁着男人不备，连夜收拾行李逃之夭夭。（动作：冷笑、逃之夭夭）
为了庆祝恢复自由，她跑去KTV狂欢。（连接：为了...）
**没想到冤家路窄**，竟撞上了江城最大的恶少。（转折：没想到；钩子：恶少）
恶少一眼认出她是季明川的玩物，竟当众羞辱，扬言要出双倍的价钱买她一夜。（细节：玩物、双倍价钱）
女人嫌他的钱脏，誓死不从。（情绪：嫌脏）
**这一举动彻底激怒了恶少**，他一声令下，竟让手下轮番上阵，谁能征服她就送谁一套房。（冲突升级：轮番上阵、送房）
就在女人即将受辱的千钧一发之际，包厢大门被一脚踹开。（高潮：千钧一发、踹开）
那个男人赶到了。
看着衣衫不整的女人，季明川怒火中烧，当场宣布让恶少家族破产。（爽点：破产）
事后，男人一把将她禁锢在怀里，逼她回家。（动作：禁锢）
女人却挑眉反问：**“你这么不想我走，该不会是假戏真做，爱上我了吧？”**（反讽钩子）
季明川刚想反驳，一道娇滴滴的声音突然打断了两人。（悬念引入）

---

### SOP 5: 有力收尾 (凤尾)

**设计思路：**
采用**“爆点一分为二法”**或**“大钩子”**。苏文文（白月光）的出现是最大的转折点，预示着“替身”生涯的结束和新折磨的开始。

**结尾文案：**
那个让男人念念不忘的“白月光”，竟然回国了。（身份揭晓）
看着正主归来，替身立刻松了一口气，笑着对白月光说：
“别误会，我和他不熟。”（反差：不熟）
说完，她转身就要离开。
可看着季明川那阴沉得快要滴水的脸色，
**这场三个人的游戏，恐怕才刚刚开始。**（大钩子：暗示后续纠缠）

---

## 最终生成的精彩解说文案

**女人甩出一张一千万的支票。**
**她做了总裁五年的替身，今天合约到期，她只想拿钱走人。**

男人却红了眼，不仅撕毁了支票，更扬言要强行续约。
女人冷笑一声，趁着男人不备，连夜收拾行李逃之夭夭。
为了庆祝恢复自由，她跑去KTV狂欢。
**没想到冤家路窄**，竟撞上了江城最大的恶少。
恶少一眼认出她是季明川的玩物，竟当众羞辱，扬言要出双倍的价钱买她一夜。
女人嫌他的钱脏，誓死不从。
**这一举动彻底激怒了恶少**，他一声令下，竟让手下轮番上阵，谁能征服她就送谁一套房。

就在女人即将受辱的千钧一发之际，包厢大门被一脚踹开。
那个男人赶到了。
看着衣衫不整的女人，季明川怒火中烧，当场宣布让恶少家族破产。
事后，男人一把将她禁锢在怀里，逼她回家。
女人却挑眉反问：**“你这么不想我走，该不会是假戏真做，爱上我了吧？”**

季明川刚想反驳，一道娇滴滴的声音突然打断了两人。
那个让男人念念不忘的“白月光”，竟然回国了。
看着正主归来，替身立刻松了一口气，笑着对白月光说：
“别误会，我和他不熟。”
说完，她转身就要离开。
可看着季明川那阴沉得快要滴水的脸色，
**这场三个人的游戏，恐怕才刚刚开始。**


    """
    # 方式2：传入字典格式
    # input_data = {
    #     "commentary_text": "待评估的文案文本..."
    # }
    
    result = skill.execute(input_data)
    print(json.dumps(result, ensure_ascii=False, indent=2))

