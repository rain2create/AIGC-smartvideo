#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
短剧解说文案创作技能
基于 BaseSkill 实现，用于生成爆款短视频解说文案
"""

import os
import json
import logging
import requests
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from base_skill import BaseSkill
from model_client import ModelClient
from skill_video_commentary_quality_eval import SkillVideoCommentaryQualityEval
from pathlib import Path
import sys

class SkillVideoCommentaryText(BaseSkill):
    """
    短剧解说文案创作技能类
    
    输入格式：
        input_data = {
            "dj": "短剧名",
            "start": 开始索引,
            "end": 结束索引
        }
    """
    
    def __init__(
        self,
        api_url: str = "https://api.360.cn/v1/chat/completions",
        api_key: str = "fk1379989049.aEmK59LGEcoiwxat0pGE18JPvtSBkrybaf481994",
        model: str = "google/gemini-3-pro-preview",
        # model: str = "openai/gpt-5.2",
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
        
        # 初始化评估技能模块
        self.eval_skill = SkillVideoCommentaryQualityEval(
            api_url=api_url,
            api_key=api_key,
            model=model,
            timeout=timeout
        )
        
        # 获取当前文件所在目录
        current_dir = Path(__file__).parent
        # 构建 markdown 文件路径
        self.markdown_path = current_dir / "prompts" /"skill_commentary_formatted.md"
        # 构建 audio/videos 目录路径（相对于 commentary 目录）
        self.audio_base_dir = "/mnt_liuhuan/search_public/aigc/smart_video_v2/audio/videos"
        # 构建 results 目录路径（相对于 commentary 目录）
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
        
        self.logger.info(f"初始化 SkillVideoCommentaryText: api_url={api_url}, model={model}")
    
    def _get_description(self) -> str:
        """获取 Skill 描述"""
        return "短剧解说文案创作技能：根据短剧对白内容，生成爆款短视频解说文案"
    
    def get_role_definition(self) -> str:
        """定义角色"""
        return "原创内容专家/影视解说文案架构师"
    
    def get_principles(self) -> str:
        """总原则"""
        return "将长篇素材进行系统性分析和结构化梳理，运用叙事技巧、钩子设计和爆点前置原则，创作出具备高完播率和高情绪共鸣的原创短视频解说文案"
    
    def get_output_style(self) -> str:
        """输出风格"""
        return "精简、精彩、具备高完播率和高情绪共鸣的原创短视频解说文案"
    
    def get_sop(self) -> str:
        """标准操作流程说明"""
        return """
        文案创作遵循以下五步递进流程：
        1. SOP 1: 结构化分析（大纲）- 完整剧情提纲与爆点标记
        2. SOP 2: 内容精简（提炼）- 核心关键字列表
        3. SOP 3: 强力开头（虎头）- 爆点前置、反差前置的第一句话
        4. SOP 4: 主体串联（豹身）- 节奏分明、无断层、具象化的文案主体
        5. SOP 5: 有力收尾（凤尾）- 悬念大钩子或下一集预告
        """
    
    def get_notes(self) -> str:
        """注意事项"""
        return "严格按照'爆款解说文案创作技能文档'的指导，每个 SOP 都严格按照给定的重要准则执行"
    
    def get_output_format(self) -> str:
        """输出格式"""
        return "直接输出文案内容"
    
    def get_output_example(self) -> str:
        """输出示例"""
        return """
        富家千金为了所谓的真爱，不惜跟亲爹断绝关系也要下嫁，结果大喜的日子直接被这穷小子一家当猴耍。
        本来女人还傻乎乎地护着老公，生怕伴娘闹得太凶，可这男人倒好，还没进门就开始跟兄弟炫耀，
        说反正老婆已经怀了孕，根本没必要给彩礼。
        """
    
    def get_system_prompt(self) -> str:
        """
        获取系统提示词：读取 markdown 文件内容
        
        Returns:
            str: 系统提示词（markdown 文件内容）
        """
        try:
            self.logger.debug(f"读取系统提示词文件: {self.markdown_path}")
            if not self.markdown_path.exists():
                raise FileNotFoundError(f"系统提示词文件不存在: {self.markdown_path}")
            
            with open(self.markdown_path, 'r', encoding='utf-8') as f:
                content = f.read()
            self.logger.info(f"成功读取系统提示词，长度: {len(content)} 字符")
            return content
        except Exception as e:
            self.logger.error(f"读取系统提示词文件失败: {str(e)}")
            raise Exception(f"读取系统提示词文件失败: {str(e)}")
    
    def _get_drama_info(self, dj: str) -> str:
        """
        获取短剧剧情简介
        
        Args:
            dj: 短剧名
            
        Returns:
            str: 格式为 "短剧名称：xxx，短剧简介：xxx"
        """
        api_url = f"https://api.kuaichuan.360.cn/cinfo?c={dj}"
        self.logger.info(f"获取短剧信息: dj={dj}, api_url={api_url}")
        
        try:
            response = requests.get(api_url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            self.logger.debug(f"API 响应: {data}")
            
            # 解析结果：['data']['info'][0]
            if 'data' in data and 'info' in data['data'] and len(data['data']['info']) > 0:
                info = data['data']['info'][0]
                name = info.get('name', '未知')
                description = info.get('description', '无简介')
                
                result = f"短剧名称：{name}，短剧简介：{description}"
                self.logger.info(f"成功获取短剧信息: name={name}")
                return result
            else:
                self.logger.warning(f"API 响应中未找到预期数据: {data}")
                return f"短剧名称：{dj}，短剧简介：暂无简介"
        
        except requests.exceptions.Timeout:
            self.logger.warning(f"获取短剧信息超时: {dj}")
            return f"短剧名称：{dj}，短剧简介：获取信息超时"
        except requests.exceptions.RequestException as e:
            self.logger.warning(f"获取短剧信息失败: {dj}, error={str(e)}")
            return f"短剧名称：{dj}，短剧简介：获取信息失败"
        except Exception as e:
            self.logger.warning(f"解析短剧信息失败: {dj}, error={str(e)}")
            return f"短剧名称：{dj}，短剧简介：解析信息失败"
    
    def _load_asr_texts(self, dj: str, start: int, end: int) -> str:
        """
        加载并整合 ASR 文本
        
        Args:
            dj: 短剧名
            start: 开始索引
            end: 结束索引
            
        Returns:
            str: 整合后的所有 ASR 文本（多集拼接成一段话）
        """
        self.logger.info(f"开始加载 ASR 文本: dj={dj}, start={start}, end={end}")
        self.audio_base_dir = Path("/mnt_liuhuan/search_public/aigc/smart_video_v2/audio/videos")
        dj_dir = self.audio_base_dir / dj
        # 触发 autofs 挂载
        import subprocess, time
        for _ in range(3):
            subprocess.run(["ls", str(dj_dir)], capture_output=True)
            if dj_dir.exists():
                break
            time.sleep(1)
        if not dj_dir.exists():
            self.logger.error(f"短剧目录不存在: {dj_dir}")
            raise FileNotFoundError(f"短剧目录不存在: {dj_dir}")
        
        all_texts = []
        loaded_count = 0
        
        # 遍历指定索引范围的所有 ASR 文件
        for index in range(start, end + 1):
            asr_file = dj_dir / f"{dj}_{index}_asr.json"
            
            if not asr_file.exists():
                self.logger.warning(f"ASR 文件不存在，已跳过: {asr_file}")
                continue
            
            try:
                self.logger.debug(f"读取 ASR 文件: {asr_file}")
                with open(asr_file, 'r', encoding='utf-8') as f:
                    asr_data = json.load(f)
                
                # 提取所有 utterances 中的 text
                utterances = asr_data.get("utterances", [])
                if not utterances:
                    # 尝试从 raw_asr_result 中提取
                    raw_result = asr_data.get("raw_asr_result", {})
                    if isinstance(raw_result, dict):
                        data = raw_result.get("data", {})
                        extra_data = data.get("extra_data")
                        if isinstance(extra_data, str):
                            extra_data = json.loads(extra_data)
                        if isinstance(extra_data, dict):
                            result = extra_data.get("result", {})
                            utterances = result.get("utterances", [])
                
                # 提取所有文本
                episode_text_count = 0
                for utt in utterances:
                    text = utt.get("text", "").strip()
                    if text:
                        all_texts.append(text)
                        episode_text_count += 1
                
                loaded_count += 1
                self.logger.debug(f"第 {index} 集加载成功，提取 {episode_text_count} 条文本")
            
            except Exception as e:
                self.logger.warning(f"读取 ASR 文件失败 {asr_file}: {str(e)}")
                continue
        
        if not all_texts:
            self.logger.error(f"未找到任何 ASR 文本内容（dj={dj}, start={start}, end={end}）")
            raise ValueError(f"未找到任何 ASR 文本内容（dj={dj}, start={start}, end={end}）")
        
        # 将所有文本拼接成一段话
        all_asr_text = " ".join(all_texts)
        self.logger.info(f"ASR 文本加载完成: 成功加载 {loaded_count} 集，共 {len(all_texts)} 条文本，总长度 {len(all_asr_text)} 字符")
        return all_asr_text
    
    def get_user_prompt(self, input_data: Any, additional_context: Optional[str] = None) -> str:
        """
        获取用户提示词
        
        Args:
            input_data: 输入数据，格式为 {"dj": str, "start": int, "end": int}
            additional_context: 额外的上下文信息（可选，如果为 None 则自动获取剧情简介）
            
        Returns:
            str: 用户提示词
        """
        self.logger.info(f"生成用户提示词: input_data={input_data}")
        # 解析输入数据
        if isinstance(input_data, dict):
            dj = input_data.get("dj")
            start = input_data.get("start")
            end = input_data.get("end")
        else:
            self.logger.error("input_data 必须是包含 'dj', 'start', 'end' 的字典")
            raise ValueError("input_data 必须是包含 'dj', 'start', 'end' 的字典")
        
        if not dj or start is None or end is None:
            self.logger.error(f"input_data 必须包含 'dj', 'start', 'end' 字段，当前: {input_data}")
            raise ValueError("input_data 必须包含 'dj', 'start', 'end' 字段")
        
        # 如果没有提供额外上下文，自动获取剧情简介
        if additional_context is None:
            additional_context = self._get_drama_info(dj)
        
        # 加载并整合 ASR 文本
        all_asr_text = self._load_asr_texts(dj, start, end)
        
        # 构建用户提示词
        user_prompt = f"""给定短剧对白内容，请严格按照"爆款解说文案创作技能文档"的指导，按照标准操作流程一步一步处理，每个 sop 都严格按照给定的重要准则执行，最终生成一段精彩的解说文案："""
        user_prompt += f"\n\n短剧信息：{additional_context}"
        user_prompt = f'{user_prompt}\n\n对白内容：{all_asr_text}'
        
        self.logger.info(f"用户提示词生成完成，长度: {len(user_prompt)} 字符")
        return user_prompt
    
    def _call_model_with_iteration(
        self, 
        input_data: Any, 
        user_prompt: str,
        max_iterations: int = 3
    ) -> Tuple[str, Dict[str, Any], List[Dict[str, str]]]:
        """
        带多轮对话的模型调用函数
        
        Args:
            input_data: 输入数据
            user_prompt: 初始用户提示词
            max_iterations: 最大迭代轮数（默认3轮）
            
        Returns:
            tuple: (最终模型返回的原始响应, 最后一次请求的 payload, 对话历史)
        """
        self.logger.info(f"开始多轮对话流程，最大轮数: {max_iterations}")
        
        # 从 input_data 中提取 dj 信息（用于保存评估结果）
        dj = None
        if isinstance(input_data, dict):
            dj = input_data.get("dj")
        
        # 获取系统提示词
        system_prompt = self.get_system_prompt()
        
        # 初始化对话历史
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        iteration = 0
        final_payload = None
        raw_response = ""  # 初始化响应变量
        
        while iteration < max_iterations:
            iteration += 1
            self.logger.info(f"第 {iteration} 轮对话开始")
            
            # 调用模型
            try:
                raw_response, payload = self.model_client.call(messages=messages)
                final_payload = payload
                
                # 将助手的响应添加到对话历史
                messages.append({"role": "assistant", "content": raw_response})
                
                self.logger.info(f"第 {iteration} 轮模型调用成功，响应长度: {len(raw_response)} 字符")
                
                # 调用评估技能模块
                self.logger.info(f"第 {iteration} 轮：调用评估技能模块")
                eval_result = self.eval_skill.execute(raw_response, dj=dj)
                eval_response = eval_result.get("raw_response", "")
                
                # 检查评估结果中是否包含 "FAIL"
                if "SUPER-优秀" not in eval_response.upper():
                    self.logger.info(f"第 {iteration} 轮评估结果不包含 SUPER，需要进行修订")
                    
                    # 如果还没达到最大轮数，继续下一轮
                    if iteration < max_iterations:
                        # 构建新的用户消息
                        revision_prompt = f"这一版解说文案的问题和优化建议如下：\n\n{eval_response}\n\n请按照优化建议进行修改"
                        messages.append({"role": "user", "content": revision_prompt})
                        self.logger.info(f"已添加修订请求，准备进行第 {iteration + 1} 轮对话")
                    else:
                        self.logger.info(f"已达到最大轮数 {max_iterations}，停止迭代")
                        break
                else:
                    self.logger.info(f"第 {iteration} 轮评估结果质量合格，停止迭代")
                    break
                    
            except Exception as e:
                self.logger.error(f"第 {iteration} 轮模型调用失败: {str(e)}")
                raise Exception(f"第 {iteration} 轮模型调用失败: {str(e)}")
        
        self.logger.info(f"多轮对话流程完成，共进行了 {iteration} 轮")
        return raw_response, final_payload, messages
    
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
            # "commentary_text": raw_response.strip(),
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
            input_data: 输入数据，格式为 {"dj": str, "start": int, "end": int}
            additional_context: 额外的上下文信息（可选，如果为 None 则自动获取剧情简介）
            
        Returns:
            Dict[str, Any]: 完整的分析结果
        """
        self.logger.info(f"开始执行 SOP 流程: input_data={input_data}")
        try:
            # 如果 additional_context 为 None，自动获取剧情简介
            if additional_context is None and isinstance(input_data, dict):
                dj = input_data.get("dj")
                if dj:
                    additional_context = self._get_drama_info(dj)
            
            # 1. 获取用户提示词
            self.logger.info("步骤 1: 获取用户提示词")
            user_prompt = self.get_user_prompt(input_data, additional_context)
            
            # 2. 调用模型（支持多轮对话，最多3轮）
            self.logger.info("步骤 2: 调用模型（多轮对话）")
            print(f"\n{'='*80}\n请求模型 - 用户提示词\n{'='*80}\n{user_prompt}\n{'='*80}\n")
            raw_response, payload, conversation_history = self._call_model_with_iteration(input_data, user_prompt)
            
            # 3. 后处理
            self.logger.info("步骤 3: 后处理")
            result = self.post_process(raw_response, payload)
            
            # 4. 添加元信息
            self.logger.info("步骤 4: 添加元信息")
            result["input_data"] = input_data
            result["additional_context"] = additional_context
            result["conversation_history"] = conversation_history
            
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
    
    def _save_result(self, result: Dict[str, Any], dj: str, start: int, end: int) -> str:
        """
        保存结果到文件
        
        Args:
            result: 要保存的结果字典
            dj: 短剧名
            start: 开始索引
            end: 结束索引
            
        Returns:
            str: 保存的 JSON 文件路径
        """
        # 创建 results/{dj} 目录
        dj_dir = self.results_base_dir / dj
        dj_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成文件名：{start}_{end}_{date}
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{start}_{end}_{date_str}"
        
        # 保存 JSON 文件
        json_file_path = dj_dir / f"{base_filename}.json"
        try:
            with open(json_file_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            self.logger.info(f"结果已保存到: {json_file_path}")
        except Exception as e:
            self.logger.error(f"保存 JSON 结果失败: {str(e)}")
            raise Exception(f"保存 JSON 结果失败: {str(e)}")
        
        # 保存 raw_response 为 markdown 文件
        if "raw_response" in result:
            md_file_path = dj_dir / f"{base_filename}.md"
            try:
                with open(md_file_path, 'w', encoding='utf-8') as f:
                    f.write(result["raw_response"])
                self.logger.info(f"raw_response 已保存到: {md_file_path}")
            except Exception as e:
                self.logger.warning(f"保存 markdown 文件失败: {str(e)}")
                # markdown 文件保存失败不影响主流程
        
        return str(json_file_path)
    
    def execute(self, input_data: Any) -> Dict[str, Any]:
        """
        组合所有信息，触发执行过程（主入口函数）
        
        Args:
            input_data: 输入数据，格式为 {"dj": str, "start": int, "end": int}
            
        Returns:
            Dict[str, Any]: 完整的分析结果
        """
        self.logger.info(f"执行入口函数被调用: input_data={input_data}")
        result = self.sop(input_data, additional_context=None)
        self.logger.info(f"执行完成，最终状态: {result.get('status')}")
        
        # 保存结果到文件
        try:
            if isinstance(input_data, dict):
                dj = input_data.get("dj")
                start = input_data.get("start")
                end = input_data.get("end")
                if dj and start is not None and end is not None:
                    saved_path = self._save_result(result, dj, start, end)
                    result["saved_path"] = saved_path
                    self.logger.info(f"结果已保存: {saved_path}")
        except Exception as e:
            self.logger.warning(f"保存结果时出现异常（不影响返回结果）: {str(e)}")
        
        return result


if __name__ == "__main__":
    # 从命令行参数读取输入
    if len(sys.argv) < 4:
        print("用法: python skill_commentary.py <dj> <start> <end>")
        print("示例: python skill_commentary.py yuhun 1 15")
        sys.exit(1)
    
    dj = sys.argv[1]
    start = int(sys.argv[2])
    end = int(sys.argv[3])
    
    input_data = {
        "dj": dj,
        "start": start,
        "end": end
    }
    
    skill = SkillVideoCommentaryText()
    result = skill.execute(input_data)
    print(json.dumps(result, ensure_ascii=False, indent=2))

