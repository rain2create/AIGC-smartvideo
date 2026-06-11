#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Skill 基类
所有 Skill 类都应该继承这个基类，实现必要的抽象方法
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any


class BaseSkill(ABC):
    """
    Skill 基类
    
    所有 Skill 类都应该继承这个基类，并实现以下抽象方法：
    - _get_description(): 获取 Skill 描述
    - get_role_definition(): 定义角色
    - get_principles(): 总原则
    - get_output_style(): 输出风格
    - get_sop(): 标准操作流程说明
    - get_notes(): 注意事项
    - get_output_format(): 输出格式
    - get_output_example(): 输出示例
    - get_system_prompt(): 获取系统提示词
    - get_user_prompt(): 用户提示词
    - call_model(): 模型调用函数
    - post_process(): 后处理函数
    - sop(): 标准操作流程函数
    - execute(): 执行入口函数
    
    注意：所有方法的 input_data 参数可以是任意类型（文件路径、文本、字典等），
    具体类型由子类根据实际需求决定。
    """
    
    @abstractmethod
    def _get_description(self) -> str:
        """
        获取 Skill 描述
        
        Returns:
            str: Skill 功能描述
        """
        pass
    
    @abstractmethod
    def get_role_definition(self) -> str:
        """
        定义角色
        
        Returns:
            str: 角色定义文本
        """
        pass
    
    @abstractmethod
    def get_principles(self) -> str:
        """
        总原则
        
        Returns:
            str: 分析原则文本
        """
        pass
    
    @abstractmethod
    def get_output_style(self) -> str:
        """
        输出风格
        
        Returns:
            str: 输出风格要求
        """
        pass
    
    @abstractmethod
    def get_sop(self) -> str:
        """
        标准操作流程 (SOP) 说明
        
        Returns:
            str: SOP 流程说明
        """
        pass
    
    @abstractmethod
    def get_notes(self) -> str:
        """
        注意事项
        
        Returns:
            str: 注意事项说明
        """
        pass
    
    @abstractmethod
    def get_output_format(self) -> str:
        """
        输出格式
        
        Returns:
            str: 输出格式说明
        """
        pass
    
    @abstractmethod
    def get_output_example(self) -> str:
        """
        输出示例
        
        Returns:
            str: 输出示例文本
        """
        pass
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """
        获取系统提示词
        
        Returns:
            str: 完整的系统提示词
        """
        pass
    
    @abstractmethod
    def get_user_prompt(self, input_data: Any, additional_context: Optional[str] = None) -> str:
        """
        获取用户提示词（输入内容，触发指令）
        
        Args:
            input_data: 输入数据（可以是文件路径、文本、字典等任意类型）
            additional_context: 额外的上下文信息（可选）
            
        Returns:
            str: 用户提示词
        """
        pass
    
    @abstractmethod
    def call_model(self, input_data: Any, user_prompt: str) -> str:
        """
        模型调用函数
        
        Args:
            input_data: 输入数据（可以是文件路径、文本、字典等任意类型）
            user_prompt: 用户提示词
            
        Returns:
            str: 模型返回的原始响应
        """
        pass
    
    @abstractmethod
    def post_process(self, raw_response: str) -> Dict[str, Any]:
        """
        后处理函数：解析模型返回的结果
        
        Args:
            raw_response: 模型返回的原始响应
            
        Returns:
            Dict[str, Any]: 解析后的结构化结果
        """
        pass
    
    @abstractmethod
    def sop(self, input_data: Any, additional_context: Optional[str] = None) -> Dict[str, Any]:
        """
        标准操作流程 (SOP) 函数
        
        Args:
            input_data: 输入数据（可以是文件路径、文本、字典等任意类型）
            additional_context: 额外的上下文信息（可选）
            
        Returns:
            Dict[str, Any]: 完整的分析结果
        """
        pass
    
    @abstractmethod
    def execute(self, input_data: Any, additional_context: Optional[str] = None) -> Dict[str, Any]:
        """
        组合所有信息，触发执行过程（主入口函数）
        
        Args:
            input_data: 输入数据（可以是文件路径、文本、字典等任意类型）
            additional_context: 额外的上下文信息（可选）
            
        Returns:
            Dict[str, Any]: 完整的分析结果
        """
        pass
