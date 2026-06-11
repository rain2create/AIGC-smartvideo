#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型请求客户端
封装模型 API 请求逻辑，提供统一的模型调用接口
"""

import logging
import requests
import time
from typing import Dict, List, Optional, Any, Tuple


class ModelClient:
    """
    模型请求客户端类
    
    负责封装模型 API 的请求逻辑，包括：
    - 构建请求 payload
    - 发送 HTTP 请求
    - 处理响应和错误
    """
    
    def __init__(
        self,
        api_url: str = "https://api.360.cn/v1/chat/completions",
        api_key: str = "fk1379989049.aEmK59LGEcoiwxat0pGE18JPvtSBkrybaf481994",
        model: str = "google/gemini-3-pro-preview",
        # model: str = "openai/gpt-5.2",
        timeout: int = 300,
        temperature: Optional[float] = 0.5,
        top_p: Optional[float] = 0.2
    ):
        """
        初始化模型客户端
        
        Args:
            api_url: API 地址
            api_key: API 密钥
            model: 模型名称
            timeout: 请求超时时间（秒）
        """
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.top_p = top_p
        
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
        
        self.logger.info(f"初始化 ModelClient: api_url={api_url}, model={model}")
    
    def build_payload(
        self,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        构建请求 payload
        
        Args:
            system_prompt: 系统提示词（与 user_prompt 一起使用，或与 messages 一起使用）
            user_prompt: 用户提示词（与 system_prompt 一起使用）
            messages: 消息列表，格式为 [{"role": "system/user/assistant", "content": "..."}, ...]
                     如果提供 messages，则忽略 system_prompt 和 user_prompt
            model: 模型名称（可选，默认使用初始化时的 model）
            
        Returns:
            Dict[str, Any]: 请求 payload
        """
        if messages is not None:
            # 使用提供的 messages 列表
            payload = {
                "model": model or self.model,
                "messages": messages
                # "temperature": 1.0
            }
            self.logger.debug(
                f"构建 payload (使用 messages): model={payload['model']}, "
                f"messages_count={len(payload['messages'])}"
            )
        elif system_prompt is not None and user_prompt is not None:
            # 使用 system_prompt 和 user_prompt
            payload = {
                "model": model or self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                # "temperature": 1.0
            }
            self.logger.debug(
                f"构建 payload (使用 system/user prompt): model={payload['model']}, "
                f"messages_count={len(payload['messages'])}, "
                f"system_prompt_length={len(system_prompt)}, "
                f"user_prompt_length={len(user_prompt)}"
            )
        else:
            raise ValueError("必须提供 messages 或同时提供 system_prompt 和 user_prompt")

        # 采样参数：优先用 call() 传入，其次用初始化默认值；都为空则不下发，走服务端默认
        final_temperature = temperature if temperature is not None else self.temperature
        final_top_p = top_p if top_p is not None else self.top_p
        if final_temperature is not None:
            payload["temperature"] = final_temperature
        if final_top_p is not None:
            payload["top_p"] = final_top_p
        
        return payload
    
    def build_headers(self) -> Dict[str, str]:
        """
        构建请求头
        
        Returns:
            Dict[str, str]: 请求头
        """
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def call(
        self,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_retries: int = 5,
        base_delay: int = 5
    ) -> Tuple[str, Dict[str, Any]]:
        """
        调用模型 API
        
        Args:
            system_prompt: 系统提示词（与 user_prompt 一起使用，或与 messages 一起使用）
            user_prompt: 用户提示词（与 system_prompt 一起使用）
            messages: 消息列表，格式为 [{"role": "system/user/assistant", "content": "..."}, ...]
                     如果提供 messages，则忽略 system_prompt 和 user_prompt
            model: 模型名称（可选，默认使用初始化时的 model）
            
        Returns:
            tuple: (模型返回的原始响应, 请求 payload)
            
        Raises:
            Exception: 当 API 请求失败时抛出异常
        """
        self.logger.info(f"开始调用模型: model={model or self.model}, api_url={self.api_url}")
        
        # 构建请求 payload
        payload = self.build_payload(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            messages=messages,
            model=model,
            temperature=temperature,
            top_p=top_p,
        )
        
        # 构建请求头
        headers = self.build_headers()
        
        for attempt in range(max_retries):
            try:
                self.logger.info(f"发送 API 请求... (第 {attempt + 1} 次)")
                response = requests.post(
                    self.api_url,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout
                )
                
                self.logger.debug(f"API 响应状态码: {response.status_code}")
                
                if response.status_code == 200:
                    response_json = response.json()
                    if "choices" in response_json and len(response_json["choices"]) > 0:
                        raw_response = response_json["choices"][0]["message"]["content"]
                        self.logger.info(f"模型调用成功，响应长度: {len(raw_response)} 字符")
                        return raw_response, payload
                    error_msg = f"API 响应格式异常: {response_json}"
                    self.logger.error(error_msg)
                    raise Exception(error_msg)
                
                if response.status_code == 429:
                    delay = base_delay * (attempt + 1)
                    self.logger.warning(f"触发频率限制(429)，{delay}s 后重试...")
                    time.sleep(delay)
                    continue
                
                error_msg = f"API 请求失败: status_code={response.status_code}, message={response.text}"
                self.logger.error(error_msg)
                raise Exception(error_msg)
            
            except requests.exceptions.Timeout:
                self.logger.warning(f"API 请求超时（{self.timeout}秒），2s 后重试...")
                time.sleep(2)
                continue
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"API 请求异常: {str(e)}，2s 后重试...")
                time.sleep(2)
                continue
        
        raise Exception(f"达到最大重试次数 ({max_retries})，模型调用失败。")

