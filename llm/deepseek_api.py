import os
import httpx
from typing import List, Dict, Any, Optional, Union
import asyncio
import json
from utils.token_counter import count_tokens, count_messages_tokens

from llm.base import BaseLLM

class DeepSeekClient(BaseLLM):
    """DeepSeek API客户端"""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        初始化DeepSeek客户端
        
        Args:
            api_key: DeepSeek API密钥，如果为None则从环境变量获取
            model: 使用的模型名称，如果为None则从环境变量获取
        """
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DeepSeek API密钥未提供")
        
        self.model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        
        # API端点
        self.api_base = "https://api.deepseek.com/v1"
        
        # 模型上下文窗口大小
        self.context_sizes = {
            "deepseek-chat": 8192,
            "deepseek-coder": 16384,
            "deepseek-chat-v2": 32768
        }
    
    async def generate_text(
        self, 
        prompt: str, 
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop: Optional[List[str]] = None
    ) -> str:
        """
        生成文本
        
        Args:
            prompt: 提示文本
            max_tokens: 生成的最大token数
            temperature: 温度参数，控制随机性
            top_p: 核采样参数
            stop: 停止生成的标记列表
        
        Returns:
            生成的文本
        """
        messages = [{"role": "user", "content": prompt}]
        return await self.generate_chat_response(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop
        )
    
    async def generate_chat_response(
        self, 
        messages: List[Dict[str, str]], 
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop: Optional[List[str]] = None
    ) -> str:
        """
        生成聊天回复
        
        Args:
            messages: 消息列表，格式为[{"role": "user", "content": "Hello"}, ...]
            max_tokens: 生成的最大token数
            temperature: 温度参数，控制随机性
            top_p: 核采样参数
            stop: 停止生成的标记列表
        
        Returns:
            生成的回复文本
        """
        # 转换消息格式
        deepseek_messages = []
        for msg in messages:
            role = msg["role"]
            # DeepSeek API使用"assistant"而不是"assistant"
            if role == "assistant":
                deepseek_messages.append({"role": "assistant", "content": msg["content"]})
            elif role == "system":
                deepseek_messages.append({"role": "system", "content": msg["content"]})
            else:  # user或其他
                deepseek_messages.append({"role": "user", "content": msg["content"]})
        
        # 构建请求数据
        request_data = {
            "model": self.model,
            "messages": deepseek_messages,
            "temperature": temperature,
            "top_p": top_p
        }
        
        if max_tokens:
            request_data["max_tokens"] = max_tokens
        
        if stop:
            request_data["stop"] = stop
        
        # 发送请求
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.api_base}/chat/completions",
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}"
                    },
                    json=request_data
                )
                
                if response.status_code != 200:
                    error_msg = f"DeepSeek API调用失败: {response.status_code} - {response.text}"
                    print(error_msg)
                    return f"抱歉，我遇到了一些问题，无法生成回复。错误: {error_msg}"
                
                response_data = response.json()
                return response_data["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"DeepSeek API调用失败: {e}")
            # 重试一次
            try:
                await asyncio.sleep(1)  # 等待1秒后重试
                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.post(
                        f"{self.api_base}/chat/completions",
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {self.api_key}"
                        },
                        json=request_data
                    )
                    
                    if response.status_code != 200:
                        error_msg = f"DeepSeek API调用失败: {response.status_code} - {response.text}"
                        print(error_msg)
                        return f"抱歉，我遇到了一些问题，无法生成回复。错误: {error_msg}"
                    
                    response_data = response.json()
                    return response_data["choices"][0]["message"]["content"]
            except Exception as e:
                print(f"DeepSeek API重试失败: {e}")
                return f"抱歉，我遇到了一些问题，无法生成回复。错误: {str(e)}"
    
    async def count_tokens(self, text: str) -> int:
        """
        计算文本的token数量
        
        Args:
            text: 要计算的文本
        
        Returns:
            token数量
        """
        # DeepSeek没有官方的token计数器，使用tiktoken的cl100k_base作为近似
        return count_tokens(text, "cl100k_base")
    
    async def count_messages_tokens(self, messages: List[Dict[str, str]]) -> int:
        """
        计算消息列表的token数量
        
        Args:
            messages: 消息列表，格式为[{"role": "user", "content": "Hello"}, ...]
        
        Returns:
            token数量
        """
        return count_messages_tokens(messages, "cl100k_base")
    
    def get_model_name(self) -> str:
        """
        获取模型名称
        
        Returns:
            模型名称
        """
        return self.model
    
    def get_model_context_size(self) -> int:
        """
        获取模型的上下文窗口大小
        
        Returns:
            上下文窗口大小（token数）
        """
        return self.context_sizes.get(self.model, 8192)  # 默认为8192