import os
import httpx
from typing import List, Dict, Any, Optional, Union
import asyncio
import json
from utils.token_counter import count_tokens, count_messages_tokens

from llm.base import BaseLLM

class OpenRouterClient(BaseLLM):
    """OpenRouter API客户端，支持访问多种LLM"""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        初始化OpenRouter客户端
        
        Args:
            api_key: OpenRouter API密钥，如果为None则从环境变量获取
            model: 使用的模型名称，如果为None则从环境变量获取
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API密钥未提供")
        
        self.model = model or os.getenv("OPENROUTER_MODEL", "openai/gpt-4-turbo")
        
        # API端点
        self.api_base = "https://openrouter.ai/api/v1"
        
        # 应用信息
        self.app_name = "Memory-Enhanced AI Chat"
        self.app_version = "1.0.0"
        
        # 模型上下文窗口大小
        self.context_sizes = {
            "openai/gpt-3.5-turbo": 4096,
            "openai/gpt-4": 8192,
            "openai/gpt-4-turbo": 128000,
            "anthropic/claude-2": 100000,
            "anthropic/claude-3-opus": 200000,
            "anthropic/claude-3-sonnet": 200000,
            "anthropic/claude-3-haiku": 200000,
            "google/gemini-pro": 32768,
            "meta-llama/llama-3-70b-instruct": 8192,
            "meta-llama/llama-3-8b-instruct": 8192,
            "deepseek/deepseek-chat": 8192,
            "deepseek/deepseek-coder": 16384
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
        # 构建请求数据
        request_data = {
            "model": self.model,
            "messages": messages,
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
                        "Authorization": f"Bearer {self.api_key}",
                        "HTTP-Referer": "https://memory-enhanced-ai-chat.example.com",  # 你的应用域名
                        "X-Title": self.app_name,
                        "X-Version": self.app_version
                    },
                    json=request_data
                )
                
                if response.status_code != 200:
                    error_msg = f"OpenRouter API调用失败: {response.status_code} - {response.text}"
                    print(error_msg)
                    return f"抱歉，我遇到了一些问题，无法生成回复。错误: {error_msg}"
                
                response_data = response.json()
                return response_data["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"OpenRouter API调用失败: {e}")
            # 重试一次
            try:
                await asyncio.sleep(1)  # 等待1秒后重试
                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.post(
                        f"{self.api_base}/chat/completions",
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {self.api_key}",
                            "HTTP-Referer": "https://memory-enhanced-ai-chat.example.com",  # 你的应用域名
                            "X-Title": self.app_name,
                            "X-Version": self.app_version
                        },
                        json=request_data
                    )
                    
                    if response.status_code != 200:
                        error_msg = f"OpenRouter API调用失败: {response.status_code} - {response.text}"
                        print(error_msg)
                        return f"抱歉，我遇到了一些问题，无法生成回复。错误: {error_msg}"
                    
                    response_data = response.json()
                    return response_data["choices"][0]["message"]["content"]
            except Exception as e:
                print(f"OpenRouter API重试失败: {e}")
                return f"抱歉，我遇到了一些问题，无法生成回复。错误: {str(e)}"
    
    async def count_tokens(self, text: str) -> int:
        """
        计算文本的token数量
        
        Args:
            text: 要计算的文本
        
        Returns:
            token数量
        """
        # 根据模型选择合适的token计数方法
        if "gpt" in self.model or "openai" in self.model:
            model_name = self.model.split("/")[-1] if "/" in self.model else self.model
            return count_tokens(text, model_name)
        else:
            # 对于非OpenAI模型，使用cl100k_base作为近似
            return count_tokens(text, "cl100k_base")
    
    async def count_messages_tokens(self, messages: List[Dict[str, str]]) -> int:
        """
        计算消息列表的token数量
        
        Args:
            messages: 消息列表，格式为[{"role": "user", "content": "Hello"}, ...]
        
        Returns:
            token数量
        """
        # 根据模型选择合适的token计数方法
        if "gpt" in self.model or "openai" in self.model:
            model_name = self.model.split("/")[-1] if "/" in self.model else self.model
            return count_messages_tokens(messages, model_name)
        else:
            # 对于非OpenAI模型，使用cl100k_base作为近似
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
        # 尝试从模型名称中提取基础模型
        model_key = self.model
        if "/" in model_key:
            model_key = self.model  # 使用完整模型名称作为键
        
        return self.context_sizes.get(model_key, 8192)  # 默认为8192