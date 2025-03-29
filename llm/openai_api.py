import os
import openai
from typing import List, Dict, Any, Optional, Union
import asyncio
from utils.token_counter import count_tokens, count_messages_tokens

from llm.base import BaseLLM

class OpenAIClient(BaseLLM):
    """OpenAI API客户端"""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        初始化OpenAI客户端
        
        Args:
            api_key: OpenAI API密钥，如果为None则从环境变量获取
            model: 使用的模型名称，如果为None则从环境变量获取
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API密钥未提供")
        
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4")
        
        # 设置OpenAI客户端
        self.client = openai.AsyncOpenAI(api_key=self.api_key)
        
        # 模型上下文窗口大小
        self.context_sizes = {
            "gpt-3.5-turbo": 4096,
            "gpt-3.5-turbo-16k": 16384,
            "gpt-4": 8192,
            "gpt-4-32k": 32768,
            "gpt-4-turbo": 128000,
            "gpt-4o": 128000
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
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI API调用失败: {e}")
            # 重试一次
            try:
                await asyncio.sleep(1)  # 等待1秒后重试
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop=stop
                )
                
                return response.choices[0].message.content
            except Exception as e:
                print(f"OpenAI API重试失败: {e}")
                return f"抱歉，我遇到了一些问题，无法生成回复。错误: {str(e)}"
    
    async def count_tokens(self, text: str) -> int:
        """
        计算文本的token数量
        
        Args:
            text: 要计算的文本
        
        Returns:
            token数量
        """
        return count_tokens(text, self.model)
    
    async def count_messages_tokens(self, messages: List[Dict[str, str]]) -> int:
        """
        计算消息列表的token数量
        
        Args:
            messages: 消息列表，格式为[{"role": "user", "content": "Hello"}, ...]
        
        Returns:
            token数量
        """
        return count_messages_tokens(messages, self.model)
    
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
        return self.context_sizes.get(self.model, 4096)  # 默认为4096