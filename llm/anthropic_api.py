import os
import anthropic
from typing import List, Dict, Any, Optional, Union
import asyncio
import re
from utils.token_counter import count_tokens

from llm.base import BaseLLM

class AnthropicClient(BaseLLM):
    """Anthropic API客户端"""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        初始化Anthropic客户端
        
        Args:
            api_key: Anthropic API密钥，如果为None则从环境变量获取
            model: 使用的模型名称，如果为None则从环境变量获取
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API密钥未提供")
        
        self.model = model or os.getenv("ANTHROPIC_MODEL", "claude-2")
        
        # 设置Anthropic客户端
        self.client = anthropic.AsyncAnthropic(api_key=self.api_key)
        
        # 模型上下文窗口大小
        self.context_sizes = {
            "claude-instant-1": 100000,
            "claude-2": 100000,
            "claude-3-opus-20240229": 200000,
            "claude-3-sonnet-20240229": 200000,
            "claude-3-haiku-20240307": 200000
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
        try:
            response = await self.client.completions.create(
                model=self.model,
                prompt=f"{anthropic.HUMAN_PROMPT} {prompt}{anthropic.AI_PROMPT}",
                max_tokens_to_sample=max_tokens or 1000,
                temperature=temperature,
                top_p=top_p,
                stop_sequences=stop or []
            )
            
            return response.completion
        except Exception as e:
            print(f"Anthropic API调用失败: {e}")
            # 重试一次
            try:
                await asyncio.sleep(1)  # 等待1秒后重试
                response = await self.client.completions.create(
                    model=self.model,
                    prompt=f"{anthropic.HUMAN_PROMPT} {prompt}{anthropic.AI_PROMPT}",
                    max_tokens_to_sample=max_tokens or 1000,
                    temperature=temperature,
                    top_p=top_p,
                    stop_sequences=stop or []
                )
                
                return response.completion
            except Exception as e:
                print(f"Anthropic API重试失败: {e}")
                return f"抱歉，我遇到了一些问题，无法生成回复。错误: {str(e)}"
    
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
        # 将消息列表转换为Anthropic格式
        prompt = ""
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if role == "user":
                prompt += f"{anthropic.HUMAN_PROMPT} {content} "
            elif role == "assistant":
                prompt += f"{anthropic.AI_PROMPT} {content} "
            elif role == "system":
                # 系统消息放在开头
                prompt = f"{content} " + prompt
        
        # 添加最后的AI提示
        prompt += anthropic.AI_PROMPT
        
        try:
            # 对于较新的Claude-3模型，使用messages API
            if self.model.startswith("claude-3"):
                # 转换为Claude-3消息格式
                claude_messages = []
                for message in messages:
                    role = message["role"]
                    content = message["content"]
                    
                    if role == "user":
                        claude_messages.append({"role": "user", "content": content})
                    elif role == "assistant":
                        claude_messages.append({"role": "assistant", "content": content})
                    elif role == "system":
                        claude_messages.append({"role": "system", "content": content})
                
                response = await self.client.messages.create(
                    model=self.model,
                    messages=claude_messages,
                    max_tokens=max_tokens or 1000,
                    temperature=temperature,
                    top_p=top_p,
                    stop_sequences=stop or []
                )
                
                return response.content[0].text
            else:
                # 对于旧版Claude模型，使用completions API
                response = await self.client.completions.create(
                    model=self.model,
                    prompt=prompt,
                    max_tokens_to_sample=max_tokens or 1000,
                    temperature=temperature,
                    top_p=top_p,
                    stop_sequences=stop or []
                )
                
                return response.completion
        except Exception as e:
            print(f"Anthropic API调用失败: {e}")
            # 重试一次
            try:
                await asyncio.sleep(1)  # 等待1秒后重试
                if self.model.startswith("claude-3"):
                    # 转换为Claude-3消息格式
                    claude_messages = []
                    for message in messages:
                        role = message["role"]
                        content = message["content"]
                        
                        if role == "user":
                            claude_messages.append({"role": "user", "content": content})
                        elif role == "assistant":
                            claude_messages.append({"role": "assistant", "content": content})
                        elif role == "system":
                            claude_messages.append({"role": "system", "content": content})
                    
                    response = await self.client.messages.create(
                        model=self.model,
                        messages=claude_messages,
                        max_tokens=max_tokens or 1000,
                        temperature=temperature,
                        top_p=top_p,
                        stop_sequences=stop or []
                    )
                    
                    return response.content[0].text
                else:
                    response = await self.client.completions.create(
                        model=self.model,
                        prompt=prompt,
                        max_tokens_to_sample=max_tokens or 1000,
                        temperature=temperature,
                        top_p=top_p,
                        stop_sequences=stop or []
                    )
                    
                    return response.completion
            except Exception as e:
                print(f"Anthropic API重试失败: {e}")
                return f"抱歉，我遇到了一些问题，无法生成回复。错误: {str(e)}"
    
    async def count_tokens(self, text: str) -> int:
        """
        计算文本的token数量
        
        Args:
            text: 要计算的文本
        
        Returns:
            token数量
        """
        # Anthropic没有官方的token计数器，使用tiktoken的cl100k_base作为近似
        return count_tokens(text, "cl100k_base")
    
    async def count_messages_tokens(self, messages: List[Dict[str, str]]) -> int:
        """
        计算消息列表的token数量
        
        Args:
            messages: 消息列表，格式为[{"role": "user", "content": "Hello"}, ...]
        
        Returns:
            token数量
        """
        # 将消息列表转换为文本
        text = ""
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if role == "user":
                text += f"Human: {content}\n"
            elif role == "assistant":
                text += f"Assistant: {content}\n"
            elif role == "system":
                text += f"System: {content}\n"
        
        # 计算token数量
        return await self.count_tokens(text)
    
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
        return self.context_sizes.get(self.model, 100000)  # 默认为100000