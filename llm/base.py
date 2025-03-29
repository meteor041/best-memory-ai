from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union

class BaseLLM(ABC):
    """大型语言模型的基础接口"""
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    async def count_tokens(self, text: str) -> int:
        """
        计算文本的token数量
        
        Args:
            text: 要计算的文本
        
        Returns:
            token数量
        """
        pass
    
    @abstractmethod
    async def count_messages_tokens(self, messages: List[Dict[str, str]]) -> int:
        """
        计算消息列表的token数量
        
        Args:
            messages: 消息列表，格式为[{"role": "user", "content": "Hello"}, ...]
        
        Returns:
            token数量
        """
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """
        获取模型名称
        
        Returns:
            模型名称
        """
        pass
    
    @abstractmethod
    def get_model_context_size(self) -> int:
        """
        获取模型的上下文窗口大小
        
        Returns:
            上下文窗口大小（token数）
        """
        pass