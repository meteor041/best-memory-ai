import os
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import json

from db.redis_client import RedisClient
from db.postgres_client import PostgresClient
from utils.token_counter import count_messages_tokens

class ShortTermMemory:
    """短期记忆管理，用于缓存当前对话的上下文"""
    
    def __init__(self, redis_client: RedisClient, postgres_client: PostgresClient):
        """
        初始化短期记忆管理器
        
        Args:
            redis_client: Redis客户端
            postgres_client: PostgreSQL客户端
        """
        self.redis_client = redis_client
        self.postgres_client = postgres_client
        self.max_messages = int(os.getenv("MAX_SHORT_TERM_MEMORY", "10"))
    
    async def get_conversation_messages(
        self, 
        conversation_id: str, 
        limit: Optional[int] = None
    ) -> List[Dict[str, str]]:
        """
        获取对话的消息
        
        Args:
            conversation_id: 对话ID
            limit: 消息数量限制，如果为None则使用配置的最大消息数
        
        Returns:
            消息列表
        """
        # 首先尝试从Redis获取
        messages = await self.redis_client.get_conversation_messages(conversation_id)
        
        # 如果Redis中没有，则从PostgreSQL获取
        if not messages:
            db_messages = await self.postgres_client.get_conversation_messages(conversation_id)
            messages = [
                {
                    "role": msg.role,
                    "content": msg.content
                }
                for msg in db_messages
            ]
            
            # 如果有消息，则缓存到Redis
            if messages:
                await self.redis_client.set_conversation_messages(
                    conversation_id=conversation_id,
                    messages=messages,
                    expiry=86400  # 1天
                )
        
        # 应用消息数量限制
        max_msgs = limit or self.max_messages
        if len(messages) > max_msgs:
            messages = messages[-max_msgs:]
        
        return messages
    
    async def add_message(
        self, 
        conversation_id: str, 
        role: str, 
        content: str,
        model: str = "gpt-4"
    ) -> Dict[str, str]:
        """
        添加消息到对话
        
        Args:
            conversation_id: 对话ID
            role: 角色（user, assistant, system）
            content: 消息内容
            model: 用于计算token的模型名称
        
        Returns:
            添加的消息
        """
        # 创建消息对象
        message = {
            "role": role,
            "content": content
        }
        
        # 计算token数量
        tokens = count_messages_tokens([message], model)
        
        # 添加到PostgreSQL
        await self.postgres_client.create_message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            tokens=tokens
        )
        
        # 添加到Redis
        await self.redis_client.add_conversation_message(
            conversation_id=conversation_id,
            message=message,
            max_messages=self.max_messages,
            expiry=86400  # 1天
        )
        
        return message
    
    async def get_messages_with_token_limit(
        self, 
        conversation_id: str, 
        max_tokens: int,
        model: str = "gpt-4",
        include_system_message: bool = True
    ) -> List[Dict[str, str]]:
        """
        获取对话消息，并确保总token数不超过限制
        
        Args:
            conversation_id: 对话ID
            max_tokens: 最大token数
            model: 用于计算token的模型名称
            include_system_message: 是否包含系统消息
        
        Returns:
            消息列表
        """
        # 获取所有消息
        all_messages = await self.get_conversation_messages(conversation_id)
        
        # 如果没有消息，返回空列表
        if not all_messages:
            return []
        
        # 如果不包含系统消息，过滤掉系统消息
        if not include_system_message:
            all_messages = [msg for msg in all_messages if msg["role"] != "system"]
        
        # 计算总token数
        total_tokens = count_messages_tokens(all_messages, model)
        
        # 如果总token数小于等于限制，直接返回所有消息
        if total_tokens <= max_tokens:
            return all_messages
        
        # 否则，从最新的消息开始，逐步添加消息，直到达到token限制
        result_messages = []
        current_tokens = 0
        
        # 首先添加系统消息（如果有）
        system_messages = [msg for msg in all_messages if msg["role"] == "system"]
        if include_system_message and system_messages:
            result_messages.extend(system_messages)
            current_tokens += count_messages_tokens(system_messages, model)
        
        # 然后从最新的消息开始添加
        non_system_messages = [msg for msg in all_messages if msg["role"] != "system"]
        for msg in reversed(non_system_messages):
            msg_tokens = count_messages_tokens([msg], model)
            if current_tokens + msg_tokens <= max_tokens:
                result_messages.insert(0, msg)
                current_tokens += msg_tokens
            else:
                break
        
        return result_messages
    
    async def clear_conversation(self, conversation_id: str) -> bool:
        """
        清除对话的短期记忆
        
        Args:
            conversation_id: 对话ID
        
        Returns:
            是否成功
        """
        # 从Redis中删除
        key = f"conversation:{conversation_id}:messages"
        await self.redis_client.delete(key)
        
        return True