import os
import json
import redis.asyncio as redis
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta

class RedisClient:
    """Redis客户端，用于缓存短期记忆和其他需要快速访问的数据"""
    
    def __init__(self):
        """初始化Redis客户端"""
        self.redis_host = os.getenv("REDIS_HOST", "localhost")
        self.redis_port = int(os.getenv("REDIS_PORT", 6379))
        self.redis_db = int(os.getenv("REDIS_DB", 0))
        self.redis_password = os.getenv("REDIS_PASSWORD", None)
        
        # 创建Redis连接池
        self.redis_pool = redis.ConnectionPool(
            host=self.redis_host,
            port=self.redis_port,
            db=self.redis_db,
            password=self.redis_password,
            decode_responses=True  # 自动将字节解码为字符串
        )
        
        # 创建Redis客户端
        self.redis = redis.Redis(connection_pool=self.redis_pool)
    
    async def ping(self) -> bool:
        """
        测试Redis连接
        
        Returns:
            连接是否成功
        """
        try:
            return await self.redis.ping()
        except Exception as e:
            print(f"Redis连接失败: {e}")
            return False
    
    async def close(self):
        """关闭Redis连接"""
        await self.redis.close()
    
    async def set_json(self, key: str, value: Dict[str, Any], expiry: Optional[int] = None) -> bool:
        """
        将JSON数据存储到Redis
        
        Args:
            key: Redis键
            value: 要存储的JSON数据
            expiry: 过期时间（秒），如果为None则不过期
        
        Returns:
            是否成功
        """
        try:
            json_data = json.dumps(value)
            if expiry:
                await self.redis.setex(key, expiry, json_data)
            else:
                await self.redis.set(key, json_data)
            return True
        except Exception as e:
            print(f"Redis设置JSON失败: {e}")
            return False
    
    async def get_json(self, key: str) -> Optional[Dict[str, Any]]:
        """
        从Redis获取JSON数据
        
        Args:
            key: Redis键
        
        Returns:
            JSON数据，如果不存在则返回None
        """
        try:
            data = await self.redis.get(key)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            print(f"Redis获取JSON失败: {e}")
            return None
    
    async def delete(self, key: str) -> bool:
        """
        删除Redis键
        
        Args:
            key: Redis键
        
        Returns:
            是否成功
        """
        try:
            await self.redis.delete(key)
            return True
        except Exception as e:
            print(f"Redis删除键失败: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """
        检查Redis键是否存在
        
        Args:
            key: Redis键
        
        Returns:
            是否存在
        """
        try:
            return await self.redis.exists(key) > 0
        except Exception as e:
            print(f"Redis检查键是否存在失败: {e}")
            return False
    
    async def set_conversation_messages(
        self, 
        conversation_id: str, 
        messages: List[Dict[str, str]], 
        expiry: int = 86400  # 默认1天
    ) -> bool:
        """
        存储对话消息到Redis
        
        Args:
            conversation_id: 对话ID
            messages: 消息列表
            expiry: 过期时间（秒）
        
        Returns:
            是否成功
        """
        key = f"conversation:{conversation_id}:messages"
        return await self.set_json(key, {"messages": messages}, expiry)
    
    async def get_conversation_messages(self, conversation_id: str) -> List[Dict[str, str]]:
        """
        从Redis获取对话消息
        
        Args:
            conversation_id: 对话ID
        
        Returns:
            消息列表，如果不存在则返回空列表
        """
        key = f"conversation:{conversation_id}:messages"
        data = await self.get_json(key)
        if data and "messages" in data:
            return data["messages"]
        return []
    
    async def add_conversation_message(
        self, 
        conversation_id: str, 
        message: Dict[str, str], 
        max_messages: int = 10,
        expiry: int = 86400  # 默认1天
    ) -> bool:
        """
        向对话添加新消息，并保持最大消息数量
        
        Args:
            conversation_id: 对话ID
            message: 新消息
            max_messages: 最大消息数量
            expiry: 过期时间（秒）
        
        Returns:
            是否成功
        """
        messages = await self.get_conversation_messages(conversation_id)
        messages.append(message)
        
        # 如果超过最大消息数量，删除最早的消息
        if len(messages) > max_messages:
            messages = messages[-max_messages:]
        
        return await self.set_conversation_messages(conversation_id, messages, expiry)
    
    async def set_user_memory_cache(
        self, 
        user_id: str, 
        memory_data: Dict[str, Any], 
        expiry: int = 3600  # 默认1小时
    ) -> bool:
        """
        缓存用户的记忆数据
        
        Args:
            user_id: 用户ID
            memory_data: 记忆数据
            expiry: 过期时间（秒）
        
        Returns:
            是否成功
        """
        key = f"user:{user_id}:memory_cache"
        return await self.set_json(key, memory_data, expiry)
    
    async def get_user_memory_cache(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        获取用户的记忆缓存
        
        Args:
            user_id: 用户ID
        
        Returns:
            记忆数据，如果不存在则返回None
        """
        key = f"user:{user_id}:memory_cache"
        return await self.get_json(key)
    
    async def invalidate_user_memory_cache(self, user_id: str) -> bool:
        """
        使用户的记忆缓存失效
        
        Args:
            user_id: 用户ID
        
        Returns:
            是否成功
        """
        key = f"user:{user_id}:memory_cache"
        return await self.delete(key)