import os
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime
import json
import asyncio

from db.postgres_client import PostgresClient
from db.redis_client import RedisClient
from memory.vector_store import VectorStore
from utils.summarizer import ConversationSummarizer
from llm.base import BaseLLM

class LongTermMemory:
    """长期记忆管理，用于存储和检索用户的长期记忆"""
    
    def __init__(
        self, 
        postgres_client: PostgresClient, 
        redis_client: RedisClient,
        vector_store: VectorStore,
        llm_client: BaseLLM
    ):
        """
        初始化长期记忆管理器
        
        Args:
            postgres_client: PostgreSQL客户端
            redis_client: Redis客户端
            vector_store: 向量数据库
            llm_client: LLM客户端
        """
        self.postgres_client = postgres_client
        self.redis_client = redis_client
        self.vector_store = vector_store
        self.llm_client = llm_client
        self.summarizer = ConversationSummarizer(llm_client)
        
        self.memory_retrieval_limit = int(os.getenv("MEMORY_RETRIEVAL_LIMIT", "5"))
    
    async def create_memory(
        self,
        user_id: str,
        content: str,
        source: Optional[str] = None,
        importance: float = 0.5,
        category: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        创建记忆
        
        Args:
            user_id: 用户ID
            content: 记忆内容
            source: 记忆来源
            importance: 重要性评分
            category: 记忆类别
            metadata: 元数据
            tags: 标签列表
        
        Returns:
            创建的记忆
        """
        # 添加到向量数据库
        vector_metadata = {
            "user_id": user_id,
            "source": source,
            "category": category,
            "importance": importance,
            "created_at": datetime.utcnow().isoformat()
        }
        
        if metadata:
            vector_metadata.update(metadata)
        
        embedding_id = await self.vector_store.add_memory(
            text=content,
            metadata=vector_metadata
        )
        
        # 添加到PostgreSQL
        memory = await self.postgres_client.create_memory(
            user_id=user_id,
            content=content,
            source=source,
            importance=importance,
            category=category,
            metadata=metadata,
            embedding_id=embedding_id
        )
        
        # 添加标签
        if tags:
            for tag in tags:
                await self.postgres_client.add_memory_tag(memory.id, tag)
        
        # 使缓存失效
        await self.redis_client.invalidate_user_memory_cache(user_id)
        
        # 返回记忆对象
        return {
            "id": memory.id,
            "user_id": memory.user_id,
            "content": memory.content,
            "source": memory.source,
            "importance": memory.importance,
            "category": memory.category,
            "metadata": memory.metadata,
            "created_at": memory.created_at.isoformat(),
            "updated_at": memory.updated_at.isoformat(),
            "embedding_id": memory.embedding_id,
            "tags": tags or []
        }
    
    async def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        获取记忆
        
        Args:
            memory_id: 记忆ID
        
        Returns:
            记忆对象，如果不存在则返回None
        """
        memory = await self.postgres_client.get_memory_by_id(memory_id)
        if not memory:
            return None
        
        # 获取标签
        tags = await self.postgres_client.get_memory_tags(memory_id)
        
        return {
            "id": memory.id,
            "user_id": memory.user_id,
            "content": memory.content,
            "source": memory.source,
            "importance": memory.importance,
            "category": memory.category,
            "metadata": memory.metadata,
            "created_at": memory.created_at.isoformat(),
            "updated_at": memory.updated_at.isoformat(),
            "embedding_id": memory.embedding_id,
            "tags": tags
        }
    
    async def update_memory(
        self,
        memory_id: str,
        content: Optional[str] = None,
        importance: Optional[float] = None,
        category: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        is_active: Optional[bool] = None,
        tags: Optional[List[str]] = None
    ) -> bool:
        """
        更新记忆
        
        Args:
            memory_id: 记忆ID
            content: 记忆内容
            importance: 重要性评分
            category: 记忆类别
            metadata: 元数据
            is_active: 是否激活
            tags: 标签列表
        
        Returns:
            是否成功
        """
        # 获取现有记忆
        memory = await self.postgres_client.get_memory_by_id(memory_id)
        if not memory:
            return False
        
        # 更新向量数据库
        if content is not None or metadata is not None:
            vector_update = {}
            
            if content is not None:
                vector_update["text"] = content
            
            if metadata is not None:
                vector_metadata = {
                    "user_id": memory.user_id,
                    "source": memory.source,
                    "category": category if category is not None else memory.category,
                    "importance": importance if importance is not None else memory.importance,
                    "updated_at": datetime.utcnow().isoformat()
                }
                
                if metadata:
                    vector_metadata.update(metadata)
                
                vector_update["metadata"] = vector_metadata
            
            if vector_update:
                await self.vector_store.update_memory(
                    id=memory.embedding_id,
                    **vector_update
                )
        
        # 更新PostgreSQL
        success = await self.postgres_client.update_memory(
            memory_id=memory_id,
            content=content,
            importance=importance,
            category=category,
            metadata=metadata,
            is_active=is_active
        )
        
        # 更新标签
        if tags is not None:
            # 获取现有标签
            existing_tags = await self.postgres_client.get_memory_tags(memory_id)
            
            # 删除不在新标签列表中的标签
            for tag in existing_tags:
                if tag not in tags:
                    await self.postgres_client.remove_memory_tag(memory_id, tag)
            
            # 添加新标签
            for tag in tags:
                if tag not in existing_tags:
                    await self.postgres_client.add_memory_tag(memory_id, tag)
        
        # 使缓存失效
        await self.redis_client.invalidate_user_memory_cache(memory.user_id)
        
        return success
    
    async def delete_memory(self, memory_id: str, soft_delete: bool = True) -> bool:
        """
        删除记忆
        
        Args:
            memory_id: 记忆ID
            soft_delete: 是否软删除
        
        Returns:
            是否成功
        """
        # 获取记忆
        memory = await self.postgres_client.get_memory_by_id(memory_id)
        if not memory:
            return False
        
        # 如果是硬删除，从向量数据库中删除
        if not soft_delete and memory.embedding_id:
            await self.vector_store.delete_memory(memory.embedding_id)
        
        # 从PostgreSQL中删除
        success = await self.postgres_client.delete_memory(memory_id, soft_delete)
        
        # 使缓存失效
        await self.redis_client.invalidate_user_memory_cache(memory.user_id)
        
        return success
    
    async def search_memories(
        self,
        user_id: str,
        query: str,
        category: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        搜索记忆
        
        Args:
            user_id: 用户ID
            query: 查询文本
            category: 记忆类别
            limit: 结果数量限制
        
        Returns:
            记忆列表
        """
        # 构建过滤条件
        filter = {"user_id": user_id}
        if category:
            filter["category"] = category
        
        # 从向量数据库中搜索
        vector_results = await self.vector_store.search_memories(
            query=query,
            filter=filter,
            limit=limit or self.memory_retrieval_limit
        )
        
        # 获取完整的记忆对象
        results = []
        for vector_result in vector_results:
            # 通过embedding_id查找PostgreSQL中的记忆
            memory = None
            # 获取用户记忆列表（这是一个普通列表，不是异步迭代器）
            memories = await self.postgres_client.get_user_memories(user_id)
            # 使用普通for循环而不是async for
            for mem in memories:
                if mem.embedding_id == vector_result["id"]:
                    memory = mem
                    break
            
            if memory and memory.is_active:
                # 获取标签
                tags = await self.postgres_client.get_memory_tags(memory.id)
                
                results.append({
                    "id": memory.id,
                    "user_id": memory.user_id,
                    "content": memory.content,
                    "source": memory.source,
                    "importance": memory.importance,
                    "category": memory.category,
                    "metadata": memory.metadata,
                    "created_at": memory.created_at.isoformat(),
                    "updated_at": memory.updated_at.isoformat(),
                    "embedding_id": memory.embedding_id,
                    "tags": tags,
                    "relevance": 1.0 - (vector_result["distance"] if vector_result.get("distance") else 0)
                })
        
        # 按相关性排序
        results.sort(key=lambda x: x.get("relevance", 0), reverse=True)
        
        return results
    
    async def get_relevant_memories(
        self,
        user_id: str,
        context: str,
        category: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        获取与上下文相关的记忆
        
        Args:
            user_id: 用户ID
            context: 上下文文本
            category: 记忆类别
            limit: 结果数量限制
        
        Returns:
            记忆列表
        """
        return await self.search_memories(
            user_id=user_id,
            query=context,
            category=category,
            limit=limit
        )
    
    async def summarize_conversation(
        self,
        conversation_id: str,
        user_id: str,
        messages: List[Dict[str, str]],
        existing_summary: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        总结对话并存储为长期记忆
        
        Args:
            conversation_id: 对话ID
            user_id: 用户ID
            messages: 消息列表
            existing_summary: 已有的摘要
        
        Returns:
            摘要
        """
        # 生成摘要
        summary = await self.summarizer.summarize_conversation(messages, existing_summary)
        
        # 更新对话摘要
        await self.postgres_client.update_conversation_summary(conversation_id, summary)
        
        # 提取关键信息
        key_info = await self.summarizer.extract_key_information(messages)
        
        # 存储关键信息为长期记忆
        if key_info.get("personal_info"):
            personal_info = key_info["personal_info"]
            if personal_info.get("preferences"):
                for pref in personal_info["preferences"]:
                    await self.create_memory(
                        user_id=user_id,
                        content=f"用户偏好: {pref}",
                        source=f"conversation:{conversation_id}",
                        importance=0.8,
                        category="preference",
                        metadata={"conversation_id": conversation_id},
                        tags=["preference"]
                    )
            
            if personal_info.get("background"):
                await self.create_memory(
                    user_id=user_id,
                    content=f"用户背景: {personal_info['background']}",
                    source=f"conversation:{conversation_id}",
                    importance=0.7,
                    category="background",
                    metadata={"conversation_id": conversation_id},
                    tags=["background"]
                )
        
        if key_info.get("tasks"):
            for task in key_info["tasks"]:
                await self.create_memory(
                    user_id=user_id,
                    content=f"任务: {task['description']}",
                    source=f"conversation:{conversation_id}",
                    importance=0.9,
                    category="task",
                    metadata={
                        "conversation_id": conversation_id,
                        "deadline": task.get("deadline"),
                        "priority": task.get("priority")
                    },
                    tags=["task"]
                )
        
        if key_info.get("important_dates"):
            for date_info in key_info["important_dates"]:
                await self.create_memory(
                    user_id=user_id,
                    content=f"重要日期: {date_info['event']} - {date_info['date']}",
                    source=f"conversation:{conversation_id}",
                    importance=0.8,
                    category="date",
                    metadata={
                        "conversation_id": conversation_id,
                        "event": date_info["event"],
                        "date": date_info["date"]
                    },
                    tags=["date"]
                )
        
        return summary
    
    async def format_memories_for_context(
        self,
        memories: List[Dict[str, Any]],
        max_tokens: int = 1000
    ) -> str:
        """
        将记忆格式化为上下文字符串
        
        Args:
            memories: 记忆列表
            max_tokens: 最大token数
        
        Returns:
            格式化的上下文字符串
        """
        if not memories:
            return ""
        
        # 按相关性排序
        memories.sort(key=lambda x: x.get("relevance", 0) if x.get("relevance") is not None else 0, reverse=True)
        
        context_parts = ["以下是与当前对话相关的记忆:"]
        
        total_tokens = 0
        token_limit_reached = False
        
        for memory in memories:
            memory_text = f"- {memory['content']}"
            if memory.get("category"):
                memory_text += f" [类别: {memory['category']}]"
            if memory.get("tags"):
                memory_text += f" [标签: {', '.join(memory['tags'])}]"
            
            # 估算token数量
            memory_tokens = await self.llm_client.count_tokens(memory_text)
            
            if total_tokens + memory_tokens > max_tokens:
                token_limit_reached = True
                break
            
            context_parts.append(memory_text)
            total_tokens += memory_tokens
        
        if token_limit_reached:
            context_parts.append("(由于token限制，部分记忆未显示)")
        
        return "\n".join(context_parts)