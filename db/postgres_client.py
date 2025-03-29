import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.future import select
from typing import List, Dict, Any, Optional, Type, TypeVar, Generic
from datetime import datetime

from db.models import Base, User, Conversation, Message, Memory, MemoryTag

# 定义泛型类型变量
T = TypeVar('T')

class PostgresClient:
    """PostgreSQL客户端，用于长期存储数据"""
    
    def __init__(self):
        """初始化PostgreSQL客户端"""
        self.postgres_host = os.getenv("POSTGRES_HOST", "localhost")
        self.postgres_port = os.getenv("POSTGRES_PORT", "5432")
        self.postgres_user = os.getenv("POSTGRES_USER", "postgres")
        self.postgres_password = os.getenv("POSTGRES_PASSWORD", "postgres")
        self.postgres_db = os.getenv("POSTGRES_DB", "memory_ai")
        
        # 创建数据库URL
        self.database_url = f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        
        # 创建异步引擎
        self.engine = create_async_engine(
            self.database_url,
            echo=False,  # 设置为True可以查看SQL语句
            future=True
        )
        
        # 创建异步会话工厂
        self.async_session = sessionmaker(
            self.engine,
            expire_on_commit=False,
            class_=AsyncSession
        )
    
    async def create_tables(self):
        """创建数据库表"""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    
    async def get_session(self) -> AsyncSession:
        """
        获取数据库会话
        
        Returns:
            异步会话
        """
        return self.async_session()
    
    async def close(self):
        """关闭数据库连接"""
        await self.engine.dispose()
    
    # 用户相关方法
    async def create_user(self, username: str) -> User:
        """
        创建用户
        
        Args:
            username: 用户名
        
        Returns:
            创建的用户对象
        """
        async with self.async_session() as session:
            user = User(username=username)
            session.add(user)
            await session.commit()
            await session.refresh(user)
            return user
    
    async def get_user_by_id(self, user_id: str) -> Optional[User]:
        """
        通过ID获取用户
        
        Args:
            user_id: 用户ID
        
        Returns:
            用户对象，如果不存在则返回None
        """
        async with self.async_session() as session:
            result = await session.execute(select(User).where(User.id == user_id))
            return result.scalars().first()
    
    async def get_user_by_username(self, username: str) -> Optional[User]:
        """
        通过用户名获取用户
        
        Args:
            username: 用户名
        
        Returns:
            用户对象，如果不存在则返回None
        """
        async with self.async_session() as session:
            result = await session.execute(select(User).where(User.username == username))
            return result.scalars().first()
    
    # 对话相关方法
    async def create_conversation(self, user_id: str, title: Optional[str] = None) -> Conversation:
        """
        创建对话
        
        Args:
            user_id: 用户ID
            title: 对话标题
        
        Returns:
            创建的对话对象
        """
        async with self.async_session() as session:
            conversation = Conversation(user_id=user_id, title=title)
            session.add(conversation)
            await session.commit()
            await session.refresh(conversation)
            return conversation
    
    async def get_conversation_by_id(self, conversation_id: str) -> Optional[Conversation]:
        """
        通过ID获取对话
        
        Args:
            conversation_id: 对话ID
        
        Returns:
            对话对象，如果不存在则返回None
        """
        async with self.async_session() as session:
            result = await session.execute(select(Conversation).where(Conversation.id == conversation_id))
            return result.scalars().first()
    
    async def get_user_conversations(self, user_id: str) -> List[Conversation]:
        """
        获取用户的所有对话
        
        Args:
            user_id: 用户ID
        
        Returns:
            对话列表
        """
        async with self.async_session() as session:
            result = await session.execute(
                select(Conversation)
                .where(Conversation.user_id == user_id)
                .order_by(Conversation.updated_at.desc())
            )
            return result.scalars().all()
    
    async def update_conversation_summary(self, conversation_id: str, summary: Dict[str, Any]) -> bool:
        """
        更新对话摘要
        
        Args:
            conversation_id: 对话ID
            summary: 摘要数据
        
        Returns:
            是否成功
        """
        async with self.async_session() as session:
            conversation = await session.get(Conversation, conversation_id)
            if conversation:
                conversation.summary = summary
                conversation.updated_at = datetime.utcnow()
                await session.commit()
                return True
            return False
    
    # 消息相关方法
    async def create_message(
        self, 
        conversation_id: str, 
        role: str, 
        content: str,
        tokens: Optional[int] = None
    ) -> Message:
        """
        创建消息
        
        Args:
            conversation_id: 对话ID
            role: 角色（user, assistant, system）
            content: 消息内容
            tokens: 消息的token数量
        
        Returns:
            创建的消息对象
        """
        async with self.async_session() as session:
            message = Message(
                conversation_id=conversation_id,
                role=role,
                content=content,
                tokens=tokens
            )
            session.add(message)
            
            # 更新对话的更新时间
            conversation = await session.get(Conversation, conversation_id)
            if conversation:
                conversation.updated_at = datetime.utcnow()
            
            await session.commit()
            await session.refresh(message)
            return message
    
    async def get_conversation_messages(self, conversation_id: str) -> List[Message]:
        """
        获取对话的所有消息
        
        Args:
            conversation_id: 对话ID
        
        Returns:
            消息列表
        """
        async with self.async_session() as session:
            result = await session.execute(
                select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.created_at)
            )
            return result.scalars().all()
    
    # 记忆相关方法
    async def create_memory(
        self,
        user_id: str,
        content: str,
        source: Optional[str] = None,
        importance: float = 0.5,
        category: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        embedding_id: Optional[str] = None
    ) -> Memory:
        """
        创建记忆
        
        Args:
            user_id: 用户ID
            content: 记忆内容
            source: 记忆来源
            importance: 重要性评分
            category: 记忆类别
            metadata: 元数据
            embedding_id: 向量数据库中的ID
        
        Returns:
            创建的记忆对象
        """
        async with self.async_session() as session:
            memory = Memory(
                user_id=user_id,
                content=content,
                source=source,
                importance=importance,
                category=category,
                metadata=metadata,
                embedding_id=embedding_id
            )
            session.add(memory)
            await session.commit()
            await session.refresh(memory)
            return memory
    
    async def get_memory_by_id(self, memory_id: str) -> Optional[Memory]:
        """
        通过ID获取记忆
        
        Args:
            memory_id: 记忆ID
        
        Returns:
            记忆对象，如果不存在则返回None
        """
        async with self.async_session() as session:
            result = await session.execute(select(Memory).where(Memory.id == memory_id))
            return result.scalars().first()
    
    async def get_user_memories(
        self, 
        user_id: str, 
        category: Optional[str] = None,
        active_only: bool = True
    ) -> List[Memory]:
        """
        获取用户的所有记忆
        
        Args:
            user_id: 用户ID
            category: 记忆类别
            active_only: 是否只返回激活的记忆
        
        Returns:
            记忆列表
        """
        async with self.async_session() as session:
            query = select(Memory).where(Memory.user_id == user_id)
            
            if category:
                query = query.where(Memory.category == category)
            
            if active_only:
                query = query.where(Memory.is_active == True)
            
            query = query.order_by(Memory.importance.desc(), Memory.updated_at.desc())
            
            result = await session.execute(query)
            return result.scalars().all()
    
    async def update_memory(
        self,
        memory_id: str,
        content: Optional[str] = None,
        importance: Optional[float] = None,
        category: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        is_active: Optional[bool] = None,
        embedding_id: Optional[str] = None
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
            embedding_id: 向量数据库中的ID
        
        Returns:
            是否成功
        """
        async with self.async_session() as session:
            memory = await session.get(Memory, memory_id)
            if memory:
                if content is not None:
                    memory.content = content
                if importance is not None:
                    memory.importance = importance
                if category is not None:
                    memory.category = category
                if metadata is not None:
                    memory.metadata = metadata
                if is_active is not None:
                    memory.is_active = is_active
                if embedding_id is not None:
                    memory.embedding_id = embedding_id
                
                memory.updated_at = datetime.utcnow()
                await session.commit()
                return True
            return False
    
    async def delete_memory(self, memory_id: str, soft_delete: bool = True) -> bool:
        """
        删除记忆
        
        Args:
            memory_id: 记忆ID
            soft_delete: 是否软删除（设置is_active为False）
        
        Returns:
            是否成功
        """
        async with self.async_session() as session:
            memory = await session.get(Memory, memory_id)
            if memory:
                if soft_delete:
                    memory.is_active = False
                    memory.updated_at = datetime.utcnow()
                    await session.commit()
                else:
                    await session.delete(memory)
                    await session.commit()
                return True
            return False
    
    # 记忆标签相关方法
    async def add_memory_tag(self, memory_id: str, tag: str) -> MemoryTag:
        """
        添加记忆标签
        
        Args:
            memory_id: 记忆ID
            tag: 标签
        
        Returns:
            创建的标签对象
        """
        async with self.async_session() as session:
            # 检查标签是否已存在
            result = await session.execute(
                select(MemoryTag)
                .where(MemoryTag.memory_id == memory_id)
                .where(MemoryTag.tag == tag)
            )
            existing_tag = result.scalars().first()
            
            if existing_tag:
                return existing_tag
            
            memory_tag = MemoryTag(memory_id=memory_id, tag=tag)
            session.add(memory_tag)
            await session.commit()
            await session.refresh(memory_tag)
            return memory_tag
    
    async def get_memory_tags(self, memory_id: str) -> List[str]:
        """
        获取记忆的所有标签
        
        Args:
            memory_id: 记忆ID
        
        Returns:
            标签列表
        """
        async with self.async_session() as session:
            result = await session.execute(
                select(MemoryTag.tag)
                .where(MemoryTag.memory_id == memory_id)
            )
            return result.scalars().all()
    
    async def remove_memory_tag(self, memory_id: str, tag: str) -> bool:
        """
        删除记忆标签
        
        Args:
            memory_id: 记忆ID
            tag: 标签
        
        Returns:
            是否成功
        """
        async with self.async_session() as session:
            result = await session.execute(
                select(MemoryTag)
                .where(MemoryTag.memory_id == memory_id)
                .where(MemoryTag.tag == tag)
            )
            memory_tag = result.scalars().first()
            
            if memory_tag:
                await session.delete(memory_tag)
                await session.commit()
                return True
            return False