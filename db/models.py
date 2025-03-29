from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, JSON, Boolean, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

Base = declarative_base()

class User(Base):
    """用户模型"""
    __tablename__ = "users"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String(50), unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # 关系
    conversations = relationship("Conversation", back_populates="user", cascade="all, delete-orphan")
    memories = relationship("Memory", back_populates="user", cascade="all, delete-orphan")

class Conversation(Base):
    """对话模型"""
    __tablename__ = "conversations"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    title = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    summary = Column(JSON, nullable=True)  # 对话摘要，JSON格式
    
    # 关系
    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")

class Message(Base):
    """消息模型"""
    __tablename__ = "messages"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id = Column(String(36), ForeignKey("conversations.id"), nullable=False)
    role = Column(String(20), nullable=False)  # user, assistant, system
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    tokens = Column(Integer, nullable=True)  # 消息的token数量
    
    # 关系
    conversation = relationship("Conversation", back_populates="messages")

class Memory(Base):
    """记忆模型"""
    __tablename__ = "memories"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    content = Column(Text, nullable=False)  # 记忆内容
    source = Column(String(50), nullable=True)  # 记忆来源（如对话ID、用户手动添加等）
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    importance = Column(Float, default=0.5)  # 重要性评分，0-1
    category = Column(String(50), nullable=True)  # 记忆类别（如个人信息、偏好、任务等）
    meta_data = Column(JSON, nullable=True)  # 元数据，JSON格式
    is_active = Column(Boolean, default=True)  # 是否激活（用于软删除）
    embedding_id = Column(String(100), nullable=True)  # 向量数据库中的ID
    
    # 关系
    user = relationship("User", back_populates="memories")

class MemoryTag(Base):
    """记忆标签模型"""
    __tablename__ = "memory_tags"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    memory_id = Column(String(36), ForeignKey("memories.id"), nullable=False)
    tag = Column(String(50), nullable=False)
    
    # 索引和约束
    __table_args__ = (
        # 复合唯一约束，确保同一记忆不会有重复标签
        {"sqlite_autoincrement": True},
    )