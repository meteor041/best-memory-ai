from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

from db.redis_client import RedisClient
from db.postgres_client import PostgresClient
from memory.long_term import LongTermMemory
from memory.vector_store import VectorStore
from llm.openai_api import OpenAIClient
from llm.deepseek_api import DeepSeekClient
from llm.openrouter_api import OpenRouterClient
from llm.anthropic_api import AnthropicClient

# 创建路由器
router = APIRouter()

# 创建客户端实例
redis_client = RedisClient()
postgres_client = PostgresClient()
vector_store = VectorStore()

# 创建LLM客户端
try:
    openai_client = OpenAIClient()
except Exception as e:
    print(f"OpenAI客户端初始化失败: {e}")
    openai_client = None

try:
    deepseek_client = DeepSeekClient()
except Exception as e:
    print(f"DeepSeek客户端初始化失败: {e}")
    deepseek_client = None

try:
    openrouter_client = OpenRouterClient()
except Exception as e:
    print(f"OpenRouter客户端初始化失败: {e}")
    openrouter_client = None

try:
    anthropic_client = AnthropicClient()
except Exception as e:
    print(f"Anthropic客户端初始化失败: {e}")
    anthropic_client = None

# 选择默认LLM客户端
default_llm_client = None
if deepseek_client:
    default_llm_client = deepseek_client
elif openrouter_client:
    default_llm_client = openrouter_client
elif openai_client:
    default_llm_client = openai_client
elif anthropic_client:
    default_llm_client = anthropic_client
else:
    raise ValueError("没有可用的LLM客户端")

# 创建长期记忆管理器
long_term_memory = LongTermMemory(postgres_client, redis_client, vector_store, default_llm_client)

# 模型
class MemoryCreate(BaseModel):
    user_id: str = Field(..., description="用户ID")
    content: str = Field(..., description="记忆内容")
    source: Optional[str] = Field(None, description="记忆来源")
    importance: float = Field(0.5, description="重要性评分", ge=0.0, le=1.0)
    category: Optional[str] = Field(None, description="记忆类别")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")
    tags: Optional[List[str]] = Field(None, description="标签列表")

class MemoryUpdate(BaseModel):
    content: Optional[str] = Field(None, description="记忆内容")
    importance: Optional[float] = Field(None, description="重要性评分", ge=0.0, le=1.0)
    category: Optional[str] = Field(None, description="记忆类别")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")
    is_active: Optional[bool] = Field(None, description="是否激活")
    tags: Optional[List[str]] = Field(None, description="标签列表")

class MemorySearch(BaseModel):
    user_id: str = Field(..., description="用户ID")
    query: str = Field(..., description="查询文本")
    category: Optional[str] = Field(None, description="记忆类别")
    limit: Optional[int] = Field(None, description="结果数量限制")

class Memory(BaseModel):
    id: str = Field(..., description="记忆ID")
    user_id: str = Field(..., description="用户ID")
    content: str = Field(..., description="记忆内容")
    source: Optional[str] = Field(None, description="记忆来源")
    importance: float = Field(..., description="重要性评分")
    category: Optional[str] = Field(None, description="记忆类别")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")
    created_at: str = Field(..., description="创建时间")
    updated_at: str = Field(..., description="更新时间")
    embedding_id: Optional[str] = Field(None, description="向量数据库中的ID")
    tags: List[str] = Field([], description="标签列表")
    relevance: Optional[float] = Field(None, description="相关性评分")

@router.post("/", response_model=Memory)
async def create_memory(memory: MemoryCreate):
    """
    创建记忆
    """
    result = await long_term_memory.create_memory(
        user_id=memory.user_id,
        content=memory.content,
        source=memory.source,
        importance=memory.importance,
        category=memory.category,
        metadata=memory.metadata,
        tags=memory.tags
    )
    
    return result

@router.get("/{memory_id}", response_model=Memory)
async def get_memory(memory_id: str):
    """
    获取记忆
    """
    memory = await long_term_memory.get_memory(memory_id)
    if not memory:
        raise HTTPException(status_code=404, detail="记忆不存在")
    
    return memory

@router.put("/{memory_id}", response_model=Memory)
async def update_memory(memory_id: str, memory: MemoryUpdate):
    """
    更新记忆
    """
    success = await long_term_memory.update_memory(
        memory_id=memory_id,
        content=memory.content,
        importance=memory.importance,
        category=memory.category,
        metadata=memory.metadata,
        is_active=memory.is_active,
        tags=memory.tags
    )
    
    if not success:
        raise HTTPException(status_code=404, detail="记忆不存在")
    
    updated_memory = await long_term_memory.get_memory(memory_id)
    return updated_memory

@router.delete("/{memory_id}")
async def delete_memory(memory_id: str, soft_delete: bool = True):
    """
    删除记忆
    """
    success = await long_term_memory.delete_memory(memory_id, soft_delete)
    if not success:
        raise HTTPException(status_code=404, detail="记忆不存在")
    
    return {"status": "success"}

@router.post("/search", response_model=List[Memory])
async def search_memories(search: MemorySearch):
    """
    搜索记忆
    """
    memories = await long_term_memory.search_memories(
        user_id=search.user_id,
        query=search.query,
        category=search.category,
        limit=search.limit
    )
    
    return memories

@router.get("/user/{user_id}", response_model=List[Memory])
async def get_user_memories(
    user_id: str,
    category: Optional[str] = None,
    active_only: bool = True
):
    """
    获取用户的所有记忆
    """
    memories = await postgres_client.get_user_memories(
        user_id=user_id,
        category=category,
        active_only=active_only
    )
    
    result = []
    for memory in memories:
        # 获取标签
        tags = await postgres_client.get_memory_tags(memory.id)
        
        result.append({
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
        })
    
    return result

@router.post("/save_memory")
async def save_memory_command(
    user_id: str,
    content: str,
    category: Optional[str] = None,
    tags: Optional[str] = None
):
    """
    保存记忆命令
    
    这是一个用户友好的API，用于通过命令保存记忆
    """
    tag_list = tags.split(",") if tags else []
    
    result = await long_term_memory.create_memory(
        user_id=user_id,
        content=content,
        source="user_command",
        importance=0.8,  # 用户手动保存的记忆默认较重要
        category=category,
        tags=tag_list
    )
    
    return {
        "status": "success",
        "message": "记忆已保存",
        "memory_id": result["id"]
    }

@router.post("/recall_memory")
async def recall_memory_command(
    user_id: str,
    query: str,
    category: Optional[str] = None,
    limit: int = 5
):
    """
    回忆记忆命令
    
    这是一个用户友好的API，用于通过命令回忆记忆
    """
    memories = await long_term_memory.search_memories(
        user_id=user_id,
        query=query,
        category=category,
        limit=limit
    )
    
    return {
        "status": "success",
        "message": f"找到 {len(memories)} 条相关记忆",
        "memories": memories
    }

@router.post("/forget_memory")
async def forget_memory_command(
    user_id: str,
    query: str,
    hard_delete: bool = False
):
    """
    忘记记忆命令
    
    这是一个用户友好的API，用于通过命令忘记记忆
    """
    # 搜索相关记忆
    memories = await long_term_memory.search_memories(
        user_id=user_id,
        query=query,
        limit=5
    )
    
    # 删除找到的记忆
    deleted_count = 0
    for memory in memories:
        success = await long_term_memory.delete_memory(
            memory_id=memory["id"],
            soft_delete=not hard_delete
        )
        if success:
            deleted_count += 1
    
    return {
        "status": "success",
        "message": f"已忘记 {deleted_count} 条记忆",
        "deleted_count": deleted_count
    }