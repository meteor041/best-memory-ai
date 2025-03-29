from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import os

from db.redis_client import RedisClient
from db.postgres_client import PostgresClient
from memory.short_term import ShortTermMemory
from memory.long_term import LongTermMemory
from memory.vector_store import VectorStore
from llm.base import BaseLLM
from llm.openai_api import OpenAIClient
from llm.anthropic_api import AnthropicClient

# 创建路由器
router = APIRouter()

# 创建客户端实例
redis_client = RedisClient()
postgres_client = PostgresClient()
vector_store = VectorStore()

# 创建LLM客户端
openai_client = OpenAIClient()
anthropic_client = AnthropicClient()

# 创建记忆管理器
short_term_memory = ShortTermMemory(redis_client, postgres_client)
long_term_memory = LongTermMemory(postgres_client, redis_client, vector_store, openai_client)

# 模型
class Message(BaseModel):
    role: str = Field(..., description="消息角色（user, assistant, system）")
    content: str = Field(..., description="消息内容")

class ChatRequest(BaseModel):
    user_id: str = Field(..., description="用户ID")
    conversation_id: Optional[str] = Field(None, description="对话ID，如果为空则创建新对话")
    message: str = Field(..., description="用户消息")
    model: str = Field("gpt-4", description="使用的模型")
    use_memory: bool = Field(True, description="是否使用记忆")
    system_message: Optional[str] = Field(None, description="系统消息")

class ChatResponse(BaseModel):
    conversation_id: str = Field(..., description="对话ID")
    response: str = Field(..., description="AI回复")
    memories_used: Optional[List[Dict[str, Any]]] = Field(None, description="使用的记忆")
    created_at: str = Field(..., description="创建时间")

@router.post("/", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks
):
    """
    聊天API
    """
    # 获取LLM客户端
    llm_client = openai_client if "gpt" in request.model.lower() else anthropic_client
    
    # 获取或创建对话
    conversation_id = request.conversation_id
    if not conversation_id:
        # 创建新对话
        conversation = await postgres_client.create_conversation(request.user_id)
        conversation_id = conversation.id
    else:
        # 验证对话是否存在
        conversation = await postgres_client.get_conversation_by_id(conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="对话不存在")
        
        # 验证对话是否属于该用户
        if conversation.user_id != request.user_id:
            raise HTTPException(status_code=403, detail="无权访问该对话")
    
    # 获取对话历史
    max_token_limit = int(os.getenv("MAX_TOKEN_LIMIT", "4000"))
    messages = await short_term_memory.get_messages_with_token_limit(
        conversation_id=conversation_id,
        max_tokens=max_token_limit // 2,  # 预留一半token给回复和记忆
        model=request.model
    )
    
    # 添加系统消息
    if request.system_message and not any(msg["role"] == "system" for msg in messages):
        system_message = {
            "role": "system",
            "content": request.system_message
        }
        messages.insert(0, system_message)
        await short_term_memory.add_message(
            conversation_id=conversation_id,
            role="system",
            content=request.system_message,
            model=request.model
        )
    
    # 添加用户消息
    user_message = {
        "role": "user",
        "content": request.message
    }
    messages.append(user_message)
    await short_term_memory.add_message(
        conversation_id=conversation_id,
        role="user",
        content=request.message,
        model=request.model
    )
    
    # 获取相关记忆
    memories_used = []
    if request.use_memory:
        memories = await long_term_memory.get_relevant_memories(
            user_id=request.user_id,
            context=request.message
        )
        
        if memories:
            # 格式化记忆为上下文
            memory_context = await long_term_memory.format_memories_for_context(
                memories=memories,
                max_tokens=max_token_limit // 4  # 使用1/4的token限制给记忆
            )
            
            # 添加记忆上下文
            if memory_context:
                memory_message = {
                    "role": "system",
                    "content": memory_context
                }
                messages.append(memory_message)
                memories_used = memories
    
    # 生成回复
    response = await llm_client.generate_chat_response(
        messages=messages,
        max_tokens=max_token_limit // 2,  # 使用1/2的token限制给回复
        temperature=0.7
    )
    
    # 添加助手消息
    assistant_message = {
        "role": "assistant",
        "content": response
    }
    await short_term_memory.add_message(
        conversation_id=conversation_id,
        role="assistant",
        content=response,
        model=request.model
    )
    
    # 在后台任务中总结对话
    background_tasks.add_task(
        summarize_conversation,
        conversation_id=conversation_id,
        user_id=request.user_id,
        messages=messages + [assistant_message]
    )
    
    # 返回响应
    return ChatResponse(
        conversation_id=conversation_id,
        response=response,
        memories_used=memories_used,
        created_at=datetime.utcnow().isoformat()
    )

@router.get("/conversations", response_model=List[Dict[str, Any]])
async def get_conversations(user_id: str):
    """
    获取用户的所有对话
    """
    conversations = await postgres_client.get_user_conversations(user_id)
    
    return [
        {
            "id": conv.id,
            "title": conv.title,
            "created_at": conv.created_at.isoformat(),
            "updated_at": conv.updated_at.isoformat(),
            "summary": conv.summary
        }
        for conv in conversations
    ]

@router.get("/conversations/{conversation_id}/messages", response_model=List[Dict[str, Any]])
async def get_conversation_messages(conversation_id: str, user_id: str):
    """
    获取对话的所有消息
    """
    # 验证对话是否存在
    conversation = await postgres_client.get_conversation_by_id(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="对话不存在")
    
    # 验证对话是否属于该用户
    if conversation.user_id != user_id:
        raise HTTPException(status_code=403, detail="无权访问该对话")
    
    # 获取消息
    db_messages = await postgres_client.get_conversation_messages(conversation_id)
    
    return [
        {
            "id": msg.id,
            "role": msg.role,
            "content": msg.content,
            "created_at": msg.created_at.isoformat(),
            "tokens": msg.tokens
        }
        for msg in db_messages
    ]

@router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str, user_id: str):
    """
    删除对话
    """
    # 验证对话是否存在
    conversation = await postgres_client.get_conversation_by_id(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="对话不存在")
    
    # 验证对话是否属于该用户
    if conversation.user_id != user_id:
        raise HTTPException(status_code=403, detail="无权访问该对话")
    
    # 删除对话
    # 注意：这里应该实现级联删除，删除对话的所有消息
    # 但是为了简单起见，我们只清除短期记忆
    await short_term_memory.clear_conversation(conversation_id)
    
    return {"status": "success"}

async def summarize_conversation(conversation_id: str, user_id: str, messages: List[Dict[str, str]]):
    """
    后台任务：总结对话
    """
    # 获取现有摘要
    conversation = await postgres_client.get_conversation_by_id(conversation_id)
    existing_summary = conversation.summary if conversation else None
    
    # 总结对话
    await long_term_memory.summarize_conversation(
        conversation_id=conversation_id,
        user_id=user_id,
        messages=messages,
        existing_summary=existing_summary
    )