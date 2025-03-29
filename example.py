import asyncio
import os
import json
from dotenv import load_dotenv
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any, List, Optional

# 加载环境变量
load_dotenv()

# 导入必要的模块
from db.postgres_client import PostgresClient
from db.redis_client import RedisClient
from memory.vector_store import VectorStore
from memory.short_term import ShortTermMemory
from memory.long_term import LongTermMemory
from llm.openai_api import OpenAIClient

# 示例用户ID
USER_ID = "user-123"

async def setup_database():
    """设置数据库"""
    postgres_client = PostgresClient()
    
    # 创建数据库表
    await postgres_client.create_tables()
    
    # 创建示例用户
    user = await postgres_client.get_user_by_username("example_user")
    if not user:
        user = await postgres_client.create_user("example_user")
        print(f"创建用户: {user.id}")
    else:
        print(f"用户已存在: {user.id}")
    
    return user.id

async def chat_example(user_id):
    """聊天示例"""
    # 创建客户端
    redis_client = RedisClient()
    postgres_client = PostgresClient()
    vector_store = VectorStore()
    llm_client = OpenAIClient()
    
    # 创建记忆管理器
    short_term = ShortTermMemory(redis_client, postgres_client)
    long_term = LongTermMemory(postgres_client, redis_client, vector_store, llm_client)
    
    # 创建对话
    conversation = await postgres_client.create_conversation(user_id, "示例对话")
    conversation_id = conversation.id
    print(f"创建对话: {conversation_id}")
    
    # 添加系统消息
    system_message = "你是一个具有长期记忆能力的AI助手，能够记住用户的偏好和重要信息。"
    await short_term.add_message(
        conversation_id=conversation_id,
        role="system",
        content=system_message
    )
    
    # 模拟对话
    messages = [
        {"role": "system", "content": system_message},
    ]
    
    # 第一轮对话
    user_message = "你好！我叫张三，我喜欢编程和旅行。"
    print(f"\n用户: {user_message}")
    
    messages.append({"role": "user", "content": user_message})
    await short_term.add_message(
        conversation_id=conversation_id,
        role="user",
        content=user_message
    )
    
    # 获取AI回复
    ai_response = await llm_client.generate_chat_response(messages)
    print(f"AI: {ai_response}")
    
    messages.append({"role": "assistant", "content": ai_response})
    await short_term.add_message(
        conversation_id=conversation_id,
        role="assistant",
        content=ai_response
    )
    
    # 总结对话并存储为长期记忆
    summary = await long_term.summarize_conversation(
        conversation_id=conversation_id,
        user_id=user_id,
        messages=messages
    )
    print(f"\n对话摘要: {json.dumps(summary, ensure_ascii=False, indent=2)}")
    
    # 第二轮对话
    user_message = "我下个月计划去日本旅行，你有什么建议吗？"
    print(f"\n用户: {user_message}")
    
    messages.append({"role": "user", "content": user_message})
    await short_term.add_message(
        conversation_id=conversation_id,
        role="user",
        content=user_message
    )
    
    # 获取相关记忆
    memories = await long_term.get_relevant_memories(
        user_id=user_id,
        context=user_message
    )
    
    if memories:
        memory_context = await long_term.format_memories_for_context(memories)
        print(f"\n相关记忆: {memory_context}")
        
        # 添加记忆上下文
        messages.append({"role": "system", "content": memory_context})
    
    # 获取AI回复
    ai_response = await llm_client.generate_chat_response(messages)
    print(f"AI: {ai_response}")
    
    messages.append({"role": "assistant", "content": ai_response})
    await short_term.add_message(
        conversation_id=conversation_id,
        role="assistant",
        content=ai_response
    )
    
    # 更新对话摘要
    summary = await long_term.summarize_conversation(
        conversation_id=conversation_id,
        user_id=user_id,
        messages=messages,
        existing_summary=summary
    )
    
    # 手动添加记忆
    memory = await long_term.create_memory(
        user_id=user_id,
        content="用户张三计划在下个月去日本旅行",
        source=f"conversation:{conversation_id}",
        importance=0.8,
        category="travel_plan",
        tags=["travel", "japan", "plan"]
    )
    print(f"\n手动添加记忆: {memory['id']}")
    
    # 第三轮对话（间隔一段时间后）
    print("\n--- 模拟一段时间后的对话 ---")
    
    user_message = "你还记得我之前说过要去哪里旅行吗？"
    print(f"\n用户: {user_message}")
    
    # 重新获取对话历史（模拟新会话）
    messages = [{"role": "system", "content": system_message}]
    
    messages.append({"role": "user", "content": user_message})
    await short_term.add_message(
        conversation_id=conversation_id,
        role="user",
        content=user_message
    )
    
    # 获取相关记忆
    memories = await long_term.get_relevant_memories(
        user_id=user_id,
        context=user_message
    )
    
    if memories:
        memory_context = await long_term.format_memories_for_context(memories)
        print(f"\n相关记忆: {memory_context}")
        
        # 添加记忆上下文
        messages.append({"role": "system", "content": memory_context})
    
    # 获取AI回复
    ai_response = await llm_client.generate_chat_response(messages)
    print(f"AI: {ai_response}")
    
    # 搜索记忆
    print("\n--- 搜索记忆 ---")
    search_results = await long_term.search_memories(
        user_id=user_id,
        query="日本旅行"
    )
    
    print(f"搜索结果: {json.dumps([{'content': mem['content'],'relevance': mem.get('relevance', 0)} for mem in search_results], ensure_ascii=False, indent=2)}")
    
    # 关闭连接
    await redis_client.close()
    await postgres_client.close()

async def main():
    """主函数"""
    # 设置数据库
    user_id = await setup_database()
    
    # 运行聊天示例
    await chat_example(user_id)

if __name__ == "__main__":
    asyncio.run(main())