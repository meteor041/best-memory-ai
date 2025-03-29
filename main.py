import os
import uvicorn
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 导入API路由
from api.chat import router as chat_router
from api.memory import router as memory_router

# 创建FastAPI应用
app = FastAPI(
    title="Memory-Enhanced AI Chat System",
    description="一个具有长期记忆能力的AI对话系统",
    version="1.0.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(chat_router, prefix="/api/chat", tags=["chat"])
app.include_router(memory_router, prefix="/api/memory", tags=["memory"])

@app.get("/")
async def root():
    """健康检查端点"""
    return {"status": "ok", "message": "Memory-Enhanced AI Chat System is running"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8083, reload=True)