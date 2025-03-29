# 具有长期记忆能力的AI对话系统

这是一个能够通过API与大型语言模型（如GPT-4、Claude等）连接的对话程序，重点增强其长期记忆能力，使其在多轮对话中保持上下文一致性，并能主动调用历史信息优化回答。

## 核心功能

### 1. 记忆功能

- **短期记忆**：实时缓存当前对话的上下文（如最近10轮对话内容）。
- **长期记忆**：
  - 支持用户自定义关键信息（如个人偏好、任务目标等）的存储与检索。
  - 自动总结对话历史，生成结构化摘要（JSON格式）。
  - 通过向量数据库（Chroma）实现语义搜索，快速关联历史信息。

### 2. API集成

- 兼容主流大模型API（OpenAI、Anthropic），支持动态切换模型。
- 设计统一的请求/响应格式，包含记忆元数据（如对话ID、时间戳、关键词）。

### 3. 用户交互

- 提供记忆管理接口（如：`/save_memory`、`/recall_memory`、`/forget_memory`）。
- 允许用户主动修正或删除错误记忆。

### 4. 性能优化

- 限制记忆存储的Token占用，避免API成本过高。
- 实现异步缓存机制，减少延迟。

## 技术栈

- **后端**：Python (FastAPI) + Redis（缓存）+ PostgreSQL（长期存储）
- **向量数据库**：Chroma
- **LLM API**：OpenAI API、Anthropic API

## 项目结构

```
.
├── api/                  # API路由
│   ├── __init__.py
│   ├── chat.py           # 聊天API
│   └── memory.py         # 记忆管理API
├── db/                   # 数据库模块
│   ├── __init__.py
│   ├── models.py         # 数据库模型
│   ├── postgres_client.py # PostgreSQL客户端
│   └── redis_client.py   # Redis客户端
├── llm/                  # LLM API集成
│   ├── __init__.py
│   ├── base.py           # 基础LLM接口
│   ├── openai_api.py     # OpenAI API集成
│   └── anthropic_api.py  # Anthropic API集成
├── memory/               # 记忆管理
│   ├── __init__.py
│   ├── short_term.py     # 短期记忆实现
│   ├── long_term.py      # 长期记忆实现
│   └── vector_store.py   # 向量数据库接口
├── utils/                # 工具模块
│   ├── __init__.py
│   ├── token_counter.py  # Token计数工具
│   └── summarizer.py     # 对话总结工具
├── .env.example          # 环境变量示例
├── main.py               # 主应用程序
├── example.py            # 示例脚本
└── requirements.txt      # 依赖项
```

## 安装与设置

1. 克隆仓库

```bash
git clone https://github.com/yourusername/memory-enhanced-ai-chat.git
cd memory-enhanced-ai-chat
```

2. 安装依赖

```bash
pip install -r requirements.txt
```

3. 配置环境变量

复制`.env.example`文件并重命名为`.env`，然后填写相应的配置：

```bash
cp .env.example .env
```

编辑`.env`文件，填写API密钥和数据库配置。

4. 设置数据库

确保PostgreSQL和Redis服务已启动，然后运行示例脚本创建必要的数据库表：

```bash
python example.py
```

5. 启动应用

```bash
uvicorn main:app --reload
```

## API使用示例

### 聊天API

```bash
curl -X POST "http://localhost:8000/api/chat/" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user-123",
    "message": "你好！我叫张三，我喜欢编程和旅行。",
    "model": "gpt-4",
    "use_memory": true
  }'
```

### 保存记忆

```bash
curl -X POST "http://localhost:8000/api/memory/save_memory" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user-123",
    "content": "我喜欢吃意大利面",
    "category": "preference",
    "tags": "food,preference,italian"
  }'
```

### 回忆记忆

```bash
curl -X POST "http://localhost:8000/api/memory/recall_memory" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user-123",
    "query": "我喜欢吃什么食物？",
    "limit": 5
  }'
```

## 示例脚本

项目包含一个`example.py`脚本，展示了系统的基本用法：

```bash
python example.py
```

这个脚本会：
1. 设置数据库并创建示例用户
2. 模拟多轮对话
3. 展示记忆的存储和检索
4. 展示AI如何利用长期记忆回答问题

## 扩展功能（待实现）

- **多模态记忆**：支持存储文本外的图片、语音等信息的元数据。
- **隐私保护**：提供端到端加密或本地存储选项。
- **主动记忆**：AI能主动提问以补全关键信息。

## 许可证

MIT