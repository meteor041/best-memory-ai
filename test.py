conversation_text = "用户：我叫小明，喜欢吃苹果和香蕉。截止明天要完成报告，优先级别高。"
prompt = f"""请从以下对话中提取关键信息，并以JSON格式返回。
        
对话内容：
{conversation_text}

请以以下JSON格式返回关键信息：
```json
{{
    "personal_info": {{
        "name": "用户名称（如果提到）",
        "preferences": ["偏好1", "偏好2", ...],
        "background": "背景信息"
    }},
    "tasks": [
        {{"description": "任务描述", "deadline": "截止日期（如果提到）", "priority": "优先级（如果提到）"}},
        ...
    ],
    "questions": ["用户提出的问题1", ...],
    "important_dates": [
        {{"event": "事件描述", "date": "日期"}}
    ]
}}
```

只返回JSON格式的信息，不要添加其他解释。如果某些字段没有相关信息，可以留空或省略。"""