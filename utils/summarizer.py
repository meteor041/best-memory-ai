from typing import List, Dict, Any, Optional
import json
from llm.base import BaseLLM

class ConversationSummarizer:
    """对话总结器，用于生成对话的结构化摘要"""
    
    def __init__(self, llm_client: BaseLLM):
        """
        初始化对话总结器
        
        Args:
            llm_client: LLM客户端实例
        """
        self.llm_client = llm_client
    
    async def summarize_conversation(
        self, 
        messages: List[Dict[str, str]], 
        existing_summary: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        总结对话内容，生成结构化摘要
        
        Args:
            messages: 消息列表，格式为[{"role": "user", "content": "Hello"}, ...]
            existing_summary: 已有的摘要，如果有的话
        
        Returns:
            结构化摘要，JSON格式
        """
        # 构建提示
        prompt = self._build_summary_prompt(messages, existing_summary)
        
        # 调用LLM生成摘要
        summary_text = await self.llm_client.generate_text(prompt)
        
        # 解析摘要为JSON格式
        try:
            summary_json = json.loads(summary_text)
            return summary_json
        except json.JSONDecodeError:
            # 如果解析失败，尝试提取JSON部分
            try:
                # 尝试找到JSON部分（通常在```json和```之间）
                import re
                json_match = re.search(r'```json\n(.*?)\n```', summary_text, re.DOTALL)
                if json_match:
                    summary_json = json.loads(json_match.group(1))
                    return summary_json
                
                # 如果没有找到，尝试直接解析可能的JSON部分
                json_match = re.search(r'(\{.*\})', summary_text, re.DOTALL)
                if json_match:
                    summary_json = json.loads(json_match.group(1))
                    return summary_json
                
                # 如果仍然失败，返回一个基本的摘要结构
                return {
                    "summary": summary_text,
                    "key_points": [],
                    "entities": [],
                    "topics": []
                }
            except Exception as e:
                # 如果所有尝试都失败，返回一个基本的摘要结构
                return {
                    "summary": summary_text,
                    "key_points": [],
                    "entities": [],
                    "topics": []
                }
    
    def _build_summary_prompt(
        self, 
        messages: List[Dict[str, str]], 
        existing_summary: Optional[str] = None
    ) -> str:
        """
        构建总结提示
        
        Args:
            messages: 消息列表
            existing_summary: 已有的摘要
        
        Returns:
            提示文本
        """
        conversation_text = "\n".join([
            f"{msg['role']}: {msg['content']}" for msg in messages
        ])
        
        prompt = f"""请总结以下对话，并以JSON格式返回结构化摘要。
        
对话内容：
{conversation_text}

"""
        
        if existing_summary:
            prompt += f"""
已有的摘要：
{existing_summary}

请基于已有摘要更新，保留重要信息并添加新的内容。
"""
        
        prompt += """
请以以下JSON格式返回摘要：
```json
{
    "summary": "对话的整体摘要",
    "key_points": ["要点1", "要点2", ...],
    "entities": [
        {"type": "person", "name": "名称", "attributes": {"属性1": "值1", ...}},
        {"type": "location", "name": "地点", "attributes": {}},
        ...
    ],
    "topics": ["主题1", "主题2", ...],
    "user_preferences": {"偏好1": "值1", ...},
    "action_items": ["待办事项1", ...]
}
```

只返回JSON格式的摘要，不要添加其他解释。"""
        
        return prompt

    async def extract_key_information(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        从对话中提取关键信息
        
        Args:
            messages: 消息列表
        
        Returns:
            关键信息，JSON格式
        """
        try:
            print("开始提取关键信息...")
            print(f"消息数量: {len(messages)}")
            
            conversation_text = "\n".join([
                f"{msg['role']}: {msg['content']}" for msg in messages
            ])
            
            print(f"对话文本长度: {len(conversation_text)}")
            
            # 使用简单字符串而不是f-string，看看是否解决问题
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
            
            print("提示构建完成，长度:", len(prompt))
            print("开始调用LLM...")
            
            # 调用LLM提取关键信息
            info_text = await self.llm_client.generate_text(prompt)
            
            print("LLM响应完成，长度:", len(info_text))
            print("开始解析JSON...")
            
            # 解析为JSON格式
            try:
                info_json = json.loads(info_text)
                return info_json
            except json.JSONDecodeError as json_err:
                print(f"JSON解析失败: {json_err}")
                # 如果解析失败，尝试提取JSON部分
                try:
                    import re
                    json_match = re.search(r'```json\n(.*?)\n```', info_text, re.DOTALL)
                    if json_match:
                        info_json = json.loads(json_match.group(1))
                        return info_json
                    
                    # 如果没有找到，尝试直接解析可能的JSON部分
                    json_match = re.search(r'(\{.*\})', info_text, re.DOTALL)
                    if json_match:
                        info_json = json.loads(json_match.group(1))
                        return info_json
                    
                    # 如果仍然失败，返回一个基本的结构
                    print("无法找到有效的JSON结构，返回默认结构")
                    return {
                        "personal_info": {},
                        "tasks": [],
                        "questions": [],
                        "important_dates": []
                    }
                except Exception as e:
                    print(f"JSON提取失败: {e}")
                    # 如果所有尝试都失败，返回一个基本的结构
                    return {
                        "personal_info": {},
                        "tasks": [],
                        "questions": [],
                        "important_dates": []
                    }
        except Exception as e:
            print(f"提取关键信息时发生错误: {e}")
            print(f"错误类型: {type(e)}")
            import traceback
            print(f"错误堆栈: {traceback.format_exc()}")
            return {
                "personal_info": {},
                "tasks": [],
                "questions": [],
                "important_dates": []
            }