import tiktoken
from typing import List, Dict, Any, Union

def count_tokens(text: str, model: str = "gpt-4") -> int:
    """
    计算文本的token数量
    
    Args:
        text: 要计算的文本
        model: 使用的模型名称
    
    Returns:
        token数量
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # 如果模型不在tiktoken的列表中，使用cl100k_base编码
        encoding = tiktoken.get_encoding("cl100k_base")
    
    return len(encoding.encode(text))

def count_messages_tokens(messages: List[Dict[str, str]], model: str = "gpt-4") -> int:
    """
    计算消息列表的token数量
    
    Args:
        messages: 消息列表，格式为[{"role": "user", "content": "Hello"}, ...]
        model: 使用的模型名称
    
    Returns:
        token数量
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    
    tokens_per_message = 3  # 每条消息的基础token数
    tokens_per_name = 1     # 如果有name字段，额外的token数
    
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    
    # 每次请求的基础token数
    num_tokens += 3
    
    return num_tokens

def truncate_text_to_token_limit(text: str, max_tokens: int, model: str = "gpt-4") -> str:
    """
    将文本截断到指定的token限制
    
    Args:
        text: 要截断的文本
        max_tokens: 最大token数
        model: 使用的模型名称
    
    Returns:
        截断后的文本
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    
    tokens = encoding.encode(text)
    
    if len(tokens) <= max_tokens:
        return text
    
    truncated_tokens = tokens[:max_tokens]
    return encoding.decode(truncated_tokens)