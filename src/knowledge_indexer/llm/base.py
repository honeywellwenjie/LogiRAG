"""
LLM 抽象基类
定义统一的 LLM 接口，方便扩展支持不同的 LLM 后端
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import re


def clean_llm_output(content: str) -> str:
    """
    清理 LLM 输出内容
    1. 处理 deepseek-r1 等推理模型的 <think>...</think> 格式
    2. 移除常见的多余前缀（如 "摘要："、"Summary:" 等）
    3. 清理 markdown 格式残留
    
    Args:
        content: LLM 原始输出内容
        
    Returns:
        str: 清理后的实际答案
    """
    if not content:
        return ""
    
    # 1. 移除 <think>...</think> 块
    cleaned = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
    
    # 2. 移除常见的多余前缀（中英文）
    # 匹配开头的 "摘要："、"**摘要：**"、"Summary:"、"描述："等
    prefix_patterns = [
        r'^\*{0,2}摘要[:：]\*{0,2}\s*',
        r'^\*{0,2}Summary[:：]\*{0,2}\s*',
        r'^\*{0,2}描述[:：]\*{0,2}\s*',
        r'^\*{0,2}Description[:：]\*{0,2}\s*',
        r'^[:：]\s*',  # 单独的冒号开头
    ]
    for pattern in prefix_patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.MULTILINE)
    
    # 3. 移除开头的换行符
    cleaned = cleaned.lstrip('\n')
    
    return cleaned.strip()


# 保留旧函数名作为别名，保持向后兼容
def extract_answer_from_reasoning(content: str) -> str:
    """别名函数，调用 clean_llm_output"""
    return clean_llm_output(content)


@dataclass
class LLMResponse:
    """LLM 响应"""
    content: str
    model: str = ""
    usage: Dict[str, int] = None
    raw_response: Any = None
    
    def __post_init__(self):
        if self.usage is None:
            self.usage = {}


@dataclass
class Message:
    """聊天消息"""
    role: str  # system, user, assistant
    content: str
    
    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


class BaseLLM(ABC):
    """LLM 抽象基类"""
    
    def __init__(
        self,
        api_key: str,
        api_base: str = None,
        model: str = "gpt-4o",
        temperature: float = 0.1,
        max_tokens: int = 4096,
        timeout: int = 60,
        **kwargs
    ):
        self.api_key = api_key
        self.api_base = api_base
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.extra_kwargs = kwargs
    
    @abstractmethod
    def chat(
        self,
        messages: List[Message],
        temperature: float = None,
        max_tokens: int = None,
        **kwargs
    ) -> LLMResponse:
        """
        发送聊天请求
        
        Args:
            messages: 消息列表
            temperature: 温度参数（可选，覆盖默认值）
            max_tokens: 最大 token 数（可选，覆盖默认值）
            
        Returns:
            LLMResponse: LLM 响应
        """
        pass
    
    def complete(
        self,
        prompt: str,
        system_prompt: str = None,
        temperature: float = None,
        max_tokens: int = None,
        **kwargs
    ) -> LLMResponse:
        """
        简化的补全接口
        
        Args:
            prompt: 用户提示
            system_prompt: 系统提示（可选）
            
        Returns:
            LLMResponse: LLM 响应
        """
        messages = []
        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))
        messages.append(Message(role="user", content=prompt))
        
        return self.chat(messages, temperature=temperature, max_tokens=max_tokens, **kwargs)
    
    def generate_summary(self, content: str, context: str = "") -> str:
        """
        生成内容摘要
        
        Args:
            content: 需要摘要的内容
            context: 上下文信息
            
        Returns:
            str: 摘要文本
        """
        system_prompt = """You are a professional document analysis assistant. Generate a concise summary for the given content.

Requirements:
1. Summarize the main content and key information
2. Be objective and accurate
3. Keep the summary between 50-150 words
4. Use the SAME language as the original content
5. Output ONLY the summary text - NO prefixes like "Summary:", "摘要：", etc.
6. Do NOT add any markdown formatting like ** or ##"""

        prompt = f"""Generate a summary for the following content:

{f"Context: {context}" if context else ""}

Content:
{content}

Output ONLY the summary text, without any prefix, label, or explanation."""

        response = self.complete(prompt, system_prompt=system_prompt)
        # 清理 LLM 输出（处理 <think> 标签和多余前缀）
        return clean_llm_output(response.content)
    
    def generate_document_description(self, content_preview: str, toc: str = "") -> str:
        """
        生成文档描述
        
        Args:
            content_preview: 文档开头内容
            toc: 目录结构
            
        Returns:
            str: 文档描述
        """
        system_prompt = """You are a professional document analysis assistant. Generate an overall description for the document based on its content and structure.

Requirements:
1. Describe the document's theme and purpose
2. Summarize the main areas covered
3. Keep the description between 100-300 words
4. Use the SAME language as the original content
5. Output ONLY the description text - NO prefixes like "Description:", "描述：", etc.
6. Do NOT add any markdown formatting like ** or ##"""

        prompt = f"""Generate a description for the following document:

Document preview:
{content_preview}

{f"Table of contents: {toc}" if toc else ""}

Output ONLY the description text, without any prefix, label, or explanation."""

        response = self.complete(prompt, system_prompt=system_prompt)
        # 清理 LLM 输出（处理 <think> 标签和多余前缀）
        return clean_llm_output(response.content)
    
    @abstractmethod
    def is_available(self) -> bool:
        """检查 LLM 服务是否可用"""
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model}, api_base={self.api_base})"



