"""
LLM 抽象基类
定义统一的 LLM 接口，方便扩展支持不同的 LLM 后端
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import re


def extract_answer_from_reasoning(content: str) -> str:
    """
    从推理模型的输出中提取实际答案
    处理 deepseek-r1 等模型的 <think>...</think> 格式
    
    Args:
        content: LLM 原始输出内容
        
    Returns:
        str: 移除思考过程后的实际答案
    """
    if not content:
        return ""
    
    # 移除 <think>...</think> 块
    # 使用 re.DOTALL 让 . 匹配换行符
    cleaned = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
    
    return cleaned.strip()


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
        system_prompt = """你是一个专业的文档分析助手。请为给定的内容生成简洁的摘要。
摘要应该：
1. 概括主要内容和关键信息
2. 保持客观准确
3. 长度控制在 50-150 字之间
4. 使用与原文相同的语言"""

        prompt = f"""请为以下内容生成摘要：

{f"上下文：{context}" if context else ""}

内容：
{content}

请直接输出摘要，不要添加任何前缀或解释。"""

        response = self.complete(prompt, system_prompt=system_prompt)
        # 处理 deepseek-r1 等推理模型的 <think> 标签
        return extract_answer_from_reasoning(response.content)
    
    def generate_document_description(self, content_preview: str, toc: str = "") -> str:
        """
        生成文档描述
        
        Args:
            content_preview: 文档开头内容
            toc: 目录结构
            
        Returns:
            str: 文档描述
        """
        system_prompt = """你是一个专业的文档分析助手。请根据文档的开头内容和目录结构，生成文档的整体描述。
描述应该：
1. 说明文档的主题和目的
2. 概括主要涵盖的内容领域
3. 长度控制在 100-200 字之间
4. 使用与原文相同的语言"""

        prompt = f"""请为以下文档生成描述：

文档开头：
{content_preview}

{f"目录结构：{toc}" if toc else ""}

请直接输出描述，不要添加任何前缀或解释。"""

        response = self.complete(prompt, system_prompt=system_prompt)
        # 处理 deepseek-r1 等推理模型的 <think> 标签
        return extract_answer_from_reasoning(response.content)
    
    @abstractmethod
    def is_available(self) -> bool:
        """检查 LLM 服务是否可用"""
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model}, api_base={self.api_base})"



