"""
OpenAI 兼容的 LLM 实现
支持 OpenAI、Azure OpenAI、Ollama、vLLM、LocalAI 等兼容 OpenAI API 的服务
"""

from typing import List, Optional
import logging

from .base import BaseLLM, LLMResponse, Message

logger = logging.getLogger(__name__)


class OpenAICompatibleLLM(BaseLLM):
    """OpenAI 兼容的 LLM 客户端"""
    
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
        super().__init__(
            api_key=api_key,
            api_base=api_base,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            **kwargs
        )
        
        # 延迟导入，避免未安装时报错
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=api_key,
                base_url=api_base,
                timeout=timeout,
            )
            self._available = True
        except ImportError:
            logger.warning("openai 库未安装，请运行: pip install openai")
            self.client = None
            self._available = False
        except Exception as e:
            logger.error(f"初始化 OpenAI 客户端失败: {e}")
            self.client = None
            self._available = False
    
    def chat(
        self,
        messages: List[Message],
        temperature: float = None,
        max_tokens: int = None,
        **kwargs
    ) -> LLMResponse:
        """发送聊天请求"""
        if not self._available or self.client is None:
            raise RuntimeError("OpenAI 客户端不可用")
        
        # 转换消息格式
        formatted_messages = [msg.to_dict() for msg in messages]
        
        # 使用传入的参数或默认值
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=formatted_messages,
                temperature=temp,
                max_tokens=tokens,
                **kwargs
            )
            
            return LLMResponse(
                content=response.choices[0].message.content,
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                },
                raw_response=response
            )
        except Exception as e:
            logger.error(f"LLM 请求失败: {e}")
            raise
    
    def is_available(self) -> bool:
        """检查服务是否可用"""
        if not self._available or self.client is None:
            return False
        
        try:
            # 发送简单请求测试连接
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5,
            )
            return True
        except Exception as e:
            logger.warning(f"LLM 服务不可用: {e}")
            return False


class OllamaLLM(OpenAICompatibleLLM):
    """Ollama LLM 客户端（基于 OpenAI 兼容 API）"""
    
    def __init__(
        self,
        model: str = "llama3",
        api_base: str = "http://localhost:11434/v1",
        api_key: str = "ollama",  # Ollama 不需要真实的 API key
        **kwargs
    ):
        super().__init__(
            api_key=api_key,
            api_base=api_base,
            model=model,
            **kwargs
        )


class LocalLLM(OpenAICompatibleLLM):
    """本地 LLM 客户端（支持 vLLM、LocalAI 等）"""
    
    def __init__(
        self,
        api_base: str,
        model: str,
        api_key: str = "local",
        **kwargs
    ):
        super().__init__(
            api_key=api_key,
            api_base=api_base,
            model=model,
            **kwargs
        )



