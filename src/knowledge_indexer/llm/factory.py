"""
LLM 工厂
根据配置创建相应的 LLM 实例
"""

from typing import Optional
import logging

from .base import BaseLLM
from .openai_compatible import OpenAICompatibleLLM, OllamaLLM, LocalLLM
from ..config import LLMConfig, IndexerConfig

logger = logging.getLogger(__name__)


class LLMFactory:
    """LLM 工厂类"""
    
    # 注册的 LLM 类型
    _providers = {
        "openai": OpenAICompatibleLLM,
        "ollama": OllamaLLM,
        "local": LocalLLM,
        "custom": OpenAICompatibleLLM,
        "azure": OpenAICompatibleLLM,
        "vllm": LocalLLM,
        "localai": LocalLLM,
    }
    
    @classmethod
    def register_provider(cls, name: str, provider_class: type):
        """注册新的 LLM 提供者"""
        cls._providers[name] = provider_class
        logger.info(f"已注册 LLM 提供者: {name}")
    
    @classmethod
    def create(
        cls,
        provider: str = "openai",
        api_key: str = None,
        api_base: str = None,
        model: str = None,
        **kwargs
    ) -> BaseLLM:
        """
        创建 LLM 实例
        
        Args:
            provider: 提供者名称 (openai, ollama, local, custom, etc.)
            api_key: API 密钥
            api_base: API 基础 URL
            model: 模型名称
            **kwargs: 其他参数
            
        Returns:
            BaseLLM: LLM 实例
        """
        provider = provider.lower()
        
        if provider not in cls._providers:
            logger.warning(f"未知的 LLM 提供者: {provider}，使用 OpenAI 兼容模式")
            provider = "custom"
        
        provider_class = cls._providers[provider]
        
        # 构建参数
        init_kwargs = {}
        
        if api_key:
            init_kwargs["api_key"] = api_key
        if api_base:
            init_kwargs["api_base"] = api_base
        if model:
            init_kwargs["model"] = model
            
        init_kwargs.update(kwargs)
        
        return provider_class(**init_kwargs)
    
    @classmethod
    def from_config(cls, config: LLMConfig) -> BaseLLM:
        """从 LLMConfig 创建 LLM 实例"""
        return cls.create(
            provider=config.provider,
            api_key=config.api_key,
            api_base=config.api_base,
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout=config.timeout,
        )
    
    @classmethod
    def from_indexer_config(cls, config: IndexerConfig) -> BaseLLM:
        """从 IndexerConfig 创建 LLM 实例"""
        return cls.from_config(config.llm)
    
    @classmethod
    def create_openai(cls, api_key: str, model: str = "gpt-4o", **kwargs) -> BaseLLM:
        """快速创建 OpenAI LLM"""
        return cls.create(
            provider="openai",
            api_key=api_key,
            model=model,
            **kwargs
        )
    
    @classmethod
    def create_ollama(cls, model: str = "llama3", base_url: str = "http://localhost:11434/v1", **kwargs) -> BaseLLM:
        """快速创建 Ollama LLM"""
        return cls.create(
            provider="ollama",
            model=model,
            api_base=base_url,
            **kwargs
        )
    
    @classmethod
    def create_custom(cls, api_key: str, api_base: str, model: str, **kwargs) -> BaseLLM:
        """快速创建自定义 LLM（兼容 OpenAI API）"""
        return cls.create(
            provider="custom",
            api_key=api_key,
            api_base=api_base,
            model=model,
            **kwargs
        )



