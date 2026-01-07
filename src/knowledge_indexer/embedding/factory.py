"""
Factory for creating embedding providers.
"""

import logging
from typing import Optional, Dict, Any
from .base import BaseEmbedding
from .sentence_transformer import SentenceTransformerEmbedding
from .openai_embedding import OpenAIEmbedding

logger = logging.getLogger(__name__)


class EmbeddingFactory:
    """
    Factory class for creating embedding providers.

    Supported providers:
    - sentence_transformer: Local sentence-transformers models
    - openai: OpenAI API embeddings
    """

    @staticmethod
    def create(
        provider: str,
        **kwargs
    ) -> BaseEmbedding:
        """
        Create an embedding provider.

        Args:
            provider: Provider type (sentence_transformer, openai)
            **kwargs: Provider-specific arguments

        Returns:
            BaseEmbedding instance
        """
        provider = provider.lower().replace("-", "_")

        if provider in ("sentence_transformer", "st", "local"):
            return SentenceTransformerEmbedding(
                model_name=kwargs.get("model"),
                device=kwargs.get("device", "cpu"),
                normalize=kwargs.get("normalize", True),
                batch_size=kwargs.get("batch_size", 32),
            )

        elif provider == "openai":
            api_key = kwargs.get("api_key")
            if not api_key:
                raise ValueError("api_key is required for OpenAI embedding")

            return OpenAIEmbedding(
                api_key=api_key,
                api_base=kwargs.get("api_base"),
                model=kwargs.get("model"),
                batch_size=kwargs.get("batch_size", 100),
            )

        else:
            raise ValueError(
                f"Unknown embedding provider: {provider}. "
                f"Supported: sentence_transformer, openai"
            )

    @staticmethod
    def from_config(config) -> BaseEmbedding:
        """
        Create an embedding provider from configuration.

        Args:
            config: EmbeddingConfig or dict with embedding settings

        Returns:
            BaseEmbedding instance
        """
        if hasattr(config, "__dict__"):
            # Convert dataclass to dict
            config_dict = {
                k: v for k, v in config.__dict__.items()
                if not k.startswith("_")
            }
        elif isinstance(config, dict):
            config_dict = config
        else:
            raise ValueError(f"Invalid config type: {type(config)}")

        provider = config_dict.pop("provider", "sentence_transformer")
        return EmbeddingFactory.create(provider, **config_dict)

    @staticmethod
    def get_default() -> BaseEmbedding:
        """
        Get the default embedding provider (local sentence-transformers).

        Returns:
            SentenceTransformerEmbedding with default settings
        """
        return SentenceTransformerEmbedding()
