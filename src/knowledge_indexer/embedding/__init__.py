"""
Embedding module for LogiRAG hybrid retrieval.

Provides embedding providers for vector-based search:
- SentenceTransformerEmbedding: Local models (default)
- OpenAIEmbedding: OpenAI API embeddings
"""

from .base import BaseEmbedding, EmbeddingResult
from .sentence_transformer import SentenceTransformerEmbedding
from .openai_embedding import OpenAIEmbedding
from .factory import EmbeddingFactory

__all__ = [
    "BaseEmbedding",
    "EmbeddingResult",
    "SentenceTransformerEmbedding",
    "OpenAIEmbedding",
    "EmbeddingFactory",
]
