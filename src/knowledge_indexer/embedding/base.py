"""
Base class for embedding providers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class EmbeddingResult:
    """Embedding result container."""
    vectors: np.ndarray  # shape: (n_texts, dim)
    model: str
    dimension: int


class BaseEmbedding(ABC):
    """
    Abstract base class for embedding providers.

    All embedding providers must implement:
    - embed_texts(): Batch embedding generation
    - dimension: Return embedding dimension
    - is_available(): Check if the provider is available
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        pass

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            numpy array of shape (n_texts, dimension)
        """
        pass

    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            numpy array of shape (dimension,)
        """
        return self.embed_texts([text])[0]

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the embedding provider is available.

        Returns:
            True if available, False otherwise
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dimension={self.dimension})"
