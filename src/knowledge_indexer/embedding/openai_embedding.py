"""
OpenAI embedding provider.

Uses OpenAI API for embedding generation.
Supports OpenAI-compatible APIs (Azure, local proxies, etc.)
"""

import logging
from typing import List, Optional
import numpy as np
from .base import BaseEmbedding

logger = logging.getLogger(__name__)


class OpenAIEmbedding(BaseEmbedding):
    """
    Embedding provider using OpenAI Embeddings API.

    Features:
    - High quality embeddings
    - Supports OpenAI-compatible APIs
    - Automatic batching for large inputs

    Models:
    - text-embedding-3-small: 1536 dims, cheap (default)
    - text-embedding-3-large: 3072 dims, better quality
    - text-embedding-ada-002: 1536 dims, legacy
    """

    DEFAULT_MODEL = "text-embedding-3-small"
    MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        api_key: str,
        api_base: str = None,
        model: str = None,
        batch_size: int = 100,
    ):
        """
        Initialize the OpenAI embedding provider.

        Args:
            api_key: OpenAI API key
            api_base: API base URL (for compatible APIs)
            model: Model name (default: text-embedding-3-small)
            batch_size: Batch size for API calls (max 100 for OpenAI)
        """
        self.api_key = api_key
        self.api_base = api_base or "https://api.openai.com/v1"
        self.model = model or self.DEFAULT_MODEL
        self.batch_size = min(batch_size, 100)  # OpenAI limit
        self._dimension = self.MODEL_DIMENSIONS.get(self.model, 1536)
        self._client = None

    def _get_client(self):
        """Get or create OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.api_base,
                )
            except ImportError:
                raise ImportError(
                    "openai library is required for OpenAI embeddings. "
                    "Install with: pip install openai"
                )
        return self._client

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        return self._dimension

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts using OpenAI API.

        Args:
            texts: List of texts to embed

        Returns:
            numpy array of shape (n_texts, dimension)
        """
        if not texts:
            return np.array([])

        client = self._get_client()
        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            # Handle empty strings
            batch = [t if t.strip() else " " for t in batch]

            try:
                response = client.embeddings.create(
                    model=self.model,
                    input=batch,
                )
                # Sort by index to maintain order
                sorted_data = sorted(response.data, key=lambda x: x.index)
                embeddings = [item.embedding for item in sorted_data]
                all_embeddings.extend(embeddings)
            except Exception as e:
                logger.error(f"OpenAI embedding API error: {e}")
                raise

        return np.array(all_embeddings)

    def is_available(self) -> bool:
        """Check if OpenAI API is available."""
        try:
            client = self._get_client()
            # Simple test call
            client.embeddings.create(
                model=self.model,
                input=["test"],
            )
            return True
        except Exception as e:
            logger.warning(f"OpenAI embedding not available: {e}")
            return False

    def __repr__(self) -> str:
        return f"OpenAIEmbedding(model={self.model}, api_base={self.api_base})"
