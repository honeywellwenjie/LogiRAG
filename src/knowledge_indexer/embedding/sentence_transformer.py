"""
Sentence Transformers embedding provider.

Uses local sentence-transformers models for embedding generation.
Default model: all-MiniLM-L6-v2 (384 dimensions, fast and lightweight)
"""

import logging
from typing import List, Optional
import numpy as np
from .base import BaseEmbedding

logger = logging.getLogger(__name__)


class SentenceTransformerEmbedding(BaseEmbedding):
    """
    Embedding provider using sentence-transformers library.

    Features:
    - Local execution (no API costs)
    - Supports CPU, CUDA, and MPS devices
    - Normalized embeddings for cosine similarity

    Recommended models:
    - all-MiniLM-L6-v2: Fast, 384 dims (default)
    - all-mpnet-base-v2: Better quality, 768 dims
    - paraphrase-multilingual-MiniLM-L12-v2: Multilingual support
    """

    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    def __init__(
        self,
        model_name: str = None,
        device: str = "cpu",
        normalize: bool = True,
        batch_size: int = 32,
    ):
        """
        Initialize the sentence transformer embedding provider.

        Args:
            model_name: Model name from HuggingFace (default: all-MiniLM-L6-v2)
            device: Device to use (cpu, cuda, mps)
            normalize: Whether to normalize embeddings (recommended for cosine similarity)
            batch_size: Batch size for encoding
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self.device = device
        self.normalize = normalize
        self.batch_size = batch_size
        self._model = None
        self._dimension = None

    def _load_model(self):
        """Lazy load the model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading sentence transformer model: {self.model_name}")
                self._model = SentenceTransformer(self.model_name, device=self.device)
                self._dimension = self._model.get_sentence_embedding_dimension()
                logger.info(f"Model loaded successfully. Dimension: {self._dimension}")
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for local embeddings. "
                    "Install with: pip install sentence-transformers"
                )
            except Exception as e:
                logger.error(f"Failed to load model {self.model_name}: {e}")
                raise

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        self._load_model()
        return self._dimension

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            numpy array of shape (n_texts, dimension)
        """
        if not texts:
            return np.array([])

        self._load_model()

        # Handle empty strings
        processed_texts = [t if t.strip() else " " for t in texts]

        embeddings = self._model.encode(
            processed_texts,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
            batch_size=self.batch_size,
        )

        return np.array(embeddings)

    def is_available(self) -> bool:
        """Check if sentence-transformers is available."""
        try:
            from sentence_transformers import SentenceTransformer
            return True
        except ImportError:
            return False

    def __repr__(self) -> str:
        return f"SentenceTransformerEmbedding(model={self.model_name}, device={self.device})"
