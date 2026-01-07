"""
Lightweight vector index for hybrid retrieval.

Features:
- In-memory vector storage with NumPy
- JSON persistence (no external vector database)
- PageIndex-style chunk aggregation scoring
- Support for both node-level and chunk-level embeddings
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VectorSearchResult:
    """Vector search result."""
    doc_name: str
    node_id: str
    score: float
    title: str = ""
    chunk_scores: List[float] = field(default_factory=list)


class VectorIndex:
    """
    Lightweight vector index for hybrid retrieval.

    Uses NumPy for vector operations and JSON for persistence.
    No external vector database required.

    Scoring formula (PageIndex style):
        NodeScore = max(node_similarity, chunk_aggregation)
        chunk_aggregation = (1/sqrt(N+1)) * sum(chunk_scores)

    This formula rewards nodes with multiple relevant chunks while
    preventing large nodes from dominating purely through volume.
    """

    def __init__(self, embedding_provider=None):
        """
        Initialize vector index.

        Args:
            embedding_provider: BaseEmbedding instance for generating embeddings
        """
        self.embedding = embedding_provider
        self.node_vectors: Dict[str, np.ndarray] = {}  # {doc:node_id -> vector}
        self.chunk_vectors: Dict[str, List[np.ndarray]] = {}  # {doc:node_id -> [vectors]}
        self.node_metadata: Dict[str, Dict[str, Any]] = {}  # {doc:node_id -> metadata}
        self._dimension: Optional[int] = None

    @property
    def dimension(self) -> Optional[int]:
        """Return embedding dimension."""
        if self._dimension is None and self.embedding:
            self._dimension = self.embedding.dimension
        return self._dimension

    def add_document(
        self,
        doc_name: str,
        index: 'DocumentIndex',
        chunk_size: int = 500,
        generate_chunks: bool = True,
    ):
        """
        Add a document to the vector index.

        Args:
            doc_name: Document name/identifier
            index: DocumentIndex instance
            chunk_size: Size of content chunks for fine-grained matching
            generate_chunks: Whether to generate chunk-level embeddings
        """
        if self.embedding is None:
            raise ValueError("Embedding provider not set")

        nodes = index.get_all_nodes()
        logger.info(f"Indexing {len(nodes)} nodes from document: {doc_name}")

        # Collect texts for batch embedding
        node_texts = []
        node_keys = []

        for node in nodes:
            key = f"{doc_name}:{node.node_id}"

            # Use summary if available, otherwise title
            text = node.summary or node.title
            if text:
                node_texts.append(text)
                node_keys.append(key)

            # Store metadata
            self.node_metadata[key] = {
                "title": node.title,
                "level": node.level,
                "doc_name": doc_name,
                "node_id": node.node_id,
            }

        # Batch generate node embeddings
        if node_texts:
            embeddings = self.embedding.embed_texts(node_texts)
            for i, key in enumerate(node_keys):
                self.node_vectors[key] = embeddings[i]

        # Generate chunk embeddings if enabled
        if generate_chunks:
            for node in nodes:
                if node.content and len(node.content) > 100:
                    key = f"{doc_name}:{node.node_id}"
                    chunks = self._split_to_chunks(node.content, chunk_size)
                    if chunks:
                        chunk_embeddings = self.embedding.embed_texts(chunks)
                        self.chunk_vectors[key] = [chunk_embeddings[i] for i in range(len(chunks))]

        self._dimension = self.embedding.dimension
        logger.info(f"Indexed {len(node_texts)} nodes, {sum(len(v) for v in self.chunk_vectors.values())} chunks")

    def search(
        self,
        query: str,
        top_k: int = 10,
        use_chunk_aggregation: bool = True,
        threshold: float = 0.0,
    ) -> List[VectorSearchResult]:
        """
        Search for relevant nodes using vector similarity.

        Args:
            query: Query text
            top_k: Maximum number of results
            use_chunk_aggregation: Use PageIndex-style chunk aggregation
            threshold: Minimum score threshold

        Returns:
            List of VectorSearchResult sorted by score
        """
        if not self.node_vectors:
            logger.warning("Vector index is empty")
            return []

        if self.embedding is None:
            raise ValueError("Embedding provider not set")

        # Generate query embedding
        query_vec = self.embedding.embed_text(query)

        results = []
        for key, node_vec in self.node_vectors.items():
            doc_name, node_id = key.split(":", 1)

            # Node-level similarity (cosine similarity for normalized vectors)
            node_score = float(np.dot(query_vec, node_vec))

            # Chunk-level aggregation (PageIndex formula)
            chunk_scores = []
            if use_chunk_aggregation and key in self.chunk_vectors:
                chunk_vecs = self.chunk_vectors[key]
                for chunk_vec in chunk_vecs:
                    chunk_scores.append(float(np.dot(query_vec, chunk_vec)))

                if chunk_scores:
                    # PageIndex formula: NodeScore = (1/sqrt(N+1)) * sum(ChunkScore)
                    n = len(chunk_scores)
                    aggregated_score = (1 / np.sqrt(n + 1)) * sum(chunk_scores)
                    final_score = max(node_score, aggregated_score)
                else:
                    final_score = node_score
            else:
                final_score = node_score

            if final_score >= threshold:
                metadata = self.node_metadata.get(key, {})
                results.append(VectorSearchResult(
                    doc_name=doc_name,
                    node_id=node_id,
                    score=final_score,
                    title=metadata.get("title", ""),
                    chunk_scores=chunk_scores,
                ))

        # Sort by score descending
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def _split_to_chunks(
        self,
        text: str,
        chunk_size: int = 500,
        overlap: int = 50,
    ) -> List[str]:
        """
        Split text into overlapping chunks.

        Args:
            text: Text to split
            chunk_size: Target chunk size in characters
            overlap: Overlap between chunks

        Returns:
            List of text chunks
        """
        if not text or len(text) <= chunk_size:
            return [text] if text else []

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size

            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings
                for sep in ['. ', '。', '\n', '! ', '? ']:
                    last_sep = text.rfind(sep, start, end)
                    if last_sep > start + chunk_size // 2:
                        end = last_sep + len(sep)
                        break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - overlap
            if start >= len(text) - overlap:
                break

        return chunks

    def save(self, path: str):
        """
        Save vector index to JSON file.

        Args:
            path: Output file path
        """
        data = {
            "version": "1.0",
            "dimension": self._dimension,
            "node_vectors": {k: v.tolist() for k, v in self.node_vectors.items()},
            "chunk_vectors": {
                k: [c.tolist() for c in v]
                for k, v in self.chunk_vectors.items()
            },
            "node_metadata": self.node_metadata,
        }

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)

        logger.info(f"Vector index saved to {path}")

    def load(self, path: str):
        """
        Load vector index from JSON file.

        Args:
            path: Input file path
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self._dimension = data.get("dimension")
        self.node_vectors = {
            k: np.array(v) for k, v in data.get("node_vectors", {}).items()
        }
        self.chunk_vectors = {
            k: [np.array(c) for c in v]
            for k, v in data.get("chunk_vectors", {}).items()
        }
        self.node_metadata = data.get("node_metadata", {})

        logger.info(f"Vector index loaded from {path}: {len(self.node_vectors)} nodes")

    def clear(self):
        """Clear all data from the index."""
        self.node_vectors.clear()
        self.chunk_vectors.clear()
        self.node_metadata.clear()
        self._dimension = None

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        return {
            "num_nodes": len(self.node_vectors),
            "num_chunks": sum(len(v) for v in self.chunk_vectors.values()),
            "dimension": self._dimension,
            "documents": list(set(
                m.get("doc_name") for m in self.node_metadata.values()
            )),
        }

    def __len__(self) -> int:
        """Return number of indexed nodes."""
        return len(self.node_vectors)

    def __repr__(self) -> str:
        stats = self.get_stats()
        return f"VectorIndex(nodes={stats['num_nodes']}, chunks={stats['num_chunks']}, dim={stats['dimension']})"
