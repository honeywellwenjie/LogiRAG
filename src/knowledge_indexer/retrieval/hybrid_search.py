"""
Hybrid search engine combining vector and reasoning-based retrieval.

Inspired by PageIndex's hybrid approach:
1. Vector Pipeline: Fast pre-filtering using embeddings
2. Reasoning Pipeline: LLM-based semantic analysis on candidates
3. Result Fusion: Deduplication + weighted score merging

Key benefit: Limits LLM prompt size by pre-filtering with vectors,
preventing token explosion when knowledge base has many documents.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple
from enum import Enum

from .tree_search import TreeSearchEngine, SearchContext, SearchResult
from .vector_index import VectorIndex, VectorSearchResult
from ..models.tree_node import DocumentIndex, TreeNode

logger = logging.getLogger(__name__)


class RetrievalMode(Enum):
    """Retrieval mode options."""
    REASONING = "reasoning"  # LLM reasoning only (original mode)
    VECTOR = "vector"        # Vector search only
    HYBRID = "hybrid"        # Combined vector + reasoning


@dataclass
class HybridSearchResult:
    """Hybrid search result combining vector and reasoning scores."""
    doc_name: str
    node_id: str
    title: str
    final_score: float
    vector_score: Optional[float] = None
    reasoning_score: Optional[float] = None
    reasoning: str = ""
    source: str = ""  # "vector", "reasoning", "both"


@dataclass
class HybridSearchConfig:
    """Configuration for hybrid search."""
    mode: RetrievalMode = RetrievalMode.HYBRID

    # Vector search parameters
    vector_top_k: int = 20  # Pre-filter top-k from vector search
    vector_threshold: float = 0.3  # Minimum vector score
    use_chunk_aggregation: bool = True

    # Reasoning parameters
    reasoning_max_candidates: int = 10  # Max nodes for LLM reasoning

    # Hybrid fusion parameters
    vector_weight: float = 0.4
    reasoning_weight: float = 0.6

    # Result parameters
    max_results: int = 10
    min_relevance: float = 0.3


class HybridSearchEngine:
    """
    Hybrid search engine combining vector and reasoning-based retrieval.

    Architecture:
    1. Vector Pipeline (fast): Pre-filter candidates using embeddings
    2. Reasoning Pipeline (accurate): LLM analysis on filtered candidates
    3. Result Fusion: Weighted combination with deduplication

    This solves the "prompt explosion" problem by limiting LLM input
    to only the most relevant candidates from vector search.
    """

    def __init__(
        self,
        tree_search_engine: TreeSearchEngine,
        vector_index: VectorIndex,
        config: HybridSearchConfig = None,
    ):
        """
        Initialize hybrid search engine.

        Args:
            tree_search_engine: TreeSearchEngine for reasoning-based search
            vector_index: VectorIndex for vector-based search
            config: HybridSearchConfig with search parameters
        """
        self.tree_search = tree_search_engine
        self.vector_index = vector_index
        self.config = config or HybridSearchConfig()

    async def search(
        self,
        query: str,
        documents: Dict[str, DocumentIndex],
        node_maps: Dict[str, Dict[str, TreeNode]],
        mode: RetrievalMode = None,
    ) -> List[HybridSearchResult]:
        """
        Execute hybrid search.

        Args:
            query: Search query
            documents: Dict of document indexes
            node_maps: Dict of node maps for each document
            mode: Override retrieval mode

        Returns:
            List of HybridSearchResult sorted by final_score
        """
        mode = mode or self.config.mode

        if mode == RetrievalMode.VECTOR:
            return await self._vector_only_search(query)
        elif mode == RetrievalMode.REASONING:
            return await self._reasoning_only_search(query, documents, node_maps)
        else:  # HYBRID
            return await self._hybrid_search(query, documents, node_maps)

    def search_sync(
        self,
        query: str,
        documents: Dict[str, DocumentIndex],
        node_maps: Dict[str, Dict[str, TreeNode]],
        mode: RetrievalMode = None,
    ) -> List[HybridSearchResult]:
        """Synchronous version of search."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
                return loop.run_until_complete(
                    self.search(query, documents, node_maps, mode)
                )
            else:
                return loop.run_until_complete(
                    self.search(query, documents, node_maps, mode)
                )
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(
                    self.search(query, documents, node_maps, mode)
                )
            finally:
                loop.close()

    async def _vector_only_search(self, query: str) -> List[HybridSearchResult]:
        """
        Vector-only search mode.

        Fast but may miss semantic nuances.
        """
        vector_results = self.vector_index.search(
            query,
            top_k=self.config.max_results,
            use_chunk_aggregation=self.config.use_chunk_aggregation,
            threshold=self.config.vector_threshold,
        )

        return [
            HybridSearchResult(
                doc_name=r.doc_name,
                node_id=r.node_id,
                title=r.title,
                final_score=r.score,
                vector_score=r.score,
                source="vector",
            )
            for r in vector_results
        ]

    async def _reasoning_only_search(
        self,
        query: str,
        documents: Dict[str, DocumentIndex],
        node_maps: Dict[str, Dict[str, TreeNode]],
    ) -> List[HybridSearchResult]:
        """
        Reasoning-only search mode (original LogiRAG behavior).

        Accurate but can hit token limits with many documents.
        """
        context = SearchContext(
            query=query,
            documents=documents,
            node_maps=node_maps,
            max_results=self.config.max_results,
            min_relevance=self.config.min_relevance,
        )

        results = await self.tree_search.search(context)

        return [
            HybridSearchResult(
                doc_name=r.doc_name,
                node_id=r.node_id,
                title=r.title,
                final_score=r.relevance_score,
                reasoning_score=r.relevance_score,
                reasoning=r.reasoning,
                source="reasoning",
            )
            for r in results
        ]

    async def _hybrid_search(
        self,
        query: str,
        documents: Dict[str, DocumentIndex],
        node_maps: Dict[str, Dict[str, TreeNode]],
    ) -> List[HybridSearchResult]:
        """
        Hybrid search combining vector pre-filtering and LLM reasoning.

        This is the key innovation for handling large knowledge bases:
        1. Vector search quickly narrows down candidates
        2. LLM reasoning only processes top candidates (avoiding prompt explosion)
        3. Results are fused with weighted scoring
        """
        # Step 1: Vector pre-filtering
        logger.info(f"Hybrid search: Vector pre-filtering (top_k={self.config.vector_top_k})")
        vector_results = self.vector_index.search(
            query,
            top_k=self.config.vector_top_k,
            use_chunk_aggregation=self.config.use_chunk_aggregation,
            threshold=self.config.vector_threshold,
        )

        if not vector_results:
            logger.warning("No vector results, falling back to reasoning-only")
            return await self._reasoning_only_search(query, documents, node_maps)

        # Step 2: Filter documents for LLM reasoning
        # Only include documents that have candidate nodes from vector search
        candidate_docs = set(r.doc_name for r in vector_results[:self.config.reasoning_max_candidates])
        filtered_documents = {
            name: doc for name, doc in documents.items()
            if name in candidate_docs
        }

        logger.info(f"Hybrid search: LLM reasoning on {len(filtered_documents)} documents")

        # Step 3: LLM reasoning on filtered documents
        reasoning_results = []
        if filtered_documents:
            context = SearchContext(
                query=query,
                documents=filtered_documents,
                node_maps={k: v for k, v in node_maps.items() if k in candidate_docs},
                max_results=self.config.reasoning_max_candidates,
                min_relevance=self.config.min_relevance,
            )

            try:
                tree_results = await self.tree_search.search(context)
                reasoning_results = [
                    HybridSearchResult(
                        doc_name=r.doc_name,
                        node_id=r.node_id,
                        title=r.title,
                        final_score=r.relevance_score,
                        reasoning_score=r.relevance_score,
                        reasoning=r.reasoning,
                        source="reasoning",
                    )
                    for r in tree_results
                ]
            except Exception as e:
                logger.warning(f"Reasoning search failed: {e}, using vector results only")

        # Step 4: Fuse results
        return self._merge_results(vector_results, reasoning_results)

    def _merge_results(
        self,
        vector_results: List[VectorSearchResult],
        reasoning_results: List[HybridSearchResult],
    ) -> List[HybridSearchResult]:
        """
        Merge vector and reasoning results with deduplication and weighted scoring.

        Nodes found by both methods get combined scores.
        """
        merged: Dict[str, HybridSearchResult] = {}

        # Add vector results
        for vr in vector_results:
            key = f"{vr.doc_name}:{vr.node_id}"
            merged[key] = HybridSearchResult(
                doc_name=vr.doc_name,
                node_id=vr.node_id,
                title=vr.title,
                final_score=vr.score * self.config.vector_weight,
                vector_score=vr.score,
                source="vector",
            )

        # Merge reasoning results
        for rr in reasoning_results:
            key = f"{rr.doc_name}:{rr.node_id}"

            if key in merged:
                # Node found by both methods - combine scores
                existing = merged[key]
                existing.reasoning_score = rr.reasoning_score
                existing.reasoning = rr.reasoning
                existing.final_score = (
                    (existing.vector_score or 0) * self.config.vector_weight +
                    rr.reasoning_score * self.config.reasoning_weight
                )
                existing.source = "both"
            else:
                # New node from reasoning only
                merged[key] = HybridSearchResult(
                    doc_name=rr.doc_name,
                    node_id=rr.node_id,
                    title=rr.title,
                    final_score=rr.reasoning_score * self.config.reasoning_weight,
                    reasoning_score=rr.reasoning_score,
                    reasoning=rr.reasoning,
                    source="reasoning",
                )

        # Sort by final score and return top results
        results = sorted(merged.values(), key=lambda x: x.final_score, reverse=True)
        return results[:self.config.max_results]

    def get_config(self) -> Dict:
        """Get current configuration as dict."""
        return {
            "mode": self.config.mode.value,
            "vector_top_k": self.config.vector_top_k,
            "vector_threshold": self.config.vector_threshold,
            "reasoning_max_candidates": self.config.reasoning_max_candidates,
            "vector_weight": self.config.vector_weight,
            "reasoning_weight": self.config.reasoning_weight,
            "max_results": self.config.max_results,
        }
