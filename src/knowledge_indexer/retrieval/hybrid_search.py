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
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple
from enum import Enum

from .tree_search import TreeSearchEngine, SearchContext, SearchResult
from .vector_index import VectorIndex, VectorSearchResult
from ..models.tree_node import DocumentIndex, TreeNode
from ..debug_utils import (
    debug_print, debug_vector_search, debug_rag_results,
    debug_rag_search_start, debug_hybrid_stage, DebugTimer
)

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

        # DEBUG: 记录搜索开始
        debug_rag_search_start(query, mode.value, len(documents))
        debug_print(
            "🔄 混合搜索引擎启动",
            {
                "查询": query,
                "模式": mode.value,
                "文档数": len(documents),
                "配置": {
                    "向量权重": self.config.vector_weight,
                    "推理权重": self.config.reasoning_weight,
                    "向量top_k": self.config.vector_top_k,
                    "推理最大候选": self.config.reasoning_max_candidates,
                }
            },
            level="start"
        )

        search_start = time.time()

        if mode == RetrievalMode.VECTOR:
            debug_print("📊 使用纯向量模式", level="info")
            results = await self._vector_only_search(query)
        elif mode == RetrievalMode.REASONING:
            debug_print("🧠 使用纯推理模式", level="info")
            results = await self._reasoning_only_search(query, documents, node_maps)
        else:  # HYBRID
            debug_print("🔀 使用混合模式 (向量+推理)", level="info")
            results = await self._hybrid_search(query, documents, node_maps)

        # DEBUG: 记录最终结果
        search_duration = time.time() - search_start
        debug_rag_results(results, mode.value, search_duration)

        return results

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
        # ============================================================
        # Step 1: Vector pre-filtering
        # ============================================================
        debug_hybrid_stage("vector_start", {
            "top_k": self.config.vector_top_k,
            "阈值": self.config.vector_threshold,
            "聚合模式": self.config.use_chunk_aggregation
        })

        logger.info(f"Hybrid search: Vector pre-filtering (top_k={self.config.vector_top_k})")

        vector_start = time.time()
        vector_results = self.vector_index.search(
            query,
            top_k=self.config.vector_top_k,
            use_chunk_aggregation=self.config.use_chunk_aggregation,
            threshold=self.config.vector_threshold,
        )
        vector_duration = time.time() - vector_start

        # DEBUG: 详细记录向量搜索结果
        debug_vector_search(query, self.config.vector_top_k, vector_results, vector_duration)

        if not vector_results:
            logger.warning("No vector results, falling back to reasoning-only")
            debug_hybrid_stage("vector_done", {
                "状态": "无结果，回退到纯推理模式",
                "耗时": f"{vector_duration:.3f}秒"
            })
            return await self._reasoning_only_search(query, documents, node_maps)

        debug_hybrid_stage("vector_done", {
            "命中数": len(vector_results),
            "耗时": f"{vector_duration:.3f}秒",
            "最高分": f"{vector_results[0].score:.4f}" if vector_results else "-",
            "最低分": f"{vector_results[-1].score:.4f}" if vector_results else "-"
        })

        # ============================================================
        # Step 2: Filter documents for LLM reasoning
        # ============================================================
        candidate_docs = set(r.doc_name for r in vector_results[:self.config.reasoning_max_candidates])
        filtered_documents = {
            name: doc for name, doc in documents.items()
            if name in candidate_docs
        }

        debug_hybrid_stage("filter_docs", {
            "原始文档数": len(documents),
            "向量命中文档": list(candidate_docs),
            "过滤后文档数": len(filtered_documents),
            "将送入LLM的节点数": min(len(vector_results), self.config.reasoning_max_candidates)
        })

        logger.info(f"Hybrid search: LLM reasoning on {len(filtered_documents)} documents")

        # ============================================================
        # Step 3: LLM reasoning on filtered documents
        # ============================================================
        debug_hybrid_stage("reasoning_start", {
            "处理文档数": len(filtered_documents),
            "最大候选数": self.config.reasoning_max_candidates
        })

        reasoning_results = []
        reasoning_duration = 0

        if filtered_documents:
            context = SearchContext(
                query=query,
                documents=filtered_documents,
                node_maps={k: v for k, v in node_maps.items() if k in candidate_docs},
                max_results=self.config.reasoning_max_candidates,
                min_relevance=self.config.min_relevance,
            )

            try:
                reasoning_start = time.time()
                tree_results = await self.tree_search.search(context)
                reasoning_duration = time.time() - reasoning_start

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

                debug_hybrid_stage("reasoning_done", {
                    "耗时": f"{reasoning_duration:.2f}秒",
                    "结果数": len(reasoning_results),
                    "命中节点": [f"{r.doc_name}:{r.node_id}({r.reasoning_score:.2f})" for r in reasoning_results[:5]]
                })

            except Exception as e:
                logger.warning(f"Reasoning search failed: {e}, using vector results only")
                debug_hybrid_stage("reasoning_done", {
                    "状态": "失败",
                    "错误": str(e)
                })

        # ============================================================
        # Step 4: Fuse results
        # ============================================================
        debug_hybrid_stage("fusion", {
            "向量结果数": len(vector_results),
            "推理结果数": len(reasoning_results),
            "向量权重": self.config.vector_weight,
            "推理权重": self.config.reasoning_weight
        })

        merged_results = self._merge_results(vector_results, reasoning_results)

        # 统计来源分布
        source_counts = {"vector": 0, "reasoning": 0, "both": 0}
        for r in merged_results:
            source_counts[r.source] = source_counts.get(r.source, 0) + 1

        debug_hybrid_stage("final", {
            "最终结果数": len(merged_results),
            "来源分布": source_counts,
            "结果列表": [
                f"{r.doc_name}:{r.node_id} [最终={r.final_score:.3f}, 向量={r.vector_score or 0:.3f}, 推理={r.reasoning_score or 0:.3f}, 来源={r.source}]"
                for r in merged_results[:5]
            ]
        })

        return merged_results

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
