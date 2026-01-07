"""
Knowledge Indexer - 一个模块化的文档索引系统
支持多种 LLM 后端，可轻松集成到其他项目中

增强功能（参考 PageIndex）：
- TOC 自动检测
- 多轮推理搜索
- Summary 感知检索
- 层级树搜索

支持的输入格式：
- Markdown 文件
- 网页 URL（自动转换为 Markdown）
"""

from .indexer.markdown_parser import MarkdownParser
from .indexer.tree_builder import TreeBuilder
from .indexer.toc_detector import TOCDetector, TOCInfo
from .llm.factory import LLMFactory
from .models.tree_node import TreeNode, DocumentIndex
from .config import IndexerConfig

# 网页索引模块（延迟导入，避免未安装依赖时报错）
def get_web_indexer():
    """获取 WebIndexer 类"""
    from .web import WebIndexer
    return WebIndexer

# 检索模块
def get_tree_search_engine():
    """获取增强版树搜索引擎"""
    from .retrieval.tree_search import TreeSearchEngine
    return TreeSearchEngine

def get_reasoning_chain():
    """获取推理链"""
    from .retrieval.reasoning import ReasoningChain
    return ReasoningChain

def get_hybrid_search_engine():
    """获取混合检索引擎（向量 + 推理）"""
    from .retrieval.hybrid_search import HybridSearchEngine
    return HybridSearchEngine

def get_vector_index():
    """获取向量索引"""
    from .retrieval.vector_index import VectorIndex
    return VectorIndex

def get_embedding_factory():
    """获取 Embedding 工厂"""
    from .embedding.factory import EmbeddingFactory
    return EmbeddingFactory

__version__ = "2.1.0"  # Bump version for hybrid retrieval feature

__all__ = [
    # Markdown 索引
    "MarkdownParser",
    "TreeBuilder",
    "TOCDetector",
    "TOCInfo",
    # LLM
    "LLMFactory",
    # 数据模型
    "TreeNode",
    "DocumentIndex",
    # 配置
    "IndexerConfig",
    # 网页索引
    "get_web_indexer",
    # 检索
    "get_tree_search_engine",
    "get_reasoning_chain",
    # Hybrid retrieval (new)
    "get_hybrid_search_engine",
    "get_vector_index",
    "get_embedding_factory",
]

