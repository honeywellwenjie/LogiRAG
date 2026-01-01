"""
树形索引构建器
将解析的 Markdown 章节构建成树形结构，并使用 LLM 生成摘要
"""

import logging
from typing import List, Optional
import os

from .markdown_parser import MarkdownParser, MarkdownSection
from ..models.tree_node import TreeNode, DocumentIndex
from ..llm.base import BaseLLM
from ..config import IndexerConfig

logger = logging.getLogger(__name__)


class TreeBuilder:
    """树形索引构建器"""
    
    def __init__(
        self,
        llm: BaseLLM = None,
        config: IndexerConfig = None,
        add_node_id: bool = True,
        add_node_summary: bool = True,
        add_doc_description: bool = True,
    ):
        """
        初始化构建器
        
        Args:
            llm: LLM 实例（用于生成摘要）
            config: 索引器配置
            add_node_id: 是否添加节点 ID
            add_node_summary: 是否生成节点摘要
            add_doc_description: 是否生成文档描述
        """
        self.llm = llm
        self.config = config or IndexerConfig()
        self.add_node_id = add_node_id
        self.add_node_summary = add_node_summary
        self.add_doc_description = add_doc_description
        
        self.parser = MarkdownParser()
        self._node_counter = 0
    
    def build_from_file(self, file_path: str, encoding: str = 'utf-8') -> DocumentIndex:
        """
        从文件构建文档索引
        
        Args:
            file_path: Markdown 文件路径
            encoding: 文件编码
            
        Returns:
            DocumentIndex: 文档索引
        """
        with open(file_path, 'r', encoding=encoding) as f:
            content = f.read()
        
        index = self.build_from_content(content)
        index.source_file = os.path.abspath(file_path)
        
        return index
    
    def build_from_content(self, content: str) -> DocumentIndex:
        """
        从内容构建文档索引
        
        Args:
            content: Markdown 文本内容
            
        Returns:
            DocumentIndex: 文档索引
        """
        self._node_counter = 0
        
        # 解析 Markdown
        sections = self.parser.parse(content)
        total_lines = len(content.split('\n'))
        
        # 获取文档标题
        doc_title = self.parser.get_document_title(content)
        
        # 构建树形结构
        root_nodes = self._build_tree(sections)
        
        # 创建文档索引
        index = DocumentIndex(
            title=doc_title,
            source_file="",
            total_lines=total_lines,
            root_nodes=root_nodes,
        )
        
        # 生成文档描述
        if self.add_doc_description and self.llm:
            try:
                preview = self.parser.get_content_preview(content, 1000)
                toc = index.get_toc(max_depth=3)
                index.description = self.llm.generate_document_description(preview, toc)
            except Exception as e:
                logger.warning(f"Failed to generate document description: {e}")
                index.description = ""
        
        return index
    
    def _build_tree(self, sections: List[MarkdownSection]) -> List[TreeNode]:
        """
        从章节列表构建树形结构
        
        Args:
            sections: 章节列表
            
        Returns:
            List[TreeNode]: 根节点列表
        """
        if not sections:
            return []
        
        # 创建节点列表
        nodes = []
        for section in sections:
            node = TreeNode(
                title=section.title,
                level=section.level,
                start_line=section.start_line,
                end_line=section.end_line,
                content=section.content,
            )
            
            # 添加节点 ID
            if self.add_node_id:
                node.node_id = self._generate_node_id()
            
            # 生成摘要
            if self.add_node_summary and self.llm and section.raw_content.strip():
                try:
                    node.summary = self.llm.generate_summary(
                        content=section.raw_content[:2000],  # 限制长度
                        context=section.title
                    )
                except Exception as e:
                    logger.warning(f"Failed to generate node summary [{section.title}]: {e}")
                    node.summary = ""
            
            nodes.append(node)
        
        # 构建父子关系
        root_nodes = self._organize_hierarchy(nodes)
        
        return root_nodes
    
    def _organize_hierarchy(self, nodes: List[TreeNode]) -> List[TreeNode]:
        """
        组织节点的层级关系
        
        根据标题级别建立父子关系：
        - 更高级别（数字更小）的节点是父节点
        - 同级别的节点是兄弟节点
        
        Args:
            nodes: 扁平的节点列表
            
        Returns:
            List[TreeNode]: 根节点列表
        """
        if not nodes:
            return []
        
        root_nodes = []
        stack = []  # 用于追踪父节点路径
        
        for node in nodes:
            # 找到合适的父节点
            while stack and stack[-1].level >= node.level:
                stack.pop()
            
            if stack:
                # 有父节点，添加为子节点
                stack[-1].children.append(node)
            else:
                # 没有父节点，作为根节点
                root_nodes.append(node)
            
            # 将当前节点加入栈
            stack.append(node)
        
        # 更新每个节点的 end_line（包含子节点）
        self._update_end_lines(root_nodes)
        
        return root_nodes
    
    def _update_end_lines(self, nodes: List[TreeNode]):
        """
        更新节点的结束行号（考虑子节点）
        """
        for node in nodes:
            if node.children:
                self._update_end_lines(node.children)
                # 父节点的结束行应该是最后一个子节点的结束行
                node.end_line = max(node.end_line, node.children[-1].end_line)
    
    def _generate_node_id(self) -> str:
        """生成节点 ID"""
        self._node_counter += 1
        return f"{self._node_counter:04d}"
    
    def build_without_llm(self, content: str) -> DocumentIndex:
        """
        不使用 LLM 构建索引（只做结构解析）
        
        Args:
            content: Markdown 文本内容
            
        Returns:
            DocumentIndex: 文档索引（不含摘要）
        """
        # 临时禁用摘要生成
        original_summary = self.add_node_summary
        original_desc = self.add_doc_description
        self.add_node_summary = False
        self.add_doc_description = False
        
        try:
            return self.build_from_content(content)
        finally:
            self.add_node_summary = original_summary
            self.add_doc_description = original_desc
    
    def enrich_with_llm(self, index: DocumentIndex, llm: BaseLLM) -> DocumentIndex:
        """
        使用 LLM 为已有索引添加摘要
        
        Args:
            index: 文档索引
            llm: LLM 实例
            
        Returns:
            DocumentIndex: 增强后的文档索引
        """
        self.llm = llm
        
        # 为每个节点生成摘要
        for node in index.get_all_nodes():
            if not node.summary and node.content.strip():
                try:
                    node.summary = llm.generate_summary(
                        content=node.content[:2000],
                        context=node.title
                    )
                except Exception as e:
                    logger.warning(f"Failed to generate node summary [{node.title}]: {e}")
        
        # 生成文档描述
        if not index.description:
            try:
                # 使用第一个根节点的内容作为预览
                if index.root_nodes:
                    preview = index.root_nodes[0].content[:1000]
                else:
                    preview = ""
                toc = index.get_toc(max_depth=3)
                index.description = llm.generate_document_description(preview, toc)
            except Exception as e:
                logger.warning(f"Failed to generate document description: {e}")
        
        return index


def build_index(
    source: str,
    llm: BaseLLM = None,
    is_file: bool = True,
    **kwargs
) -> DocumentIndex:
    """
    便捷函数：构建文档索引
    
    Args:
        source: 文件路径或 Markdown 内容
        llm: LLM 实例（可选）
        is_file: source 是否为文件路径
        **kwargs: 其他参数传递给 TreeBuilder
        
    Returns:
        DocumentIndex: 文档索引
    """
    builder = TreeBuilder(llm=llm, **kwargs)
    
    if is_file:
        return builder.build_from_file(source)
    else:
        return builder.build_from_content(source)



