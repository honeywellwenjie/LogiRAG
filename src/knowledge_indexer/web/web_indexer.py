"""
网页索引器
将网页内容转换为结构化索引
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime

from .fetcher import WebFetcher, WebPage
from .html_to_markdown import HTMLToMarkdown, SimpleHTMLToMarkdown
from ..indexer.tree_builder import TreeBuilder
from ..models.tree_node import DocumentIndex
from ..llm.base import BaseLLM

logger = logging.getLogger(__name__)


class WebIndexer:
    """网页索引器 - 抓取网页并生成结构化索引"""
    
    def __init__(
        self,
        llm: BaseLLM = None,
        fetcher: WebFetcher = None,
        add_node_id: bool = True,
        add_node_summary: bool = True,
        add_doc_description: bool = True,
        use_llm_for_conversion: bool = True,
    ):
        """
        初始化网页索引器
        
        Args:
            llm: LLM 实例
            fetcher: 网页抓取器实例
            add_node_id: 是否添加节点 ID
            add_node_summary: 是否生成节点摘要
            add_doc_description: 是否生成文档描述
            use_llm_for_conversion: 是否使用 LLM 进行 HTML 到 Markdown 的转换
        """
        self.llm = llm
        self.fetcher = fetcher or WebFetcher()
        self.add_node_id = add_node_id
        self.add_node_summary = add_node_summary
        self.add_doc_description = add_doc_description
        self.use_llm_for_conversion = use_llm_for_conversion
        
        # 创建转换器
        if use_llm_for_conversion and llm:
            self.converter = HTMLToMarkdown(llm)
        else:
            self.converter = SimpleHTMLToMarkdown()
        
        # 创建树构建器
        self.tree_builder = TreeBuilder(
            llm=llm,
            add_node_id=add_node_id,
            add_node_summary=add_node_summary,
            add_doc_description=add_doc_description,
        )
    
    def index_url(self, url: str) -> DocumentIndex:
        """
        索引网页 URL
        
        Args:
            url: 网页 URL
            
        Returns:
            DocumentIndex: 文档索引
        """
        logger.info(f"开始索引网页: {url}")
        
        # 1. 抓取网页
        page = self.fetcher.fetch_with_metadata(url)
        
        if not page.is_success:
            raise RuntimeError(f"抓取网页失败: {url} (状态码: {page.status_code})")
        
        # 2. 转换为 Markdown
        if self.use_llm_for_conversion and isinstance(self.converter, HTMLToMarkdown):
            markdown = self.converter.convert(
                html=page.html,
                title=page.title,
                url=url,
            )
        else:
            markdown = self.converter.convert(
                html=page.html,
                title=page.title,
            )
        
        page.markdown = markdown
        
        # 3. 构建索引
        index = self.tree_builder.build_from_content(markdown)
        
        # 4. 添加网页元数据
        index.source_file = url
        index.metadata.update({
            "type": "web",
            "url": url,
            "domain": page.domain,
            "fetched_at": datetime.now().isoformat(),
            "original_title": page.title,
            **page.metadata,
        })
        
        # 如果没有生成描述，使用元数据中的 description
        if not index.description and page.metadata.get('description'):
            index.description = page.metadata['description']
        
        logger.info(f"网页索引完成: {url} (节点数: {len(index.get_all_nodes())})")
        
        return index
    
    def index_html(
        self,
        html: str,
        url: str = "",
        title: str = "",
    ) -> DocumentIndex:
        """
        索引 HTML 内容
        
        Args:
            html: HTML 内容
            url: 来源 URL（可选）
            title: 标题（可选）
            
        Returns:
            DocumentIndex: 文档索引
        """
        # 转换为 Markdown
        if self.use_llm_for_conversion and isinstance(self.converter, HTMLToMarkdown):
            markdown = self.converter.convert(
                html=html,
                title=title,
                url=url,
            )
        else:
            markdown = self.converter.convert(
                html=html,
                title=title,
            )
        
        # 构建索引
        index = self.tree_builder.build_from_content(markdown)
        
        # 添加元数据
        if url:
            index.source_file = url
        index.metadata.update({
            "type": "html",
            "url": url,
            "indexed_at": datetime.now().isoformat(),
        })
        
        return index
    
    def get_page_with_markdown(self, url: str) -> WebPage:
        """
        抓取网页并转换为 Markdown（不生成索引）
        
        Args:
            url: 网页 URL
            
        Returns:
            WebPage: 包含 Markdown 的网页数据
        """
        page = self.fetcher.fetch_with_metadata(url)
        
        if page.is_success:
            if self.use_llm_for_conversion and isinstance(self.converter, HTMLToMarkdown):
                page.markdown = self.converter.convert(
                    html=page.html,
                    title=page.title,
                    url=url,
                )
            else:
                page.markdown = self.converter.convert(
                    html=page.html,
                    title=page.title,
                )
        
        return page


def index_web(
    url: str,
    llm: BaseLLM = None,
    use_llm_for_conversion: bool = True,
    **kwargs
) -> DocumentIndex:
    """
    便捷函数：索引网页
    
    Args:
        url: 网页 URL
        llm: LLM 实例
        use_llm_for_conversion: 是否使用 LLM 转换 HTML
        **kwargs: 其他参数传递给 WebIndexer
        
    Returns:
        DocumentIndex: 文档索引
    """
    indexer = WebIndexer(
        llm=llm,
        use_llm_for_conversion=use_llm_for_conversion,
        **kwargs
    )
    return indexer.index_url(url)



