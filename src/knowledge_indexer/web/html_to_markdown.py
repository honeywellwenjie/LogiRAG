"""
HTML 转 Markdown 模块
使用 LLM 将网页 HTML 转换为结构化的 Markdown
"""

import logging
import re
from typing import Optional

from ..llm.base import BaseLLM

logger = logging.getLogger(__name__)


class HTMLToMarkdown:
    """HTML 转 Markdown 转换器（使用 LLM）"""
    
    # 转换提示词
    SYSTEM_PROMPT = """你是一个专业的文档转换助手。你的任务是将网页 HTML 内容转换为结构化的 Markdown 格式。

转换规则：
1. 保留原文的层级结构，使用正确的 Markdown 标题层级（# ## ### 等）
2. 保留重要的文本内容，去除广告、导航、页脚等无关内容
3. 代码块使用 ``` 包裹，并标注语言类型
4. 列表使用 - 或数字标记
5. 链接保留为 [文本](URL) 格式
6. 图片转换为 ![描述](URL) 格式
7. 表格使用 Markdown 表格语法
8. 保持原文的段落结构
9. 不要添加原文没有的内容
10. 输出纯 Markdown，不要添加任何解释

重要：
- 确保生成的 Markdown 具有清晰的标题层级结构
- 第一个标题应该是文档的主标题，使用 # 
- 子标题使用 ## ### 等，按照内容的逻辑层级组织"""

    CONVERT_PROMPT = """请将以下网页内容转换为结构化的 Markdown 格式：

网页标题：{title}
网页 URL：{url}

网页内容：
{content}

请直接输出 Markdown 格式的内容，不要添加任何解释或前缀。"""

    def __init__(self, llm: BaseLLM, max_content_length: int = 50000):
        """
        初始化转换器
        
        Args:
            llm: LLM 实例
            max_content_length: 最大内容长度（字符）
        """
        self.llm = llm
        self.max_content_length = max_content_length
    
    def convert(
        self,
        html: str,
        title: str = "",
        url: str = "",
        use_text: bool = False,
    ) -> str:
        """
        将 HTML 转换为 Markdown
        
        Args:
            html: HTML 内容
            title: 网页标题
            url: 网页 URL
            use_text: 是否使用纯文本而非 HTML
            
        Returns:
            str: Markdown 格式的内容
        """
        # 预处理 HTML
        if use_text:
            content = self._extract_text(html)
        else:
            content = self._clean_html(html)
        
        # 截断过长内容
        if len(content) > self.max_content_length:
            content = content[:self.max_content_length] + "\n...[内容已截断]..."
            logger.warning(f"内容过长，已截断至 {self.max_content_length} 字符")
        
        # 使用 LLM 转换
        prompt = self.CONVERT_PROMPT.format(
            title=title or "未知",
            url=url or "未知",
            content=content,
        )
        
        try:
            response = self.llm.complete(
                prompt=prompt,
                system_prompt=self.SYSTEM_PROMPT,
                temperature=0.1,
            )
            
            markdown = response.content.strip()
            
            # 清理可能的代码块包裹
            markdown = self._clean_markdown_output(markdown)
            
            logger.info(f"HTML 转 Markdown 完成，生成 {len(markdown)} 字符")
            return markdown
            
        except Exception as e:
            logger.error(f"LLM 转换失败: {e}")
            # 回退到简单转换
            return self._fallback_convert(html, title)
    
    def _clean_html(self, html: str) -> str:
        """清理 HTML，移除脚本、样式等"""
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            
            # 移除不需要的标签
            for tag in soup(['script', 'style', 'nav', 'footer', 'header', 
                           'aside', 'iframe', 'noscript', 'svg', 'form']):
                tag.decompose()
            
            # 移除注释
            from bs4 import Comment
            for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
                comment.extract()
            
            # 移除空标签
            for tag in soup.find_all():
                if not tag.get_text(strip=True) and tag.name not in ['img', 'br', 'hr']:
                    tag.decompose()
            
            # 获取 body 内容
            body = soup.find('body')
            if body:
                return str(body)
            return str(soup)
            
        except ImportError:
            # 没有 BeautifulSoup，使用正则清理
            html = re.sub(r'<script[^>]*>[\s\S]*?</script>', '', html, flags=re.IGNORECASE)
            html = re.sub(r'<style[^>]*>[\s\S]*?</style>', '', html, flags=re.IGNORECASE)
            html = re.sub(r'<!--[\s\S]*?-->', '', html)
            return html
    
    def _extract_text(self, html: str) -> str:
        """提取纯文本"""
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            
            for tag in soup(['script', 'style']):
                tag.decompose()
            
            return soup.get_text(separator='\n')
        except ImportError:
            text = re.sub(r'<[^>]+>', '', html)
            return text
    
    def _clean_markdown_output(self, markdown: str) -> str:
        """清理 LLM 输出的 Markdown"""
        # 移除可能的代码块包裹
        if markdown.startswith('```markdown'):
            markdown = markdown[11:]
        elif markdown.startswith('```md'):
            markdown = markdown[5:]
        elif markdown.startswith('```'):
            markdown = markdown[3:]
        
        if markdown.endswith('```'):
            markdown = markdown[:-3]
        
        return markdown.strip()
    
    def _fallback_convert(self, html: str, title: str = "") -> str:
        """回退转换（不使用 LLM）"""
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            
            lines = []
            
            # 添加标题
            if title:
                lines.append(f"# {title}")
                lines.append("")
            
            # 简单提取标题和段落
            for tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li']):
                text = tag.get_text(strip=True)
                if not text:
                    continue
                
                if tag.name.startswith('h'):
                    level = int(tag.name[1])
                    # 如果已有标题，降一级
                    if title and level == 1:
                        level = 2
                    lines.append(f"{'#' * level} {text}")
                elif tag.name == 'li':
                    lines.append(f"- {text}")
                else:
                    lines.append(text)
                lines.append("")
            
            return '\n'.join(lines)
            
        except Exception as e:
            logger.error(f"回退转换失败: {e}")
            return f"# {title}\n\n{self._extract_text(html)}"


class SimpleHTMLToMarkdown:
    """简单的 HTML 转 Markdown（不使用 LLM）"""
    
    def convert(self, html: str, title: str = "", url: str = "") -> str:
        """
        将 HTML 转换为 Markdown（基于规则）
        
        Args:
            html: HTML 内容
            title: 网页标题
            
        Returns:
            str: Markdown 格式的内容
        """
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            
            # 移除不需要的标签
            for tag in soup(['script', 'style', 'nav', 'footer', 'aside', 'iframe']):
                tag.decompose()
            
            lines = []
            
            # 添加标题
            if title:
                lines.append(f"# {title}")
                lines.append("")
            
            # 查找主要内容区域
            main = soup.find('main') or soup.find('article') or soup.find('body')
            if not main:
                main = soup
            
            # 处理内容
            self._process_element(main, lines, title_offset=1 if title else 0)
            
            return '\n'.join(lines)
            
        except ImportError:
            logger.warning("BeautifulSoup 未安装，使用简单转换")
            text = re.sub(r'<[^>]+>', '', html)
            return f"# {title}\n\n{text}" if title else text
    
    def _process_element(self, element, lines: list, title_offset: int = 0):
        """递归处理 HTML 元素"""
        from bs4 import NavigableString
        
        for child in element.children:
            if isinstance(child, NavigableString):
                text = str(child).strip()
                if text:
                    lines.append(text)
                continue
            
            tag_name = child.name
            text = child.get_text(strip=True)
            
            if not text and tag_name not in ['hr', 'br']:
                continue
            
            # 处理不同标签
            if tag_name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                level = int(tag_name[1]) + title_offset
                level = min(level, 6)
                lines.append(f"{'#' * level} {text}")
                lines.append("")
            
            elif tag_name == 'p':
                lines.append(text)
                lines.append("")
            
            elif tag_name in ['ul', 'ol']:
                for i, li in enumerate(child.find_all('li', recursive=False)):
                    li_text = li.get_text(strip=True)
                    if tag_name == 'ol':
                        lines.append(f"{i+1}. {li_text}")
                    else:
                        lines.append(f"- {li_text}")
                lines.append("")
            
            elif tag_name == 'pre' or tag_name == 'code':
                code = child.get_text()
                lang = child.get('class', [''])[0] if child.get('class') else ''
                lang = lang.replace('language-', '') if lang else ''
                lines.append(f"```{lang}")
                lines.append(code)
                lines.append("```")
                lines.append("")
            
            elif tag_name == 'blockquote':
                for line in text.split('\n'):
                    lines.append(f"> {line}")
                lines.append("")
            
            elif tag_name == 'a':
                href = child.get('href', '')
                if href and not href.startswith('#'):
                    lines.append(f"[{text}]({href})")
            
            elif tag_name == 'img':
                src = child.get('src', '')
                alt = child.get('alt', '')
                if src:
                    lines.append(f"![{alt}]({src})")
            
            elif tag_name == 'hr':
                lines.append("---")
                lines.append("")
            
            elif tag_name in ['div', 'section', 'article', 'main']:
                self._process_element(child, lines, title_offset)

