"""
网页抓取模块
负责获取网页内容
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from urllib.parse import urlparse
import re

logger = logging.getLogger(__name__)


@dataclass
class WebPage:
    """网页数据"""
    url: str
    title: str = ""
    html: str = ""
    text: str = ""  # 纯文本内容
    markdown: str = ""  # 转换后的 Markdown
    status_code: int = 0
    headers: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def domain(self) -> str:
        """获取域名"""
        parsed = urlparse(self.url)
        return parsed.netloc
    
    @property
    def is_success(self) -> bool:
        """是否成功获取"""
        return 200 <= self.status_code < 300


class WebFetcher:
    """网页抓取器"""
    
    DEFAULT_HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    }
    
    def __init__(
        self,
        timeout: int = 30,
        headers: Dict[str, str] = None,
        verify_ssl: bool = True,
    ):
        """
        初始化抓取器
        
        Args:
            timeout: 请求超时时间（秒）
            headers: 自定义请求头
            verify_ssl: 是否验证 SSL 证书
        """
        self.timeout = timeout
        self.headers = {**self.DEFAULT_HEADERS, **(headers or {})}
        self.verify_ssl = verify_ssl
        
        # 尝试导入 requests
        try:
            import requests
            self._requests = requests
        except ImportError:
            logger.warning("requests 库未安装，请运行: pip install requests")
            self._requests = None
    
    def fetch(self, url: str) -> WebPage:
        """
        抓取网页
        
        Args:
            url: 网页 URL
            
        Returns:
            WebPage: 网页数据
        """
        if self._requests is None:
            raise RuntimeError("requests 库未安装")
        
        page = WebPage(url=url)
        
        try:
            response = self._requests.get(
                url,
                headers=self.headers,
                timeout=self.timeout,
                verify=self.verify_ssl,
            )
            
            page.status_code = response.status_code
            page.headers = dict(response.headers)
            
            if page.is_success:
                # 处理编码
                response.encoding = response.apparent_encoding or 'utf-8'
                page.html = response.text
                
                # 提取标题和文本
                page.title = self._extract_title(page.html)
                page.text = self._extract_text(page.html)
                
                logger.info(f"成功抓取: {url} (标题: {page.title})")
            else:
                logger.warning(f"抓取失败: {url} (状态码: {page.status_code})")
                
        except Exception as e:
            logger.error(f"抓取异常: {url} - {e}")
            page.metadata["error"] = str(e)
        
        return page
    
    def _extract_title(self, html: str) -> str:
        """从 HTML 提取标题"""
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            
            # 优先使用 og:title
            og_title = soup.find('meta', property='og:title')
            if og_title and og_title.get('content'):
                return og_title['content'].strip()
            
            # 使用 title 标签
            title_tag = soup.find('title')
            if title_tag:
                return title_tag.get_text().strip()
            
            # 使用 h1
            h1_tag = soup.find('h1')
            if h1_tag:
                return h1_tag.get_text().strip()
                
        except ImportError:
            # 没有 BeautifulSoup，使用正则
            match = re.search(r'<title[^>]*>([^<]+)</title>', html, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        except Exception as e:
            logger.warning(f"提取标题失败: {e}")
        
        return "Untitled"
    
    def _extract_text(self, html: str) -> str:
        """从 HTML 提取纯文本"""
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            
            # 移除脚本和样式
            for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                tag.decompose()
            
            # 获取文本
            text = soup.get_text(separator='\n')
            
            # 清理空白行
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            return '\n'.join(lines)
            
        except ImportError:
            # 简单清理
            text = re.sub(r'<script[^>]*>[\s\S]*?</script>', '', html, flags=re.IGNORECASE)
            text = re.sub(r'<style[^>]*>[\s\S]*?</style>', '', html, flags=re.IGNORECASE)
            text = re.sub(r'<[^>]+>', '', text)
            return text.strip()
        except Exception as e:
            logger.warning(f"提取文本失败: {e}")
            return ""
    
    def fetch_with_metadata(self, url: str) -> WebPage:
        """
        抓取网页并提取元数据
        
        Args:
            url: 网页 URL
            
        Returns:
            WebPage: 包含元数据的网页数据
        """
        page = self.fetch(url)
        
        if page.is_success and page.html:
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(page.html, 'html.parser')
                
                # 提取元数据
                metadata = {}
                
                # Open Graph 标签
                for meta in soup.find_all('meta', property=re.compile(r'^og:')):
                    key = meta.get('property', '').replace('og:', '')
                    metadata[f"og_{key}"] = meta.get('content', '')
                
                # description
                desc_tag = soup.find('meta', attrs={'name': 'description'})
                if desc_tag:
                    metadata['description'] = desc_tag.get('content', '')
                
                # keywords
                kw_tag = soup.find('meta', attrs={'name': 'keywords'})
                if kw_tag:
                    metadata['keywords'] = kw_tag.get('content', '')
                
                # author
                author_tag = soup.find('meta', attrs={'name': 'author'})
                if author_tag:
                    metadata['author'] = author_tag.get('content', '')
                
                page.metadata.update(metadata)
                
            except ImportError:
                pass
            except Exception as e:
                logger.warning(f"提取元数据失败: {e}")
        
        return page



