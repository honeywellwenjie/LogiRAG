"""
多层网页爬虫
支持递归爬取网页链接
"""

import logging
import re
from dataclasses import dataclass, field
from typing import List, Set, Optional, Callable
from urllib.parse import urljoin, urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from .fetcher import WebFetcher, WebPage

logger = logging.getLogger(__name__)


@dataclass
class CrawlResult:
    """爬取结果"""
    url: str
    pages: List[WebPage] = field(default_factory=list)
    failed_urls: List[str] = field(default_factory=list)
    total_links_found: int = 0
    
    @property
    def success_count(self) -> int:
        return len(self.pages)
    
    @property
    def failed_count(self) -> int:
        return len(self.failed_urls)


class WebCrawler:
    """多层网页爬虫"""
    
    # 不爬取的文件扩展名
    SKIP_EXTENSIONS = {
        '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.webp', '.ico',
        '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
        '.zip', '.rar', '.7z', '.tar', '.gz',
        '.mp3', '.mp4', '.avi', '.mov', '.wmv', '.flv',
        '.css', '.js', '.json', '.xml',
        '.exe', '.dmg', '.apk',
    }
    
    def __init__(
        self,
        fetcher: WebFetcher = None,
        max_pages: int = 100,
        same_domain_only: bool = True,
        max_workers: int = 5,
        respect_robots: bool = False,
    ):
        """
        初始化爬虫
        
        Args:
            fetcher: 网页抓取器
            max_pages: 最大爬取页面数
            same_domain_only: 是否只爬取同一域名
            max_workers: 并发线程数
            respect_robots: 是否遵守 robots.txt（暂未实现）
        """
        self.fetcher = fetcher or WebFetcher()
        self.max_pages = max_pages
        self.same_domain_only = same_domain_only
        self.max_workers = max_workers
        self.respect_robots = respect_robots
        
        self._visited_urls: Set[str] = set()
        self._base_domain: str = ""
    
    def crawl(
        self,
        start_url: str,
        level: int = 1,
        url_filter: Callable[[str], bool] = None,
    ) -> CrawlResult:
        """
        爬取网页
        
        Args:
            start_url: 起始 URL
            level: 爬取深度（0=仅当前页，1=当前页+链接，2=再深一层...，255=无限）
            url_filter: URL 过滤函数，返回 True 表示爬取
            
        Returns:
            CrawlResult: 爬取结果
        """
        self._visited_urls.clear()
        parsed = urlparse(start_url)
        self._base_domain = parsed.netloc
        
        result = CrawlResult(url=start_url)
        
        # 爬取起始页面
        logger.info(f"Starting crawl: {start_url} (depth: {level})")
        
        urls_to_crawl = [(start_url, 0)]  # (url, current_depth)
        
        while urls_to_crawl and len(result.pages) < self.max_pages:
            current_batch = []
            next_batch = []
            
            # 收集当前批次要爬取的 URL
            for url, depth in urls_to_crawl:
                if url in self._visited_urls:
                    continue
                if len(result.pages) + len(current_batch) >= self.max_pages:
                    break
                    
                self._visited_urls.add(url)
                current_batch.append((url, depth))
            
            if not current_batch:
                break
            
            # 并发爬取
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_url = {
                    executor.submit(self._fetch_page, url): (url, depth)
                    for url, depth in current_batch
                }
                
                for future in as_completed(future_to_url):
                    url, depth = future_to_url[future]
                    try:
                        page = future.result()
                        if page and page.is_success:
                            result.pages.append(page)
                            logger.info(f"[{len(result.pages)}/{self.max_pages}] Crawled: {url}")
                            
                            # 如果还需要继续深入
                            max_depth = 255 if level == 0xFF else level
                            if depth < max_depth:
                                links = self._extract_links(page.html, url)
                                result.total_links_found += len(links)
                                
                                for link in links:
                                    if self._should_crawl(link, url_filter):
                                        next_batch.append((link, depth + 1))
                        else:
                            result.failed_urls.append(url)
                            logger.warning(f"Crawl failed: {url}")
                            
                    except Exception as e:
                        result.failed_urls.append(url)
                        logger.error(f"Crawl exception: {url} - {e}")
            
            urls_to_crawl = next_batch
        
        logger.info(f"Crawl completed: success {result.success_count}, failed {result.failed_count}")
        return result
    
    def _fetch_page(self, url: str) -> Optional[WebPage]:
        """抓取单个页面"""
        try:
            return self.fetcher.fetch_with_metadata(url)
        except Exception as e:
            logger.error(f"Fetch failed: {url} - {e}")
            return None
    
    def _extract_links(self, html: str, base_url: str) -> List[str]:
        """从 HTML 中提取链接"""
        links = []
        
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            
            for a_tag in soup.find_all('a', href=True):
                href = a_tag['href']
                
                # 跳过锚点和 javascript
                if href.startswith('#') or href.startswith('javascript:'):
                    continue
                
                # 转换为绝对 URL
                absolute_url = urljoin(base_url, href)
                
                # 规范化 URL（移除 fragment）
                parsed = urlparse(absolute_url)
                normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                if parsed.query:
                    normalized += f"?{parsed.query}"
                
                links.append(normalized)
                
        except ImportError:
            # 没有 BeautifulSoup，使用正则
            pattern = re.compile(r'href=["\']([^"\']+)["\']', re.IGNORECASE)
            for match in pattern.finditer(html):
                href = match.group(1)
                if not href.startswith('#') and not href.startswith('javascript:'):
                    absolute_url = urljoin(base_url, href)
                    links.append(absolute_url)
        
        return list(set(links))  # 去重
    
    def _should_crawl(self, url: str, url_filter: Callable[[str], bool] = None) -> bool:
        """判断是否应该爬取该 URL"""
        if url in self._visited_urls:
            return False
        
        parsed = urlparse(url)
        
        # 只爬取 http/https
        if parsed.scheme not in ('http', 'https'):
            return False
        
        # 检查域名
        if self.same_domain_only and parsed.netloc != self._base_domain:
            return False
        
        # 检查文件扩展名
        path_lower = parsed.path.lower()
        for ext in self.SKIP_EXTENSIONS:
            if path_lower.endswith(ext):
                return False
        
        # 自定义过滤器
        if url_filter and not url_filter(url):
            return False
        
        return True
    
    def crawl_and_convert(
        self,
        start_url: str,
        level: int = 1,
        converter = None,
        url_filter: Callable[[str], bool] = None,
    ) -> CrawlResult:
        """
        爬取并转换为 Markdown
        
        Args:
            start_url: 起始 URL
            level: 爬取深度
            converter: HTML 转 Markdown 转换器
            url_filter: URL 过滤函数
            
        Returns:
            CrawlResult: 爬取结果（pages 中包含 markdown 字段）
        """
        result = self.crawl(start_url, level, url_filter)
        
        if converter:
            for page in result.pages:
                try:
                    page.markdown = converter.convert(
                        html=page.html,
                        title=page.title,
                        url=page.url,
                    )
                except Exception as e:
                    logger.error(f"转换 Markdown 失败: {page.url} - {e}")
                    page.markdown = ""
        
        return result



