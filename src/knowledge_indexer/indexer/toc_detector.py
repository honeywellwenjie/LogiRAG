"""
TOC (Table of Contents) 检测器
自动检测 Markdown 文档中的目录结构，辅助建立更准确的索引

参考 PageIndex 的 TOC 检测功能
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class TOCEntry:
    """目录条目"""
    title: str
    level: int
    page_or_line: Optional[int] = None  # 页码或行号
    link: Optional[str] = None  # 锚点链接


@dataclass
class TOCInfo:
    """目录信息"""
    entries: List[TOCEntry]
    start_line: int
    end_line: int
    is_auto_generated: bool = False  # 是否为自动生成的目录（如 [TOC] 标记）
    confidence: float = 0.0  # 检测置信度


class TOCDetector:
    """
    目录检测器
    
    检测 Markdown 文档中的目录结构，包括：
    1. 显式目录（如 [TOC] 标记生成的）
    2. 链接列表形式的目录
    3. 纯文本形式的目录
    """
    
    # 目录标题关键词
    TOC_TITLE_PATTERNS = [
        r'^#+\s*(table\s+of\s+contents?|contents?|目录|索引|toc)\s*$',
        r'^(table\s+of\s+contents?|contents?|目录|索引)\s*$',
    ]
    
    # [TOC] 标记
    TOC_MARKER_PATTERN = re.compile(r'^\s*\[toc\]\s*$', re.IGNORECASE)
    
    # Markdown 链接目录项 - [Title](#anchor) 或 [Title](page.md#anchor)
    TOC_LINK_PATTERN = re.compile(
        r'^\s*[-*+]?\s*\[([^\]]+)\]\(([^)]*)\)\s*$'
    )
    
    # 带编号的目录项 - 1. Title 或 1.1 Title
    NUMBERED_TOC_PATTERN = re.compile(
        r'^\s*(\d+\.)+\s*(.+?)(?:\s+\.{2,}\s*\d+)?\s*$'
    )
    
    # 带点线的目录项 - Title .... 12
    DOTTED_TOC_PATTERN = re.compile(
        r'^\s*(.+?)\s*\.{2,}\s*(\d+)\s*$'
    )
    
    def __init__(self, check_lines: int = 100):
        """
        Args:
            check_lines: 检查文档开头多少行来寻找目录
        """
        self.check_lines = check_lines
    
    def detect(self, content: str) -> Optional[TOCInfo]:
        """
        检测文档中的目录
        
        Args:
            content: Markdown 文档内容
            
        Returns:
            TOCInfo 或 None（如果没有检测到目录）
        """
        lines = content.split('\n')
        
        # 限制检查范围
        check_range = min(len(lines), self.check_lines)
        
        # 1. 检查 [TOC] 标记
        toc_info = self._detect_toc_marker(lines[:check_range])
        if toc_info:
            toc_info.is_auto_generated = True
            return toc_info
        
        # 2. 检查目录标题后的链接列表
        toc_info = self._detect_link_toc(lines[:check_range])
        if toc_info and toc_info.confidence > 0.5:
            return toc_info
        
        # 3. 检查编号或点线形式的目录
        toc_info = self._detect_text_toc(lines[:check_range])
        if toc_info and toc_info.confidence > 0.5:
            return toc_info
        
        return None
    
    def _detect_toc_marker(self, lines: List[str]) -> Optional[TOCInfo]:
        """检测 [TOC] 标记"""
        for i, line in enumerate(lines):
            if self.TOC_MARKER_PATTERN.match(line):
                return TOCInfo(
                    entries=[],
                    start_line=i + 1,
                    end_line=i + 1,
                    is_auto_generated=True,
                    confidence=1.0
                )
        return None
    
    def _detect_link_toc(self, lines: List[str]) -> Optional[TOCInfo]:
        """检测链接形式的目录"""
        toc_start = None
        entries = []
        
        for i, line in enumerate(lines):
            # 检查是否是目录标题
            if toc_start is None:
                for pattern in self.TOC_TITLE_PATTERNS:
                    if re.match(pattern, line.strip(), re.IGNORECASE):
                        toc_start = i
                        break
                continue
            
            # 已找到目录标题，开始收集目录项
            if not line.strip():
                continue
            
            # 检查是否是链接形式的目录项
            match = self.TOC_LINK_PATTERN.match(line)
            if match:
                title = match.group(1).strip()
                link = match.group(2).strip()
                
                # 计算缩进级别
                indent = len(line) - len(line.lstrip())
                level = indent // 2 + 1
                
                entries.append(TOCEntry(
                    title=title,
                    level=level,
                    link=link
                ))
            elif entries:  # 已有目录项但当前行不匹配，目录结束
                # 检查是否是新的标题（目录结束）
                if re.match(r'^#+\s+', line):
                    break
        
        if entries and len(entries) >= 3:  # 至少 3 个条目才认为是有效目录
            return TOCInfo(
                entries=entries,
                start_line=toc_start + 1 if toc_start else 1,
                end_line=toc_start + len(entries) + 2 if toc_start else len(entries) + 1,
                confidence=min(1.0, len(entries) / 5)  # 条目越多置信度越高
            )
        
        return None
    
    def _detect_text_toc(self, lines: List[str]) -> Optional[TOCInfo]:
        """检测纯文本形式的目录"""
        toc_start = None
        entries = []
        
        for i, line in enumerate(lines):
            # 检查是否是目录标题
            if toc_start is None:
                for pattern in self.TOC_TITLE_PATTERNS:
                    if re.match(pattern, line.strip(), re.IGNORECASE):
                        toc_start = i
                        break
                continue
            
            if not line.strip():
                continue
            
            # 检查编号形式
            match = self.NUMBERED_TOC_PATTERN.match(line)
            if match:
                title = match.group(2).strip()
                level = len(match.group(1).split('.')) - 1
                entries.append(TOCEntry(title=title, level=max(1, level)))
                continue
            
            # 检查点线形式
            match = self.DOTTED_TOC_PATTERN.match(line)
            if match:
                title = match.group(1).strip()
                page = int(match.group(2))
                entries.append(TOCEntry(title=title, level=1, page_or_line=page))
                continue
            
            # 遇到其他内容，目录结束
            if entries and re.match(r'^#+\s+', line):
                break
        
        if entries and len(entries) >= 3:
            return TOCInfo(
                entries=entries,
                start_line=toc_start + 1 if toc_start else 1,
                end_line=toc_start + len(entries) + 2 if toc_start else len(entries) + 1,
                confidence=min(1.0, len(entries) / 5)
            )
        
        return None
    
    def use_toc_for_indexing(self, toc: TOCInfo, sections: list) -> list:
        """
        使用检测到的目录优化章节结构
        
        如果文档有显式目录，可以用它来：
        1. 验证自动检测的章节结构
        2. 补充遗漏的章节
        3. 调整章节层级
        
        Args:
            toc: 检测到的目录信息
            sections: 自动解析的章节列表
            
        Returns:
            优化后的章节列表
        """
        if not toc or not toc.entries:
            return sections
        
        # 创建目录标题到条目的映射
        toc_map = {}
        for entry in toc.entries:
            # 标准化标题（去除空白和标点）
            normalized = re.sub(r'\s+', ' ', entry.title.lower().strip())
            toc_map[normalized] = entry
        
        # 用目录信息优化章节
        for section in sections:
            normalized_title = re.sub(r'\s+', ' ', section.title.lower().strip())
            if normalized_title in toc_map:
                toc_entry = toc_map[normalized_title]
                # 如果目录有层级信息，使用它
                if toc_entry.level > 0:
                    section.level = toc_entry.level
        
        return sections
    
    def get_toc_summary(self, toc: TOCInfo) -> str:
        """生成目录摘要，用于 LLM 索引"""
        if not toc or not toc.entries:
            return ""
        
        lines = ["Document Table of Contents:"]
        for entry in toc.entries:
            indent = "  " * (entry.level - 1)
            lines.append(f"{indent}- {entry.title}")
        
        return "\n".join(lines)


