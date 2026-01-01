"""
Markdown 解析器
解析 Markdown 文件，提取标题层级结构
"""

import re
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class MarkdownSection:
    """Markdown 章节"""
    title: str
    level: int  # 1-6
    start_line: int  # 起始行号（从 1 开始）
    end_line: int  # 结束行号
    content: str = ""  # 章节内容（包括标题行）
    raw_content: str = ""  # 不含标题的原始内容
    
    def __repr__(self):
        return f"MarkdownSection(level={self.level}, title='{self.title}', lines={self.start_line}-{self.end_line})"


class MarkdownParser:
    """Markdown 解析器"""
    
    # 匹配 ATX 风格标题 (# Title)
    HEADING_PATTERN = re.compile(r'^(#{1,6})\s+(.+?)(?:\s*#*\s*)?$')
    
    # 匹配 Setext 风格标题 (Title\n===== or Title\n-----)
    SETEXT_H1_PATTERN = re.compile(r'^=+\s*$')
    SETEXT_H2_PATTERN = re.compile(r'^-+\s*$')
    
    def __init__(self):
        pass
    
    def parse(self, content: str) -> List[MarkdownSection]:
        """
        解析 Markdown 内容，提取所有章节
        
        Args:
            content: Markdown 文本内容
            
        Returns:
            List[MarkdownSection]: 章节列表（按文档顺序）
        """
        lines = content.split('\n')
        total_lines = len(lines)
        
        # 提取所有标题及其位置
        headings = self._extract_headings(lines)
        
        if not headings:
            # 没有标题，整个文档作为一个章节
            return [MarkdownSection(
                title="Document",
                level=0,
                start_line=1,
                end_line=total_lines,
                content=content,
                raw_content=content
            )]
        
        # 构建章节列表
        sections = []
        for i, (line_num, level, title) in enumerate(headings):
            # 确定章节结束位置
            if i + 1 < len(headings):
                end_line = headings[i + 1][0] - 1
            else:
                end_line = total_lines
            
            # 提取章节内容
            section_lines = lines[line_num - 1:end_line]
            content_text = '\n'.join(section_lines)
            
            # 提取不含标题的内容
            raw_lines = lines[line_num:end_line]  # 从标题后一行开始
            raw_content = '\n'.join(raw_lines).strip()
            
            sections.append(MarkdownSection(
                title=title,
                level=level,
                start_line=line_num,
                end_line=end_line,
                content=content_text,
                raw_content=raw_content
            ))
        
        return sections
    
    def _extract_headings(self, lines: List[str]) -> List[Tuple[int, int, str]]:
        """
        提取所有标题
        
        Returns:
            List of (line_number, level, title)
        """
        headings = []
        i = 0
        in_code_block = False
        
        while i < len(lines):
            line = lines[i]
            
            # 检查代码块边界
            if line.strip().startswith('```'):
                in_code_block = not in_code_block
                i += 1
                continue
            
            # 在代码块内，跳过所有内容
            if in_code_block:
                i += 1
                continue
            
            # 检查 ATX 风格标题
            match = self.HEADING_PATTERN.match(line)
            if match:
                level = len(match.group(1))
                title = match.group(2).strip()
                headings.append((i + 1, level, title))  # 行号从 1 开始
                i += 1
                continue
            
            # 检查 Setext 风格标题
            if i + 1 < len(lines) and line.strip():
                next_line = lines[i + 1]
                if self.SETEXT_H1_PATTERN.match(next_line):
                    headings.append((i + 1, 1, line.strip()))
                    i += 2
                    continue
                elif self.SETEXT_H2_PATTERN.match(next_line) and not line.startswith('-'):
                    headings.append((i + 1, 2, line.strip()))
                    i += 2
                    continue
            
            i += 1
        
        return headings
    
    def parse_file(self, file_path: str, encoding: str = 'utf-8') -> List[MarkdownSection]:
        """
        解析 Markdown 文件
        
        Args:
            file_path: 文件路径
            encoding: 文件编码
            
        Returns:
            List[MarkdownSection]: 章节列表
        """
        with open(file_path, 'r', encoding=encoding) as f:
            content = f.read()
        return self.parse(content)
    
    def get_document_title(self, content: str) -> str:
        """
        获取文档标题（第一个一级标题或第一个标题）
        
        Args:
            content: Markdown 文本
            
        Returns:
            str: 文档标题
        """
        lines = content.split('\n')
        headings = self._extract_headings(lines)
        
        if not headings:
            return "Untitled Document"
        
        # 优先返回一级标题
        for line_num, level, title in headings:
            if level == 1:
                return title
        
        # 否则返回第一个标题
        return headings[0][2]
    
    def get_content_preview(self, content: str, max_chars: int = 500) -> str:
        """
        获取文档开头预览
        
        Args:
            content: Markdown 文本
            max_chars: 最大字符数
            
        Returns:
            str: 文档预览
        """
        # 移除代码块
        clean_content = re.sub(r'```[\s\S]*?```', '', content)
        # 移除行内代码
        clean_content = re.sub(r'`[^`]+`', '', clean_content)
        # 移除链接
        clean_content = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', clean_content)
        # 移除图片
        clean_content = re.sub(r'!\[([^\]]*)\]\([^)]+\)', '', clean_content)
        
        preview = clean_content[:max_chars].strip()
        if len(content) > max_chars:
            preview += "..."
        
        return preview
    
    def extract_code_blocks(self, content: str) -> List[Tuple[str, str]]:
        """
        提取代码块
        
        Returns:
            List of (language, code)
        """
        pattern = re.compile(r'```(\w*)\n([\s\S]*?)```', re.MULTILINE)
        return [(m.group(1), m.group(2)) for m in pattern.finditer(content)]
    
    def get_structure_overview(self, sections: List[MarkdownSection]) -> str:
        """
        生成结构概览
        
        Args:
            sections: 章节列表
            
        Returns:
            str: 结构概览文本
        """
        lines = []
        for section in sections:
            indent = "  " * (section.level - 1)
            lines.append(f"{indent}- {section.title} (lines {section.start_line}-{section.end_line})")
        return '\n'.join(lines)

