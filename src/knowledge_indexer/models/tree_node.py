"""
树节点数据模型
定义文档索引的核心数据结构
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import json


@dataclass
class TreeNode:
    """文档树节点"""
    title: str
    level: int  # 标题层级 1-6
    start_line: int  # 起始行号
    end_line: int  # 结束行号
    content: str = ""  # 节点原始内容
    summary: str = ""  # LLM 生成的摘要
    node_id: str = ""  # 节点唯一标识
    children: List["TreeNode"] = field(default_factory=list)
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self, include_content: bool = False) -> Dict[str, Any]:
        """转换为字典格式"""
        result = {
            "title": self.title,
            "node_id": self.node_id,
            "level": self.level,
            "start_line": self.start_line,
            "end_line": self.end_line,
        }
        
        if self.summary:
            result["summary"] = self.summary
            
        if include_content and self.content:
            result["content"] = self.content
            
        if self.metadata:
            result["metadata"] = self.metadata
            
        if self.children:
            result["children"] = [child.to_dict(include_content) for child in self.children]
            
        return result
    
    def to_json(self, include_content: bool = False, indent: int = 2) -> str:
        """转换为 JSON 字符串"""
        return json.dumps(self.to_dict(include_content), ensure_ascii=False, indent=indent)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TreeNode":
        """从字典创建节点"""
        children_data = data.pop("children", [])
        node = cls(**data)
        node.children = [cls.from_dict(child) for child in children_data]
        return node
    
    def get_all_nodes(self) -> List["TreeNode"]:
        """获取所有节点（包括自身和所有子节点）"""
        nodes = [self]
        for child in self.children:
            nodes.extend(child.get_all_nodes())
        return nodes
    
    def find_node_by_id(self, node_id: str) -> Optional["TreeNode"]:
        """根据 ID 查找节点"""
        if self.node_id == node_id:
            return self
        for child in self.children:
            found = child.find_node_by_id(node_id)
            if found:
                return found
        return None
    
    def get_path(self, target_id: str, path: List[str] = None) -> Optional[List[str]]:
        """获取到目标节点的路径"""
        if path is None:
            path = []
        
        current_path = path + [self.title]
        
        if self.node_id == target_id:
            return current_path
            
        for child in self.children:
            result = child.get_path(target_id, current_path)
            if result:
                return result
        return None


@dataclass
class DocumentIndex:
    """文档索引"""
    title: str  # 文档标题
    description: str = ""  # 文档描述
    source_file: str = ""  # 源文件路径
    total_lines: int = 0  # 总行数
    root_nodes: List[TreeNode] = field(default_factory=list)
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self, include_content: bool = False) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "title": self.title,
            "description": self.description,
            "source_file": self.source_file,
            "total_lines": self.total_lines,
            "metadata": self.metadata,
            "nodes": [node.to_dict(include_content) for node in self.root_nodes]
        }
    
    def to_json(self, include_content: bool = False, indent: int = 2) -> str:
        """转换为 JSON 字符串"""
        return json.dumps(self.to_dict(include_content), ensure_ascii=False, indent=indent)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentIndex":
        """从字典创建文档索引"""
        nodes_data = data.pop("nodes", [])
        index = cls(**data)
        index.root_nodes = [TreeNode.from_dict(node) for node in nodes_data]
        return index
    
    @classmethod
    def from_json(cls, json_str: str) -> "DocumentIndex":
        """从 JSON 字符串创建文档索引"""
        return cls.from_dict(json.loads(json_str))
    
    def get_all_nodes(self) -> List[TreeNode]:
        """获取所有节点"""
        nodes = []
        for root in self.root_nodes:
            nodes.extend(root.get_all_nodes())
        return nodes
    
    def find_node_by_id(self, node_id: str) -> Optional[TreeNode]:
        """根据 ID 查找节点"""
        for root in self.root_nodes:
            found = root.find_node_by_id(node_id)
            if found:
                return found
        return None
    
    def get_toc(self, max_depth: int = None) -> str:
        """生成目录结构（用于 LLM）"""
        lines = []
        
        def _add_node(node: TreeNode, depth: int = 0):
            if max_depth and depth >= max_depth:
                return
            indent = "  " * depth
            lines.append(f"{indent}- [{node.node_id}] {node.title}")
            for child in node.children:
                _add_node(child, depth + 1)
        
        for root in self.root_nodes:
            _add_node(root)
        
        return "\n".join(lines)



