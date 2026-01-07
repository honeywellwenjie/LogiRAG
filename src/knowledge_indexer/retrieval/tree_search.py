"""
增强版树搜索引擎
参考 PageIndex 的 reasoning-based retrieval，实现多轮推理和层级搜索

特性：
1. 多轮推理 - 先定位高层节点，再深入子节点
2. 置信度评分 - 为每个节点评估相关性
3. 上下文感知 - 考虑节点的 summary 和父子关系
4. 回溯验证 - 验证选择的节点是否包含答案
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from ..llm.base import BaseLLM
from ..models.tree_node import DocumentIndex, TreeNode
from ..debug_utils import (
    debug_print, debug_reasoning_round, debug_rag_results,
    debug_llm_call, debug_llm_response, DebugTimer
)

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """搜索结果"""
    doc_name: str
    node_id: str
    title: str
    relevance_score: float  # 0-1 的相关性分数
    reasoning: str  # 选择该节点的理由
    path: List[str] = field(default_factory=list)  # 从根到该节点的路径


@dataclass
class SearchContext:
    """搜索上下文"""
    query: str
    documents: Dict[str, DocumentIndex]
    node_maps: Dict[str, Dict[str, TreeNode]]
    max_results: int = 10
    min_relevance: float = 0.3


class TreeSearchEngine:
    """
    增强版树搜索引擎
    
    实现类似 PageIndex 的 reasoning-based retrieval：
    1. 第一轮：扫描所有文档的顶层结构，选择相关文档和高层章节
    2. 第二轮：深入选中的章节，找到具体的目标节点
    3. 第三轮（可选）：验证选中节点的相关性
    """
    
    def __init__(self, llm: BaseLLM, max_rounds: int = 2):
        """
        Args:
            llm: LLM 实例
            max_rounds: 最大推理轮数
        """
        self.llm = llm
        self.max_rounds = max_rounds
    
    async def search(self, context: SearchContext) -> List[SearchResult]:
        """
        执行多轮树搜索

        Args:
            context: 搜索上下文

        Returns:
            List[SearchResult]: 排序后的搜索结果
        """
        if not context.documents:
            return []

        # DEBUG: 开始树搜索
        debug_print(
            "🌳 树搜索引擎开始",
            {
                "查询": context.query,
                "文档数量": len(context.documents),
                "最大结果数": context.max_results,
                "最小相关度": context.min_relevance,
                "最大推理轮数": self.max_rounds
            },
            level="start"
        )

        search_start = time.time()

        # 第一轮：高层扫描
        logger.info(f"Round 1: High-level scan for query: {context.query}")
        debug_print("🔍 第一轮: 高层扫描", {"目标": "识别相关文档和顶层章节"}, level="search")

        high_level_candidates = await self._round1_high_level_scan(context)

        # DEBUG: 第一轮结果
        debug_print(
            "📋 第一轮结果",
            {
                "候选数量": len(high_level_candidates),
                "候选列表": high_level_candidates[:5]  # 最多显示5个
            },
            level="result"
        )

        if not high_level_candidates:
            logger.warning("No candidates found in round 1")
            debug_print("⚠️ 第一轮未找到候选节点", level="warning")
            return []

        # 第二轮：深入搜索
        logger.info(f"Round 2: Deep search in {len(high_level_candidates)} candidates")
        debug_print(
            "🔍 第二轮: 深入搜索",
            {"候选数量": len(high_level_candidates), "目标": "在候选节点中找到最具体的答案位置"},
            level="search"
        )

        detailed_results = await self._round2_deep_search(context, high_level_candidates)

        # 排序和过滤
        results = sorted(detailed_results, key=lambda x: x.relevance_score, reverse=True)
        results = [r for r in results if r.relevance_score >= context.min_relevance]
        final_results = results[:context.max_results]

        # DEBUG: 最终结果
        search_duration = time.time() - search_start
        debug_print(
            "🏁 树搜索完成",
            {
                "总耗时": f"{search_duration:.2f}秒",
                "原始结果数": len(detailed_results),
                "过滤后结果数": len(results),
                "返回结果数": len(final_results),
                "最终结果": [
                    {
                        "文档": r.doc_name,
                        "节点": r.node_id,
                        "标题": r.title,
                        "分数": round(r.relevance_score, 4),
                        "理由": r.reasoning[:100] if r.reasoning else ""
                    }
                    for r in final_results
                ]
            },
            level="end"
        )

        return final_results
    
    def search_sync(self, context: SearchContext) -> List[SearchResult]:
        """同步版本的搜索"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
                return loop.run_until_complete(self.search(context))
            else:
                return loop.run_until_complete(self.search(context))
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.search(context))
            finally:
                loop.close()
    
    async def _round1_high_level_scan(self, context: SearchContext) -> List[Dict]:
        """
        第一轮：高层扫描

        只看文档描述和顶层章节标题+摘要，选择可能相关的区域
        """
        # 构建高层结构视图
        high_level_view = self._build_high_level_view(context.documents)

        prompt = f"""You are an expert document retrieval system. Given a question and document structures, identify the most relevant sections.

## Question
{context.query}

## Documents Overview
{json.dumps(high_level_view, indent=2, ensure_ascii=False)}

## Task
Analyze each document and its top-level sections. For each potentially relevant section, assess how likely it contains information to answer the question.

Consider:
1. Document title and description - does this document likely contain the answer?
2. Section titles and summaries - which sections are most relevant?
3. Think about what specific information the question is asking for

## Response Format
Return a JSON object:
```json
{{
    "analysis": "<Brief analysis of which documents/sections are relevant and why>",
    "candidates": [
        {{
            "doc_name": "document_name",
            "node_id": "section_id",
            "relevance": 0.9,
            "reason": "This section likely contains..."
        }}
    ]
}}
```

Select up to 5 most relevant sections. Be selective - only include sections with relevance >= 0.5.
Return ONLY the JSON, no other text."""

        # DEBUG: 记录第一轮推理请求
        debug_reasoning_round(
            round_num=1,
            candidates_count=sum(len(doc.get('sections', [])) for doc in high_level_view.get('documents', [])),
            prompt=prompt,
            response="(等待 LLM 响应...)"
        )

        try:
            start_time = time.time()
            response = self.llm.complete(prompt, temperature=0.1)
            duration = time.time() - start_time

            # DEBUG: 记录第一轮推理响应
            debug_reasoning_round(
                round_num=1,
                candidates_count=sum(len(doc.get('sections', [])) for doc in high_level_view.get('documents', [])),
                prompt=f"(已记录，耗时 {duration:.2f}秒)",
                response=response.content
            )

            result = self._parse_json_response(response.content)
            return result.get('candidates', [])
        except Exception as e:
            logger.error(f"Round 1 search failed: {e}")
            debug_print(f"❌ 第一轮推理失败: {e}", level="error")
            # Fallback: 返回所有顶层节点
            return self._fallback_candidates(context.documents)
    
    async def _round2_deep_search(
        self,
        context: SearchContext,
        candidates: List[Dict]
    ) -> List[SearchResult]:
        """
        第二轮：深入搜索

        对每个候选区域进行详细搜索，找到最相关的具体节点
        """
        results = []

        for idx, candidate in enumerate(candidates):
            doc_name = candidate.get('doc_name')
            node_id = candidate.get('node_id')
            base_relevance = candidate.get('relevance', 0.5)

            # DEBUG: 处理每个候选
            debug_print(
                f"🔎 第二轮: 处理候选 {idx + 1}/{len(candidates)}",
                {"文档": doc_name, "节点": node_id, "基础相关度": base_relevance},
                level="search"
            )

            if doc_name not in context.node_maps:
                debug_print(f"⚠️ 文档不存在: {doc_name}", level="warning")
                continue

            node_map = context.node_maps[doc_name]
            if node_id not in node_map:
                debug_print(f"⚠️ 节点不存在: {node_id}", level="warning")
                continue

            node = node_map[node_id]

            # 如果节点没有子节点，直接返回
            if not node.children:
                debug_print(
                    f"📌 叶子节点，直接使用",
                    {"标题": node.title, "相关度": base_relevance},
                    level="info"
                )
                results.append(SearchResult(
                    doc_name=doc_name,
                    node_id=node_id,
                    title=node.title,
                    relevance_score=base_relevance,
                    reasoning=candidate.get('reason', ''),
                    path=[node.title]
                ))
                continue

            # 有子节点，深入搜索
            subtree_view = self._build_subtree_view(node, doc_name)

            prompt = f"""You are searching within a document section to find the most specific answer location.

## Question
{context.query}

## Current Section
Title: {node.title}
Summary: {node.summary or 'No summary'}

## Subsections
{json.dumps(subtree_view, indent=2, ensure_ascii=False)}

## Task
Find the most specific subsection(s) that contain the answer.
- If the answer is in a specific subsection, select that subsection
- If the answer spans multiple subsections or is in the parent section itself, select the parent
- Consider both title and summary when making decisions

## Response Format
```json
{{
    "selected_nodes": [
        {{
            "node_id": "specific_node_id",
            "relevance": 0.95,
            "reason": "This subsection specifically discusses..."
        }}
    ]
}}
```

Return ONLY the JSON."""

            # DEBUG: 记录第二轮推理
            debug_reasoning_round(
                round_num=2,
                candidates_count=len(node.children),
                prompt=prompt,
                response="(等待 LLM 响应...)"
            )

            try:
                start_time = time.time()
                response = self.llm.complete(prompt, temperature=0.1)
                duration = time.time() - start_time

                # DEBUG: 记录响应
                debug_reasoning_round(
                    round_num=2,
                    candidates_count=len(node.children),
                    prompt=f"(已记录，耗时 {duration:.2f}秒)",
                    response=response.content
                )

                sub_result = self._parse_json_response(response.content)

                for selected in sub_result.get('selected_nodes', []):
                    sel_node_id = selected.get('node_id')
                    if sel_node_id in node_map:
                        sel_node = node_map[sel_node_id]
                        debug_print(
                            f"✅ 选中子节点",
                            {"节点": sel_node_id, "标题": sel_node.title, "相关度": selected.get('relevance')},
                            level="success"
                        )
                        results.append(SearchResult(
                            doc_name=doc_name,
                            node_id=sel_node_id,
                            title=sel_node.title,
                            relevance_score=selected.get('relevance', base_relevance),
                            reasoning=selected.get('reason', ''),
                            path=self._get_node_path(sel_node_id, node_map)
                        ))
                    else:
                        # 如果返回的 node_id 无效，使用父节点
                        debug_print(
                            f"⚠️ 返回的节点ID无效，使用父节点",
                            {"无效ID": sel_node_id, "使用": node_id},
                            level="warning"
                        )
                        results.append(SearchResult(
                            doc_name=doc_name,
                            node_id=node_id,
                            title=node.title,
                            relevance_score=base_relevance * 0.8,
                            reasoning=candidate.get('reason', ''),
                            path=[node.title]
                        ))

            except Exception as e:
                logger.warning(f"Round 2 deep search failed for {node_id}: {e}")
                debug_print(f"❌ 第二轮深入搜索失败: {e}", {"节点": node_id}, level="error")
                # Fallback: 使用候选节点本身
                results.append(SearchResult(
                    doc_name=doc_name,
                    node_id=node_id,
                    title=node.title,
                    relevance_score=base_relevance,
                    reasoning=candidate.get('reason', ''),
                    path=[node.title]
                ))

        return results
    
    def _build_high_level_view(self, documents: Dict[str, DocumentIndex]) -> Dict:
        """构建高层视图（只包含文档描述和顶层章节）"""
        view = {"documents": []}
        
        for doc_name, index in documents.items():
            doc_view = {
                "doc_name": doc_name,
                "title": index.title,
                "description": index.description[:500] if index.description else "",
                "sections": []
            }
            
            # 只包含顶层节点（level 1-2）
            for node in index.root_nodes:
                section_view = {
                    "node_id": node.node_id,
                    "title": node.title,
                    "level": node.level,
                    "summary": node.summary[:200] if node.summary else "",
                    "has_children": len(node.children) > 0
                }
                doc_view["sections"].append(section_view)
                
                # 也包含二级节点
                for child in node.children[:5]:  # 限制数量
                    child_view = {
                        "node_id": child.node_id,
                        "title": child.title,
                        "level": child.level,
                        "summary": child.summary[:150] if child.summary else "",
                        "has_children": len(child.children) > 0
                    }
                    doc_view["sections"].append(child_view)
            
            view["documents"].append(doc_view)
        
        return view
    
    def _build_subtree_view(self, node: TreeNode, doc_name: str) -> Dict:
        """构建子树视图"""
        def node_to_dict(n: TreeNode, depth: int = 0) -> Dict:
            result = {
                "node_id": n.node_id,
                "title": n.title,
                "level": n.level,
                "summary": n.summary[:200] if n.summary else "",
            }
            if n.children and depth < 2:  # 限制深度
                result["children"] = [node_to_dict(c, depth + 1) for c in n.children]
            return result
        
        return {
            "parent": {
                "node_id": node.node_id,
                "title": node.title,
                "summary": node.summary[:200] if node.summary else "",
            },
            "children": [node_to_dict(c) for c in node.children]
        }
    
    def _get_node_path(self, node_id: str, node_map: Dict[str, TreeNode]) -> List[str]:
        """获取节点路径"""
        # 简化实现：只返回节点标题
        if node_id in node_map:
            return [node_map[node_id].title]
        return []
    
    def _fallback_candidates(self, documents: Dict[str, DocumentIndex]) -> List[Dict]:
        """Fallback：返回所有顶层节点作为候选"""
        candidates = []
        for doc_name, index in documents.items():
            for node in index.root_nodes[:3]:  # 限制数量
                candidates.append({
                    "doc_name": doc_name,
                    "node_id": node.node_id,
                    "relevance": 0.5,
                    "reason": "Fallback selection"
                })
        return candidates
    
    def _parse_json_response(self, content: str) -> Dict:
        """解析 LLM 返回的 JSON"""
        content = content.strip()
        
        # 移除 markdown 代码块
        if content.startswith('```'):
            lines = content.split('\n')
            if lines[0].strip().startswith('```'):
                lines = lines[1:]
            if lines and lines[-1].strip() == '```':
                lines = lines[:-1]
            content = '\n'.join(lines)
        
        # 尝试直接解析
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        
        # 尝试提取 JSON
        start = content.find('{')
        end = content.rfind('}') + 1
        if start >= 0 and end > start:
            try:
                return json.loads(content[start:end])
            except json.JSONDecodeError:
                pass
        
        logger.warning(f"Failed to parse JSON: {content[:200]}")
        return {}


class SimpleTreeSearch:
    """
    简化版树搜索（单轮）
    
    用于对比测试或作为 fallback
    """
    
    def __init__(self, llm: BaseLLM):
        self.llm = llm
    
    def search(
        self, 
        query: str, 
        tree_structure: Dict,
        max_results: int = 5
    ) -> List[Dict]:
        """单轮树搜索"""
        prompt = f"""You are given a question and tree structures of multiple documents.
Each node contains a node_id, title, level, and summary.
Your task is to find all nodes that are likely to contain the answer to the question.

Question: {query}

Document tree structures:
{json.dumps(tree_structure, indent=2, ensure_ascii=False)}

Important: 
- Pay attention to node summaries - they describe what information each section contains
- Select the most SPECIFIC nodes that would answer the question
- Include child nodes if they are more specific than parent nodes
- Consider nodes related to: contact info, personal details, work history, skills, etc.

Please reply in the following JSON format:
{{
    "thinking": "<Your step-by-step reasoning about which nodes are relevant>",
    "node_list": [
        {{"doc_name": "document_name", "node_id": "node_id_1", "relevance": 0.9}},
        {{"doc_name": "document_name", "node_id": "node_id_2", "relevance": 0.8}}
    ]
}}

Select up to {max_results} most relevant nodes.
Return ONLY the JSON structure."""

        try:
            response = self.llm.complete(prompt, temperature=0.1)
            content = response.content.strip()
            
            # 解析 JSON
            if content.startswith('```'):
                lines = content.split('\n')
                if lines[0].strip().startswith('```'):
                    lines = lines[1:]
                if lines and lines[-1].strip() == '```':
                    lines = lines[:-1]
                content = '\n'.join(lines)
            
            try:
                result = json.loads(content)
                return result.get('node_list', [])
            except json.JSONDecodeError:
                start = content.find('{')
                end = content.rfind('}') + 1
                if start >= 0 and end > start:
                    result = json.loads(content[start:end])
                    return result.get('node_list', [])
                raise
                
        except Exception as e:
            logger.error(f"Simple tree search failed: {e}")
            return []


