"""
推理链模块
用于复杂查询的多步推理

参考 PageIndex 的 reasoning-based retrieval
"""

import json
import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from ..llm.base import BaseLLM

logger = logging.getLogger(__name__)


@dataclass
class ReasoningStep:
    """推理步骤"""
    step_number: int
    thought: str  # 思考过程
    action: str  # 采取的行动
    observation: str  # 观察结果
    conclusion: Optional[str] = None  # 结论（最后一步）


class ReasoningChain:
    """
    推理链
    
    实现多步推理来处理复杂查询：
    1. 分解问题
    2. 逐步检索
    3. 综合答案
    """
    
    def __init__(self, llm: BaseLLM, max_steps: int = 5):
        self.llm = llm
        self.max_steps = max_steps
    
    def decompose_query(self, query: str) -> List[str]:
        """
        分解复杂查询为子问题
        
        Args:
            query: 原始查询
            
        Returns:
            子问题列表
        """
        prompt = f"""Analyze this question and determine if it needs to be broken down into sub-questions.

Question: {query}

If this is a simple, direct question, return it as-is.
If this is a complex question that requires multiple pieces of information, break it down.

Response format:
```json
{{
    "is_complex": true/false,
    "sub_questions": ["sub-question 1", "sub-question 2", ...]
}}
```

Return ONLY the JSON."""

        try:
            response = self.llm.complete(prompt, temperature=0.1)
            result = self._parse_json(response.content)
            
            if result.get('is_complex') and result.get('sub_questions'):
                return result['sub_questions']
            return [query]
            
        except Exception as e:
            logger.warning(f"Query decomposition failed: {e}")
            return [query]
    
    def reason_about_relevance(
        self,
        query: str,
        node_title: str,
        node_summary: str,
        parent_context: str = ""
    ) -> Dict[str, Any]:
        """
        推理节点与查询的相关性
        
        Args:
            query: 查询
            node_title: 节点标题
            node_summary: 节点摘要
            parent_context: 父节点上下文
            
        Returns:
            {"relevant": bool, "confidence": float, "reason": str}
        """
        prompt = f"""Determine if this document section is relevant to answering the question.

Question: {query}

Section Title: {node_title}
Section Summary: {node_summary}
{f"Parent Context: {parent_context}" if parent_context else ""}

Think step by step:
1. What information is the question asking for?
2. What information does this section contain (based on title and summary)?
3. Is there a match?

Response format:
```json
{{
    "thinking": "step by step analysis",
    "relevant": true/false,
    "confidence": 0.0-1.0,
    "reason": "brief explanation"
}}
```

Return ONLY the JSON."""

        try:
            response = self.llm.complete(prompt, temperature=0.1)
            result = self._parse_json(response.content)
            return {
                "relevant": result.get("relevant", False),
                "confidence": result.get("confidence", 0.5),
                "reason": result.get("reason", ""),
                "thinking": result.get("thinking", "")
            }
        except Exception as e:
            logger.warning(f"Relevance reasoning failed: {e}")
            return {
                "relevant": True,  # 默认认为相关
                "confidence": 0.5,
                "reason": "Unable to determine",
                "thinking": ""
            }
    
    def verify_answer(
        self,
        query: str,
        retrieved_content: str
    ) -> Dict[str, Any]:
        """
        验证检索到的内容是否能回答问题
        
        Args:
            query: 查询
            retrieved_content: 检索到的内容
            
        Returns:
            {"contains_answer": bool, "answer_location": str, "confidence": float}
        """
        prompt = f"""Verify if the retrieved content can answer the question.

Question: {query}

Retrieved Content:
{retrieved_content[:2000]}

Check:
1. Does this content contain the information needed to answer the question?
2. Is the answer explicit or does it need to be inferred?
3. How confident are you?

Response format:
```json
{{
    "contains_answer": true/false,
    "answer_type": "explicit" | "implicit" | "partial" | "none",
    "key_information": "the specific info that answers the question",
    "confidence": 0.0-1.0
}}
```

Return ONLY the JSON."""

        try:
            response = self.llm.complete(prompt, temperature=0.1)
            return self._parse_json(response.content)
        except Exception as e:
            logger.warning(f"Answer verification failed: {e}")
            return {
                "contains_answer": True,
                "answer_type": "unknown",
                "confidence": 0.5
            }
    
    def _parse_json(self, content: str) -> Dict:
        """解析 JSON 响应"""
        content = content.strip()
        
        # 移除代码块标记
        if content.startswith('```'):
            lines = content.split('\n')
            if lines[0].strip().startswith('```'):
                lines = lines[1:]
            if lines and lines[-1].strip() == '```':
                lines = lines[:-1]
            content = '\n'.join(lines)
        
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            start = content.find('{')
            end = content.rfind('}') + 1
            if start >= 0 and end > start:
                return json.loads(content[start:end])
            raise


