"""
调试工具模块 - 提供详细的调试日志输出
用于追踪 RAG 系统的完整运行流程
"""

import logging
import json
import time
from typing import Any, Dict, List, Optional
from functools import wraps
from datetime import datetime

# 配置调试日志格式
DEBUG_SEPARATOR = "=" * 80
DEBUG_SUBSEP = "-" * 60

# 创建专用的调试 logger
debug_logger = logging.getLogger("logirag.debug")
debug_logger.setLevel(logging.DEBUG)

# 确保有处理器
if not debug_logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '\n%(asctime)s [DEBUG] %(message)s',
        datefmt='%H:%M:%S'
    )
    handler.setFormatter(formatter)
    debug_logger.addHandler(handler)


def debug_print(title: str, content: Any = None, level: str = "info"):
    """
    打印调试信息

    Args:
        title: 标题
        content: 内容（可以是字符串、字典、列表等）
        level: 日志级别 (info, success, warning, error)
    """
    level_icons = {
        "info": "ℹ️",
        "success": "✅",
        "warning": "⚠️",
        "error": "❌",
        "start": "🚀",
        "end": "🏁",
        "llm": "🤖",
        "search": "🔍",
        "result": "📋",
    }

    icon = level_icons.get(level, "📌")

    print(f"\n{DEBUG_SEPARATOR}")
    print(f"{icon} {title}")
    print(DEBUG_SUBSEP)

    if content is not None:
        if isinstance(content, dict):
            print(json.dumps(content, indent=2, ensure_ascii=False, default=str))
        elif isinstance(content, list):
            for i, item in enumerate(content):
                if isinstance(item, dict):
                    print(f"[{i}] {json.dumps(item, indent=2, ensure_ascii=False, default=str)}")
                else:
                    print(f"[{i}] {item}")
        else:
            print(str(content))

    print(DEBUG_SEPARATOR)


def debug_request(endpoint: str, method: str, data: Dict):
    """记录用户请求"""
    debug_print(
        f"📥 用户请求 [{method}] {endpoint}",
        {
            "时间": datetime.now().isoformat(),
            "端点": endpoint,
            "方法": method,
            "请求数据": data
        },
        level="start"
    )


def debug_llm_call(purpose: str, prompt: str, system_prompt: str = None, model: str = None):
    """记录 LLM 调用 - 请求"""
    content = {
        "目的": purpose,
        "模型": model or "未知",
        "时间": datetime.now().isoformat(),
    }

    if system_prompt:
        content["系统提示词"] = system_prompt[:500] + "..." if len(system_prompt) > 500 else system_prompt

    content["用户提示词"] = prompt[:2000] + "..." if len(prompt) > 2000 else prompt

    debug_print(f"🤖 LLM 调用请求 - {purpose}", content, level="llm")


def debug_llm_response(purpose: str, response: str, usage: Dict = None, duration: float = None):
    """记录 LLM 调用 - 响应"""
    content = {
        "目的": purpose,
        "响应长度": len(response) if response else 0,
    }

    if duration:
        content["耗时"] = f"{duration:.2f}秒"

    if usage:
        content["Token 使用"] = usage

    # 显示完整响应或截断
    if response:
        content["响应内容"] = response[:3000] + "..." if len(response) > 3000 else response

    debug_print(f"🤖 LLM 调用响应 - {purpose}", content, level="llm")


def debug_rag_search_start(query: str, mode: str, documents_count: int):
    """记录 RAG 搜索开始"""
    debug_print(
        "🔍 RAG 搜索开始",
        {
            "查询": query,
            "检索模式": mode,
            "文档数量": documents_count,
            "时间": datetime.now().isoformat()
        },
        level="search"
    )


def debug_vector_search(query: str, top_k: int, results: List, duration: float = None):
    """记录向量搜索结果"""
    result_summary = []
    for r in results[:10]:  # 最多显示10个
        result_summary.append({
            "文档": getattr(r, 'doc_name', 'unknown'),
            "节点ID": getattr(r, 'node_id', 'unknown'),
            "标题": getattr(r, 'title', 'unknown'),
            "分数": round(getattr(r, 'score', 0), 4)
        })

    debug_print(
        "🔍 向量搜索结果",
        {
            "查询": query[:100],
            "请求数量": top_k,
            "返回数量": len(results),
            "耗时": f"{duration:.2f}秒" if duration else "未知",
            "结果列表": result_summary
        },
        level="search"
    )


def debug_reasoning_round(round_num: int, candidates_count: int, prompt: str, response: str):
    """记录推理轮次"""
    debug_print(
        f"🧠 推理第 {round_num} 轮",
        {
            "候选节点数": candidates_count,
            "提示词": prompt[:1500] + "..." if len(prompt) > 1500 else prompt,
            "LLM 响应": response[:1500] + "..." if len(response) > 1500 else response
        },
        level="llm"
    )


def debug_rag_results(results: List, mode: str, duration: float = None):
    """记录 RAG 搜索结果"""
    result_summary = []
    for r in results[:10]:
        if hasattr(r, 'doc_name'):
            # HybridSearchResult 或 SearchResult
            result_summary.append({
                "文档": r.doc_name,
                "节点ID": r.node_id,
                "标题": getattr(r, 'title', 'unknown'),
                "最终分数": round(getattr(r, 'final_score', getattr(r, 'relevance_score', 0)), 4),
                "向量分数": round(getattr(r, 'vector_score', 0) or 0, 4),
                "推理分数": round(getattr(r, 'reasoning_score', 0) or 0, 4),
                "来源": getattr(r, 'source', mode),
                "理由": getattr(r, 'reasoning', '')[:100]
            })
        elif isinstance(r, dict):
            result_summary.append({
                "文档": r.get('doc_name', 'unknown'),
                "节点ID": r.get('node_id', 'unknown'),
                "相关度": round(r.get('relevance', 0), 4)
            })

    debug_print(
        "📋 RAG 搜索结果汇总",
        {
            "检索模式": mode,
            "结果数量": len(results),
            "耗时": f"{duration:.2f}秒" if duration else "未知",
            "命中列表": result_summary
        },
        level="result"
    )


def debug_context_retrieval(contexts: List[Dict]):
    """记录上下文提取"""
    context_summary = []
    total_chars = 0

    for ctx in contexts:
        content = ctx.get('content', '')
        content_len = len(content)
        total_chars += content_len
        context_summary.append({
            "文档": ctx.get('doc_name', 'unknown'),
            "节点ID": ctx.get('node_id', 'unknown'),
            "标题": ctx.get('title', 'unknown'),
            "内容长度": content_len,
            "相关度": round(ctx.get('relevance', 0), 4),
            "内容预览": content[:200] + "..." if len(content) > 200 else content
        })

    debug_print(
        "📚 上下文提取结果",
        {
            "提取节点数": len(contexts),
            "总字符数": total_chars,
            "估计Token": total_chars // 3,
            "节点详情": context_summary
        },
        level="result"
    )


def debug_chat_response(query: str, response: str, context_used: bool, duration: float = None):
    """记录聊天响应"""
    debug_print(
        "💬 聊天响应生成",
        {
            "用户问题": query,
            "使用知识库": "是" if context_used else "否",
            "响应长度": len(response),
            "耗时": f"{duration:.2f}秒" if duration else "未知",
            "响应内容": response
        },
        level="end"
    )


def debug_response(endpoint: str, status: str, data: Dict, duration: float = None):
    """记录最终响应"""
    summary = {
        "端点": endpoint,
        "状态": status,
        "耗时": f"{duration:.2f}秒" if duration else "未知",
    }

    # 添加关键响应数据的摘要
    if 'context' in data:
        summary["上下文长度"] = len(data.get('context', ''))
    if 'nodes' in data:
        summary["命中节点数"] = len(data.get('nodes', []))
    if 'source_files' in data:
        summary["来源文档"] = data.get('source_files', [])
    if 'mode' in data:
        summary["检索模式"] = data.get('mode')
    if 'thinking' in data:
        thinking = data.get('thinking', '')
        summary["推理过程"] = thinking[:500] + "..." if len(thinking) > 500 else thinking

    debug_print(f"📤 响应完成 - {endpoint}", summary, level="end")


class DebugTimer:
    """调试计时器上下文管理器"""

    def __init__(self, name: str):
        self.name = name
        self.start_time = None
        self.duration = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.duration = time.time() - self.start_time
        return False

    def elapsed(self) -> float:
        if self.duration is not None:
            return self.duration
        return time.time() - self.start_time


def debug_decorator(name: str):
    """调试装饰器 - 记录函数执行时间和参数"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            debug_print(f"⏱️ 开始执行: {name}", {"参数": str(kwargs)[:500]}, level="start")
            start = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start
                debug_print(f"⏱️ 执行完成: {name}", {"耗时": f"{duration:.2f}秒"}, level="end")
                return result
            except Exception as e:
                duration = time.time() - start
                debug_print(f"⏱️ 执行失败: {name}", {"错误": str(e), "耗时": f"{duration:.2f}秒"}, level="error")
                raise
        return wrapper
    return decorator
