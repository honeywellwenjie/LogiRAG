"""
调试工具模块 - 提供详细的调试日志输出
用于追踪 RAG 系统的完整运行流程

优化目标：
- 可读性：JSON 格式化显示，换行符正确渲染
- 简洁性：去除时间戳，聚焦关键信息
- 明确性：清晰展示混合模式各阶段命中情况
"""

import logging
import json
import re
import time
from typing import Any, Dict, List, Optional
from functools import wraps

# 配置调试日志格式
DEBUG_SEPARATOR = "=" * 70
DEBUG_SUBSEP = "-" * 50

# 创建专用的调试 logger（无时间戳）
debug_logger = logging.getLogger("logirag.debug")
debug_logger.setLevel(logging.DEBUG)

# 确保有处理器
if not debug_logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('\n[DEBUG] %(message)s')
    handler.setFormatter(formatter)
    debug_logger.addHandler(handler)


def _format_json_readable(data: Any, indent: int = 2) -> str:
    """
    将数据格式化为可读的格式
    - JSON 字符串中的 \\n 转为真正的换行
    - 移除多余的转义符号
    """
    if data is None:
        return "null"

    try:
        # 先转为 JSON 字符串
        json_str = json.dumps(data, indent=indent, ensure_ascii=False, default=str)
        # 不做额外处理，保持 JSON 格式
        return json_str
    except Exception:
        return str(data)


def _format_llm_response(response: str) -> str:
    """
    格式化 LLM 响应，使其更可读
    - 提取并格式化 JSON 块
    - 换行符正确显示
    """
    if not response:
        return "(空响应)"

    # 尝试提取 JSON 块
    json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response)
    if json_match:
        json_str = json_match.group(1)
        try:
            parsed = json.loads(json_str)
            return _format_parsed_llm_json(parsed)
        except json.JSONDecodeError:
            pass

    # 尝试直接解析为 JSON
    try:
        parsed = json.loads(response)
        return _format_parsed_llm_json(parsed)
    except json.JSONDecodeError:
        pass

    # 普通文本：换行符正确显示
    return response.replace('\\n', '\n')


def _format_parsed_llm_json(parsed: dict) -> str:
    """格式化解析后的 LLM JSON 响应"""
    lines = []

    # 分析部分
    if 'analysis' in parsed:
        lines.append("【分析】")
        lines.append(parsed['analysis'])
        lines.append("")

    # 候选节点
    if 'candidates' in parsed:
        lines.append("【候选节点】")
        for i, c in enumerate(parsed['candidates'], 1):
            doc = c.get('doc_name', 'unknown')
            node = c.get('node_id', 'unknown')
            rel = c.get('relevance', 0)
            reason = c.get('reason', '')
            lines.append(f"  {i}. [{doc}:{node}] 相关度={rel}")
            if reason:
                lines.append(f"     原因: {reason[:100]}...")
        lines.append("")

    # 选中节点
    if 'selected_nodes' in parsed:
        lines.append("【选中节点】")
        for i, s in enumerate(parsed['selected_nodes'], 1):
            node = s.get('node_id', 'unknown')
            rel = s.get('relevance', 0)
            reason = s.get('reason', '')
            lines.append(f"  {i}. [{node}] 相关度={rel}")
            if reason:
                lines.append(f"     原因: {reason[:100]}...")
        lines.append("")

    if lines:
        return '\n'.join(lines)
    else:
        # 回退到格式化 JSON
        return json.dumps(parsed, indent=2, ensure_ascii=False)


def debug_print(title: str, content: Any = None, level: str = "info"):
    """
    打印调试信息

    Args:
        title: 标题
        content: 内容（可以是字符串、字典、列表等）
        level: 日志级别 (info, success, warning, error, start, end, llm, search, result)
    """
    level_icons = {
        "info": "ℹ️ ",
        "success": "✅",
        "warning": "⚠️ ",
        "error": "❌",
        "start": "🚀",
        "end": "🏁",
        "llm": "🤖",
        "search": "🔍",
        "result": "📋",
        "vector": "📊",
        "hybrid": "🔀",
    }

    icon = level_icons.get(level, "📌")

    print(f"\n{DEBUG_SEPARATOR}")
    print(f"{icon} {title}")
    print(DEBUG_SUBSEP)

    if content is not None:
        if isinstance(content, dict):
            _print_dict_readable(content)
        elif isinstance(content, list):
            for i, item in enumerate(content):
                if isinstance(item, dict):
                    print(f"[{i}]")
                    _print_dict_readable(item, indent=2)
                else:
                    print(f"[{i}] {item}")
        else:
            print(str(content))

    print(DEBUG_SEPARATOR)


def _print_dict_readable(d: dict, indent: int = 0):
    """可读地打印字典，特殊处理 LLM 响应"""
    prefix = " " * indent

    for key, value in d.items():
        if key in ("LLM 响应", "响应内容", "LLM响应") and isinstance(value, str):
            # 特殊处理 LLM 响应
            print(f"{prefix}{key}:")
            formatted = _format_llm_response(value)
            for line in formatted.split('\n'):
                print(f"{prefix}  {line}")
        elif key in ("用户提示词", "系统提示词", "提示词") and isinstance(value, str):
            # 提示词：显示前200字符
            preview = value[:200] + "..." if len(value) > 200 else value
            print(f"{prefix}{key}: {preview}")
        elif isinstance(value, dict):
            print(f"{prefix}{key}:")
            _print_dict_readable(value, indent + 2)
        elif isinstance(value, list):
            if len(value) == 0:
                print(f"{prefix}{key}: []")
            elif all(isinstance(x, dict) for x in value):
                print(f"{prefix}{key}:")
                for i, item in enumerate(value):
                    print(f"{prefix}  [{i}]")
                    _print_dict_readable(item, indent + 4)
            else:
                print(f"{prefix}{key}: {value}")
        else:
            print(f"{prefix}{key}: {value}")


def debug_llm_call(purpose: str, prompt: str, system_prompt: str = None, model: str = None):
    """记录 LLM 调用 - 请求（简化版）"""
    print(f"\n{DEBUG_SUBSEP}")
    print(f"🤖 LLM 请求: {purpose}")
    print(f"   模型: {model or '未知'}")
    print(f"   提示词长度: {len(prompt)} 字符")
    print(DEBUG_SUBSEP)


def debug_llm_response(purpose: str, response: str, usage: Dict = None, duration: float = None):
    """记录 LLM 调用 - 响应（格式化显示）"""
    print(f"\n{DEBUG_SUBSEP}")
    print(f"🤖 LLM 响应: {purpose}")
    if duration:
        print(f"   耗时: {duration:.2f}秒")
    if usage:
        print(f"   Token: {usage}")
    print(DEBUG_SUBSEP)

    # 格式化显示响应内容
    formatted = _format_llm_response(response)
    print(formatted)
    print(DEBUG_SUBSEP)


def debug_rag_search_start(query: str, mode: str, documents_count: int):
    """记录 RAG 搜索开始"""
    print(f"\n{'=' * 70}")
    print(f"🔍 RAG 搜索开始")
    print(f"   查询: {query}")
    print(f"   模式: {mode}")
    print(f"   文档数: {documents_count}")
    print(f"{'=' * 70}")


def debug_vector_search(query: str, top_k: int, results: List, duration: float = None):
    """记录向量搜索结果（详细版）"""
    print(f"\n{DEBUG_SEPARATOR}")
    print(f"📊 向量搜索完成")
    print(DEBUG_SUBSEP)
    print(f"   查询: {query[:50]}...")
    print(f"   请求 top_k: {top_k}")
    print(f"   返回数量: {len(results)}")
    if duration:
        print(f"   耗时: {duration:.3f}秒")
    print(DEBUG_SUBSEP)

    if results:
        print("   命中节点:")
        for i, r in enumerate(results[:10], 1):
            doc = getattr(r, 'doc_name', 'unknown')
            node = getattr(r, 'node_id', 'unknown')
            title = getattr(r, 'title', 'unknown')
            score = getattr(r, 'score', 0)
            print(f"   {i:2d}. [{doc}:{node}] score={score:.4f}")
            print(f"       标题: {title[:40]}...")
    else:
        print("   (无命中结果)")
    print(DEBUG_SEPARATOR)


def debug_reasoning_round(round_num: int, candidates_count: int, prompt: str, response: str):
    """记录推理轮次（格式化 LLM 响应）"""
    print(f"\n{DEBUG_SEPARATOR}")
    print(f"🧠 推理第 {round_num} 轮")
    print(DEBUG_SUBSEP)
    print(f"   候选节点数: {candidates_count}")
    print(f"   提示词长度: {len(prompt)} 字符")
    print(DEBUG_SUBSEP)
    print("LLM 响应:")
    formatted = _format_llm_response(response)
    print(formatted)
    print(DEBUG_SEPARATOR)


def debug_rag_results(results: List, mode: str, duration: float = None):
    """记录 RAG 搜索结果（混合模式详细版）"""
    print(f"\n{'=' * 70}")
    print(f"📋 RAG 搜索结果汇总")
    print(f"{'=' * 70}")
    print(f"   检索模式: {mode}")
    print(f"   结果数量: {len(results)}")
    if duration:
        print(f"   总耗时: {duration:.2f}秒")
    print(DEBUG_SUBSEP)

    if results:
        print("命中详情:")
        print(f"{'序号':<4} {'文档':<25} {'节点':<8} {'最终分':<8} {'向量分':<8} {'推理分':<8} {'来源':<8}")
        print("-" * 70)

        for i, r in enumerate(results[:10], 1):
            if hasattr(r, 'doc_name'):
                doc = r.doc_name[:24]
                node = r.node_id
                final = getattr(r, 'final_score', getattr(r, 'relevance_score', 0))
                vec = getattr(r, 'vector_score', None)
                reas = getattr(r, 'reasoning_score', None)
                source = getattr(r, 'source', mode)

                vec_str = f"{vec:.4f}" if vec else "-"
                reas_str = f"{reas:.4f}" if reas else "-"

                print(f"{i:<4} {doc:<25} {node:<8} {final:<8.4f} {vec_str:<8} {reas_str:<8} {source:<8}")

                # 显示推理原因（如果有）
                reasoning = getattr(r, 'reasoning', '')
                if reasoning:
                    print(f"     原因: {reasoning[:60]}...")
    else:
        print("   (无命中结果)")

    print(f"{'=' * 70}")


def debug_hybrid_stage(stage: str, info: dict):
    """记录混合搜索的各个阶段"""
    stage_icons = {
        "vector_start": "📊 向量预过滤开始",
        "vector_done": "📊 向量预过滤完成",
        "filter_docs": "📁 文档过滤",
        "reasoning_start": "🧠 LLM推理开始",
        "reasoning_done": "🧠 LLM推理完成",
        "fusion": "🔀 结果融合",
        "final": "✅ 最终结果",
    }

    title = stage_icons.get(stage, f"📌 {stage}")
    print(f"\n{DEBUG_SUBSEP}")
    print(f"{title}")

    for key, value in info.items():
        if isinstance(value, list) and len(value) > 5:
            print(f"   {key}: [{len(value)} 项]")
            for item in value[:3]:
                print(f"     - {item}")
            print(f"     ... 还有 {len(value) - 3} 项")
        else:
            print(f"   {key}: {value}")

    print(DEBUG_SUBSEP)


def debug_context_retrieval(contexts: List[Dict]):
    """记录上下文提取"""
    total_chars = sum(len(ctx.get('content', '')) for ctx in contexts)

    print(f"\n{DEBUG_SEPARATOR}")
    print(f"📚 上下文提取结果")
    print(DEBUG_SUBSEP)
    print(f"   提取节点数: {len(contexts)}")
    print(f"   总字符数: {total_chars}")
    print(f"   估计Token: {total_chars // 3}")
    print(DEBUG_SUBSEP)

    for i, ctx in enumerate(contexts[:5], 1):
        doc = ctx.get('doc_name', 'unknown')
        node = ctx.get('node_id', 'unknown')
        title = ctx.get('title', 'unknown')
        rel = ctx.get('relevance', 0)
        content = ctx.get('content', '')

        print(f"{i}. [{doc}:{node}] 相关度={rel:.4f}")
        print(f"   标题: {title}")
        print(f"   内容: {content[:100]}...")
        print()

    print(DEBUG_SEPARATOR)


def debug_chat_response(query: str, response: str, context_used: bool, duration: float = None):
    """记录聊天响应"""
    print(f"\n{DEBUG_SEPARATOR}")
    print(f"💬 聊天响应生成")
    print(DEBUG_SUBSEP)
    print(f"   用户问题: {query}")
    print(f"   使用知识库: {'是' if context_used else '否'}")
    print(f"   响应长度: {len(response)} 字符")
    if duration:
        print(f"   耗时: {duration:.2f}秒")
    print(DEBUG_SUBSEP)
    print("响应内容:")
    print(response)
    print(DEBUG_SEPARATOR)


def debug_response(endpoint: str, status: str, data: Dict, duration: float = None):
    """记录最终响应"""
    print(f"\n{DEBUG_SEPARATOR}")
    print(f"📤 响应完成 - {endpoint}")
    print(DEBUG_SUBSEP)
    print(f"   状态: {status}")
    if duration:
        print(f"   耗时: {duration:.2f}秒")

    if 'context' in data:
        print(f"   上下文长度: {len(data.get('context', ''))} 字符")
    if 'nodes' in data:
        print(f"   命中节点数: {len(data.get('nodes', []))}")
    if 'source_files' in data:
        print(f"   来源文档: {data.get('source_files', [])}")
    if 'mode' in data:
        print(f"   检索模式: {data.get('mode')}")

    print(DEBUG_SEPARATOR)


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
    """调试装饰器 - 记录函数执行时间"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            print(f"\n⏱️  开始: {name}")
            start = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start
                print(f"⏱️  完成: {name} ({duration:.2f}秒)")
                return result
            except Exception as e:
                duration = time.time() - start
                print(f"⏱️  失败: {name} ({duration:.2f}秒) - {e}")
                raise
        return wrapper
    return decorator
