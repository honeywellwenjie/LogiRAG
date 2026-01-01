#!/usr/bin/env python3
"""
RAG API Server based on Knowledge Indexer
Provides API endpoint for querying knowledge base using reasoning-based retrieval

增强功能（参考 PageIndex）：
1. 多轮推理搜索
2. 节点 summary 感知
3. 层级搜索
4. 置信度评分

Port: 3003
"""

import os
import json
import asyncio
import logging
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# 添加 src 到路径
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from knowledge_indexer import LLMFactory, IndexerConfig, TreeBuilder, DocumentIndex
from knowledge_indexer.llm.base import BaseLLM
from knowledge_indexer.retrieval.tree_search import TreeSearchEngine, SimpleTreeSearch, SearchContext
from knowledge_indexer.retrieval.reasoning import ReasoningChain

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
if os.path.exists('/app/result'):
    KNOWLEDGE_BASE_DIR = '/app/knowledge_base'
    RESULTS_DIR = '/app/result'
else:
    PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
    KNOWLEDGE_BASE_DIR = os.path.join(PROJECT_ROOT, 'knowledge_base')
    RESULTS_DIR = os.path.join(PROJECT_ROOT, 'result')

os.makedirs(KNOWLEDGE_BASE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Global variables
llm: BaseLLM = None
document_indexes: dict = {}
node_maps: dict = {}
search_engine: TreeSearchEngine = None
simple_search: SimpleTreeSearch = None
chat_llm: BaseLLM = None  # Separate LLM for chat responses


def get_llm() -> BaseLLM:
    """Get or create LLM instance for RAG search and indexing"""
    global llm
    if llm is None:
        try:
            config = IndexerConfig.from_file()
            llm = LLMFactory.from_config(config.rag_llm)
            logger.info(f"RAG LLM initialized: {config.rag_llm.provider}/{config.rag_llm.model}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise
    return llm


def get_chat_llm() -> BaseLLM:
    """Get or create LLM instance for chat responses (can be different from RAG LLM)"""
    global chat_llm
    if chat_llm is None:
        try:
            config = IndexerConfig.from_file()
            # Check if chat_llm is configured separately
            if hasattr(config, 'chat_llm') and config.chat_llm:
                chat_llm = LLMFactory.from_config(config.chat_llm)
                logger.info(f"Chat LLM initialized: {config.chat_llm.provider}/{config.chat_llm.model}")
            else:
                # Fall back to main LLM
                chat_llm = get_llm()
                logger.info("Chat LLM: using main LLM config")
        except Exception as e:
            logger.error(f"Failed to initialize Chat LLM: {e}, falling back to main LLM")
            chat_llm = get_llm()
    return chat_llm


def get_search_engine() -> TreeSearchEngine:
    """Get or create search engine"""
    global search_engine
    if search_engine is None:
        search_engine = TreeSearchEngine(get_llm(), max_rounds=2)
    return search_engine


def get_simple_search() -> SimpleTreeSearch:
    """Get or create simple search"""
    global simple_search
    if simple_search is None:
        simple_search = SimpleTreeSearch(get_llm())
    return simple_search


def create_node_mapping(index: DocumentIndex) -> dict:
    """Create a mapping from node_id to node for quick lookup"""
    node_map = {}
    for node in index.get_all_nodes():
        if node.node_id:
            node_map[node.node_id] = node
    return node_map


def load_all_indexes():
    """Load all index files from results directory"""
    global document_indexes, node_maps
    
    for filename in os.listdir(RESULTS_DIR):
        if filename.endswith('_index.json'):
            filepath = os.path.join(RESULTS_DIR, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                index = DocumentIndex.from_dict(data)
                doc_name = filename.replace('_index.json', '')
                document_indexes[doc_name] = index
                node_maps[doc_name] = create_node_mapping(index)
                logger.info(f"Loaded index: {filename} ({len(index.get_all_nodes())} nodes)")
            except Exception as e:
                logger.error(f"Failed to load index {filename}: {e}")
    
    for filename in os.listdir(KNOWLEDGE_BASE_DIR):
        if filename.endswith('.md'):
            doc_name = filename.replace('.md', '')
            if doc_name not in document_indexes:
                try:
                    md_path = os.path.join(KNOWLEDGE_BASE_DIR, filename)
                    builder = TreeBuilder(add_node_id=True, add_node_summary=False)
                    index = builder.build_from_file(md_path)
                    document_indexes[doc_name] = index
                    node_maps[doc_name] = create_node_mapping(index)
                    logger.info(f"Generated index for: {filename} ({len(index.get_all_nodes())} nodes)")
                except Exception as e:
                    logger.error(f"Failed to generate index for {filename}: {e}")
    
    logger.info(f"Total documents loaded: {len(document_indexes)}")


def get_enhanced_tree_structure() -> dict:
    """
    获取增强版树结构（包含 summary）
    这是与 PageIndex 对齐的关键改进
    """
    combined = {"documents": []}
    
    for doc_name, index in document_indexes.items():
        doc_info = {
            "doc_name": doc_name,
            "title": index.title,
            "description": index.description[:300] if index.description else "",
            "sections": []
        }
        
        def add_node(node, depth=0):
            node_info = {
                "node_id": node.node_id,
                "title": node.title,
                "level": node.level,
                "summary": node.summary[:200] if node.summary else "",
                "line_range": f"{node.start_line}-{node.end_line}"
            }
            doc_info["sections"].append(node_info)
            
            # 递归添加子节点（限制深度）
            if depth < 3 and node.children:
                for child in node.children:
                    add_node(child, depth + 1)
        
        for root_node in index.root_nodes:
            add_node(root_node)
        
        combined["documents"].append(doc_info)
    
    return combined


async def enhanced_tree_search(query: str, use_multi_round: bool = True) -> dict:
    """
    增强版树搜索 - 参考 PageIndex 的 reasoning-based retrieval
    
    Args:
        query: 查询
        use_multi_round: 是否使用多轮推理
        
    Returns:
        {"thinking": str, "node_list": list}
    """
    llm_instance = get_llm()
    tree_structure = get_enhanced_tree_structure()
    
    # 增强版 prompt - 强调 summary 的重要性
    search_prompt = f"""You are an expert document retrieval system performing reasoning-based search.

## Question
{query}

## Document Tree Structures
Each node has: node_id, title, level, summary, and line_range.
**Important**: The 'summary' field describes what information each section contains. Use it to determine relevance.

{json.dumps(tree_structure, indent=2, ensure_ascii=False)}

## Search Strategy
1. First, identify which documents might contain the answer based on their title and description
2. Then, look at section titles AND summaries to find the most relevant sections
3. Prefer more specific (deeper level) nodes if they match the query
4. Consider that personal information (email, nationality, location) is often in "Contact" or similar sections
5. Work history and experience is usually in "Experience" or "Professional Experience" sections

## Reasoning Process
Think step by step:
1. What specific information is the question asking for?
2. Which document is most likely to contain this information?
3. Which section's summary mentions this type of information?
4. What is the most specific node that contains the answer?

## Response Format
```json
{{
    "thinking": "Step-by-step reasoning about which nodes are relevant...",
    "node_list": [
        {{"doc_name": "document_name", "node_id": "most_specific_node_id", "relevance": 0.95}},
        {{"doc_name": "document_name", "node_id": "another_node_id", "relevance": 0.8}}
    ]
}}
```

Select up to 5 most relevant nodes. Prioritize specificity over breadth.
Return ONLY the JSON structure."""

    try:
        response = llm_instance.complete(search_prompt, temperature=0.1)
        result = response.content.strip()
        
        # 处理 deepseek-r1 等推理模型的 <think>...</think> 标签
        import re
        result = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL).strip()
        
        # 清理 markdown 代码块
        if result.startswith('```'):
            lines = result.split('\n')
            if lines[0].strip().startswith('```'):
                lines = lines[1:]
            if lines and lines[-1].strip() == '```':
                lines = lines[:-1]
            result = '\n'.join(lines)
        
        # 解析 JSON
        try:
            result_json = json.loads(result)
            return result_json
        except json.JSONDecodeError:
            start = result.find('{')
            end = result.rfind('}') + 1
            if start >= 0 and end > start:
                return json.loads(result[start:end])
            raise ValueError(f"Invalid JSON: {result[:200]}")
            
    except Exception as e:
        logger.error(f"Enhanced tree search error: {e}")
        raise ValueError(f"Search error: {str(e)}")


async def multi_round_search(query: str) -> dict:
    """
    多轮推理搜索（完整版 PageIndex 风格）
    """
    try:
        engine = get_search_engine()
        context = SearchContext(
            query=query,
            documents=document_indexes,
            node_maps=node_maps,
            max_results=10,
            min_relevance=0.3
        )
        
        results = await engine.search(context)
        
        if not results:
            return {"thinking": "No relevant nodes found", "node_list": []}
        
        # 转换为兼容格式
        node_list = [
            {
                "doc_name": r.doc_name,
                "node_id": r.node_id,
                "relevance": r.relevance_score
            }
            for r in results
        ]
        
        thinking = "\n".join([
            f"- {r.doc_name}/{r.node_id}: {r.reasoning}" 
            for r in results[:3]
        ])
        
        return {"thinking": thinking, "node_list": node_list}
        
    except Exception as e:
        logger.error(f"Multi-round search failed: {e}")
        # Fallback 到简单搜索
        return await enhanced_tree_search(query, use_multi_round=False)


def get_node_with_children_content(node) -> str:
    """Get content from node and all its children recursively"""
    parts = []
    
    if node.content:
        parts.append(node.content)
    elif node.summary:
        parts.append(node.summary)
    
    if hasattr(node, 'children') and node.children:
        for child in node.children:
            child_content = get_node_with_children_content(child)
            if child_content:
                parts.append(child_content)
    
    return "\n\n".join(parts)


def retrieve_context(node_list: list) -> list:
    """Extract text content from retrieved nodes (including children)"""
    contexts = []
    seen_nodes = set()  # 避免重复
    
    for item in node_list:
        doc_name = item.get('doc_name')
        node_id = item.get('node_id')
        
        key = f"{doc_name}:{node_id}"
        if key in seen_nodes:
            continue
        seen_nodes.add(key)
        
        if doc_name in node_maps and node_id in node_maps[doc_name]:
            node = node_maps[doc_name][node_id]
            content = get_node_with_children_content(node)
            if content:
                contexts.append({
                    'doc_name': doc_name,
                    'node_id': node_id,
                    'title': node.title,
                    'content': content,
                    'relevance': item.get('relevance', 0.5)
                })
    
    # 按相关性排序
    contexts.sort(key=lambda x: x.get('relevance', 0), reverse=True)
    
    return contexts


def get_knowledge_base_stats() -> dict:
    """获取知识库统计信息"""
    total_chars = 0
    total_nodes = 0
    for index in document_indexes.values():
        for node in index.get_all_nodes():
            if node.content:
                total_chars += len(node.content)
            total_nodes += 1
    # 粗略估算 token 数（中文约 2 字符 = 1 token，英文约 4 字符 = 1 token）
    estimated_tokens = total_chars // 3
    return {
        'total_chars': total_chars,
        'estimated_tokens': estimated_tokens,
        'total_nodes': total_nodes,
        'total_documents': len(document_indexes)
    }


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint - 仅返回服务状态"""
    return jsonify({
        'status': 'ok',
        'service': 'Knowledge Indexer RAG Server',
        'version': '2.0',
        'port': 3003
    })


@app.route('/fstats', methods=['GET'])
def fstats():
    """知识库统计信息接口"""
    kb_stats = get_knowledge_base_stats()
    return jsonify({
        'documents_loaded': len(document_indexes),
        'total_chars': kb_stats['total_chars'],
        'estimated_tokens': kb_stats['estimated_tokens'],
        'total_nodes': kb_stats['total_nodes'],
        'features': ['multi-round-search', 'summary-aware', 'reasoning-based']
    })


@app.route('/query', methods=['POST'])
def query():
    """
    Main query endpoint - compatible with wenjie-webui RAG interface
    
    增强功能：
    - 使用节点 summary 进行更准确的搜索
    - 多轮推理（可选）
    - 置信度评分
    """
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'Missing "query" field'}), 400
        
        query_text = data['query']
        use_enhanced = data.get('enhanced', True)  # 默认使用增强搜索
        use_multi_round = data.get('multi_round', False)  # 多轮搜索（更慢但更准确）
        
        logger.info(f"Query: {query_text} (enhanced={use_enhanced}, multi_round={use_multi_round})")
        
        if not document_indexes:
            return jsonify({
                'error': 'No documents loaded',
                'context': '',
                'answer': ''
            }), 500
        
        # 执行搜索
        try:
            import nest_asyncio
            try:
                nest_asyncio.apply()
            except:
                pass
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                if use_multi_round:
                    search_result = loop.run_until_complete(multi_round_search(query_text))
                else:
                    search_result = loop.run_until_complete(enhanced_tree_search(query_text))
            finally:
                loop.close()
                
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return jsonify({'error': f'Search failed: {str(e)}'}), 500
        
        # 提取上下文
        node_list = search_result.get('node_list', [])
        if not node_list:
            logger.warning(f"No relevant nodes for: {query_text}")
            return jsonify({
                'context': '',
                'answer': '',
                'nodes': [],
                'source_files': [],
                'thinking': search_result.get('thinking', 'No relevant information found.')
            })
        
        contexts = retrieve_context(node_list)
        
        # 组合上下文
        combined_context = "\n\n".join([
            f"## [{ctx['doc_name']}] {ctx['title']}\n{ctx['content']}"
            for ctx in contexts
        ])
        
        logger.info(f"Retrieved {len(contexts)} nodes")
        
        # 提取唯一的源文件列表（兼容 wenjie-webui）
        source_files = list(set(ctx['doc_name'] for ctx in contexts))
        
        # 获取知识库统计（用于 wenjie-webui 显示）
        kb_stats = get_knowledge_base_stats()
        
        return jsonify({
            'context': combined_context,
            'answer': combined_context,
            'nodes': [f"{ctx['doc_name']}:{ctx['node_id']}" for ctx in contexts],
            'source_files': source_files,
            'thinking': search_result.get('thinking', ''),
            'relevance_scores': {
                f"{ctx['doc_name']}:{ctx['node_id']}": ctx.get('relevance', 0.5)
                for ctx in contexts
            },
            # 知识库统计（用于 Full Knowledge Base 显示）
            'knowledge_base_size': {
                'chars': kb_stats['total_chars'],
                'tokens': kb_stats['estimated_tokens']
            }
        })
        
    except Exception as e:
        logger.error(f"Query error: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/reload', methods=['POST'])
def reload():
    """Reload all indexes"""
    global document_indexes, node_maps, search_engine, simple_search
    document_indexes = {}
    node_maps = {}
    search_engine = None
    simple_search = None
    try:
        load_all_indexes()
        return jsonify({
            'status': 'ok',
            'message': 'Indexes reloaded',
            'documents_loaded': len(document_indexes)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/documents', methods=['GET'])
def list_documents():
    """List all loaded documents with details"""
    docs = []
    for doc_name, index in document_indexes.items():
        # 计算有 summary 的节点数
        nodes_with_summary = sum(1 for n in index.get_all_nodes() if n.summary)
        
        docs.append({
            'name': doc_name,
            'title': index.title,
            'total_nodes': len(index.get_all_nodes()),
            'nodes_with_summary': nodes_with_summary,
            'description': index.description[:200] if index.description else '',
            'has_full_summary': nodes_with_summary == len(index.get_all_nodes())
        })
    return jsonify({'documents': docs})


@app.route('/index', methods=['POST'])
def index_document():
    """Index a new markdown document with LLM summaries"""
    try:
        data = request.get_json()
        
        if 'content' in data:
            content = data['content']
            doc_name = data.get('name', 'untitled')
        elif 'file' in data:
            filepath = data['file']
            if not os.path.exists(filepath):
                return jsonify({'error': f'File not found: {filepath}'}), 400
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            doc_name = os.path.splitext(os.path.basename(filepath))[0]
        else:
            return jsonify({'error': 'Missing "content" or "file" field'}), 400
        
        # 默认使用 LLM 生成 summary（这是与 PageIndex 对齐的关键）
        use_llm = data.get('use_llm', True)
        include_content = data.get('include_content', True)
        
        llm_instance = get_llm() if use_llm else None
        
        builder = TreeBuilder(
            llm=llm_instance,
            add_node_id=True,
            add_node_summary=use_llm,
            add_doc_description=use_llm,
        )
        
        index = builder.build_from_content(content)
        
        document_indexes[doc_name] = index
        node_maps[doc_name] = create_node_mapping(index)
        
        index_path = os.path.join(RESULTS_DIR, f"{doc_name}_index.json")
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(index.to_json(include_content=include_content))
        
        return jsonify({
            'status': 'ok',
            'doc_name': doc_name,
            'nodes': len(index.get_all_nodes()),
            'nodes_with_summary': sum(1 for n in index.get_all_nodes() if n.summary),
            'index_file': index_path
        })
        
    except Exception as e:
        logger.error(f"Error indexing document: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/analyze', methods=['POST'])
def analyze_query():
    """
    分析查询并返回推理过程（调试用）
    """
    try:
        data = request.get_json()
        query_text = data.get('query', '')
        
        if not query_text:
            return jsonify({'error': 'Missing query'}), 400
        
        # 使用推理链分析
        reasoning = ReasoningChain(get_llm())
        
        # 分解查询
        sub_questions = reasoning.decompose_query(query_text)
        
        # 获取树结构
        tree_structure = get_enhanced_tree_structure()
        
        return jsonify({
            'query': query_text,
            'sub_questions': sub_questions,
            'tree_structure': tree_structure,
            'documents': list(document_indexes.keys())
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============ 文件上传功能 ============

ALLOWED_EXTENSIONS = {'md', 'markdown', 'txt'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload', methods=['GET'])
def upload_page():
    """File upload page with debug panel"""
    # 获取 LLM 配置信息
    llm_info = {'provider': 'unknown', 'model': 'unknown', 'api_base': 'unknown'}
    try:
        config = IndexerConfig.from_file()
        llm_info = {
            'provider': config.rag_llm.provider,
            'model': config.rag_llm.model,
            'api_base': config.rag_llm.api_base
        }
    except:
        pass
    
    html = f'''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Knowledge Base Upload</title>
    <style>
        :root {{
            --primary: #4facfe;
            --primary-dark: #00f2fe;
            --bg-dark: #1a1a2e;
            --bg-darker: #16213e;
            --text-light: #e0e0e0;
            --text-muted: #888;
            --success: #2ed573;
            --warning: #ffd700;
            --info: #87ceeb;
            --error: #ff4757;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-darker);
            min-height: 100vh;
            color: var(--text-light);
        }}
        .page-container {{
            display: flex;
            height: 100vh;
        }}
        /* Left Debug Panel */
        .debug-panel {{
            width: 380px;
            background: var(--bg-dark);
            border-right: 1px solid #2d2d44;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }}
        .debug-header {{
            padding: 1rem 1.25rem;
            background: linear-gradient(135deg, var(--bg-darker) 0%, var(--bg-dark) 100%);
            border-bottom: 1px solid #2d2d44;
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }}
        .status-dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--success);
            animation: pulse 2s infinite;
        }}
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.5; }}
        }}
        .debug-header h3 {{
            color: var(--success);
            font-size: 1rem;
            font-weight: 600;
            font-family: 'Fira Code', monospace;
        }}
        .debug-content {{
            flex: 1;
            overflow-y: auto;
            padding: 1rem;
            font-family: 'Fira Code', monospace;
            font-size: 0.8rem;
            line-height: 1.6;
        }}
        .debug-section {{
            margin-bottom: 1rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid #2d2d44;
        }}
        .debug-label {{
            color: #7b68ee;
            font-weight: 600;
            margin-bottom: 0.5rem;
            display: block;
        }}
        .debug-value {{ color: #b8b8b8; word-wrap: break-word; }}
        .debug-value.success {{ color: var(--success); }}
        .debug-value.warning {{ color: var(--warning); }}
        .debug-value.info {{ color: var(--info); }}
        .debug-value.error {{ color: var(--error); }}
        .debug-empty {{
            color: var(--text-muted);
            text-align: center;
            padding: 2rem;
            font-style: italic;
        }}
        .process-log {{
            background: var(--bg-darker);
            border-radius: 8px;
            padding: 0.75rem;
            margin-top: 0.5rem;
            max-height: 300px;
            overflow-y: auto;
        }}
        .log-entry {{
            padding: 0.25rem 0;
            border-bottom: 1px solid #2d2d44;
        }}
        .log-entry:last-child {{
            border-bottom: none;
        }}
        .log-time {{
            color: var(--text-muted);
            font-size: 0.7rem;
        }}
        .log-msg {{
            color: var(--text-light);
        }}
        .log-msg.success {{ color: var(--success); }}
        .log-msg.error {{ color: var(--error); }}
        .log-msg.info {{ color: var(--info); }}
        /* Right Upload Area */
        .upload-area-container {{
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }}
        .container {{
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 40px;
            max-width: 500px;
            width: 100%;
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 25px 50px rgba(0, 0, 0, 0.3);
        }}
        h1 {{
            color: #fff;
            text-align: center;
            margin-bottom: 10px;
            font-size: 1.8rem;
        }}
        .subtitle {{
            color: rgba(255, 255, 255, 0.6);
            text-align: center;
            margin-bottom: 30px;
            font-size: 0.9rem;
        }}
        .upload-area {{
            border: 2px dashed rgba(255, 255, 255, 0.3);
            border-radius: 15px;
            padding: 40px 20px;
            text-align: center;
            transition: all 0.3s ease;
            cursor: pointer;
            margin-bottom: 20px;
        }}
        .upload-area:hover, .upload-area.dragover {{
            border-color: #4facfe;
            background: rgba(79, 172, 254, 0.1);
        }}
        .upload-area svg {{
            width: 50px;
            height: 50px;
            fill: rgba(255, 255, 255, 0.5);
            margin-bottom: 15px;
        }}
        .upload-area p {{
            color: rgba(255, 255, 255, 0.7);
            margin-bottom: 10px;
        }}
        .upload-area .hint {{
            color: rgba(255, 255, 255, 0.4);
            font-size: 0.8rem;
        }}
        input[type="file"] {{ display: none; }}
        .options {{
            display: flex;
            gap: 20px;
            margin-bottom: 20px;
        }}
        .option {{
            display: flex;
            align-items: center;
            gap: 8px;
            color: rgba(255, 255, 255, 0.8);
            font-size: 0.9rem;
        }}
        .option input[type="checkbox"] {{
            width: 18px;
            height: 18px;
            accent-color: #4facfe;
        }}
        button {{
            width: 100%;
            padding: 15px;
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            border: none;
            border-radius: 10px;
            color: #fff;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 10px 30px rgba(79, 172, 254, 0.4);
        }}
        button:disabled {{
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }}
        .status {{
            margin-top: 20px;
            padding: 15px;
            border-radius: 10px;
            display: none;
        }}
        .status.success {{
            display: block;
            background: rgba(46, 213, 115, 0.2);
            border: 1px solid rgba(46, 213, 115, 0.5);
            color: #2ed573;
        }}
        .status.error {{
            display: block;
            background: rgba(255, 71, 87, 0.2);
            border: 1px solid rgba(255, 71, 87, 0.5);
            color: #ff4757;
        }}
        .status.loading {{
            display: block;
            background: rgba(79, 172, 254, 0.2);
            border: 1px solid rgba(79, 172, 254, 0.5);
            color: #4facfe;
        }}
        .file-name {{
            color: #2ed573;
            margin-top: 15px;
            font-size: 1.1rem;
            font-weight: 600;
            padding: 8px 16px;
            background: rgba(46, 213, 115, 0.15);
            border-radius: 8px;
            display: inline-block;
        }}
        .file-name:empty {{
            display: none;
        }}
        .nav-link {{
            display: block;
            text-align: center;
            margin-top: 20px;
            color: var(--primary);
            text-decoration: none;
        }}
        .nav-link:hover {{
            text-decoration: underline;
        }}
        @media (max-width: 992px) {{
            .page-container {{ flex-direction: column; }}
            .debug-panel {{ width: 100%; height: 250px; border-right: none; border-bottom: 1px solid #2d2d44; }}
        }}
    </style>
</head>
<body>
    <div class="page-container">
        <!-- Left Debug Panel -->
        <div class="debug-panel">
            <div class="debug-header">
                <div class="status-dot"></div>
                <h3>Index Debug Log</h3>
            </div>
            <div class="debug-content" id="debugLog">
                <div class="debug-section">
                    <span class="debug-label">[LLM Configuration]</span>
                    <div class="debug-value">
                        <div>Provider: <span class="warning">{llm_info['provider']}</span></div>
                        <div>Model: <span class="success">{llm_info['model']}</span></div>
                        <div>API Base: <span class="info">{llm_info['api_base']}</span></div>
                    </div>
                </div>
                <div class="debug-section">
                    <span class="debug-label">[Status]</span>
                    <span class="debug-value info">Ready - Waiting for file upload...</span>
                </div>
                <div class="debug-section" id="processSection" style="display:none;">
                    <span class="debug-label">[Processing Log]</span>
                    <div class="process-log" id="processLog"></div>
                </div>
                <div class="debug-section" id="resultSection" style="display:none;">
                    <span class="debug-label">[Index Result]</span>
                    <div class="debug-value" id="indexResult"></div>
                </div>
            </div>
        </div>

        <!-- Right Upload Area -->
        <div class="upload-area-container">
            <div class="container">
                <h1>📚 Knowledge Base Upload</h1>
                <p class="subtitle">Upload Markdown files to the knowledge base</p>
                
                <form id="uploadForm" enctype="multipart/form-data">
                    <div class="upload-area" id="dropZone">
                        <svg viewBox="0 0 24 24"><path d="M19.35 10.04C18.67 6.59 15.64 4 12 4 9.11 4 6.6 5.64 5.35 8.04 2.34 8.36 0 10.91 0 14c0 3.31 2.69 6 6 6h13c2.76 0 5-2.24 5-5 0-2.64-2.05-4.78-4.65-4.96zM14 13v4h-4v-4H7l5-5 5 5h-3z"/></svg>
                        <p>Click or drag files here</p>
                        <p class="hint">Supports .md, .markdown, .txt files</p>
                        <p class="file-name" id="fileName"></p>
                    </div>
                    <input type="file" id="fileInput" name="file" accept=".md,.markdown,.txt">
                    
                    <div class="options">
                        <label class="option">
                            <input type="checkbox" id="useLlm" checked>
                            Use LLM for summaries
                        </label>
                        <label class="option">
                            <input type="checkbox" id="includeContent" checked>
                            Include content
                        </label>
                    </div>
                    
                    <button type="submit" id="submitBtn">Upload & Index</button>
                </form>
                
                <div class="status" id="status"></div>
                <a href="/demo" class="nav-link">← Back to Demo</a>
            </div>
        </div>
    </div>

    <script>
        const dropZone = document.getElementById('dropZone');
        const fileInput = document.getElementById('fileInput');
        const fileName = document.getElementById('fileName');
        const form = document.getElementById('uploadForm');
        const status = document.getElementById('status');
        const submitBtn = document.getElementById('submitBtn');
        const processSection = document.getElementById('processSection');
        const processLog = document.getElementById('processLog');
        const resultSection = document.getElementById('resultSection');
        const indexResult = document.getElementById('indexResult');

        function addLog(msg, type = '') {{
            const entry = document.createElement('div');
            entry.className = 'log-entry';
            const time = new Date().toLocaleTimeString();
            entry.innerHTML = `<span class="log-time">${{time}}</span> <span class="log-msg ${{type}}">${{msg}}</span>`;
            processLog.appendChild(entry);
            processLog.scrollTop = processLog.scrollHeight;
        }}

        dropZone.addEventListener('click', () => fileInput.click());
        
        dropZone.addEventListener('dragover', (e) => {{
            e.preventDefault();
            dropZone.classList.add('dragover');
        }});
        
        dropZone.addEventListener('dragleave', () => {{
            dropZone.classList.remove('dragover');
        }});
        
        dropZone.addEventListener('drop', (e) => {{
            e.preventDefault();
            dropZone.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {{
                fileInput.files = files;
                fileName.textContent = files[0].name;
                addLog(`File selected: ${{files[0].name}}`, 'info');
            }}
        }});
        
        fileInput.addEventListener('change', () => {{
            if (fileInput.files.length > 0) {{
                fileName.textContent = fileInput.files[0].name;
                processSection.style.display = 'block';
                processLog.innerHTML = '';
                addLog(`File selected: ${{fileInput.files[0].name}}`, 'info');
            }}
        }});

        form.addEventListener('submit', async (e) => {{
            e.preventDefault();
            
            if (!fileInput.files.length) {{
                status.className = 'status error';
                status.textContent = 'Please select a file';
                addLog('Error: No file selected', 'error');
                return;
            }}
            
            submitBtn.disabled = true;
            status.className = 'status loading';
            status.textContent = 'Uploading and indexing...';
            processSection.style.display = 'block';
            resultSection.style.display = 'none';
            
            const useLlm = document.getElementById('useLlm').checked;
            const includeContent = document.getElementById('includeContent').checked;
            
            addLog('Starting upload...', 'info');
            addLog(`Options: use_llm=${{useLlm}}, include_content=${{includeContent}}`, 'info');
            
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            formData.append('use_llm', useLlm);
            formData.append('include_content', includeContent);
            
            try {{
                addLog('Sending file to server...', 'info');
                const startTime = Date.now();
                
                const response = await fetch('/upload', {{
                    method: 'POST',
                    body: formData
                }});
                
                const result = await response.json();
                const duration = ((Date.now() - startTime) / 1000).toFixed(2);
                
                if (response.ok) {{
                    const llmInfo = result.llm_info || {{}};
                    addLog(`Upload completed in ${{duration}}s`, 'success');
                    addLog(`Document: ${{result.doc_name}}`, 'success');
                    addLog(`Total nodes: ${{result.nodes}}`, 'success');
                    addLog(`Summary nodes: ${{result.nodes_with_summary}}`, 'success');
                    if (llmInfo.model) {{
                        addLog(`LLM used: ${{llmInfo.provider}}/${{llmInfo.model}}`, 'info');
                    }}
                    
                    status.className = 'status success';
                    status.innerHTML = `✅ Upload successful!<br>Document: ${{result.doc_name}}<br>Nodes: ${{result.nodes}}<br>Summary nodes: ${{result.nodes_with_summary}}`;
                    
                    // Show result section
                    resultSection.style.display = 'block';
                    indexResult.innerHTML = `
                        <div>Document: <span class="success">${{result.doc_name}}</span></div>
                        <div>Total Nodes: <span class="warning">${{result.nodes}}</span></div>
                        <div>Summary Nodes: <span class="info">${{result.nodes_with_summary}}</span></div>
                        <div>Processing Time: <span class="info">${{duration}}s</span></div>
                        <div>Index File: <span class="info">${{result.index_file || 'N/A'}}</span></div>
                        <div style="margin-top:10px; padding-top:10px; border-top:1px solid #2d2d44;">
                            <div>LLM Provider: <span class="warning">${{llmInfo.provider || 'N/A'}}</span></div>
                            <div>LLM Model: <span class="success">${{llmInfo.model || 'N/A'}}</span></div>
                        </div>
                    `;
                    
                    fileInput.value = '';
                    fileName.textContent = '';
                }} else {{
                    addLog(`Error: ${{result.error || 'Upload failed'}}`, 'error');
                    status.className = 'status error';
                    status.textContent = '❌ ' + (result.error || 'Upload failed');
                }}
            }} catch (err) {{
                addLog(`Network error: ${{err.message}}`, 'error');
                status.className = 'status error';
                status.textContent = '❌ Network error: ' + err.message;
            }} finally {{
                submitBtn.disabled = false;
            }}
        }});
    </script>
</body>
</html>
'''
    return html


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and indexing"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Unsupported file format. Please upload .md, .markdown or .txt files'}), 400
        
        # 安全的文件名
        filename = secure_filename(file.filename)
        doc_name = os.path.splitext(filename)[0]
        
        # 文件路径
        md_path = os.path.join(RESULTS_DIR, filename)
        index_path = os.path.join(RESULTS_DIR, f"{doc_name}_index.json")
        
        # 如果文件已存在，先删除旧文件和索引
        if os.path.exists(md_path):
            os.remove(md_path)
            logger.info(f"Deleted old file: {md_path}")
        if os.path.exists(index_path):
            os.remove(index_path)
            logger.info(f"Deleted old index: {index_path}")
        
        # 从内存中移除旧索引
        if doc_name in document_indexes:
            del document_indexes[doc_name]
        if doc_name in node_maps:
            del node_maps[doc_name]
        
        # 保存新文件到 result 目录
        file.save(md_path)
        logger.info(f"File saved: {md_path}")
        
        # 读取文件内容
        with open(md_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 获取选项
        use_llm = request.form.get('use_llm', 'true').lower() == 'true'
        include_content = request.form.get('include_content', 'true').lower() == 'true'
        
        # 创建索引
        llm_instance = get_llm() if use_llm else None
        
        builder = TreeBuilder(
            llm=llm_instance,
            add_node_id=True,
            add_node_summary=use_llm and llm_instance is not None,
            add_doc_description=use_llm and llm_instance is not None,
        )
        
        index = builder.build_from_content(content)
        
        # 保存索引
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(index.to_json(include_content=include_content))
        
        # 更新内存中的索引
        document_indexes[doc_name] = index
        node_maps[doc_name] = create_node_mapping(index)
        
        logger.info(f"Index created: {index_path} ({len(index.get_all_nodes())} nodes)")
        
        # 获取 LLM 配置信息
        llm_info = {'provider': 'N/A', 'model': 'N/A', 'api_base': 'N/A'}
        if use_llm:
            try:
                config = IndexerConfig.from_file()
                llm_info = {
                    'provider': config.rag_llm.provider,
                    'model': config.rag_llm.model,
                    'api_base': config.rag_llm.api_base
                }
            except:
                pass
        
        return jsonify({
            'status': 'ok',
            'doc_name': doc_name,
            'nodes': len(index.get_all_nodes()),
            'nodes_with_summary': sum(1 for n in index.get_all_nodes() if n.summary),
            'md_file': md_path,
            'index_file': index_path,
            'llm_info': llm_info
        })
        
    except Exception as e:
        logger.error(f"Upload error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


# ============ 聊天机器人演示 ============

@app.route('/demo', methods=['GET'])
def demo_page():
    """Chatbot demo page"""
    html = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LogiRAG Demo</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #4facfe;
            --primary-dark: #00f2fe;
            --bg-dark: #1a1a2e;
            --bg-darker: #16213e;
            --text-light: #e0e0e0;
            --text-muted: #888;
            --success: #2ed573;
            --warning: #ffd700;
            --info: #87ceeb;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg-darker);
            min-height: 100vh;
            color: var(--text-light);
        }
        .container {
            display: flex;
            height: 100vh;
            gap: 0;
        }
        /* Left RAG Log Panel */
        .rag-panel {
            width: 380px;
            background: var(--bg-dark);
            border-right: 1px solid #2d2d44;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        .rag-header {
            padding: 1rem 1.25rem;
            background: linear-gradient(135deg, var(--bg-darker) 0%, var(--bg-dark) 100%);
            border-bottom: 1px solid #2d2d44;
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--success);
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .rag-header h3 {
            color: var(--success);
            font-size: 1rem;
            font-weight: 600;
            font-family: 'Fira Code', monospace;
        }
        .rag-content {
            flex: 1;
            overflow-y: auto;
            padding: 1rem;
            font-family: 'Fira Code', monospace;
            font-size: 0.8rem;
            line-height: 1.6;
        }
        .rag-section {
            margin-bottom: 1rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid #2d2d44;
        }
        .rag-label {
            color: #7b68ee;
            font-weight: 600;
            margin-bottom: 0.5rem;
            display: block;
        }
        .rag-value { color: #b8b8b8; word-wrap: break-word; }
        .rag-value.success { color: var(--success); }
        .rag-value.warning { color: var(--warning); }
        .rag-value.info { color: var(--info); }
        .rag-comparison {
            background: var(--bg-darker);
            border-radius: 8px;
            padding: 0.75rem;
            margin-top: 0.5rem;
        }
        .rag-row {
            display: flex;
            justify-content: space-between;
            padding: 0.25rem 0;
        }
        .rag-row-label { color: var(--text-muted); }
        .rag-row-value { color: var(--success); font-weight: 600; }
        .rag-saved {
            background: linear-gradient(90deg, rgba(46, 213, 115, 0.2) 0%, transparent 100%);
            border-left: 3px solid var(--success);
            padding: 0.5rem 0.75rem;
            margin-top: 0.5rem;
            border-radius: 0 4px 4px 0;
        }
        .rag-saved .percent {
            font-size: 1.25rem;
            font-weight: 700;
            color: var(--success);
        }
        .rag-empty {
            color: var(--text-muted);
            text-align: center;
            padding: 2rem;
            font-style: italic;
        }
        /* Right Chat Area */
        .chat-area {
            flex: 1;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        .chat-header {
            padding: 1.25rem 1.5rem;
            background: #fff;
            border-bottom: 1px solid #eee;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .chat-header-left {
            display: flex;
            align-items: center;
            gap: 1rem;
        }
        .chat-header .logo-icon {
            width: 56px;
            height: 56px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
            border-radius: 16px;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
            position: relative;
            overflow: hidden;
        }
        .chat-header .logo-icon::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent);
            animation: shimmer 3s infinite;
        }
        @keyframes shimmer {
            0% { transform: translateX(-100%) rotate(45deg); }
            100% { transform: translateX(100%) rotate(45deg); }
        }
        .chat-header .logo-icon svg {
            width: 32px;
            height: 32px;
            fill: white;
            filter: drop-shadow(0 2px 4px rgba(0,0,0,0.2));
        }
        .chat-title {
            font-size: 1.25rem;
            font-weight: 700;
            color: #333;
        }
        .chat-subtitle {
            color: #666;
            font-size: 0.85rem;
            margin-top: 0.25rem;
        }
        .nav-links {
            display: flex;
            gap: 1rem;
        }
        .nav-links a {
            padding: 0.5rem 1rem;
            background: var(--primary);
            color: white;
            text-decoration: none;
            border-radius: 8px;
            font-weight: 500;
            font-size: 0.9rem;
            transition: all 0.2s;
        }
        .nav-links a:hover {
            background: var(--primary-dark);
            transform: translateY(-1px);
        }
        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 2rem;
            background: #f8f9fa;
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }
        .message {
            display: flex;
            gap: 1rem;
            animation: fadeIn 0.3s ease;
        }
        .message.user { flex-direction: row-reverse; }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .message-avatar {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 0.8rem;
            flex-shrink: 0;
        }
        .message.user .message-avatar {
            background: var(--primary);
            color: white;
        }
        .message.bot .message-avatar {
            background: #e9ecef;
            color: #333;
        }
        .message-content {
            max-width: 70%;
            padding: 1rem;
            border-radius: 12px;
            line-height: 1.6;
            word-wrap: break-word;
        }
        .message.user .message-content {
            background: var(--primary);
            color: white;
        }
        .message.bot .message-content {
            background: white;
            color: #333;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .message-time {
            font-size: 0.75rem;
            color: #999;
            margin-top: 0.25rem;
        }
        .chat-input-area {
            padding: 1.5rem;
            background: white;
            border-top: 1px solid #eee;
            display: flex;
            gap: 1rem;
        }
        .chat-input {
            flex: 1;
            padding: 0.875rem;
            border: 2px solid #e9ecef;
            border-radius: 8px;
            font-family: inherit;
            font-size: 1rem;
            resize: none;
            transition: border-color 0.2s;
        }
        .chat-input:focus {
            outline: none;
            border-color: var(--primary);
        }
        .send-btn {
            padding: 0.875rem 2rem;
            background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 600;
            transition: all 0.2s;
        }
        .send-btn:hover:not(:disabled) {
            transform: translateY(-1px);
            box-shadow: 0 5px 20px rgba(79, 172, 254, 0.4);
        }
        .send-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        .loading {
            display: none;
            padding: 1rem;
            text-align: center;
            color: #666;
        }
        .loading.show { display: block; }
        @media (max-width: 992px) {
            .container { flex-direction: column; }
            .rag-panel { width: 100%; height: 250px; border-right: none; border-bottom: 1px solid #2d2d44; }
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- RAG Log Panel -->
        <div class="rag-panel">
            <div class="rag-header">
                <div class="status-dot"></div>
                <h3>RAG Debug Log</h3>
            </div>
            <div class="rag-content" id="ragLog">
                <div class="rag-empty">
                    Waiting for conversation...<br>
                    RAG debug info will appear after sending a message
                </div>
            </div>
        </div>

        <!-- Chat Area -->
        <div class="chat-area">
            <div class="chat-header">
                <div class="chat-header-left">
                    <div class="logo-icon">
                        <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                            <!-- Futuristic AI Brain Icon -->
                            <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 17.93c-3.95-.49-7-3.85-7-7.93 0-.62.08-1.21.21-1.79L9 15v1c0 1.1.9 2 2 2v1.93zm6.9-2.54c-.26-.81-1-1.39-1.9-1.39h-1v-3c0-.55-.45-1-1-1H8v-2h2c.55 0 1-.45 1-1V7h2c1.1 0 2-.9 2-2v-.41c2.93 1.19 5 4.06 5 7.41 0 2.08-.8 3.97-2.1 5.39z"/>
                            <!-- Neural network dots -->
                            <circle cx="12" cy="12" r="1.5" opacity="0.8"/>
                            <circle cx="8" cy="10" r="1" opacity="0.6"/>
                            <circle cx="16" cy="10" r="1" opacity="0.6"/>
                            <circle cx="10" cy="15" r="0.8" opacity="0.5"/>
                            <circle cx="14" cy="15" r="0.8" opacity="0.5"/>
                        </svg>
                    </div>
                    <div>
                        <h1 class="chat-title">LogiRAG Demo</h1>
                        <p class="chat-subtitle">Reasoning-based RAG with Tree Indexing</p>
                    </div>
                </div>
                <div class="nav-links">
                    <a href="/upload">📤 Upload</a>
                    <a href="/fstats">📊 Stats</a>
                </div>
            </div>

            <div class="chat-messages" id="messages">
                <div class="message bot">
                    <div class="message-avatar">Bot</div>
                    <div>
                        <div class="message-content">
                            Hello! I'm a knowledge base assistant. I can answer questions based on the indexed documents. How can I help you?
                        </div>
                        <div class="message-time" id="welcomeTime"></div>
                    </div>
                </div>
            </div>

            <div class="loading" id="loading">Thinking...</div>

            <div class="chat-input-area">
                <textarea class="chat-input" id="input" placeholder="Type your question..." rows="1"></textarea>
                <button class="send-btn" id="sendBtn">Send</button>
            </div>
        </div>
    </div>

    <script>
        const messages = document.getElementById('messages');
        const input = document.getElementById('input');
        const sendBtn = document.getElementById('sendBtn');
        const loading = document.getElementById('loading');
        const ragLog = document.getElementById('ragLog');

        document.getElementById('welcomeTime').textContent = new Date().toLocaleTimeString();

        // Auto-resize input
        input.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 120) + 'px';
        });

        // Send message
        async function sendMessage() {
            const text = input.value.trim();
            if (!text) return;

            addMessage(text, 'user');
            input.value = '';
            input.style.height = 'auto';
            input.disabled = true;
            sendBtn.disabled = true;
            loading.classList.add('show');

            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: text })
                });

                const data = await response.json();

                if (response.ok) {
                    addMessage(data.reply, 'bot');
                    updateRAGLog(data.debug);
                } else {
                    addMessage('Sorry, an error occurred: ' + (data.error || 'Unknown error'), 'bot');
                }
            } catch (err) {
                addMessage('Network error, please try again', 'bot');
            } finally {
                input.disabled = false;
                sendBtn.disabled = false;
                loading.classList.remove('show');
                input.focus();
            }
        }

        function addMessage(text, type) {
            const div = document.createElement('div');
            div.className = `message ${type}`;
            div.innerHTML = `
                <div class="message-avatar">${type === 'user' ? 'You' : 'Bot'}</div>
                <div>
                    <div class="message-content">${escapeHtml(text).replace(/\\n/g, '<br>')}</div>
                    <div class="message-time">${new Date().toLocaleTimeString()}</div>
                </div>
            `;
            messages.appendChild(div);
            messages.scrollTop = messages.scrollHeight;
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        function updateRAGLog(debug) {
            if (!debug) return;
            
            const kbChars = debug.kb_size ? debug.kb_size.chars : 0;
            const kbTokens = debug.kb_size ? debug.kb_size.tokens : 0;
            const ctxChars = debug.context_size || 0;
            const ctxTokens = Math.floor(ctxChars / 3);
            const savedChars = kbChars - ctxChars;
            const savedTokens = kbTokens - ctxTokens;
            const savedPercent = kbChars > 0 ? ((savedChars / kbChars) * 100).toFixed(1) : '0.0';
            
            const ragLlmInfo = debug.rag_llm_info || debug.llm_info || {};
            const chatLlmInfo = debug.chat_llm_info || debug.llm_info || {};
            const isSameLlm = ragLlmInfo.model === chatLlmInfo.model && ragLlmInfo.api_base === chatLlmInfo.api_base;
            ragLog.innerHTML = `
                <div class="rag-section">
                    <span class="rag-label">[Timestamp]</span>
                    <span class="rag-value info">${new Date().toLocaleString()}</span>
                </div>
                <div class="rag-section">
                    <span class="rag-label">[RAG LLM]</span>
                    <div class="rag-value">
                        <div>Provider: <span class="warning">${ragLlmInfo.provider || 'unknown'}</span></div>
                        <div>Model: <span class="success">${ragLlmInfo.model || 'unknown'}</span></div>
                        <div>API Base: <span class="info">${ragLlmInfo.api_base || 'unknown'}</span></div>
                    </div>
                </div>
                <div class="rag-section">
                    <span class="rag-label">[Chat LLM]</span>
                    <div class="rag-value">
                        ${isSameLlm ? '<div><span class="info">(Same as RAG LLM)</span></div>' : `
                        <div>Provider: <span class="warning">${chatLlmInfo.provider || 'unknown'}</span></div>
                        <div>Model: <span class="success">${chatLlmInfo.model || 'unknown'}</span></div>
                        <div>API Base: <span class="info">${chatLlmInfo.api_base || 'unknown'}</span></div>
                        `}
                    </div>
                </div>
                <div class="rag-section">
                    <span class="rag-label">[Query]</span>
                    <span class="rag-value">${escapeHtml(debug.query)}</span>
                </div>
                <div class="rag-section">
                    <span class="rag-label">[RAG Results]</span>
                    <div class="rag-value">
                        <div>Status: <span class="success">${debug.nodes && debug.nodes.length > 0 ? 'Success' : 'No matches'}</span></div>
                        <div>RAG Server: <span class="info">${window.location.origin}/query</span></div>
                        <div>Matched Nodes: <span class="warning">${debug.nodes ? debug.nodes.join(', ') : 'None'}</span></div>
                        <div>Source Files: <span class="info">${debug.source_files ? debug.source_files.join(', ') : 'None'}</span></div>
                    </div>
                </div>
                <div class="rag-section">
                    <span class="rag-label">[Reasoning Process]</span>
                    <span class="rag-value">${escapeHtml(debug.thinking || 'N/A')}</span>
                </div>
                <div class="rag-section">
                    <span class="rag-label">[Full Context Comparison]</span>
                    <div class="rag-comparison">
                        <div class="rag-row">
                            <span class="rag-row-label">Full Knowledge Base:</span>
                            <span class="rag-row-value">${kbChars.toLocaleString()} chars (${kbTokens.toLocaleString()} tokens)</span>
                        </div>
                        <div class="rag-row">
                            <span class="rag-row-label">RAG Context Sent:</span>
                            <span class="rag-row-value">${ctxChars.toLocaleString()} chars (${ctxTokens.toLocaleString()} tokens)</span>
                        </div>
                    </div>
                    <div class="rag-saved">
                        <div>Tokens Saved: <span class="percent">${savedPercent}%</span> (${savedTokens.toLocaleString()} tokens)</div>
                    </div>
                </div>
                <div class="rag-section">
                    <span class="rag-label">[LLM Response]</span>
                    <span class="rag-value">${escapeHtml(debug.llm_response || 'N/A').substring(0, 500)}${debug.llm_response && debug.llm_response.length > 500 ? '...' : ''}</span>
                </div>
            `;
        }

        sendBtn.addEventListener('click', sendMessage);
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });

        input.focus();
    </script>
</body>
</html>
'''
    return html


@app.route('/chat', methods=['POST'])
def chat():
    """Chat API - combines RAG retrieval and LLM response generation"""
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'error': 'Missing message field'}), 400

        user_message = data['message']
        logger.info(f"Chat message: {user_message}")

        # 1. Use RAG to retrieve relevant context
        try:
            import nest_asyncio
            try:
                nest_asyncio.apply()
            except:
                pass

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                search_result = loop.run_until_complete(enhanced_tree_search(user_message))
            finally:
                loop.close()
        except Exception as e:
            logger.error(f"RAG search failed: {e}")
            search_result = {'node_list': [], 'thinking': str(e)}

        # 2. Extract context
        node_list = search_result.get('node_list', [])
        contexts = retrieve_context(node_list) if node_list else []

        combined_context = "\\n\\n".join([
            f"## {ctx['title']}\\n{ctx['content']}"
            for ctx in contexts
        ]) if contexts else ""

        # 3. Build LLM prompt with current date/time
        from datetime import datetime
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        current_date = datetime.now().strftime("%B %d, %Y")
        current_year = datetime.now().year
        
        kb_stats = get_knowledge_base_stats()

        if combined_context:
            system_prompt = f"""You are an intelligent assistant based on a knowledge base. Please answer the user's question based on the following knowledge base content.
If there is no relevant information in the knowledge base, please honestly state that.

**Current Date/Time Information:**
- Current Date: {current_date}
- Current Year: {current_year}
- Current Timestamp: {current_datetime}

Use the current date above for any time-related calculations (e.g., calculating years of experience, age, etc.).

Knowledge Base Content:
{combined_context}"""
        else:
            system_prompt = f"""You are an intelligent assistant. No relevant content was found in the knowledge base for this question, please inform the user.

**Current Date/Time Information:**
- Current Date: {current_date}
- Current Year: {current_year}
- Current Timestamp: {current_datetime}"""

        # 4. Call Chat LLM to generate response (may be different from RAG LLM)
        try:
            chat_llm_instance = get_chat_llm()
            response = chat_llm_instance.complete(
                prompt=user_message,
                system_prompt=system_prompt,
                temperature=0.7
            )
            reply = response.content
            
            # 处理 deepseek-r1 等推理模型的 <think>...</think> 标签
            import re
            reply = re.sub(r'<think>.*?</think>', '', reply, flags=re.DOTALL).strip()

        except Exception as e:
            logger.error(f"Chat LLM call failed: {e}")
            reply = f"Sorry, an error occurred while generating response: {str(e)}"

        # 5. Return result with enhanced debug info
        # 获取 LLM 模型信息 (分别显示 RAG LLM 和 Chat LLM)
        rag_llm_info = {
            'provider': 'unknown',
            'model': 'unknown',
            'api_base': 'unknown'
        }
        chat_llm_info = {
            'provider': 'unknown',
            'model': 'unknown',
            'api_base': 'unknown'
        }
        try:
            config = IndexerConfig.from_file()
            rag_llm_info = {
                'provider': config.rag_llm.provider,
                'model': config.rag_llm.model,
                'api_base': config.rag_llm.api_base
            }
            # Chat LLM info
            if hasattr(config, 'chat_llm') and config.chat_llm:
                chat_llm_info = {
                    'provider': config.chat_llm.provider,
                    'model': config.chat_llm.model,
                    'api_base': config.chat_llm.api_base
                }
            else:
                chat_llm_info = rag_llm_info  # Same as RAG LLM
        except:
            pass
        
        return jsonify({
            'reply': reply,
            'debug': {
                'query': user_message,
                'nodes': [f"{ctx['doc_name']}:{ctx['node_id']}" for ctx in contexts],
                'source_files': list(set(ctx['doc_name'] for ctx in contexts)),
                'thinking': search_result.get('thinking', ''),
                'context_size': len(combined_context),
                'kb_size': {
                    'chars': kb_stats['total_chars'],
                    'tokens': kb_stats['estimated_tokens']
                },
                'llm_response': reply,
                'rag_llm_info': rag_llm_info,    # LLM used for RAG search
                'chat_llm_info': chat_llm_info,  # LLM used for chat response
                'llm_info': chat_llm_info        # Keep for backward compatibility
            }
        })

    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    try:
        load_all_indexes()
        logger.info("Indexes loaded successfully")
        
        # 检查 summary 覆盖率
        total_nodes = sum(len(idx.get_all_nodes()) for idx in document_indexes.values())
        nodes_with_summary = sum(
            sum(1 for n in idx.get_all_nodes() if n.summary)
            for idx in document_indexes.values()
        )
        logger.info(f"Summary coverage: {nodes_with_summary}/{total_nodes} nodes")
        
    except Exception as e:
        logger.error(f"Failed to load indexes: {str(e)}")
        logger.warning("Server will start but queries may fail")
    
    port = int(os.getenv('RAG_SERVER_PORT', 3003))
    logger.info(f"Starting Enhanced RAG API server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
