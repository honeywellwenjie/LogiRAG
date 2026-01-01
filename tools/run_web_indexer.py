#!/usr/bin/env python3
"""
网页索引生成器
用法: python run_web_indexer.py --url https://example.com
      python run_web_indexer.py --url https://example.com --level 1  # 爬取链接
      python run_web_indexer.py --url https://example.com --level 255  # 爬取所有链接

将网页内容转换为结构化的 JSON 索引
结果会自动保存到 result/ 目录
"""

import argparse
import os
import sys
import re
from datetime import datetime
from urllib.parse import urlparse

# 获取项目根目录（从 tools 目录向上一级）
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# 添加 src 到路径
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from knowledge_indexer import LLMFactory, IndexerConfig
from knowledge_indexer.web import WebIndexer, WebFetcher, WebCrawler
from knowledge_indexer.web.html_to_markdown import HTMLToMarkdown, SimpleHTMLToMarkdown

# 默认输出目录（相对于项目根目录）
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "result")


def url_to_filename(url: str) -> str:
    """将 URL 转换为安全的文件名"""
    parsed = urlparse(url)
    # 使用域名和路径
    name = parsed.netloc + parsed.path
    # 移除不安全字符
    name = re.sub(r'[^\w\-_.]', '_', name)
    # 移除多余的下划线
    name = re.sub(r'_+', '_', name)
    # 移除首尾下划线
    name = name.strip('_')
    return name or "webpage"


def crawl_single_page(args, indexer, output_dir):
    """爬取单个页面（原有逻辑）"""
    print(f"正在抓取: {args.url}", file=sys.stderr)
    
    # 用于保存 Markdown 内容
    markdown_content = None
    
    # 始终获取 Markdown，以便保存原始内容
    page = indexer.get_page_with_markdown(args.url)
    
    if not page.is_success:
        print(f"错误: 抓取失败 (状态码: {page.status_code})", file=sys.stderr)
        sys.exit(1)
    
    markdown_content = page.markdown
    
    # 如果指定了 output-markdown，保存到指定位置
    if args.output_markdown:
        with open(args.output_markdown, 'w', encoding='utf-8') as f:
            f.write(page.markdown)
        print(f"Markdown 已保存到: {args.output_markdown}", file=sys.stderr)
    
    # 生成索引
    index = indexer.tree_builder.build_from_content(page.markdown)
    index.source_file = args.url
    index.metadata["type"] = "web"
    index.metadata["url"] = args.url
    index.metadata["domain"] = page.domain
    index.metadata["fetched_at"] = datetime.now().isoformat()
    index.metadata["original_title"] = page.title
    
    return index, markdown_content, [page]


def crawl_multi_level(args, fetcher, converter, tree_builder, output_dir):
    """多层爬取"""
    level = args.level
    level_desc = "无限深度" if level == 0xFF else f"{level} 层"
    print(f"正在爬取: {args.url} (深度: {level_desc})", file=sys.stderr)
    
    # 创建爬虫
    crawler = WebCrawler(
        fetcher=fetcher,
        max_pages=args.max_pages,
        same_domain_only=not args.allow_external,
        max_workers=args.workers,
    )
    
    # 爬取
    result = crawler.crawl_and_convert(
        start_url=args.url,
        level=level,
        converter=converter,
    )
    
    print(f"爬取完成: 成功 {result.success_count} 页, 失败 {result.failed_count} 页", file=sys.stderr)
    
    if not result.pages:
        print("错误: 没有成功爬取任何页面", file=sys.stderr)
        sys.exit(1)
    
    # 保存所有页面的 Markdown
    os.makedirs(output_dir, exist_ok=True)
    saved_files = []
    
    for page in result.pages:
        if page.markdown:
            filename = url_to_filename(page.url)
            md_path = os.path.join(output_dir, f"{filename}.md")
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(page.markdown)
            saved_files.append(md_path)
    
    print(f"已保存 {len(saved_files)} 个 Markdown 文件到 {output_dir}/", file=sys.stderr)
    
    # 为每个页面生成索引
    indexes = []
    for page in result.pages:
        if page.markdown:
            try:
                index = tree_builder.build_from_content(page.markdown)
                index.source_file = page.url
                index.metadata["type"] = "web"
                index.metadata["url"] = page.url
                index.metadata["domain"] = page.domain
                index.metadata["fetched_at"] = datetime.now().isoformat()
                index.metadata["original_title"] = page.title
                indexes.append(index)
            except Exception as e:
                print(f"警告: 生成索引失败 {page.url}: {e}", file=sys.stderr)
    
    # 保存所有索引
    for idx, index in enumerate(indexes):
        filename = url_to_filename(index.metadata.get("url", f"page_{idx}"))
        json_path = os.path.join(output_dir, f"{filename}_index.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            f.write(index.to_json(include_content=args.include_content, indent=2))
    
    print(f"已保存 {len(indexes)} 个索引文件到 {output_dir}/", file=sys.stderr)
    
    # 返回第一个页面的索引作为主索引
    main_index = indexes[0] if indexes else None
    main_markdown = result.pages[0].markdown if result.pages else None
    
    return main_index, main_markdown, result.pages


def main():
    parser = argparse.ArgumentParser(
        description="网页索引生成器 - 抓取网页并生成树形结构索引",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 爬取单个页面
  python run_web_indexer.py --url https://example.com

  # 爬取页面及其所有链接（1层）
  python run_web_indexer.py --url https://example.com --level 1

  # 爬取2层链接
  python run_web_indexer.py --url https://example.com --level 2

  # 无限深度爬取（谨慎使用）
  python run_web_indexer.py --url https://example.com --level 255 --max-pages 50
"""
    )
    
    # 输入
    parser.add_argument(
        "--url",
        type=str,
        required=True,
        help="要索引的网页 URL"
    )
    
    # 爬取深度
    parser.add_argument(
        "--level",
        type=int,
        default=0,
        help="爬取深度：0=仅当前页，1=当前页+链接，2=再深一层，255=无限深度（默认: 0）"
    )
    
    parser.add_argument(
        "--max-pages",
        type=int,
        default=100,
        help="最大爬取页面数（默认: 100）"
    )
    
    parser.add_argument(
        "--allow-external",
        action="store_true",
        help="允许爬取外部域名的链接（默认只爬取同一域名）"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="并发爬取线程数（默认: 5）"
    )
    
    # 配置文件
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="配置文件路径（YAML 格式）"
    )
    
    # 输出选项
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="输出 JSON 文件路径（默认自动保存到 result/ 目录）"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"输出目录（默认: {DEFAULT_OUTPUT_DIR}）"
    )
    
    parser.add_argument(
        "--output-markdown",
        type=str,
        default=None,
        help="同时输出转换后的 Markdown 文件（仅单页模式）"
    )
    
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="不保存文件，仅输出到终端（仅单页模式）"
    )
    
    parser.add_argument(
        "--include-content",
        action="store_true",
        help="在输出中包含原始内容"
    )
    
    # LLM 配置（命令行参数覆盖配置文件）
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        choices=["openai", "ollama", "custom"],
        help="LLM 提供者（覆盖配置文件）"
    )
    
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API 密钥（覆盖配置文件）"
    )
    
    parser.add_argument(
        "--api-base",
        type=str,
        default=None,
        help="API 基础 URL（覆盖配置文件）"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="模型名称（覆盖配置文件）"
    )
    
    # 索引选项
    parser.add_argument(
        "--no-llm-convert",
        action="store_true",
        help="不使用 LLM 进行 HTML 到 Markdown 的转换（使用规则转换）"
    )
    
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="不生成节点摘要"
    )
    
    parser.add_argument(
        "--no-description",
        action="store_true",
        help="不生成文档描述"
    )
    
    parser.add_argument(
        "--no-node-id",
        action="store_true",
        help="不添加节点 ID"
    )
    
    # 抓取选项
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="请求超时时间（秒，覆盖配置文件）"
    )
    
    parser.add_argument(
        "--no-verify-ssl",
        action="store_true",
        help="不验证 SSL 证书"
    )
    
    args = parser.parse_args()
    
    # 加载配置（从项目根目录查找配置文件）
    config_file = args.config
    if not config_file:
        # 按优先级查找配置文件
        for name in ["config.local.yaml", "config.yaml"]:
            path = os.path.join(PROJECT_ROOT, name)
            if os.path.exists(path):
                config_file = path
                break
    
    try:
        if config_file:
            config = IndexerConfig.from_file(config_file)
            print(f"使用配置文件: {config_file}", file=sys.stderr)
        else:
            config = IndexerConfig()
    except FileNotFoundError as e:
        if args.config:
            print(f"错误: {e}", file=sys.stderr)
            sys.exit(1)
        config = IndexerConfig()
    
    # 命令行参数覆盖配置文件
    if args.provider:
        config.llm.provider = args.provider
    if args.api_key:
        config.llm.api_key = args.api_key
    if args.api_base:
        config.llm.api_base = args.api_base
    if args.model:
        config.llm.model = args.model
    
    # 创建 LLM
    llm = None
    use_llm_for_conversion = not args.no_llm_convert
    
    if use_llm_for_conversion or not args.no_summary:
        if not config.llm.api_key and config.llm.provider != "ollama":
            if use_llm_for_conversion:
                print("警告: 未提供 API 密钥，将使用规则转换", file=sys.stderr)
                use_llm_for_conversion = False
            else:
                print("警告: 未提供 API 密钥，将不生成摘要", file=sys.stderr)
        else:
            try:
                llm = LLMFactory.from_config(config.llm)
                print(f"使用 LLM: {config.llm.provider}/{config.llm.model}", file=sys.stderr)
            except Exception as e:
                print(f"警告: 创建 LLM 失败: {e}", file=sys.stderr)
                if use_llm_for_conversion:
                    print("将使用规则转换代替 LLM 转换", file=sys.stderr)
                    use_llm_for_conversion = False
    
    # 创建抓取器
    fetcher = WebFetcher(
        timeout=args.timeout or 30,
        verify_ssl=not args.no_verify_ssl,
    )
    
    # 创建转换器
    if use_llm_for_conversion and llm:
        converter = HTMLToMarkdown(llm)
    else:
        converter = SimpleHTMLToMarkdown()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        if args.level == 0:
            # 单页模式
            indexer = WebIndexer(
                llm=llm,
                fetcher=fetcher,
                add_node_id=not args.no_node_id,
                add_node_summary=not args.no_summary and llm is not None,
                add_doc_description=not args.no_description and llm is not None,
                use_llm_for_conversion=use_llm_for_conversion and llm is not None,
            )
            index, markdown_content, pages = crawl_single_page(args, indexer, args.output_dir)
            
            # 输出结果
            result = index.to_json(include_content=args.include_content, indent=2)
            
            # 确定输出路径
            if args.output:
                output_path = args.output
            elif not args.no_save:
                base_name = url_to_filename(args.url)
                output_path = os.path.join(args.output_dir, f"{base_name}_index.json")
            else:
                output_path = None
            
            # 保存文件
            if output_path:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(result)
                print(f"索引已保存到: {output_path}", file=sys.stderr)
                
                # 同时保存 Markdown 原始内容到 result 目录
                if markdown_content and not args.output_markdown:
                    base_name = url_to_filename(args.url)
                    md_path = os.path.join(args.output_dir, f"{base_name}.md")
                    with open(md_path, 'w', encoding='utf-8') as f:
                        f.write(markdown_content)
                    print(f"Markdown 已保存到: {md_path}", file=sys.stderr)
            
            # 如果指定了 --no-save，输出到终端
            if args.no_save:
                print(result)
            
            # 输出统计信息
            all_nodes = index.get_all_nodes()
            print(f"\n统计信息:", file=sys.stderr)
            print(f"  - 文档标题: {index.title}", file=sys.stderr)
            print(f"  - 来源 URL: {args.url}", file=sys.stderr)
            print(f"  - 节点数量: {len(all_nodes)}", file=sys.stderr)
            print(f"  - 根节点数: {len(index.root_nodes)}", file=sys.stderr)
            if output_path:
                print(f"  - 输出文件: {output_path}", file=sys.stderr)
        
        else:
            # 多层爬取模式
            from knowledge_indexer.indexer.tree_builder import TreeBuilder
            tree_builder = TreeBuilder(
                llm=llm,
                add_node_id=not args.no_node_id,
                add_node_summary=not args.no_summary and llm is not None,
                add_doc_description=not args.no_description and llm is not None,
            )
            
            index, markdown_content, pages = crawl_multi_level(
                args, fetcher, converter, tree_builder, args.output_dir
            )
            
            # 输出统计信息
            print(f"\n统计信息:", file=sys.stderr)
            print(f"  - 起始 URL: {args.url}", file=sys.stderr)
            print(f"  - 爬取深度: {args.level}", file=sys.stderr)
            print(f"  - 爬取页面数: {len(pages)}", file=sys.stderr)
            print(f"  - 输出目录: {args.output_dir}", file=sys.stderr)
            
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
