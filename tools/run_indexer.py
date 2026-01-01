#!/usr/bin/env python3
"""
Markdown 文档索引生成器
用法: python run_indexer.py --md_path /path/to/document.md
      python run_indexer.py --md_path /path/to/document.md --config config.yaml

结果会自动保存到 result/ 目录
"""

import argparse
import os
import sys
import shutil
from datetime import datetime

# 获取项目根目录（从 tools 目录向上一级）
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# 添加 src 到路径
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from knowledge_indexer import (
    TreeBuilder,
    LLMFactory,
    IndexerConfig,
)

# 默认输出目录（相对于项目根目录）
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "result")


def main():
    parser = argparse.ArgumentParser(
        description="Markdown 文档索引生成器 - 生成树形结构索引"
    )
    
    # 输入文件
    parser.add_argument(
        "--md_path",
        type=str,
        required=True,
        help="Markdown 文件路径"
    )
    
    # 配置文件
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="配置文件路径（YAML 格式）。如未指定，按顺序查找 config.local.yaml、config.yaml"
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
        "--no-save",
        action="store_true",
        help="不保存文件，仅输出到终端"
    )
    
    parser.add_argument(
        "--include-content",
        action="store_true",
        help="在输出中包含原始内容"
    )
    
    # LLM 配置（命令行参数会覆盖配置文件）
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
        "--no-summary",
        action="store_true",
        help="不生成节点摘要（不使用 LLM）"
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
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.md_path):
        print(f"错误: 文件不存在: {args.md_path}", file=sys.stderr)
        sys.exit(1)
    
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
    
    # 创建 LLM（如果需要）
    llm = None
    if not args.no_summary:
        if not config.llm.api_key and config.llm.provider != "ollama":
            print("警告: 未提供 API 密钥，将不生成摘要", file=sys.stderr)
            print("提示: 在 config.yaml 中配置 api_key 或使用 --api-key 参数", file=sys.stderr)
        else:
            try:
                llm = LLMFactory.from_config(config.llm)
                print(f"使用 LLM: {config.llm.provider}/{config.llm.model}", file=sys.stderr)
            except Exception as e:
                print(f"警告: 创建 LLM 失败: {e}", file=sys.stderr)
                print("将不生成摘要", file=sys.stderr)
    
    # 创建构建器
    builder = TreeBuilder(
        llm=llm,
        add_node_id=not args.no_node_id,
        add_node_summary=not args.no_summary and llm is not None,
        add_doc_description=not args.no_description and llm is not None,
    )
    
    # 构建索引
    print(f"正在处理: {args.md_path}", file=sys.stderr)
    index = builder.build_from_file(args.md_path)
    
    # 输出结果
    result = index.to_json(include_content=args.include_content, indent=2)
    
    # 确定输出路径
    if args.output:
        output_path = args.output
    elif not args.no_save:
        # 自动生成输出路径：result/原文件名_index.json
        os.makedirs(args.output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(args.md_path))[0]
        output_path = os.path.join(args.output_dir, f"{base_name}_index.json")
    else:
        output_path = None
    
    # 保存文件
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(result)
        print(f"索引已保存到: {output_path}", file=sys.stderr)
        
        # 复制原始文件到输出目录
        src_file = os.path.abspath(args.md_path)
        dst_file = os.path.join(args.output_dir, os.path.basename(args.md_path))
        if src_file != os.path.abspath(dst_file):  # 避免复制到自身
            shutil.copy2(src_file, dst_file)
            print(f"原始文件已复制到: {dst_file}", file=sys.stderr)
    
    # 如果指定了 --no-save 或者用户需要查看，输出到终端
    if args.no_save:
        print(result)
    
    # 输出统计信息
    all_nodes = index.get_all_nodes()
    print(f"\n统计信息:", file=sys.stderr)
    print(f"  - 文档标题: {index.title}", file=sys.stderr)
    print(f"  - 总行数: {index.total_lines}", file=sys.stderr)
    print(f"  - 节点数量: {len(all_nodes)}", file=sys.stderr)
    print(f"  - 根节点数: {len(index.root_nodes)}", file=sys.stderr)
    if output_path:
        print(f"  - 输出文件: {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
