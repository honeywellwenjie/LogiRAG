<div align="center">

# 🧠 LogiRAG

**Reasoning-based RAG with Tree Indexing**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-brightgreen.svg)](https://www.docker.com/)

*No Vector DB • No Chunking • Human-like Retrieval • Multi-LLM Support*

[English](#-introduction) | [中文](#-简介-1)

</div>

---

## 📢 Introduction

**LogiRAG** is an open-source, reasoning-based RAG (Retrieval-Augmented Generation) system inspired by [PageIndex](https://github.com/VectifyAI/PageIndex). It builds a hierarchical tree index from documents and uses LLM reasoning to navigate and retrieve relevant content—just like how humans read documents.

### Why LogiRAG?

Traditional vector-based RAG relies on **semantic similarity**, but **similarity ≠ relevance**. When working with professional documents that require domain expertise and multi-step reasoning, similarity search often falls short.

LogiRAG uses **reasoning-based retrieval**:
1. Build a "Table of Contents" **tree structure** from documents
2. Use LLM to **reason** through the tree to find relevant sections

---

## ✨ Features

### Core Features (Inspired by PageIndex)
| Feature | Description |
|---------|-------------|
| 🚫 **No Vector DB** | Uses document structure and LLM reasoning, not vector similarity |
| 🚫 **No Chunking** | Documents organized into natural sections, not artificial chunks |
| 🧠 **Human-like Retrieval** | Simulates how experts navigate complex documents |
| 📊 **Explainable** | Traceable reasoning process with section references |

### 🚀 LogiRAG Unique Features

| Feature | Description |
|---------|-------------|
| 🌐 **Web Scraping** | Crawl and index web pages with multi-level link following |
| 🖥️ **Web UI** | Built-in chat demo and file upload interface |
| 🤖 **Multi-LLM Support** | Works with OpenAI, Ollama, DeepSeek, Azure, vLLM, LocalAI, and any OpenAI-compatible API |
| 🐳 **Docker Ready** | One-command deployment with Docker Compose |
| 📤 **File Upload** | Drag-and-drop file upload with automatic indexing |
| 💬 **Chat Demo** | Interactive chat interface with RAG debug panel |
| 📊 **Context Savings** | Shows token savings (typically 95%+ reduction) |
| 🔄 **Hot Reload** | Update knowledge base without restart |

---

## 🖼️ Screenshots

### Chat Demo with RAG Debug Panel

<img src="docs/images/logirag_demo.png" alt="LogiRAG Demo" width="100%">

- **Left Panel**: RAG Debug Log showing reasoning process, matched nodes, and context statistics
- **Right Panel**: Chat interface with knowledge-based responses
- **99%+ Token Savings**: Only relevant sections are sent to LLM

### File Upload Interface

<img src="docs/images/logirag_upload.png" alt="LogiRAG Upload" width="100%">

- Drag-and-drop file upload
- Optional LLM summary generation
- Automatic indexing

---

## 🛠️ Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/LogiRAG.git
cd LogiRAG
```

### 2. Configure LLM

```bash
cp config.example.yaml config.yaml
# Edit config.yaml with your LLM settings
```

**Example configurations:**

<details>
<summary>OpenAI</summary>

```yaml
llm:
  provider: openai
  api_key: "sk-your-api-key"
  api_base: "https://api.openai.com/v1"
  model: "gpt-4o"
```
</details>

<details>
<summary>Ollama (Local)</summary>

```yaml
llm:
  provider: ollama
  api_base: "http://localhost:11434/v1"
  model: "llama3"
```
</details>

<details>
<summary>DeepSeek</summary>

```yaml
llm:
  provider: openai
  api_key: "sk-your-deepseek-key"
  api_base: "https://api.deepseek.com/v1"
  model: "deepseek-chat"
```
</details>

<details>
<summary>vLLM / LocalAI / LM Studio</summary>

```yaml
llm:
  provider: openai
  api_key: "not-needed"
  api_base: "http://localhost:8000/v1"
  model: "your-local-model"
```
</details>

### 3. Start with Docker (Recommended)

```bash
./tools/restart-rag.sh
```

Or manually:

```bash
cd server
docker compose up -d
```

### 4. Access Web Interface

| Interface | URL | Description |
|-----------|-----|-------------|
| 💬 Chat Demo | http://localhost:3003/demo | Interactive chat with RAG |
| 📤 Upload | http://localhost:3003/upload | Upload knowledge files |
| 📊 Stats | http://localhost:3003/fstats | Knowledge base statistics |
| ❤️ Health | http://localhost:3003/health | Service health check |

---

## 📚 Usage

### Index a Markdown File

```bash
python tools/run_indexer.py --md_path /path/to/document.md
```

### Index a Web Page

```bash
# Single page
python tools/run_web_indexer.py --url https://example.com

# Crawl with depth
python tools/run_web_indexer.py --url https://example.com --level 2 --max-pages 50
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/query` | POST | RAG query with reasoning |
| `/upload` | POST | Upload and index files |
| `/chat` | POST | Chat with knowledge base |
| `/reload` | POST | Reload all indexes |
| `/fstats` | GET | Knowledge base statistics |
| `/health` | GET | Health check |

**Query Example:**

```bash
curl -X POST http://localhost:3003/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is LogiRAG?"}'
```

---

## 📁 Project Structure

```
LogiRAG/
├── src/
│   └── knowledge_indexer/     # Core indexing library
│       ├── indexer/           # Document parsing & tree building
│       ├── llm/               # Multi-LLM support
│       ├── retrieval/         # Reasoning-based search
│       └── web/               # Web scraping
├── server/
│   ├── rag_server.py          # Flask API server
│   ├── Dockerfile             # Docker configuration
│   └── docker-compose.yml     # Docker Compose
├── tools/
│   ├── run_indexer.py         # CLI for Markdown indexing
│   ├── run_web_indexer.py     # CLI for web scraping
│   └── restart-rag.sh         # Service restart script
├── result/                    # Generated indexes (gitignored)
├── config.example.yaml        # Configuration template
└── README.md
```

---

## 🔧 Configuration

### Full Configuration Options

```yaml
# LLM Configuration
llm:
  provider: openai          # openai, ollama
  api_key: "your-key"       # API key
  api_base: "https://..."   # API endpoint
  model: "gpt-4o"           # Model name
  temperature: 0.1          # Response randomness
  max_tokens: 4096          # Max response tokens
  timeout: 60               # Request timeout (seconds)

# Indexer Configuration
indexer:
  add_node_id: true         # Add unique node IDs
  add_node_summary: true    # Generate node summaries
  add_doc_description: true # Generate document descriptions
  max_depth: 6              # Maximum tree depth

# Web Scraping Configuration
web:
  timeout: 30               # Request timeout
  verify_ssl: true          # Verify SSL certificates
  use_llm_for_conversion: true  # Use LLM for HTML→Markdown
```

---

## 🆚 Comparison with PageIndex

| Feature | PageIndex | LogiRAG |
|---------|-----------|---------|
| Tree Indexing | ✅ | ✅ |
| Reasoning-based Retrieval | ✅ | ✅ |
| PDF Support | ✅ | ❌ (Markdown/Text) |
| Web Scraping | ❌ | ✅ |
| Multi-level Crawling | ❌ | ✅ |
| Web UI (Chat) | ❌ | ✅ |
| File Upload UI | ❌ | ✅ |
| Docker Deployment | ❌ | ✅ |
| Multi-LLM Support | OpenAI only | ✅ All OpenAI-compatible |
| Local Models | ❌ | ✅ Ollama, vLLM, etc. |
| RAG Debug Panel | ❌ | ✅ |
| Open Source | ✅ MIT | ✅ MIT |

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Inspired by [PageIndex](https://github.com/VectifyAI/PageIndex) by VectifyAI
- Thanks to all contributors and users

---

## 📢 简介

**LogiRAG** 是一个开源的、基于推理的 RAG（检索增强生成）系统，灵感来自 [PageIndex](https://github.com/VectifyAI/PageIndex)。它从文档构建层次化的树形索引，并使用 LLM 推理来导航和检索相关内容——就像人类阅读文档一样。

### 为什么选择 LogiRAG？

传统的基于向量的 RAG 依赖于**语义相似性**，但**相似 ≠ 相关**。在处理需要专业知识和多步推理的专业文档时，相似性搜索往往不够用。

LogiRAG 使用**基于推理的检索**：
1. 从文档构建"目录"式的**树形结构**
2. 使用 LLM **推理**遍历树来找到相关章节

### ✨ 特性亮点

| 特性 | 描述 |
|------|------|
| 🚫 **无向量数据库** | 使用文档结构和 LLM 推理，而非向量相似性 |
| 🚫 **无分块** | 文档按自然章节组织，而非人为切分 |
| 🧠 **类人检索** | 模拟专家浏览复杂文档的方式 |
| 🌐 **网页爬取** | 支持多层链接跟踪的网页爬取和索引 |
| 🖥️ **Web 界面** | 内置聊天演示和文件上传界面 |
| 🤖 **多 LLM 支持** | 支持 OpenAI、Ollama、DeepSeek、Azure、vLLM、LocalAI 等 |
| 🐳 **Docker 就绪** | 一键 Docker Compose 部署 |
| 💬 **聊天演示** | 带 RAG 调试面板的交互式聊天界面 |
| 📊 **节省 Token** | 通常可节省 95%+ 的 Token |

### 🚀 快速开始

```bash
# 1. 克隆仓库
git clone https://github.com/yourusername/LogiRAG.git
cd LogiRAG

# 2. 配置 LLM
cp config.example.yaml config.yaml
# 编辑 config.yaml 填入你的 API 密钥

# 3. 启动服务
./tools/restart-rag.sh

# 4. 访问界面
# 聊天演示: http://localhost:3003/demo
# 文件上传: http://localhost:3003/upload
```

---

## ⭐ Star History

如果你觉得这个项目有用，请给它一个 ⭐！

你的 Star 帮助更多人发现这个项目，也激励我们持续开发。

[![Star this repo](https://img.shields.io/github/stars/yourusername/LogiRAG?style=social)](https://github.com/yourusername/LogiRAG)

---

<div align="center">

**[⬆ 返回顶部](#-logirag)**

Made with ❤️ by the LogiRAG Community

</div>
