<div align="center">

# 🧠 LogiRAG

**基于推理的 RAG 系统，采用树形索引**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-brightgreen.svg)](https://www.docker.com/)

*无向量数据库 • 无分块 • 类人检索 • 多LLM支持*

[English](README.md) | [中文](#-简介)

</div>

---

## 📢 简介

**LogiRAG** 是一个开源的、基于推理的 RAG（检索增强生成）系统，灵感来自 [PageIndex](https://github.com/VectifyAI/PageIndex)。它从文档构建层次化的树形索引，并使用 LLM 推理来导航和检索相关内容——就像人类阅读文档一样。

### 为什么选择 LogiRAG？

传统的基于向量的 RAG 依赖于**语义相似性**，但**相似 ≠ 相关**。在处理需要专业知识和多步推理的专业文档时，相似性搜索往往不够用。

LogiRAG 使用**基于推理的检索**：
1. 从文档构建"目录"式的**树形结构**
2. 使用 LLM **推理**遍历树来找到相关章节

---

## ✨ 特性

### 核心特性（灵感来自 PageIndex）
| 特性 | 描述 |
|------|------|
| 🚫 **无向量数据库** | 使用文档结构和 LLM 推理，而非向量相似性 |
| 🚫 **无分块** | 文档按自然章节组织，而非人为切分 |
| 🧠 **类人检索** | 模拟专家浏览复杂文档的方式 |
| 📊 **可解释性** | 可追溯的推理过程，带章节引用 |

### 🚀 LogiRAG 独有特性

| 特性 | 描述 |
|------|------|
| 🌐 **网页爬取** | 支持多层链接跟踪的网页爬取和索引 |
| 🖥️ **Web 界面** | 内置聊天演示和文件上传界面 |
| 🤖 **多 LLM 支持** | 支持 OpenAI、Ollama、DeepSeek、Azure、vLLM、LocalAI 等所有 OpenAI 兼容 API |
| 🐳 **Docker 就绪** | 一键 Docker Compose 部署 |
| 📤 **文件上传** | 拖拽上传文件，自动索引 |
| 💬 **聊天演示** | 带 RAG 调试面板的交互式聊天界面 |
| 📊 **Token 节省** | 显示 Token 节省量（通常可节省 95%+） |
| 🔄 **热重载** | 无需重启即可更新知识库 |

---

## 🖼️ 截图

### 聊天演示（带 RAG 调试面板）

<img src="docs/images/logirag_demo.png" alt="LogiRAG Demo" width="100%">

- **左侧面板**：RAG 调试日志，显示推理过程、匹配节点和上下文统计
- **右侧面板**：基于知识库的聊天界面
- **99%+ Token 节省**：只有相关章节被发送给 LLM

### 文件上传界面

<img src="docs/images/logirag_upload.png" alt="LogiRAG Upload" width="100%">

- 拖拽上传文件
- 可选 LLM 摘要生成
- 自动索引

---

## 🛠️ 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/yourusername/LogiRAG.git
cd LogiRAG
```

### 2. 配置 LLM

```bash
cp config.example.yaml config.yaml
# 编辑 config.yaml 填入你的 LLM 设置
```

**配置示例：**

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
<summary>Ollama（本地）</summary>

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

### 3. 使用 Docker 启动（推荐）

```bash
./tools/restart-rag.sh
```

或手动启动：

```bash
cd server
docker compose up -d
```

### 4. 访问 Web 界面

| 界面 | URL | 描述 |
|------|-----|------|
| 💬 聊天演示 | http://localhost:3003/demo | 交互式 RAG 聊天 |
| 📤 上传 | http://localhost:3003/upload | 上传知识文件 |
| 📊 统计 | http://localhost:3003/fstats | 知识库统计 |
| ❤️ 健康检查 | http://localhost:3003/health | 服务健康状态 |

---

## 📚 使用方法

### 索引 Markdown 文件

```bash
python tools/run_indexer.py --md_path /path/to/document.md
```

### 索引网页

```bash
# 单个页面
python tools/run_web_indexer.py --url https://example.com

# 多层爬取
python tools/run_web_indexer.py --url https://example.com --level 2 --max-pages 50
```

### API 接口

| 接口 | 方法 | 描述 |
|------|------|------|
| `/query` | POST | RAG 推理查询 |
| `/upload` | POST | 上传并索引文件 |
| `/chat` | POST | 与知识库对话 |
| `/reload` | POST | 重新加载所有索引 |
| `/fstats` | GET | 知识库统计 |
| `/health` | GET | 健康检查 |

**查询示例：**

```bash
curl -X POST http://localhost:3003/query \
  -H "Content-Type: application/json" \
  -d '{"query": "什么是 LogiRAG?"}'
```

---

## 📁 项目结构

```
LogiRAG/
├── src/
│   └── knowledge_indexer/     # 核心索引库
│       ├── indexer/           # 文档解析 & 树构建
│       ├── llm/               # 多 LLM 支持
│       ├── retrieval/         # 基于推理的搜索
│       └── web/               # 网页爬取
├── server/
│   ├── rag_server.py          # Flask API 服务器
│   ├── Dockerfile             # Docker 配置
│   └── docker-compose.yml     # Docker Compose
├── tools/
│   ├── run_indexer.py         # Markdown 索引 CLI
│   ├── run_web_indexer.py     # 网页爬取 CLI
│   └── restart-rag.sh         # 服务重启脚本
├── result/                    # 生成的索引（gitignore）
├── config.example.yaml        # 配置模板
└── README.md
```

---

## 🔧 配置

### 完整配置选项

```yaml
# LLM 配置
llm:
  provider: openai          # openai, ollama
  api_key: "your-key"       # API 密钥
  api_base: "https://..."   # API 端点
  model: "gpt-4o"           # 模型名称
  temperature: 0.1          # 响应随机性
  max_tokens: 4096          # 最大响应 token
  timeout: 60               # 请求超时（秒）

# 索引器配置
indexer:
  add_node_id: true         # 添加唯一节点 ID
  add_node_summary: true    # 生成节点摘要
  add_doc_description: true # 生成文档描述
  max_depth: 6              # 最大树深度

# 网页爬取配置
web:
  timeout: 30               # 请求超时
  verify_ssl: true          # 验证 SSL 证书
  use_llm_for_conversion: true  # 使用 LLM 转换 HTML→Markdown
```

---

## 🆚 与 PageIndex 对比

| 特性 | PageIndex | LogiRAG |
|------|-----------|---------|
| 树形索引 | ✅ | ✅ |
| 基于推理的检索 | ✅ | ✅ |
| PDF 支持 | ✅ | ❌（Markdown/文本） |
| 网页爬取 | ❌ | ✅ |
| 多层爬取 | ❌ | ✅ |
| Web UI（聊天） | ❌ | ✅ |
| 文件上传界面 | ❌ | ✅ |
| Docker 部署 | ❌ | ✅ |
| 多 LLM 支持 | 仅 OpenAI | ✅ 所有 OpenAI 兼容 |
| 本地模型 | ❌ | ✅ Ollama, vLLM 等 |
| RAG 调试面板 | ❌ | ✅ |
| 开源 | ✅ MIT | ✅ MIT |

---

## 🤝 贡献

欢迎贡献！请随时提交 Pull Request。

1. Fork 仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

---

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

---

## 🙏 致谢

- 灵感来自 VectifyAI 的 [PageIndex](https://github.com/VectifyAI/PageIndex)
- 感谢所有贡献者和用户

---

## ⭐ Star 历史

如果你觉得这个项目有用，请给它一个 ⭐！

你的 Star 帮助更多人发现这个项目，也激励我们持续开发。

[![Star this repo](https://img.shields.io/github/stars/yourusername/LogiRAG?style=social)](https://github.com/yourusername/LogiRAG)

---

<div align="center">

**[⬆ 返回顶部](#-logirag)**

Made with ❤️ by the LogiRAG Community

</div>

