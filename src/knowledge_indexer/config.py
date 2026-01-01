"""
配置管理模块
支持环境变量、YAML 配置文件和直接配置
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """加载 YAML 配置文件"""
    try:
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        raise RuntimeError("pyyaml 未安装，请运行: pip install pyyaml")
    except FileNotFoundError:
        raise FileNotFoundError(f"配置文件不存在: {config_path}")


@dataclass
class LLMConfig:
    """LLM 配置"""
    provider: str = "openai"  # openai, ollama, custom
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    model: str = "gpt-4o"
    temperature: float = 0.1
    max_tokens: int = 4096
    timeout: int = 60
    
    def __post_init__(self):
        # 从环境变量读取默认值
        if self.api_key is None:
            self.api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        if self.api_base is None:
            self.api_base = os.getenv("LLM_API_BASE") or os.getenv("OPENAI_API_BASE")
        
        # 根据 provider 设置默认 base URL
        if self.api_base is None:
            if self.provider == "openai":
                self.api_base = "https://api.openai.com/v1"
            elif self.provider == "ollama":
                self.api_base = "http://localhost:11434/v1"


@dataclass 
class IndexerConfig:
    """索引器配置"""
    # RAG LLM (用于 RAG 搜索和索引生成)
    rag_llm: LLMConfig = field(default_factory=LLMConfig)
    
    # Chat LLM (可选，用于聊天回复，如不配置则使用 rag_llm)
    chat_llm: Optional[LLMConfig] = None
    
    # 向后兼容：llm 属性指向 rag_llm
    @property
    def llm(self) -> LLMConfig:
        return self.rag_llm
    
    # 索引生成配置
    max_tokens_per_node: int = 20000
    max_depth: int = 6
    add_node_id: bool = True
    add_node_summary: bool = True
    add_doc_description: bool = True
    
    # 语言设置
    language: str = "zh"  # zh, en
    
    @classmethod
    def from_env(cls) -> "IndexerConfig":
        """从环境变量创建配置"""
        llm_config = LLMConfig(
            provider=os.getenv("LLM_PROVIDER", "openai"),
            api_key=os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY"),
            api_base=os.getenv("LLM_API_BASE") or os.getenv("OPENAI_API_BASE"),
            model=os.getenv("LLM_MODEL", "gpt-4o"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
        )
        return cls(llm=llm_config)
    
    @classmethod
    def for_openai(cls, api_key: str, model: str = "gpt-4o", **kwargs) -> "IndexerConfig":
        """创建 OpenAI 配置"""
        llm_config = LLMConfig(
            provider="openai",
            api_key=api_key,
            model=model,
            **kwargs
        )
        return cls(llm=llm_config)
    
    @classmethod
    def for_ollama(cls, model: str = "llama3", base_url: str = "http://localhost:11434/v1", **kwargs) -> "IndexerConfig":
        """创建 Ollama 配置"""
        llm_config = LLMConfig(
            provider="ollama",
            api_base=base_url,
            model=model,
            api_key="ollama",  # Ollama 不需要真实的 API key
            **kwargs
        )
        return cls(llm=llm_config)
    
    @classmethod
    def for_custom(cls, api_key: str, api_base: str, model: str, **kwargs) -> "IndexerConfig":
        """创建自定义 LLM 配置（兼容 OpenAI API 的服务）"""
        llm_config = LLMConfig(
            provider="custom",
            api_key=api_key,
            api_base=api_base,
            model=model,
            **kwargs
        )
        return cls(llm=llm_config)
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "IndexerConfig":
        """从 YAML 配置文件创建配置"""
        data = load_yaml_config(config_path)
        
        # 解析 RAG LLM 配置 (用于 RAG 搜索和索引)
        # 支持新的 rag_llm 和旧的 llm 配置名
        rag_llm_data = data.get("rag_llm") or data.get("llm", {})
        rag_llm_config = LLMConfig(
            provider=rag_llm_data.get("provider", "openai"),
            api_key=rag_llm_data.get("api_key"),
            api_base=rag_llm_data.get("api_base"),
            model=rag_llm_data.get("model", "gpt-4o"),
            temperature=float(rag_llm_data.get("temperature", 0.1)),
            max_tokens=int(rag_llm_data.get("max_tokens", 4096)),
            timeout=int(rag_llm_data.get("timeout", 60)),
        )
        
        # 解析 Chat LLM 配置 (可选，用于聊天回复)
        chat_llm_config = None
        chat_llm_data = data.get("chat_llm", {})
        if chat_llm_data:
            chat_llm_config = LLMConfig(
                provider=chat_llm_data.get("provider", "openai"),
                api_key=chat_llm_data.get("api_key"),
                api_base=chat_llm_data.get("api_base"),
                model=chat_llm_data.get("model", "gpt-4o"),
                temperature=float(chat_llm_data.get("temperature", 0.7)),
                max_tokens=int(chat_llm_data.get("max_tokens", 4096)),
                timeout=int(chat_llm_data.get("timeout", 60)),
            )
        
        # 解析索引器配置
        indexer_data = data.get("indexer", {})
        
        return cls(
            rag_llm=rag_llm_config,
            chat_llm=chat_llm_config,
            add_node_id=indexer_data.get("add_node_id", True),
            add_node_summary=indexer_data.get("add_node_summary", True),
            add_doc_description=indexer_data.get("add_doc_description", True),
            max_depth=indexer_data.get("max_depth", 6),
        )
    
    @classmethod
    def from_file(cls, config_path: str = None) -> "IndexerConfig":
        """
        从配置文件创建配置
        
        按以下顺序查找配置文件：
        1. 指定的 config_path
        2. config.local.yaml（本地配置，不提交到 git）
        3. config.yaml
        4. 环境变量
        """
        # 查找配置文件
        search_paths = []
        if config_path:
            search_paths.append(config_path)
        
        # 在当前目录和项目根目录查找
        for base_dir in [os.getcwd(), os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))]:
            search_paths.extend([
                os.path.join(base_dir, "config.local.yaml"),
                os.path.join(base_dir, "config.yaml"),
            ])
        
        for path in search_paths:
            if os.path.exists(path):
                return cls.from_yaml(path)
        
        # 没有找到配置文件，使用环境变量
        return cls.from_env()


@dataclass
class WebConfig:
    """网页抓取配置"""
    timeout: int = 30
    verify_ssl: bool = True
    use_llm_for_conversion: bool = True
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WebConfig":
        """从字典创建配置"""
        return cls(
            timeout=data.get("timeout", 30),
            verify_ssl=data.get("verify_ssl", True),
            use_llm_for_conversion=data.get("use_llm_for_conversion", True),
        )

