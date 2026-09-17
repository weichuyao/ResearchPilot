"""
config settings
"""

# from pydantic import BaseSettings
from ctypes import DEFAULT_MODE
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import find_dotenv



class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=find_dotenv(),
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
        validate_default=False,
    )
    
    APP_NAME: str = "科研领航"
    DEBUG: bool = True
    DATABASE_URL: str | None = None
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    DEV: bool = True

    # --- 日志（改造 #7）---
    # 日志目录，相对 backend/（见 core/logging_config.py 里锚定的理由）。
    # 容器里这个目录由 compose 的 bind mount 提供（./backend/logs:/app/logs），
    # 重启容器后日志还在 —— 落文件要的就是这个。
    LOG_DIR: str = "logs"
    # 根日志级别。DEBUG 会让 httpx 之外的大部分库的细节进来，排查用。
    LOG_LEVEL: str = "INFO"
    
    DEEPSEEK_API_KEY: str | None = None
    OLLAMA_BASE_URL: str | None = None
    OLLAMA_MODEL: str | None = None
    
    DEFAULT_MODEL: str | None = None
    
    EMBEDDING_MODEL: str | None = None

    CHROMA_PATH: str | None = None

    # --- 向量库后端（改造 #6 之二）---
    # "chroma"（默认，嵌入式文件，行为不变）| "qdrant"（独立服务，对比/切换用）。
    # 对比实验的口径见 reference/transformation-06-qdrant-design.md。
    VECTOR_STORE: str = "chroma"
    QDRANT_URL: str = "http://127.0.0.1:6333"
    # True 时 Qdrant 放弃 HNSW 近似做全量比对 —— 名次确定性的对照开关。
    # 636 块的语料上代价可忽略；Chroma 没有这个能力，这正是对比的痛点之一。
    QDRANT_EXACT: bool = False

    # --- MCP（改造 #4）---
    # 目前只有 arXiv 一条。ENABLED=false 时一个子进程都不拉（评估/测试用）。
    MCP_ARXIV_ENABLED: bool = False
    # stdio 传输的启动命令，空格切分。uvx 首次运行会下载包，之后走缓存。
    MCP_ARXIV_COMMAND: str = "uvx arxiv-mcp-server"

    # 当前代码的 git 提交号，给 /health 用。
    #
    # 本机跑的时候 /health 直接读 .git 目录就行（见 api/system_routes.py）。
    # 但在容器里**读不到** —— `.git` 在仓库根目录，而镜像的构建上下文是 backend/，
    # 它根本不在里面。于是 /health 会报 null，恰好丢掉它最想要回答的那个问题
    # （"线上跑的是哪份代码"）。
    #
    # 所以构建时用 --build-arg GIT_REV=... 传进来，这里作为首选来源。
    GIT_REV: str | None = None

    # --- 文档上传（改造 #5）---
    # 上传的 PDF 落盘目录，相对 backend/。用内容哈希命名，不用用户给的文件名
    # （用户给的名字不可信，也不该出现在磁盘路径里）。
    UPLOAD_DIR: str = "resource/uploads"
    # 单个文件大小上限。超了回 413，而且必须**边读边算**，不能先整个读进内存再判。
    MAX_UPLOAD_MB: int = 50

    # --- 规划节点专用的改写模型（改造 #11：微调闭环）---
    #
    # ## 为什么规划节点要能和生成节点用不同的模型
    #
    # 图上四类节点对模型的要求不同：analyze / refine 是**窄任务**——
    # 中文问题进、固定 JSON schema 的英文检索查询出，不需要世界知识，
    # 格式约束强；而 synthesize 要读十几个块、产出带页码引用的长回答。
    #
    # 所以「用一个小模型替掉一次 API 调用」是划算的工程决策，
    # 而「用一个小模型替掉生成环节」是事故。这个字段让前者可配置，
    # 后者不受影响。
    #
    # 取值是 ai/models.py 里的模型名（如 "local-rewriter"）。留空 = 沿用
    # DEFAULT_MODEL，即完全保持旧行为（改造前的所有评估报告都是这个口径）。
    ANALYZE_MODEL: str | None = None

    # 本地改写模型在 Ollama 里的 tag（ANALYZE_MODEL=local-rewriter 时使用）。
    # 微调产物的落地方式见 finetune/README.md：LLaMA-Factory 导出 GGUF →
    # ollama create。默认值只是占位，实际必须与 ollama create 时起的名字一致。
    OLLAMA_REWRITE_MODEL: str = "research-rewriter"

    # --- 语料侧术语补全（改造 #11 第三档）---
    #
    # 小模型学会了「输出英文查询」但**背不出论文里的字面术语**（实测：
    # 它把 A²RNet 的 `instance bank` 换成了泛化描述，还把 TAR 编造成
    # "Temporal Adaptive Representation"）。那是领域知识缺失，不是改写能力
    # 缺失 —— 微调补不上，加训练轮数只会加重过拟合与幻觉。
    #
    # 所以把术语交给语料：从 BM25 索引里挖出「这篇论文确实在用的稀有词组」
    # 补进查询。术语有了语料出处，幻觉问题一并消失。
    # 见 ai/rag/term_expand.py 的模块文档。
    #
    # 默认关闭：它是实验性增强，开启前请用 rewrite_bench 对比。
    TERM_EXPAND_ENABLED: bool = False
    TERM_EXPAND_MAX_TERMS: int = 4

    def is_dev(self):
        return self.DEV



settings = Settings()