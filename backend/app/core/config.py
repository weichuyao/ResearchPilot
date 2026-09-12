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
    
    APP_NAME: str = "ResearchPilot"
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

    def is_dev(self):
        return self.DEV



settings = Settings()