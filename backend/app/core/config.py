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
    
    APP_NAME: str = "AI ChatKit"
    DEBUG: bool = True
    DATABASE_URL: str | None = None
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    DEV: bool = True
    
    DEEPSEEK_API_KEY: str | None = None
    OLLAMA_BASE_URL: str | None = None
    OLLAMA_MODEL: str | None = None
    
    DEFAULT_MODEL: str | None = None
    
    EMBEDDING_MODEL: str | None = None
    
    CHROMA_PATH: str | None = None

    # --- 文档上传（改造 #5）---
    # 上传的 PDF 落盘目录，相对 backend/。用内容哈希命名，不用用户给的文件名
    # （用户给的名字不可信，也不该出现在磁盘路径里）。
    UPLOAD_DIR: str = "resource/uploads"
    # 单个文件大小上限。超了回 413，而且必须**边读边算**，不能先整个读进内存再判。
    MAX_UPLOAD_MB: int = 50

    def is_dev(self):
        return self.DEV



settings = Settings()