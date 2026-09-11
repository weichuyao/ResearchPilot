import os

import chromadb
from chromadb.config import Settings
from langchain_ollama import OllamaEmbeddings

from langchain_chroma import Chroma
from core.config import settings
from chromadb.api.shared_system_client import SharedSystemClient


# CHROMA_PATH 在 .env 里是相对路径（resource/chroma_db）。相对路径会按「进程的工作目录」
# 解析，所以从不同目录启动就会写出不同的数据目录 —— 项目里已经因此留下过
# backend/app/resource/ 这样的残留。这里统一锚定到 backend/，消除这个隐患。
_BACKEND_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
CHROMA_PATH = settings.CHROMA_PATH or "resource/chroma_db"
if not os.path.isabs(CHROMA_PATH):
    CHROMA_PATH = os.path.join(_BACKEND_ROOT, CHROMA_PATH)


client = chromadb.PersistentClient(path=CHROMA_PATH, settings=Settings(anonymized_telemetry=False))

embeddings = OllamaEmbeddings(
    model=settings.EMBEDDING_MODEL,
)

# 原始 baseline 的员工手册 collection。保留不动，让 tests/rag/ 下的旧脚本还能跑。
hand_book_vector_store = Chroma(
    collection_name="handbook",
    persist_directory=CHROMA_PATH,
    embedding_function=embeddings,
    client=client,
    create_collection_if_not_exists=True,
)

# ResearchPilot 的论文知识库。检索工具用的是这一个。
document_vector_store = Chroma(
    collection_name="papers",
    persist_directory=CHROMA_PATH,
    embedding_function=embeddings,
    client=client,
    create_collection_if_not_exists=True,
)
