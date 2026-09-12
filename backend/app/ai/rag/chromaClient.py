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
    # ⚠️ 这一行是改造 #9 做 Docker 时才补上的 —— 之前这里是
    # `OllamaEmbeddings(model=settings.EMBEDDING_MODEL)`，**没有传 base_url**。
    #
    # 后果：项目里明明定义了 `settings.OLLAMA_BASE_URL`（ai/llm.py 的聊天模型用了它），
    # 但 embedding 那一路被忽略了，只能走 langchain_ollama 的默认值 127.0.0.1:11434。
    # 宿主机上跑没问题，**一放进容器就指向容器自己**，连不上 Ollama。
    #
    # 这类"设置项存在但一半代码没读它"的缺口很难发现：不报错、不告警，
    # 只在换环境时以"连接被拒绝"的形式出现，而且很容易被误认为是网络问题。
    base_url=settings.OLLAMA_BASE_URL or None,
)

# ResearchPilot 的论文知识库。检索工具用的是这一个。
#
# 后端由 settings.VECTOR_STORE 选择（改造 #6 之二）：
#   chroma（默认）—— 嵌入式文件，行为与历史一致；
#   qdrant        —— 独立服务（compose 的 qdrant），对比/切换用。
# 两个后端实现同一个五方法接口（get / similarity_search_with_relevance_scores /
# add_documents / delete / count），hybrid / ingest / rank_bench 一行不改。
# 对比口径与结论见 reference/transformation-06-qdrant-design.md。
if settings.VECTOR_STORE == "qdrant":
    from ai.rag.qdrantClient import QdrantVectorStore

    document_vector_store: object = QdrantVectorStore(
        url=settings.QDRANT_URL,
        collection_name="papers",
        embeddings=embeddings,
        exact=settings.QDRANT_EXACT,
    )
else:
    class _PaperChroma(Chroma):
        """补一个 count()：hybrid 的自检原来直接摸 client.get_collection，
        那是 Chroma 专有的。计数走统一接口之后，/health 的语义才不依赖后端。"""

        def count(self) -> int:
            return self._collection.count()

    document_vector_store: object = _PaperChroma(
        collection_name="papers",
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
        client=client,
        create_collection_if_not_exists=True,
    )

