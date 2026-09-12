"""把 Chroma 里的 papers collection 原样复制到 Qdrant（改造 #6 之二）。

**向量不重算** —— 从 Chroma get(include=["embeddings"]) 取出逐位上传。
embedding 的任何重算都会引入差异，毁掉对比实验的公平性（设计文档第一节）。

用法（先 `docker compose up -d qdrant`）：

    python app/ai/rag/copy_to_qdrant.py            # 全量复制
    python app/ai/rag/copy_to_qdrant.py --verify   # 复制后抽样核对分数
"""

from __future__ import annotations

import argparse
import os
import sys

# app/ai/rag/ 下四层才到 backend/（与 chromaClient.py 的锚定同一套路）
_BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(_BACKEND_ROOT, "app"))
os.chdir(_BACKEND_ROOT)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true", help="复制后抽样核对两边的相似度分数")
    args = parser.parse_args()

    from core.config import settings
    from ai.rag.chromaClient import document_vector_store as chroma_store, embeddings

    if settings.VECTOR_STORE == "qdrant":
        raise SystemExit("VECTOR_STORE 当前是 qdrant —— 复制方向是从 Chroma 到 Qdrant，先切回 chroma")

    from ai.rag.qdrantClient import QdrantVectorStore

    got = chroma_store.get(include=["documents", "metadatas", "embeddings"])
    n = len(got["ids"])
    print(f"从 Chroma 读出 {n} 块")

    qdrant = QdrantVectorStore(
        url=settings.QDRANT_URL, collection_name="papers", embeddings=embeddings
    )
    from langchain_core.documents import Document
    from qdrant_client import QdrantClient, models

    client = QdrantClient(url=settings.QDRANT_URL, timeout=60)
    # 迁移工具语义 = 重建：度量或参数变了就必须重传，残集合只会制造假对比
    client.delete_collection("papers")
    qdrant.ensure_collection(len(got["embeddings"][0]))

    # 直接走原生 upsert 批量通道（适配器的 add_documents 会重新嵌入，这里向量是现成的）
    batch = 128
    for start in range(0, n, batch):
        end = min(start + batch, n)
        points = [
            models.PointStruct(
                id=chroma_id_to_point(got["ids"][i]),
                vector=got["embeddings"][i],
                payload={"content": got["documents"][i], "metadata": got["metadatas"][i]},
            )
            for i in range(start, end)
        ]
        client.upsert(collection_name="papers", points=points, wait=True)
        print(f"  已上传 {end}/{n}")

    final = client.count(collection_name="papers", exact=True).count
    print(f"Qdrant 现有 {final} 块（Chroma：{n}）")
    if final != n:
        raise SystemExit("数量不一致 —— 停下来查，别继续")

    if args.verify:
        verify(client, embeddings)


def chroma_id_to_point(chroma_id: str) -> str:
    # 与 qdrantClient.delete 同一套映射：get 返回的字符串 id 就是 point id 的命名空间输入
    import uuid

    return str(uuid.uuid5(uuid.NAMESPACE_URL, chroma_id))


def verify(client, embeddings) -> None:
    """抽样核对：两边走各自完整链路（含分数换算）后，分数与名次应逐位一致。"""
    from core.config import settings as _settings

    from ai.rag.chromaClient import document_vector_store as chroma_store
    from ai.rag.qdrantClient import QdrantVectorStore

    adapter = QdrantVectorStore(
        url=_settings.QDRANT_URL, collection_name="papers", embeddings=embeddings
    )
    for query in ("SAP 模块推断的三类语义属性", "ViV-ReID dataset annotation pipeline"):
        chroma_hits = chroma_store.similarity_search_with_relevance_scores(query, k=3)
        qdrant_hits = adapter.similarity_search_with_relevance_scores(query, k=3)
        print(f"\nquery: {query}")
        for (qdoc, qscore), (cdoc, cscore) in zip(qdrant_hits, chroma_hits):
            same = qdoc.page_content == cdoc.page_content
            delta = abs(qscore - cscore)
            print(f"  qdrant {qscore:.6f} | chroma {cscore:.6f} | 差 {delta:.2e} | 同块={same}")


if __name__ == "__main__":
    main()
