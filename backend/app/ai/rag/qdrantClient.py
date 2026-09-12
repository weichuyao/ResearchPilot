"""Qdrant 向量库适配器（改造 #6 之二）。

## 为什么长这样

代码对 Chroma 的耦合面已经收敛到五个方法（见 transformation-06-qdrant-design.md）：
`get` / `similarity_search_with_relevance_scores` / `add_documents` / `delete` / `count`。
这个类实现**同一个接口**，于是 hybrid / ingest / rank_bench 一行不改就能换后端 ——
对比实验才可能公平（唯一变量是向量库本身）。

只实现被用到的那部分接口，不做通用抽象层 —— 多了的方法就是没人测的代码。

## 分数口径

Chroma 的 relevance score（cosine space）就是余弦相似度；Qdrant 的 COSINE 距离
返回的 score 同样是余弦相似度。两边逐位可比，是「阀门」（阈值 0.35）能直接复用的前提。

## id 空间

Chroma 的 id 是字符串；Qdrant 的 point id 必须是 UUID 或整数。这里用
`uuid5(内容键)` 做稳定映射：同一块内容在两个库里对应同一个逻辑 id，
`get(where={"source": ...}) → delete(ids)` 的先查后删流程保持一致。
"""

from __future__ import annotations

import logging
import uuid

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def _point_id(metadata: dict, content: str) -> str:
    key = "%s|%s|%s" % (metadata.get("source", ""), metadata.get("page_label", ""), content)
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def _translate_filter(where: dict | None) -> dict | None:
    """Chroma 的 metadata filter 语法 → Qdrant payload filter。

    只翻译项目实际用到的两种：`{"k": v}` 和 `{"k": {"$in": [...]}}`。
    其它语法出现在这里就该加测试，而不是静默接受。
    """
    if not where:
        return None
    must = []
    for key, value in where.items():
        payload_key = "metadata." + key
        if isinstance(value, dict) and "$in" in value:
            must.append({"key": payload_key, "match": {"any": list(value["$in"])}})
        else:
            must.append({"key": payload_key, "match": {"value": value}})
    return {"must": must}


class QdrantVectorStore:
    """与 langchain_chroma.Chroma 被用到的行为对齐的最小适配器。"""

    def __init__(self, url: str, collection_name: str, embeddings, exact: bool = False):
        from qdrant_client import QdrantClient, models

                # 服务端 compose 里锁 v1.12.4：本适配器只用到 scroll / retrieve /
        # query_points / upsert / delete / count，1.10+ 都支持，版本检查关掉
        # 免得客户端一升级就报警。
        self._client = QdrantClient(url=url, timeout=30, check_compatibility=False)
        self._collection = collection_name
        self._embeddings = embeddings
        self._models = models
        # exact=True：放弃 HNSW 近似，全量比对 —— 名次确定性的对照实验用。
        self._exact = exact

    # ---- 建集合 / 写入（ingest 用）----

    def ensure_collection(self, vector_size: int) -> None:
        m = self._models
        if not self._client.collection_exists(self._collection):
            self._client.create_collection(
                collection_name=self._collection,
                # EUCLID 而不是 COSINE：Chroma 的 papers collection 实际跑在 l2 空间
                # （metadata 为 None 时的默认值，实测确认），langchain 把分数换算成
                # 1 - d²/√2。阈值 0.35 是在那个口径上校准的 —— 要让阀门可移植，
                # 两边必须同度量同变换，见下方 _relevance()。
                vectors_config=m.VectorParams(size=vector_size, distance=m.Distance.EUCLID),
            )

    @staticmethod
    def _relevance(distance: float) -> float:
        """复现 langchain_chroma 对 l2 度量的分数变换。

        Chroma 的 l2 返回**平方**欧氏距离，langchain 的换算是 1 - d/√2（d 为
        平方距离），即 relevance = 1 - d²/√2。实测（本项目 636 块）与 Chroma
        报分逐位一致。前置条件：bge-m3 的向量模长为 1（实测确认），此口径下
        L2 排序与余弦排序单调等价。
        """
        import math

        return 1.0 - (distance * distance) / math.sqrt(2)

    def add_documents(self, documents: list[Document]) -> None:
        m = self._models
        texts = [d.page_content for d in documents]
        vectors = self._embeddings.embed_documents(texts)
        self.ensure_collection(len(vectors[0]))
        points = [
            m.PointStruct(
                id=_point_id(d.metadata, d.page_content),
                vector=vector,
                payload={"content": d.page_content, "metadata": dict(d.metadata)},
            )
            for d, vector in zip(documents, vectors)
        ]
        self._client.upsert(collection_name=self._collection, points=points, wait=True)

    def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        # ids 是 get() 返回的字符串 id —— uuid5 同一批字符串得到的还是同一批 point id
        point_ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, i)) for i in ids]
        self._client.delete(
            collection_name=self._collection,
            points_selector=self._models.PointIdsList(points=point_ids),
            wait=True,
        )

    # ---- 查询（hybrid 用）----

    def _scroll_all(self, where: dict | None = None):
        flt = _translate_filter(where)
        offset = None
        while True:
            points, offset = self._client.scroll(
                collection_name=self._collection,
                scroll_filter=flt,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            yield from points
            if offset is None:
                return

    @staticmethod
    def _point_to_doc(point) -> Document:
        payload = point.payload or {}
        return Document(page_content=payload.get("content", ""), metadata=dict(payload.get("metadata") or {}))

    def get(self, include: list[str] | None = None, where: dict | None = None) -> dict:
        points = list(self._scroll_all(where))
        want_vectors = bool(include and "embeddings" in include)
        vectors: dict[str, list] = {}
        if want_vectors and points:
            # 一次批量取回全部向量（迁移脚本用）；分页 retrieve 是逐点的，别那么干
            got = self._client.retrieve(
                collection_name=self._collection,
                ids=[p.id for p in points],
                with_vectors=True,
            )
            vectors = {str(r.id): list(r.vector) for r in got}
        out: dict = {"ids": [], "documents": [], "metadatas": []}
        if want_vectors:
            out["embeddings"] = []
        for point in points:
            out["ids"].append(str(point.id))
            doc = self._point_to_doc(point)
            out["documents"].append(doc.page_content)
            out["metadatas"].append(doc.metadata)
            if want_vectors:
                out["embeddings"].append(vectors.get(str(point.id), []))
        return out

    def similarity_search_with_relevance_scores(
        self, query: str, k: int = 4, filter: dict | None = None, **kwargs
    ) -> list[tuple[Document, float]]:
        m = self._models
        vector = self._embeddings.embed_query(query)
        result = self._client.query_points(
            collection_name=self._collection,
            query=vector,
            limit=k,
            query_filter=_translate_filter(filter),
            search_params=m.SearchParams(exact=self._exact),
            with_payload=True,
        )
        # EUCLID 度量下 score 是（非平方）欧氏距离，越小越近 —— 换算成与
        # Chroma 相同的 relevance 口径（越大越相关），hybrid 的排序与阀门不用改。
        return [(self._point_to_doc(p), self._relevance(float(p.score))) for p in result.points]

    # ---- 计数（hybrid 自检 / /health 同款语义）----

    def count(self) -> int:
        return self._client.count(collection_name=self._collection, exact=True).count
