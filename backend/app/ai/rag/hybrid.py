"""混合检索：向量召回 + BM25 关键词召回，用 RRF 融合。

## 为什么需要 BM25

纯向量检索靠语义相似度，在两类情况下都会吃亏：

  1. **词汇完全不重合**：问题说 "annotation pipeline"，论文写的是
     "Grounding DINO is employed to generate the bounding boxes"。
     语义虽然相关，但相似度分数低到过不了阈值（实测 0.3453，阈值 0.35）。
  2. **精确术语**：模型名、数据集名、缩写（如 DPSM、AIS、ViV-ReID）。
     关键词命中比语义相似度更可靠。

## 为什么用 RRF 融合，而不是加权分数

余弦相似度是 0~1，BM25 分数无上界，两者**量纲不同，无法直接相加**。
要加权就必须先归一化，而归一化方式本身就是新的主观参数，而且 BM25 的分布形态
随查询长度变化，归一化很脆。

RRF（Reciprocal Rank Fusion）**只用排名、不用分数**：

    RRF(d) = Σ_i  1 / (k + rank_i(d))          k 取 60（论文推荐值）

天然回避了量纲问题，也不需要调权重。

## 为什么自己实现 BM25 而不装 rank_bm25

这个项目的环境有两个 venv（`.venv` 是 3.13、`.venv-py311` 才是实际在用的），
且都没装 pip、`pyproject.toml` 声明的 >=3.13 与实际运行的 3.11 不一致。
为一个 30 KB 的纯 Python 包去动依赖环境和 lockfile，风险大于收益。

BM25 是标准公式，自己实现还能顺带解决一个现成库做不到的事：
**把候选集按 metadata 过滤之后再做排序**（限定单篇论文检索时需要）。

## 阈值在这里的角色

RRF 的分数是排名算出来的（0.016 这种量级），和余弦相似度不是一个量纲。
所以原来的相关性阈值不再用于排序过滤，而是**降级成一个「阀门」**：

    向量 top-1 分数 < RELEVANCE_THRESHOLD  ->  判定这个话题不在库里
    过了阀门  ->  交给 RRF 排序

这样既保住了「查不到就说未找到」的行为，又让排序用上了关键词信号。

## 并列分数必须显式打破（否则检索结果不可复现）

这个 bug 是被 rank_bench.py 抓出来的：同一个问题连跑三次，A09 的期望段落
一次排第 6、一次排第 7。查下来不是 embedding 的问题（同一段文本嵌入三次，
1024 维逐位相同），而是 **Chroma 对分数并列的条目，返回顺序在进程之间不稳定**：

    两次运行：前 6 名的分数和内容哈希完全一致
             第 20 名的整体顺序哈希不同

Chroma 用的是 HNSW 近似检索，并列项之间没有定义的先后。而原来的代码
`for doc, score in vector_results:` 直接吃返回顺序、RRF 也只按分数排序
（Python 的 sort 是稳定的，于是并列项保持插入顺序）—— 也就是说，
**排序结果里混进了 Chroma 内部的、进程间不稳定的顺序**。

后果不只是「结果会飘」：评估脚本拿它做前后对比时，一名的差异分不清是改动
带来的还是这个不确定性带来的。所以这里三处都改成显式按 `(-分数, key)` 排序，
并列项一律由稳定的 key 决定先后：

  1. 向量召回结果的排序
  2. BM25 的排序
  3. RRF 融合后的排序
"""

from __future__ import annotations

import hashlib
import re
import threading

import numpy as np

from langchain_core.documents import Document

from ai.rag.chromaClient import document_vector_store


# ---- 检索参数 ---------------------------------------------------------------
VECTOR_TOP_K = 20   # 向量先宽召回，给融合留空间
BM25_TOP_K = 20     # 关键词同样宽召回
FINAL_TOP_N = 10    # 融合后交给模型的条数（与原 RETRIEVE_K 持平，便于对比）
RRF_K = 60          # RRF 的平滑常数，取论文推荐值

# BM25 参数（取文献常用值）
BM25_K1 = 1.5
BM25_B = 0.75

# 文档频率上限：出现频率超过这个比例的**索引词**会被整体剔除。
# 这是数据驱动的停用词识别 —— 不去手写词表，而是让语料自己说话。
MAX_DF_RATIO = 0.20

# 手写停用词表：补 DF 过滤抓不到的那一类。
# 英文虚词（the/of/is…）靠 DF 就能识别，但**疑问词**在这个语料里很罕见，
# 所以 IDF 很高、单靠它们也能凑出可观分数 —— 实测 "how" 一个词就贡献了 5.67 分，
# 而 "how to cook braised pork belly" 本该被拦住。稀有 ≠ 相关。
STOPWORDS = frozenset("""
a an the and or but if then than that this these those it its is are was were be been being
of in on at to for from by with without into over under about
we our you your they their he she his her
do does did can could should would may might will shall
not no nor so such very more most much many any all some each other another
what which who whom whose when where why how whether
there here also thus however therefore
one two three first second third
""".split())

# 分词：ASCII 单词/数字整体成一个词，CJK 逐字成词。
# 逐字对中文很粗，但这批语料是英文论文，中文查询主要由向量那一路负责；
# 这里保留逐字只是为了以后语料换成中文时不至于完全失效。
_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


def doc_key(metadata: dict, content: str) -> str:
    """块的稳定标识。用于把向量检索的结果和 BM25 的排名对应起来。"""
    raw = "%s|%s|%s" % (
        metadata.get("source", ""),
        metadata.get("page_label", ""),
        hashlib.md5((content or "").encode("utf-8")).hexdigest(),
    )
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


class Bm25Index:
    """BM25（Lucene 变体）。

    IDF 用 log(1 + (N - n + 0.5) / (n + 0.5))，这个形式恒为正，
    避免高频词拿到负 IDF 从而把总分拉低。
    """

    def __init__(self, documents: list[Document], k1: float = BM25_K1, b: float = BM25_B):
        self.documents = documents
        self.k1 = k1
        self.b = b

        tokenized = [tokenize(doc.page_content) for doc in documents]
        self.doc_len = np.array([len(tokens) for tokens in tokenized], dtype=np.float64)
        self.avgdl = float(self.doc_len.mean()) if len(self.doc_len) else 1.0
        if self.avgdl <= 0:
            self.avgdl = 1.0

        vocab: dict[str, int] = {}
        for tokens in tokenized:
            for token in set(tokens):
                vocab.setdefault(token, len(vocab))
        self.vocab = vocab

        doc_freq = np.zeros(len(vocab), dtype=np.float64)
        self.term_freq: list[dict[int, int]] = []
        for tokens in tokenized:
            counts: dict[int, int] = {}
            for token in tokens:
                term_id = vocab[token]
                counts[term_id] = counts.get(term_id, 0) + 1
            self.term_freq.append(counts)
            for term_id in counts:
                doc_freq[term_id] += 1

        # ---- 索引层停用词过滤 ----
        # 出现在超过 MAX_DF_RATIO 比例的块里的词，以及手写停用词表里的词，
        # 都从索引中整体剔除：它们对区分「相关/不相关」没有贡献，
        # 却会在长查询里累积成可观的分数（停用词污染）。
        n_docs_total = max(len(documents), 1)
        self.allowed_ids = {
            term_id
            for term, term_id in vocab.items()
            if term not in STOPWORDS and doc_freq[term_id] / n_docs_total <= MAX_DF_RATIO
        }
        for counts in self.term_freq:
            for term_id in [t for t in counts if t not in self.allowed_ids]:
                del counts[term_id]

        n_docs = max(len(documents), 1)
        self.idf = np.log(1.0 + (n_docs - doc_freq + 0.5) / (doc_freq + 0.5))

        # 长度归一化项预先算好
        self.length_norm = self.k1 * (1.0 - self.b + self.b * self.doc_len / self.avgdl)

    def scores(self, query: str) -> np.ndarray:
        """返回每个文档的 BM25 分数（与 self.documents 同序）。"""
        out = np.zeros(len(self.documents), dtype=np.float64)
        query_ids = {
            self.vocab[token]
            for token in tokenize(query)
            if token in self.vocab and self.vocab[token] in self.allowed_ids
        }
        if not query_ids:
            return out
        for index, counts in enumerate(self.term_freq):
            total = 0.0
            for term_id in query_ids:
                freq = counts.get(term_id, 0)
                if freq:
                    total += self.idf[term_id] * (freq * (self.k1 + 1.0)) / (freq + self.length_norm[index])
            out[index] = total
        return out


# ---- 索引缓存 ---------------------------------------------------------------
# 语料有 414 块 / 30 万字符，全量建索引只要几十毫秒，但每次查询都重建就浪费了。
# 用集合大小当签名：重新导入之后大小大概率变化，索引随之重建。
_index_lock = threading.Lock()
_index_cache: dict = {"signature": None, "index": None, "documents": [], "keys": []}


def _load_corpus() -> tuple[list[Document], list[str], int]:
    got = document_vector_store.get(include=["documents", "metadatas"])
    contents = got.get("documents") or []
    metadatas = got.get("metadatas") or []
    documents = [
        Document(page_content=content or "", metadata=metadata or {})
        for content, metadata in zip(contents, metadatas)
    ]
    keys = [doc_key(doc.metadata, doc.page_content) for doc in documents]
    count = document_vector_store.count()
    return documents, keys, count


def get_index() -> tuple[Bm25Index, list[Document], list[str]]:
    with _index_lock:
        if _index_cache["index"] is not None:
            try:
                current = document_vector_store.count()
            except Exception:
                current = None
            if current == _index_cache["signature"]:
                return _index_cache["index"], _index_cache["documents"], _index_cache["keys"]

        documents, keys, count = _load_corpus()
        index = Bm25Index(documents)
        _index_cache.update(
            {"signature": count, "index": index, "documents": documents, "keys": keys}
        )
        return index, documents, keys


def invalidate_index() -> None:
    """导入/删除文档后调用，让下次查询重建索引。"""
    with _index_lock:
        _index_cache.update({"signature": None, "index": None, "documents": [], "keys": []})


# ---- 融合 -------------------------------------------------------------------
def rrf_fuse(ranked_lists: list[list[str]], k: int = RRF_K) -> dict[str, float]:
    """Reciprocal Rank Fusion。入参是若干「按排名从高到低」的 key 列表。"""
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, key in enumerate(ranked, start=1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    return scores


def hybrid_search(query: str, allowed_sources: list[str] | None = None,
                  top_n: int = FINAL_TOP_N):
    """向量 + BM25 混合检索。

    返回 (hits, vector_top1_score)：
      hits              —— [(Document, 来源标记, 向量分数或 None)]，按 RRF 排序
      vector_top1_score —— 向量那一路的最高分，供调用方做「阀门」判断
    """
    where = None
    if allowed_sources:
        where = ({"source": allowed_sources[0]} if len(allowed_sources) == 1
                 else {"source": {"$in": allowed_sources}})

    # 1) 向量召回（带分数，同时拿到阀门要用的 top-1 分数）
    vector_results = document_vector_store.similarity_search_with_relevance_scores(
        query, k=VECTOR_TOP_K, filter=where
    )
    rows = []
    for doc, score in vector_results:
        key = doc_key(doc.metadata, doc.page_content)
        rows.append((key, float(score), doc))

    # 显式排序：并列项由 key 决定先后，不能依赖 Chroma 的返回顺序（见模块文档）。
    rows.sort(key=lambda row: (-row[1], row[0]))

    vector_score_by_key: dict[str, float] = {key: score for key, score, _doc in rows}
    vector_rank: list[str] = [key for key, _score, _doc in rows]
    doc_by_key: dict[str, Document] = {key: doc for key, _score, doc in rows}

    vector_top1 = rows[0][1] if rows else 0.0

    # 2) BM25 召回
    index, documents, keys = get_index()
    scores = index.scores(query)
    candidates = range(len(documents))
    if allowed_sources:
        allowed = set(allowed_sources)
        candidates = [i for i in range(len(documents)) if documents[i].metadata.get("source") in allowed]

    ranked = sorted(
        (i for i in candidates if scores[i] > 0), key=lambda i: (-scores[i], keys[i])
    )[:BM25_TOP_K]
    bm25_rank = [keys[i] for i in ranked]
    for i in ranked:
        doc_by_key.setdefault(keys[i], documents[i])

    # 3) RRF 融合
    fused = rrf_fuse([vector_rank, bm25_rank])

    ordered = sorted(fused.items(), key=lambda item: (-item[1], item[0]))[:top_n]
    hits = []
    for key, _rrf in ordered:
        doc = doc_by_key.get(key)
        if doc is None:
            continue
        if key in vector_score_by_key:
            hits.append((doc, "hybrid" if key in bm25_rank else "vector",
                         vector_score_by_key[key]))
        else:
            hits.append((doc, "keyword", None))

    return hits, vector_top1
