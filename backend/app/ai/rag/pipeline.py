"""检索管线的可复用入口：混合召回 → 阀门 → 重排。

## 为什么单独抽一个模块

`search_documents` 工具和 #3 的 research workflow 都要走同一条检索管线，但需要的
**返回形态不同**：工具要的是拼好的文本（给模型看），工作流要的是**结构化的命中结果**
（要拿分数判证据充分性、要数命中了几篇论文）。

分成两处实现的话，「阈值、重排条数、什么算拒绝」就会有两份会各自漂移的副本 ——
今天已经因为「工具返回格式改了、评估脚本的解析正则没跟着改」导致
`retrieval_recall` 静默掉到 0.25 一次。同一类错误不值得再犯第二遍。

所以：**管线在这里，格式化在调用方。**

## 阀门为什么还在

向量 top-1 分数决定「这个话题在不在库里」，RRF 只决定「哪几段最相关」。
这个分工见 ai/rag/hybrid.py 的模块说明。重排不参与这个判断 ——
它的分数是 logit，没有校准到能当阈值用（实测同一问题下相关段落可以是 -1.02）。
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document

from ai.rag.hybrid import hybrid_search
from ai.rag.rerank import rerank_hits, score_pairs


# 检索相关性阈值。校准过程见 design-decisions.md 决策一。
#
#   不该返回的最高 0.2685（"今天的天气怎么样"）
#   应返回的最低   0.3752（"什么是 hard positive problem"）
#   取 0.35 落在空隙中，15 个真实问法的查询全部判断正确（10 通过 / 5 拦截）
#
# ⚠️ 当时是 15 个查询的小样本。之后补了 20 题评估集，但阈值没有重新校准 ——
#    换 embedding 模型或换语料都必须重测。
RELEVANCE_THRESHOLD = 0.35

# 二段式检索的两个数字。
#
# 第一段：混合召回、过阀门之后，留多少条进重排。交叉编码器每个 (查询, 段落) 对都是
# 一次完整前向，实测 10 条约 0.5s，所以候选池不放大。
RERANK_CANDIDATES = 10

# 第二段：重排之后真正交给模型的条数。
#
# 这里曾经设成 5，用评估集测出来**不可行**：省了 39% 上下文（20972→12857），
# 但 A 类 verdict 从 1.0 掉到 0.917。查原始回答，原因不是重排排错，是**截断** ——
# 一个答案经常横跨相邻两块，裁掉续块就只剩一半。
#
# 那个「跨块」的根因后来查清了：是 PDF 抽取把图表文字混进了正文流
# （见 design-decisions.md 决策七，已用 pdfminer 修掉）。但**条数仍然保持 10** ——
# 修完抽取之后依然存在「枚举跨度 931 字符 > chunk_size 800」的情况，
# 而 12 道 A 类题换 39% 上下文这个账不划算：对科研问答来说，答错比多花 token 贵。
FINAL_TOP_N = 10

# 重排之前的「候选池条数」和「最终条数」是两件事，调用方可以只覆盖后者。
DEFAULT_CANDIDATES = RERANK_CANDIDATES
DEFAULT_TOP_N = FINAL_TOP_N


@dataclass
class Outcome:
    """一次检索的结果。**结构化**，供工作流判证据充分性用。"""

    query: str
    hits: list[tuple[Document, str, float | None]]
    vector_top1: float
    rejected: bool
    allowed_sources: list[str] | None = None

    @property
    def papers(self) -> list[str]:
        """命中的论文标题（去重，保持首次出现顺序）。"""
        seen: list[str] = []
        for doc, _origin, _score in self.hits:
            title = doc.metadata.get("paper_title")
            if title and title not in seen:
                seen.append(title)
        return seen

    @property
    def pages(self) -> list[tuple[str, str]]:
        """(论文标题, 页码) 对，去重。"""
        seen: list[tuple[str, str]] = []
        for doc, _origin, _score in self.hits:
            meta = doc.metadata or {}
            key = (str(meta.get("paper_title") or "?"), str(meta.get("page_label") or "?"))
            if key not in seen:
                seen.append(key)
        return seen

    def top_score(self) -> float | None:
        scores = [s for _d, _o, s in self.hits if s is not None]
        return max(scores) if scores else None


def retrieve(
    query: str,
    allowed_sources: list[str] | None = None,
    candidates: int = DEFAULT_CANDIDATES,
    top_n: int = DEFAULT_TOP_N,
) -> Outcome:
    """混合召回 → 阀门 → 重排。同步函数（向量编码和重排都是同步的）。

    阀门刻意放在重排**之前**：既然要拒答了，就没必要再花 0.5s 跑一次重排。
    """
    hits, vector_top1 = hybrid_search(query, allowed_sources=allowed_sources, top_n=candidates)

    if vector_top1 < RELEVANCE_THRESHOLD or not hits:
        return Outcome(query=query, hits=[], vector_top1=vector_top1,
                       rejected=True, allowed_sources=allowed_sources)

    ordered = rerank_hits(query, hits, top_n=top_n)
    return Outcome(query=query, hits=ordered, vector_top1=vector_top1,
                   rejected=False, allowed_sources=allowed_sources)


def evidence_score(question: str, hits: list[tuple[Document, str, float | None]]) -> float | None:
    """用交叉编码器给「**原始问题** × 检索到的段落」打分，取最高分。

    判定「检索到的材料是不是真的相关」要用这个，而不是向量余弦 —— 理由是实测的：

    | | 应回答（A 类）最低 | 应拒绝（B/C 类）最高 | 错几题 |
    |---|---|---|---|
    | 向量 top-1 | 0.353 | 0.470 | 6+ |
    | 交叉编码器（对原问题打分） | **-0.314** | **0.794** | **2** |

    B 类是「语料里有这个主题、但没有这个事实」（例如问论文是否用 Mamba）。
    向量余弦只回答「像不像」，所以任何关于 ReID 的句子都能拿高分；
    交叉编码器把问题和段落**放在一起读**，所以分得开得多。

    ⚠️ 但它**也不是干净的分界线** —— 上表那 2 题（A06 的 -0.314 与 B05 的 0.794）
    就是反例。所以调用方必须按「保守使用」来设计阈值，见 EVIDENCE_RETRY_THRESHOLD。

    ⚠️ 另一个坑：**必须用原始问题打分，不能用改写后的检索查询。**
    改写是为召回优化的（往查询里塞满领域词汇），拿它当相关性判据必然虚高 ——
    实测「这四篇论文是否采用 Mamba」被改写成
    "Mamba state space model SSM architecture for ship re-identification person
    re-identification" 之后，在 re-ID 语料上必然高分，哪怕语料里根本没有 Mamba。
    """
    if not hits:
        return None
    scores = score_pairs(question, [doc.page_content for doc, _o, _s in hits])
    return float(max(scores))


# 证据判定的阈值。
#
# 取 -3.0，是**刻意偏保守**的：它只用来触发「再查一轮」，不是最终结论。
# 两边的代价不对称 ——
#
#   · 误触发重试：多花一轮调用（约 +1s），而且下一轮大概率还是查不到
#   · 漏触发重试：不要紧，synthesize 本来就会读材料做最终判断
#
# 所以宁可少重试、不可误重试。在 20 题评估集上的效果：
#
#   12 道 A 类题最低是 A06 的 -0.314，**全部不受影响**，不会被误判成"要重试"
#   明显不相关的会被触发：B01 -8.137 / B04 -3.705 / B06 -3.191 / C01 -7.334 / C02 -6.155
#   落在中间的 B02 -1.258 / B03 -0.340 / B05 0.794 不触发 —— 这三题交给 synthesize，
#   它本来就能答对（实测 B 类 verdict 一直是 NOT_FOUND_IN_CORPUS）
EVIDENCE_RETRY_THRESHOLD = -3.0


# 「没找到」时返回给模型的文案。
#
# 关键：这段字**不是给用户看的，是给模型看的**。它必须把两件事说清楚 ——
#   ① 现在这个结论的适用范围（"no passage in the knowledge base"）
#   ② 模型被允许说什么、禁止说什么（可以报告未收录，不可以推断不存在）
# 写短了就退回 baseline 的「无根据的否定」缺陷（见 reference/baseline-defects.md）。
NO_HITS_MESSAGE = (
    "No relevant documents found: no passage in the knowledge base is sufficiently "
    "related to this query. When you answer, you may only state that the documents do "
    "not contain relevant information. You must not conclude that the fact does not "
    "exist, and you must not add anything that is not in the documents."
)


def format_hits(hits: list[tuple[Document, str, float | None]]) -> str:
    """把命中结果拼成给模型看的文本块。

    格式：`[source N | 论文标题 | p.X | 匹配方式]` + 正文。
    末段说明「怎么匹配上的」，格式会演进 —— 评估脚本解析它时不要把格式写死
    （run_eval.py 顶部有这条教训）。
    """
    import os

    blocks = []
    for index, (doc, origin, score) in enumerate(hits, start=1):
        meta = doc.metadata or {}
        title = meta.get("paper_title") or os.path.basename(str(meta.get("source", "unknown")))
        page = meta.get("page_label") or meta.get("page", "?")
        if score is None:
            how = "matched on exact terms"
        elif origin == "hybrid":
            how = "relevance %.2f + exact terms" % score
        else:
            how = "relevance %.2f" % score
        blocks.append(
            "[source %d | %s | p.%s | %s]\n%s" % (index, title, page, how, doc.page_content)
        )
    return "\n\n".join(blocks)
