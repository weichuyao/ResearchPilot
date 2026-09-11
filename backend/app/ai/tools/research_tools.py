"""ResearchPilot 的工具集。

只放模型可以申请调用的能力。这里有两类工具，分工是刻意的：

  · list_papers        —— 结构化查询：精确、可枚举、能过滤（作者/年份）
  · search_documents   —— 二段式检索：
                           召回  向量（按意思找）+ BM25（按精确术语找），RRF 融合
                           重排  交叉编码器精排，再截断
                         实现与取舍见 ai/rag/hybrid.py 和 ai/rag/rerank.py

语义检索答不了「库里有哪几篇」「2023 年之后的」「作者是谁」这类问题，
那些需要结构化查询；反过来结构化查询也找不到「和这个问题意思相近的段落」。
两者互补，缺一不可。

search_documents 支持把检索限制在**单篇论文**内（paper 参数）：
用户点名某篇论文时，这个过滤能把其他论文的匹配整个排除掉。

原先的 get_user_info / get_user_department（查 SQLite 里的员工和部门）属于 OA 企业
助手的遗留，已随业务切换移除。
"""

import os
from typing import Optional

from langchain_core.tools import tool

from ai.rag.hybrid import hybrid_search
from ai.rag.rerank import rerank_hits
from db.database import async_session_maker
from db.models.paper import Paper
from db.repository.paper_repo import PaperRepository


# 检索相关性阈值。
#
# 校准过程（两轮，第二轮推翻了第一轮）：
#   第一轮：用 4 个手挑的查询（措辞贴近论文原文），得到 0.366 / 0.566 的空隙，
#           据此取了 0.5。结果是错的 —— 那批查询有个共同偏差：都塞进了论文自己的
#           词汇，分数被抬高，不能代表真实提问。
#   第二轮：用 15 个真实问法的查询实测，得到真正的空隙：
#           不该返回的最高 0.2685（"今天的天气怎么样"）
#           应返回的最低   0.3752（"什么是 hard positive problem"）
#           取 0.35 落在空隙中，15 个查询全部判断正确（10 通过 / 5 拦截）。
#
# ⚠️ 仍然是 15 个查询的小样本。正式做法是用评估集校准（改造 #8），
#    并且每次换 embedding 模型或换语料都要重新测。
RELEVANCE_THRESHOLD = 0.35

# 二段式检索的两个数字。
#
# 第一段：混合召回、过阀门之后，留多少条进重排。候选池越大重排越慢 ——
# 交叉编码器每个 (查询, 段落) 对都是一次完整前向，实测 10 条约 0.5s。
RERANK_CANDIDATES = 10

# 第二段：重排之后真正交给模型的条数。
#
# 这里曾经设成 5，用评估集测出来是**不可行**的，记录如下：
#
#   hybrid@10      上下文 20972 字符  A 类 verdict 1.0    概念覆盖 1.0    延迟 4.1s
#   hybrid+rerank@5 上下文 12857 字符  A 类 verdict 0.917  概念覆盖 0.944  延迟 5.5s
#
# 省了 39% 上下文，但掉了 1 题。查那题的原始回答，原因不是重排排错，是**截断** ——
# 但「截断切在哪儿」这件事，第一次的归因是错的，修正如下。
#
# 【错误归因】一开始写成「分块边界把枚举切断了，补上邻块就能修」。
# 【实际原因】是 **PDF 抽取出来的文本本身就被图表文字污染**。
#
#   A08 问「SAP 模块推断的三类语义属性」。块 23 的结尾确实是
#   "...inferred from the global feature vector: (i) ship type, (ii) imaging
#   perspective, and"，紧接着就是 Fig. 3 的一堆图形标签：
#
#       Instance Bank / SAP / Semantic Features / Conv / MP / AP / HCP ...
#
#   然后是块 24（几乎整块是 "CLS Prob / Bow 0.0000 / Port 0.7414" 这类
#   表格数字），块 25 才出现 "(iii) loading and equipment configuration"。
#
#   也就是说，正文那句话在 **PDF 自己的文本流里**就被图 3 的标签、图 3 的图注、
#   一张概率表和图 4 的图注隔开了约 1600 字符。pypdf 的流式抽取忠实地还原了
#   这个顺序 —— 即使分块完美（按句子切），这句话照样是断的。
#
#   更麻烦的后果是**检索**：块 25 真正含答案，但它 767 字符里有 500 多字符是数字，
#   embedding 被稀释，所以它排不进候选 —— 评估里的 retrieval_recall 却是 1.0，
#   因为那一页的另一块（块 23，含 "ship type"）被召回了。**页级召回是个会骗人的代理指标。**
#
# 所以结论是：**重排的分数没有校准到能当裁剪依据**，而这份语料里一个答案经常
# 被图表文字打断、横跨好几块。用 12 道 A 类题去换 39% 的上下文，一题 = 8.3%，
# 对科研问答来说答错比多花 token 贵得多。
#
# 于是收回到 10：重排只负责排序，不负责裁剪。
#
# 真正对症的修法是**改良抽取**，把图表文字从正文流里去掉。已经验证了一个可行的
# 信号：pypdf 的 visitor_text 回调能拿到每个文本片段的**字号** —— A²RNet 第 3 页上
# 正文是 9.96pt，而 Fig. 3 的标签全是 24pt（图内的文字被放大了 2.4 倍）。
# 字号能干净地把两者分开，而且不需要任何新依赖。还没做，见 design-decisions.md 决策六。
FINAL_TOP_N = 10

_NO_HITS_MESSAGE = (
    "No relevant documents found: no passage in the knowledge base is sufficiently "
    "related to this query. When you answer, you may only state that the documents do "
    "not contain relevant information. You must not conclude that the fact does not "
    "exist, and you must not add anything that is not in the documents."
)


def _format_paper(paper: Paper) -> str:
    parts = [paper.title]
    if paper.year:
        parts.append("(%s)" % paper.year)
    if paper.venue:
        parts.append("| %s" % paper.venue)
    if paper.authors:
        parts.append("| %s" % paper.authors)
    parts.append("| %d chunks indexed" % paper.chunk_count)
    return "  ".join(parts)


@tool
async def list_papers(
    collection: Optional[str] = None,
    year_from: Optional[int] = None,
    author: Optional[str] = None,
) -> str:
    """List the papers held in the knowledge base, with their metadata.

    Use this for questions about WHICH papers exist, who wrote them, what year they are
    from, or how large the corpus is. This is an exact lookup, not a similarity search.

    Args:
        collection: optional knowledge base name to restrict to.
        year_from: optional, only papers published in this year or later.
        author: optional, matches part of an author name.
    """
    async with async_session_maker() as session:
        papers = await PaperRepository.list_papers(
            session=session, year_from=year_from, author=author
        )

    if collection:
        async with async_session_maker() as session:
            from db.repository.collection_repo import CollectionRepository

            found = await CollectionRepository.get_by_name(session=session, name=collection)
        if not found:
            return "No knowledge base named %r. Known papers:\n%s" % (
                collection,
                "\n".join(_format_paper(p) for p in papers) or "  (none)",
            )
        papers = [p for p in papers if p.collection_id == found.id]

    if not papers:
        return (
            "No papers in the knowledge base match that filter. This means the corpus does "
            "not contain such a paper; it does NOT mean no such paper exists anywhere."
        )

    return "Found %d paper(s) in the knowledge base:\n%s" % (
        len(papers),
        "\n".join(_format_paper(paper) for paper in papers),
    )


@tool
async def search_documents(query: str, paper: Optional[str] = None) -> str:
    """Search the research paper knowledge base and return the most relevant passages.

    The search is two-stage. It first recalls candidates in two ways at once —
    semantic similarity and BM25 keyword matching — and merges them, so it finds
    passages both by meaning and by exact terms (model names, dataset names,
    abbreviations). It then re-scores those candidates with a cross-encoder that
    reads the query and the passage together, and returns only the best few.

    Each passage is prefixed with the paper title, page number, and how it was matched,
    so you can tell the user where the information came from.

    Args:
        query: what to look for, in natural language.
        paper: optional. Part of a paper's title. When set, the search is restricted to
            that single paper, so passages from the other papers cannot be returned.
            Use it whenever the user names a specific paper.
    """
    allowed_sources = None
    if paper:
        async with async_session_maker() as session:
            matches = await PaperRepository.find_by_title(session=session, keyword=paper, limit=5)
        if not matches:
            return (
                "No paper in the knowledge base has a title matching %r, so the search was "
                "not restricted. Known titles:\n%s"
                % (paper, await _all_titles())
            )
        allowed_sources = [m.source_file for m in matches]

    hits, vector_top1 = hybrid_search(query, allowed_sources=allowed_sources,
                                      top_n=RERANK_CANDIDATES)

    # 阀门：向量那一路的最高分不过阈值 -> 判定这个话题不在知识库里。
    # 注意 RRF 的分数是排名算出来的，不能拿来当相关性判据，所以「该不该回答」
    # 仍然由向量的绝对分数决定，「哪几段最相关」才交给 RRF。
    #
    # 阀门刻意放在重排**之前**：既然都要拒答了，就没必要再花 0.5s 跑一次重排。
    if vector_top1 < RELEVANCE_THRESHOLD or not hits:
        return _NO_HITS_MESSAGE

    hits = rerank_hits(query, hits, top_n=FINAL_TOP_N)

    blocks = []
    for index, (doc, origin, score) in enumerate(hits, start=1):
        title = doc.metadata.get("paper_title") or os.path.basename(
            str(doc.metadata.get("source", "unknown"))
        )
        page = doc.metadata.get("page_label") or doc.metadata.get("page", "?")
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


async def _all_titles() -> str:
    async with async_session_maker() as session:
        papers = await PaperRepository.list_papers(session=session)
    return "\n".join("  - %s" % p.title for p in papers) or "  (knowledge base is empty)"
