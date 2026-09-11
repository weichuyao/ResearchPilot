"""ResearchPilot 的工具集。

只放模型可以申请调用的能力。这里有两类工具，分工是刻意的：

  · list_papers        —— 结构化查询：精确、可枚举、能过滤（作者/年份）
  · search_documents   —— 混合检索：向量（按意思找）+ BM25（按精确术语找），
                          用 RRF 融合。实现与取舍见 ai/rag/hybrid.py

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

# 先粗召回多少条，再用阈值筛。召回放宽、筛选收紧，避免阈值把真答案一刀切掉。
RETRIEVE_K = 10

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

    The search is hybrid: it combines semantic similarity with BM25 keyword matching,
    so it finds passages both by meaning and by exact terms (model names, dataset names,
    abbreviations).

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

    hits, vector_top1 = hybrid_search(query, allowed_sources=allowed_sources)

    # 阀门：向量那一路的最高分不过阈值 -> 判定这个话题不在知识库里。
    # 注意 RRF 的分数是排名算出来的，不能拿来当相关性判据，所以「该不该回答」
    # 仍然由向量的绝对分数决定，「哪几段最相关」才交给 RRF。
    if vector_top1 < RELEVANCE_THRESHOLD or not hits:
        return _NO_HITS_MESSAGE

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
