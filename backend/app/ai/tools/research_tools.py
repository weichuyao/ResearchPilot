"""ResearchPilot 的工具集。

只放模型可以申请调用的能力。这里有两类工具，分工是刻意的：

  · list_papers        —— 结构化查询：精确、可枚举、能过滤（作者/年份）
  · search_documents   —— 二段式检索：
                           召回  向量（按意思找）+ BM25（按精确术语找），RRF 融合
                           重排  交叉编码器精排，再截断
                         管线的实现与全部取舍见 ai/rag/pipeline.py

语义检索答不了「库里有哪几篇」「2023 年之后的」「作者是谁」这类问题，
那些需要结构化查询；反过来结构化查询也找不到「和这个问题意思相近的段落」。
两者互补，缺一不可。

search_documents 支持把检索限制在**单篇论文**内（paper 参数）：
用户点名某篇论文时，这个过滤能把其他论文的匹配整个排除掉。

这个文件**只管「怎么把结果说给模型听」**。「怎么检索」在 ai/rag/pipeline.py ——
那里是工具和 research workflow 共用的，避免阈值/重排/拒绝规则出现两份实现。

原先的 get_user_info / get_user_department（查 SQLite 里的员工和部门）属于 OA 企业
助手的遗留，已随业务切换移除。
"""

from typing import Optional

from langchain_core.tools import tool

from ai.rag.pipeline import NO_HITS_MESSAGE, format_hits, retrieve
from db.database import async_session_maker
from db.models.paper import Paper
from db.repository.paper_repo import PaperRepository


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


async def resolve_paper_sources(paper: str) -> list[str] | None:
    """把「论文标题的一部分」解析成 Chroma 的 source 列表。

    返回 None 表示知识库里没有匹配的论文（调用方自己决定怎么处理）。
    抽成公开函数是因为 research workflow 也要做同一件事。
    """
    async with async_session_maker() as session:
        matches = await PaperRepository.find_by_title(session=session, keyword=paper, limit=5)
    if not matches:
        return None
    return [m.source_file for m in matches]


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
        allowed_sources = await resolve_paper_sources(paper)
        if allowed_sources is None:
            return (
                "No paper in the knowledge base has a title matching %r, so the search was "
                "not restricted. Known titles:\n%s" % (paper, await all_titles())
            )

    outcome = retrieve(query, allowed_sources=allowed_sources)
    if outcome.rejected:
        return NO_HITS_MESSAGE
    return format_hits(outcome.hits)


async def all_titles() -> str:
    async with async_session_maker() as session:
        papers = await PaperRepository.list_papers(session=session)
    return "\n".join("  - %s" % p.title for p in papers) or "  (knowledge base is empty)"
