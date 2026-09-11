"""ResearchPilot 的工具集。

这里只放模型可以申请调用的能力。原先的 get_user_info / get_user_department
（查 SQLite 里的员工和部门）属于 OA 企业助手的遗留，已经在切换到科研文献助手时移除。
对应的 HTTP 接口仍然保留在 api/employee_routers.py，但它们直接走 repository，
不再通过工具层绕一圈。
"""

import os

from langchain_core.tools import tool

from ai.rag.chromaClient import document_vector_store


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


@tool
async def search_documents(query: str) -> str:
    """Search the research paper knowledge base and return the most relevant passages.

    Each passage is prefixed with the paper title, page number and relevance score,
    so you can tell the user where the information came from.
    """
    results = document_vector_store.similarity_search_with_relevance_scores(query, k=RETRIEVE_K)

    hits = [(doc, score) for doc, score in results if score >= RELEVANCE_THRESHOLD]

    if not hits:
        return (
            "No relevant documents found: no passage in the knowledge base is sufficiently "
            "related to this query. When you answer, you may only state that the documents do "
            "not contain relevant information. You must not conclude that the fact does not "
            "exist, and you must not add anything that is not in the documents."
        )

    blocks = []
    for index, (doc, score) in enumerate(hits, start=1):
        title = doc.metadata.get("paper_title") or os.path.basename(
            str(doc.metadata.get("source", "unknown"))
        )
        page = doc.metadata.get("page_label") or doc.metadata.get("page", "?")
        blocks.append(
            "[source %d | %s | p.%s | relevance %.2f]\n%s"
            % (index, title, page, score, doc.page_content)
        )
    return "\n\n".join(blocks)
