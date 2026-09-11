from langchain_core.tools import BaseTool, tool
import os

from db.models.employee import Employee
from db.repository.employee_repo import EmployeeRepository
from db.models.department import Department
from db.repository.department_repo import DepartmentRepository
from db.database import async_session_maker
from dataclasses import dataclass, asdict
from fastapi import Depends
from ai.rag.chromaClient import hand_book_vector_store


@tool
async def get_user_info(user_name: str) -> dict:
    """Obtain user information"""
    async with async_session_maker() as session:
        employee =  await EmployeeRepository.get_employee_by_name(session, name=user_name)
        
        if not employee:
            return {"error": "user not found"}

        return asdict(employee)

@tool
async def get_user_department(user_name: str) -> dict:
    """Obtain the information of the user department"""
   
    async with async_session_maker() as session: 
        employee = await EmployeeRepository.get_employee_by_name(session, name=user_name)
        
        if not employee:
            return {"error": "user not found"}

        department = await DepartmentRepository.get_department(session, department_id=employee.department_id)
        
        if not department:
            return {"error": "department not found"}
        
        return asdict(department)

# 检索相关性阈值。
# 实测分布：应当返回的最低分 0.566，不应当返回的最高分 0.366。
# 0.5 落在两者之间，四个实测 case（vacation benefits / annual leave policy /
# 宠物保险 / 无关乱码）全部判断正确。校准方法见 reference/transformation-03-research-workflow-design.md。
RELEVANCE_THRESHOLD = 0.5

# 先粗召回多少条，再用阈值筛。召回放宽、筛选收紧，避免阈值把真答案一刀切掉。
RETRIEVE_K = 10


@tool
async def search_documents(query: str) -> str:
    """Search the research document knowledge base and return the most relevant passages.

    Each passage is prefixed with its source file, page number and relevance score,
    so you can tell the user where the information came from.
    """
    results = hand_book_vector_store.similarity_search_with_relevance_scores(query, k=RETRIEVE_K)

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
        source = os.path.basename(str(doc.metadata.get("source", "unknown")))
        page = doc.metadata.get("page_label") or doc.metadata.get("page", "?")
        blocks.append(
            "[source %d | %s | page %s | relevance %.2f]\n%s"
            % (index, source, page, score, doc.page_content)
        )
    return "\n\n".join(blocks)