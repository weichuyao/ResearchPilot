"""会话 API（改造 #5 之二）。

路由层只做编排：列表/删除走 conversation_repo（索引表），
消息历史走 checkpointer（`aget_state`）。分工见 db/models/conversation.py 的文档。

为什么消息历史不落 conversation 表：同一份数据存两处必然漂移 ——
项目里 agent 列表、上传格式两个「手工镜像」都是这么坏的。
"""

import logging

from fastapi import APIRouter, HTTPException
from langchain_core.runnables import RunnableConfig

from ai.agent.agents import DEFAULT_AGENT, get_agent
from db.database import async_session_maker
from db.models.conversation import Conversation
from db.repository.conversation_repo import ConversationRepository
from utils.chat_utils import langchain_to_chat_message
from api.schema.chatSchema import ChatMessage

logger = logging.getLogger(__name__)

conversation_router = APIRouter(prefix="/conversations", tags=["conversations"])


@conversation_router.get("")
async def list_conversations() -> list[dict]:
    async with async_session_maker() as session:
        rows = await ConversationRepository.list_all(session=session)
    return [
        {
            "thread_id": row.thread_id,
            "title": row.title,
            "agent_id": row.agent_id,
            "last_message_at": row.last_message_at,
        }
        for row in rows
    ]


@conversation_router.get("/{thread_id}/messages")
async def get_messages(thread_id: str) -> list[ChatMessage]:
    async with async_session_maker() as session:
        row = await ConversationRepository.get_by_thread(session, thread_id)
    if row is None:
        raise HTTPException(status_code=404, detail="会话不存在")

    # 必须用**当时的那个图**读：不同图的 state schema 不同，
    # thread 是绑定在图上的。会话行里存 agent_id 就是为了这里。
    agent = get_agent(row.agent_id or DEFAULT_AGENT)
    config = RunnableConfig(configurable={"thread_id": thread_id})
    state = await agent.aget_state(config=config)
    messages = (state.values or {}).get("messages", [])

    output = []
    for message in messages:
        try:
            output.append(langchain_to_chat_message(message))
        except Exception as exc:
            # 单条消息解析失败只降级那一条，不让整个历史 500
            logger.warning("会话 %s 的历史消息解析失败：%s", thread_id, exc)
    return output


@conversation_router.delete("/{thread_id}", status_code=204)
async def delete_conversation(thread_id: str):
    """删会话 = 删索引行 + 删 checkpoint 里的 thread。两半都要删。

    顺序和文档删除接口同一条思路：先删 checkpoint（挂的那一半），
    再删索引行 —— checkpoint 删除失败时索引还在，可以重试；
    反过来索引没了，这个 thread 就变成谁也找不到的孤儿。
    """
    async with async_session_maker() as session:
        row = await ConversationRepository.get_by_thread(session, thread_id)
        if row is None:
            raise HTTPException(status_code=404, detail="会话不存在")
        agent_id = row.agent_id or DEFAULT_AGENT

    # checkpointer 不走业务 session —— 它的表是 LangGraph 自己的。
    agent = get_agent(agent_id)
    try:
        await agent.checkpointer.adelete_thread(thread_id)
    except Exception as exc:
        logger.error("删除会话 %s 的 checkpoint 失败：%s", thread_id, exc)
        raise HTTPException(status_code=503, detail="会话记录删除失败，请重试")

    async with async_session_maker() as session:
        await ConversationRepository.delete(session, thread_id=thread_id)
