"""会话索引的仓储层。

与 paper_repo 同一条规矩：每个方法显式收 session。这里全部是主循环上的
异步操作（索引表很小，不需要丢线程池）。
"""

from datetime import datetime

from sqlalchemy import delete, select
from sqlmodel import desc

from db.models.conversation import Conversation


class ConversationRepository:
    @classmethod
    async def list_all(cls, session, limit: int = 200) -> list[Conversation]:
        """最近活跃在前。limit 防御：会话没有分页需求，但也不该无界。"""
        result = await session.execute(
            select(Conversation).order_by(desc(Conversation.last_message_at)).limit(limit)
        )
        return list(result.scalars().all())

    @classmethod
    async def get_by_thread(cls, session, thread_id: str) -> Conversation | None:
        result = await session.execute(
            select(Conversation).where(Conversation.thread_id == thread_id)
        )
        return result.scalars().first()

    @classmethod
    async def upsert(
        cls, session, thread_id: str, title: str, agent_id: str
    ) -> Conversation:
        """每轮对话成功后调用。

        标题只在**首次**写入（就是"首条用户消息"的字面含义），
        之后的轮次只刷新 last_message_at —— 用户改口第二条消息不该把会话改名。
        """
        row = await cls.get_by_thread(session, thread_id)
        now = datetime.now().isoformat(timespec="seconds")
        if row is None:
            row = Conversation(
                thread_id=thread_id,
                title=title[:50] or "新会话",
                agent_id=agent_id,
                last_message_at=now,
            )
            session.add(row)
        else:
            row.last_message_at = now
        await session.commit()
        await session.refresh(row)
        return row

    @classmethod
    async def update_title(cls, session, thread_id: str, title: str) -> Conversation | None:
        row = await cls.get_by_thread(session, thread_id)
        if row is None:
            return None
        row.title = title
        await session.commit()
        await session.refresh(row)
        return row

    @classmethod
    async def delete(cls, session, thread_id: str) -> bool:
        """返回是否真的删了行。checkpoint 那一半由路由层调 adelete_thread。"""
        result = await session.execute(
            delete(Conversation).where(Conversation.thread_id == thread_id)
        )
        await session.commit()
        return result.rowcount > 0
