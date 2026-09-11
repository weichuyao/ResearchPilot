"""论文记录的数据访问层。

注意这里每个方法都要求显式传入 session —— department_repo 那两个方法少传 session、
导致实参位置错位抛 TypeError 的坑，这里不再犯（且调用方一律用关键字参数传 session）。
"""

from datetime import datetime
from typing import List, Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import func, select

from db.models.paper import Paper


class PaperRepository:

    @classmethod
    async def get_by_source_file(
        cls, session: AsyncSession, source_file: str
    ) -> Optional[Paper]:
        result = await session.execute(
            select(Paper).where(Paper.source_file == source_file)
        )
        return result.scalars().first()

    @classmethod
    async def get_by_title(cls, session: AsyncSession, title: str) -> Optional[Paper]:
        result = await session.execute(select(Paper).where(Paper.title == title))
        return result.scalars().first()

    @classmethod
    async def find_by_title(
        cls, session: AsyncSession, keyword: str, limit: int = 5
    ) -> List[Paper]:
        """模糊匹配标题。

        用于把用户/模型说的论文名（可能只写简称，例如「ViV-ReID」）映射到具体记录。
        """
        statement = (
            select(Paper)
            .where(Paper.title.ilike("%" + keyword + "%"))
            .order_by(Paper.id)
            .limit(limit)
        )
        result = await session.execute(statement)
        return list(result.scalars().all())

    @classmethod
    async def list_papers(
        cls,
        session: AsyncSession,
        collection_id: Optional[int] = None,
        year_from: Optional[int] = None,
        author: Optional[str] = None,
        limit: int = 50,
    ) -> List[Paper]:
        statement = select(Paper)
        if collection_id is not None:
            statement = statement.where(Paper.collection_id == collection_id)
        if year_from is not None:
            statement = statement.where(Paper.year >= year_from)
        if author:
            statement = statement.where(Paper.authors.ilike("%" + author + "%"))
        statement = statement.order_by(Paper.year.desc(), Paper.id).limit(limit)
        result = await session.execute(statement)
        return list(result.scalars().all())

    @classmethod
    async def count(cls, session: AsyncSession) -> int:
        result = await session.execute(select(func.count()).select_from(Paper))
        return int(result.scalar() or 0)

    @classmethod
    async def upsert(cls, session: AsyncSession, paper: Paper) -> Paper:
        """按 source_file 更新或插入 —— 重复导入同一篇不会产生第二条记录。"""
        existing = await cls.get_by_source_file(session, source_file=paper.source_file)
        if existing:
            payload = paper.model_dump(exclude={"id", "create_time", "edit_time"})
            for key, value in payload.items():
                setattr(existing, key, value)
            existing.edit_time = datetime.now()
            await session.commit()
            await session.refresh(existing)
            return existing

        session.add(paper)
        await session.commit()
        await session.refresh(paper)
        return paper
