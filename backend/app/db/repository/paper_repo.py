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
    async def get_by_id(cls, session: AsyncSession, paper_id: int) -> Optional[Paper]:
        result = await session.execute(select(Paper).where(Paper.id == paper_id))
        return result.scalars().first()

    @classmethod
    async def get_by_source_file(
        cls, session: AsyncSession, source_file: str
    ) -> Optional[Paper]:
        result = await session.execute(
            select(Paper).where(Paper.source_file == source_file)
        )
        return result.scalars().first()

    @classmethod
    async def update_progress(
        cls,
        session: AsyncSession,
        paper_id: int,
        status: int,
        error: str = "",
        chunk_count: Optional[int] = None,
    ) -> Optional[Paper]:
        """更新导入进度。

        为什么单独一个方法而不是让调用方 setattr 完 commit：上传的后台任务会**并发**
        改同一批记录，散落在各处的「读-改-写」很容易漏掉 edit_time 或忘记 commit，
        而漏 commit 的表现是「接口返回成功但状态没变」——又是一次静默失败。
        """
        paper = await cls.get_by_id(session=session, paper_id=paper_id)
        if paper is None:
            return None
        paper.status = status
        if error:
            paper.error = error[:500]
        elif status != 2:
            # 成功或重新开始时清掉上一次的失败原因，避免旧错误一直挂在记录上
            paper.error = ""
        if chunk_count is not None:
            paper.chunk_count = chunk_count
        if status == 1:
            paper.indexed_at = datetime.now()
        paper.edit_time = datetime.now()
        await session.commit()
        await session.refresh(paper)
        return paper

    @classmethod
    async def count_by_status(cls, session: AsyncSession) -> dict:
        """按 status 统计篇数。给 /health 用。"""
        statement = select(Paper.status, func.count()).group_by(Paper.status)
        result = await session.execute(statement)
        return {int(row[0]): int(row[1]) for row in result.all()}

    @classmethod
    async def sum_chunks(cls, session: AsyncSession) -> int:
        result = await session.execute(select(func.coalesce(func.sum(Paper.chunk_count), 0)))
        return int(result.scalar() or 0)

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
    async def delete_not_in(
        cls, session: AsyncSession, collection_id: int, keep_source_files: list[str]
    ) -> list[str]:
        """删掉这个知识库里、但不在 keep 列表中的论文记录，返回被删的 source_file。

        ## 为什么需要它

        `ingest(reset=True)` 会先清空整个 Chroma collection 重建，但**不会**动 paper 表 ——
        而 paper 表里的行是靠 upsert 写入的，只增不减。于是从语料目录里移走一篇论文之后：

            向量库：4 篇 411 块        ← 干净
            paper 表：5 篇 434 块     ← 还留着那篇

        后果是 `list_papers` 会列出一篇**根本搜不到**的论文，而 `/health` 的
        `index.papers` 也会虚高。这种不一致不会报错，只会让人对数据失去信任。

        实测就是这么发现的：把评估集文档从语料目录移走、重跑导入之后，
        Chroma 是 411 而 paper 表还是 434。
        """
        statement = select(Paper).where(Paper.collection_id == collection_id)
        if keep_source_files:
            statement = statement.where(Paper.source_file.notin_(keep_source_files))
        result = await session.execute(statement)
        stale = list(result.scalars().all())
        for paper in stale:
            await session.delete(paper)
        if stale:
            await session.commit()
        return [p.source_file for p in stale]

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
