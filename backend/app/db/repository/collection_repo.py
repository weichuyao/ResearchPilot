"""知识库记录的数据访问层。"""

from typing import List, Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from db.models.collection import Collection


class CollectionRepository:

    @classmethod
    async def get_by_name(cls, session: AsyncSession, name: str) -> Optional[Collection]:
        result = await session.execute(select(Collection).where(Collection.name == name))
        return result.scalars().first()

    @classmethod
    async def get(cls, session: AsyncSession, collection_id: int) -> Optional[Collection]:
        result = await session.execute(
            select(Collection).where(Collection.id == collection_id)
        )
        return result.scalars().first()

    @classmethod
    async def get_all(cls, session: AsyncSession) -> List[Collection]:
        result = await session.execute(select(Collection).order_by(Collection.id))
        return list(result.scalars().all())

    @classmethod
    async def get_or_create(
        cls, session: AsyncSession, name: str, description: str = ""
    ) -> Collection:
        existing = await cls.get_by_name(session, name=name)
        if existing:
            return existing
        collection = Collection(name=name, description=description)
        session.add(collection)
        await session.commit()
        await session.refresh(collection)
        return collection
