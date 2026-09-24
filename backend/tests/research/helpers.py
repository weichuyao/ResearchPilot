from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

# Register every application table before create_all, matching app startup.
import db.models  # noqa: F401,E402


@asynccontextmanager
async def isolated_session(path: Path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}")

    @event.listens_for(engine.sync_engine, "connect")
    def _foreign_keys(dbapi_connection, _connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    async with engine.begin() as connection:
        await connection.run_sync(SQLModel.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    try:
        async with maker() as session:
            yield session
    finally:
        await engine.dispose()


def literature_evidence(question_id: str, **overrides):
    from db.models.research import Evidence
    from research.enums import EvidenceType, SearchIntent

    values = {
        "research_question_id": question_id,
        "evidence_type": EvidenceType.LITERATURE.value,
        "statement": "The paper reports an improvement under the stated protocol.",
        "excerpt": "The treatment improves mAP by 0.6 points over the control.",
        "source_id": "paper-1.pdf",
        "source_type": "paper",
        "source_title": "A fixed fixture paper",
        "page": "7",
        "locator_prefix": "p",
        "locator_kind": "page",
        "chunk_id": "chunk-fixture-001",
        "search_intent": SearchIntent.PRIMARY.value,
        "provenance": {"query": "treatment mAP improvement", "rank": 1, "origin": "hybrid"},
    }
    values.update(overrides)
    return Evidence(**values)

