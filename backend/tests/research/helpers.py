from __future__ import annotations

import asyncio
import contextlib
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


@contextlib.contextmanager
def api_client(path: Path):
    """FastAPI TestClient over a throwaway SQLite file, wired like app startup.

    Routes own the approval → entity-state side effects, so the rejection paths
    can only be proven through HTTP; calling the repository directly would skip
    exactly the code we care about.

    Yields ``(client, session_maker)``: some fixtures (evidence rows) have no
    public REST entry on purpose, so tests seed them through the repository
    against the same file.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from api.research_routes import research_router
    from db.database import get_async_session

    engine = create_async_engine(f"sqlite+aiosqlite:///{path}")

    @event.listens_for(engine.sync_engine, "connect")
    def _foreign_keys(dbapi_connection, _connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    async def create_schema():
        async with engine.begin() as connection:
            await connection.run_sync(SQLModel.metadata.create_all)

    asyncio.run(create_schema())
    maker = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    async def session_override():
        async with maker() as session:
            yield session

    app = FastAPI()
    app.include_router(research_router)
    app.dependency_overrides[get_async_session] = session_override
    try:
        with TestClient(app) as client:
            yield client, maker
    finally:
        asyncio.run(engine.dispose())
