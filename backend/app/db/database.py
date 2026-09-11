"""
database connection
"""

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio.engine import create_async_engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlmodel import SQLModel
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker,Session
from typing import AsyncGenerator

from core.config import settings
from functools import wraps
from typing import Annotated
from fastapi import Depends
import logging
import os


# connect_args = {"check_same_thread": False}
async_engine = create_async_engine(settings.DATABASE_URL)
# engine = create_engine(settings.DATABASE_URL);

async_session_maker = async_sessionmaker(
    async_engine, expire_on_commit=False, class_=AsyncSession
)

# sessionmaker = sessionmaker(
#     engine, expire_on_commit=False, class_=Session
# )

# def get_seesion():
#     with sessionmaker() as session:
#         yield session

async def create_db_and_tables():
    # 显式导入模型包：SQLModel.metadata.create_all 只会为「已经被 import 过」的模型建表。
    # 漏导入一个模块，那张表就会静默地不存在，然后在第一次查询时才炸。
    import db.models  # noqa: F401

    async with async_engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
        await conn.run_sync(_add_missing_columns)


def _add_missing_columns(conn) -> None:
    """给**已存在**的表补上新加的列。

    ## 为什么需要这个

    `metadata.create_all` 只会建**不存在的表** —— 表已经在了，它就什么都不做。
    所以给 `Paper` 加一个字段（比如改造 #5 的 `error`）之后，
    旧数据库不会自动获得这一列，第一次查询就会 `no such column` 炸掉。
    而不巧的是本项目的向量库和关系库都是**可重建的运行时产物**，
    最省事的办法一直是「删掉 database.db 重跑 ingest」——但那要求使用者知道这件事，
    而且会丢掉所有人的论文记录。

    ## 为什么手写而不是上 Alembic

    SQLite 从 3.2 起就支持 `ALTER TABLE ... ADD COLUMN`，而我们要补的恰好都是
    **可空的、带默认值的新列** —— 这是唯一一种 SQLite 能原地完成的 schema 变更。
    引入 Alembic 会带来 migration 文件、版本表、和 `create_all` 的双轨制问题，
    收益在当前规模下是负的。

    ⚠️ 这是**临时手段**，只覆盖「加可空列」这一种情况。改造 #6 换 PostgreSQL 时
    应该同时引入真正的 migration 工具（Alembic），那时删列、改类型、建索引才有的谈。
    """
    from sqlalchemy import inspect, text

    inspector = inspect(conn)
    for table_name, table in SQLModel.metadata.tables.items():
        if not inspector.has_table(table_name):
            continue
        existing = {col["name"] for col in inspector.get_columns(table_name)}
        for column in table.columns:
            if column.name in existing:
                continue
            ddl = "ALTER TABLE %s ADD COLUMN %s %s" % (
                table_name, column.name, column.type.compile(conn.dialect)
            )
            conn.execute(text(ddl))
            logging.getLogger(__name__).info(
                "schema: 给 %s 补上新列 %s", table_name, column.name
            )

# create_db_and_tables();

async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_maker() as session:
        yield session
        
SessionDep = Annotated[AsyncSession, Depends(get_async_session)]

def db_session(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        with async_session_maker() as session:
            try:
                result = f(session, *args, **kwargs)
                session.commit()
                return result
            except:
                session.rollback()
                raise

    return wrapper