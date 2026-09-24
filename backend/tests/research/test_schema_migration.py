"""补列迁移的真实形状：给已有行的表加带默认值的列。

锁住 `docs/final_audit.md` §13 记的那条隐患。要点是**这种坏数据只有一条来路**：
模型里 `review_status` 是 NOT NULL，所以新建的库永远不会有 `NULL` 行；
`NULL` 只能来自"表已经有数据之后再补列"——旧版 `_add_missing_columns` 发的是
不带 DEFAULT 的 `ALTER TABLE ADD COLUMN`，而模型里的 `default=` 只在 ORM 插新行时生效，
于是**已有行拿到 NULL**。而 `NULL != "PROPOSED"` 会被判成"已复核过"，记录就此锁死。

所以夹具必须按这条来路造：先铺数据 → 删列（回到旧库形状）→ 补列 → 看已有行拿到什么。
"""

from __future__ import annotations

import asyncio
import sqlite3

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

import db.models  # noqa: F401
from db.database import _add_missing_columns
from db.repository.research_repository import ResearchRepository, _pending_review
from research.enums import ReviewStatus
from research.validators import ResearchValidationError

TABLE = "research_observation_hypothesis"
REVIEW_COLUMNS = ("review_status", "reviewed_by", "reviewed_at")


def _supports_drop_column() -> bool:
    return sqlite3.sqlite_version_info >= (3, 35, 0)


async def seed_chain(session, suffix: str, with_link: bool = True) -> tuple[str, str]:
    """最小外键链：问题 → 假设 → 实验 → 观察 (→ 观察关系)。返回 (观察 id, 假设 id)。

    裸 SQL 是刻意的：本测试的对象是 schema 而不是领域规则，走仓储会牵进审批链。
    """
    q, h, e, o = (f"RQ-{suffix}", f"H-{suffix}", f"EXP-{suffix}", f"O-{suffix}")
    await session.execute(text(
        "INSERT INTO research_question (id,title,description,background,scope,out_of_scope,"
        "status,created_at,updated_at) VALUES "
        "(:i,'RQ','Migration shape','','','','ACTIVE','2026-01-01','2026-01-01')"), {"i": q})
    await session.execute(text(
        "INSERT INTO research_hypothesis (id,research_question_id,statement,rationale,"
        "prediction,status,parent_hypothesis_id,created_at,updated_at) VALUES "
        "(:i,:q,'H','','P','TESTABLE',NULL,'2026-01-01','2026-01-01')"), {"i": h, "q": q})
    await session.execute(text(
        "INSERT INTO research_experiment (id,research_question_id,purpose,independent_variable,"
        "dependent_variables,control,treatment,controlled_variables,metrics,success_criteria,"
        "status,created_at,updated_at) VALUES "
        "(:i,:q,'p','i','[]','off','on','[]','[\"mAP\"]','{}','COMPLETED','2026-01-01','2026-01-01')"),
        {"i": e, "q": q})
    await session.execute(text(
        "INSERT INTO research_observation (id,experiment_id,measured_results,"
        "derived_statistics,description,created_at,updated_at) VALUES "
        "(:i,:e,'{}','{}','d','2026-01-01','2026-01-01')"), {"i": o, "e": e})
    if with_link:
        # 显式写 PROPOSED：ORM 插新行时会带上模型默认值，裸 SQL 不会。
        await session.execute(text(
            "INSERT INTO %s (observation_id,hypothesis_id,relation,review_status) "
            "VALUES (:o,:h,'SUPPORT','PROPOSED')" % TABLE), {"o": o, "h": h})
    await session.commit()
    return o, h


async def _drop_and_readd(conn, *, with_default: bool) -> None:
    """回到"旧库形状"，再按两种版本补列。"""
    # SQLite 会因为索引还在而拒绝 DROP COLUMN，先按名字删掉这三列上的索引
    indexed = [row[0] for row in (await conn.exec_driver_sql(
        "select name from sqlite_master where type='index' and tbl_name='%s'" % TABLE)).all()
        if not row[0].startswith("sqlite_autoindex")]
    for name in indexed:
        await conn.exec_driver_sql("DROP INDEX IF EXISTS %s" % name)
    for column in REVIEW_COLUMNS:
        await conn.exec_driver_sql("ALTER TABLE %s DROP COLUMN %s" % (TABLE, column))
    if with_default:
        await conn.run_sync(_add_missing_columns)
        return
    # 旧版行为：不带 DEFAULT、也不建索引的补列，已有行必然拿到 NULL
    await conn.exec_driver_sql(
        "ALTER TABLE %s ADD COLUMN review_status VARCHAR(24)" % TABLE)
    await conn.exec_driver_sql(
        "ALTER TABLE %s ADD COLUMN reviewed_by VARCHAR(200)" % TABLE)
    await conn.exec_driver_sql(
        "ALTER TABLE %s ADD COLUMN reviewed_at TIMESTAMP" % TABLE)


def _engine(tmp_path, name):
    return create_async_engine(f"sqlite+aiosqlite:///{tmp_path / name}")


@pytest.mark.skipif(not _supports_drop_column(), reason="SQLite 不支持 DROP COLUMN")
def test_added_column_is_backfilled_instead_of_left_null(tmp_path):
    """补列之后，已有行必须拿到 PROPOSED 而不是 NULL。"""
    async def scenario():
        engine = _engine(tmp_path, "backfill.db")
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)
        maker = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
        async with maker() as session:
            await seed_chain(session, "mig")          # 先铺数据（列还在）
        async with engine.begin() as conn:
            await _drop_and_readd(conn, with_default=True)   # 删列 → 用现在的机制补回来
        async with engine.connect() as conn:
            statuses = [row[0] for row in (await conn.execute(text(
                "select review_status from %s" % TABLE))).all()]
            indexes = {row[0] for row in (await conn.execute(text(
                "select name from sqlite_master where type='index' and tbl_name='%s'"
                % TABLE))).all()}
        assert statuses, "夹具行没铺出来，这个测试等于没测"
        assert statuses == [ReviewStatus.PROPOSED.value], (
            "补列后已有行不是 PROPOSED —— §13 那条锁死隐患：%r" % (statuses,))
        # 只补列不补索引，旧库上这个字段就是全表扫；审核列表恰恰按它过滤
        assert "ix_%s_review_status" % TABLE in indexes, (
            "补列时没有重建列上的索引：%s" % sorted(indexes))
        await engine.dispose()

    asyncio.run(scenario())


@pytest.mark.skipif(not _supports_drop_column(), reason="SQLite 不支持 DROP COLUMN")
def test_legacy_null_row_is_reproducible_via_old_backfill(tmp_path):
    """确认"NULL 行"确实是旧补列路径的产物 —— 否则下一条测试是空判。"""
    async def scenario():
        engine = _engine(tmp_path, "legacy-null.db")
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)
        maker = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
        async with maker() as session:
            await seed_chain(session, "legacy")
        async with engine.begin() as conn:
            await _drop_and_readd(conn, with_default=False)
        async with engine.connect() as conn:
            statuses = [row[0] for row in (await conn.execute(text(
                "select review_status from %s" % TABLE))).all()]
        assert statuses == [None], "旧路径没能复现出 NULL，后面的容忍测试就失去意义"
        await engine.dispose()

    asyncio.run(scenario())


@pytest.mark.skipif(not _supports_drop_column(), reason="SQLite 不支持 DROP COLUMN")
def test_null_review_status_is_still_reviewable(tmp_path):
    """已经是 NULL 的历史行不能被判成"已复核过"而锁死。"""
    async def scenario():
        engine = _engine(tmp_path, "null-status.db")
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)
        maker = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
        async with maker() as session:
            observation_id, hypothesis_id = await seed_chain(session, "null")
        async with engine.begin() as conn:
            await _drop_and_readd(conn, with_default=False)
        async with maker() as session:
            repository = ResearchRepository(session)
            reviewed = await repository.review_observation_relation(
                observation_id, hypothesis_id, "CONFIRMED", reviewer="pi")
            assert reviewed.review_status == ReviewStatus.CONFIRMED.value
            assert reviewed.reviewed_by == "pi"
            # NULL 只是"未复核"，不是"可以反复改"
            with pytest.raises(ResearchValidationError, match="already been reviewed"):
                await repository.review_observation_relation(
                    observation_id, hypothesis_id, "REJECTED", reviewer="pi")
        await engine.dispose()

    asyncio.run(scenario())


def test_evidence_relation_treats_null_as_unreviewed(tmp_path):
    """文献证据关系走同一个 helper —— 两处判断以前是同一个写法，必须一起修。

    这张表的 `review_status` 也是 NOT NULL，所以复现 NULL 同样只能走补列路径；
    为控制测试成本，这里直接测共用的 helper 本身，不重造整条链。
    """
    assert _pending_review(None) is True
    assert _pending_review("") is True
    assert _pending_review(ReviewStatus.PROPOSED.value) is True
    assert _pending_review(ReviewStatus.CONFIRMED.value) is False
    assert _pending_review(ReviewStatus.REJECTED.value) is False
