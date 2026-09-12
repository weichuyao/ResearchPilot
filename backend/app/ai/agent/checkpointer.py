"""会话记忆的持久化（改造 #5 之二）。

## 数据放在哪

LangGraph 的 checkpointer 按 thread_id 全量保存每一轮消息 —— 它本来就是
消息的 source of truth。所以这里不建 message 表，只让 checkpointer 落到
PostgreSQL（与业务表同一个库，改造 #6 的成果直接复用）。

## 生命周期（两段式 + 惰性，实测踩出来的）

`AsyncPostgresSaver.__init__` 会抓**当前运行中的事件循环** —— 在没有循环的
上下文里（TestClient 导入应用、脚本 import）构造它会直接 RuntimeError。
所以两件事必须都延迟：

1. **Saver 实例**：`get_checkpointer()` 首次调用才创建，而它只在两个地方被调 ——
   main 的 startup 事件（此时循环已在跑）和 agent 图的惰性编译（同样在循环里，
   见 agents.py）。uvicorn 的启动顺序是「先建循环 → 再导入应用 → 跑 startup」，
   所以这两处一定有循环；而任何**导入期**路径（测试、脚本）都碰不到它。
2. **连接**：`AsyncConnectionPool(open=False)` 构造不连接，`startup_checkpointer()`
   里才 open + `saver.setup()`（幂等 DDL），shutdown 里 close。

## 降级口径

`DATABASE_URL` 不是 postgres 时退回 `MemorySaver`：行为等于改造前（进程内存），
但**必须打 WARNING** —— 降级要看得见，这是这个项目一贯的规矩
（重排模型缺失静默降级那次一样，/health 专门为它加了状态字段）。
"""

import asyncio
import logging
import sys

from core.config import settings

logger = logging.getLogger(__name__)

# Windows 上 psycopg 异步模式只支持 SelectorEventLoop。这个模块是唯一引入 psycopg
# 的地方，所以在导入时就把策略定好 —— 之后创建的一切循环（uvicorn 的、
# run_eval 的 asyncio.run、TestClient 的 portal）都会是 Selector。
# 实测教训：run_eval 独立进程里没这条，池的连接在 Proactor 上全部静默失败，
# 30 秒后以 PoolTimeout 暴露，看起来像网络问题。
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def _is_postgres(url: str | None) -> bool:
    return bool(url) and url.startswith("postgresql")


if _is_postgres(settings.DATABASE_URL):
    # +asyncpg 方言标记是 SQLAlchemy 的，psycopg 不认识，剥掉。
    _DSN = settings.DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from psycopg_pool import AsyncConnectionPool

    # min_size=1：聊天是低频操作，不值得为它常驻 4 条连接（池的默认值）。
    # autocommit=True 必须显式给：saver.setup() 的迁移里有 CREATE INDEX CONCURRENTLY，
    # 它不能在事务块里跑 —— 池默认 autocommit=False，启动直接报
    # ActiveSqlTransaction（实测踩过）。from_conn_string 的官方用法内部就是这么配的。
    _POOL = AsyncConnectionPool(
        conninfo=_DSN, open=False, min_size=1, max_size=5, kwargs={"autocommit": True}
    )
    _SAVER: AsyncPostgresSaver | None = None
    _POOL_OPENED = False
    _BACKEND = "postgres"

    def get_checkpointer() -> AsyncPostgresSaver:
        """返回 Postgres checkpointer。**必须在事件循环内调用**（见模块 docstring）。"""
        global _SAVER
        if _SAVER is None:
            _SAVER = AsyncPostgresSaver(_POOL)
        return _SAVER

else:
    from langgraph.checkpoint.memory import MemorySaver

    _SAVER: MemorySaver | None = MemorySaver()
    _BACKEND = "memory"

    def get_checkpointer() -> MemorySaver:
        return _SAVER  # type: ignore[return-value]


async def startup_checkpointer() -> None:
    """打开连接池 + 建表（幂等）。

    服务路径由 main.py 的 startup 事件调用；**脚本路径（run_eval）也必须调** ——
    它是独立进程，没有 startup 事件，不调的话池根本没打开，第一笔 checkpoint
    就会抛 PoolClosed（实测踩过）。
    """
    if _BACKEND == "postgres":
        global _POOL_OPENED
        saver = get_checkpointer()  # 循环内创建，__init__ 才能抓到循环
        if not _POOL_OPENED:
            await _POOL.open()
            _POOL_OPENED = True
        await saver.setup()
        logger.info("checkpointer: PostgreSQL —— 会话跨重启持久")
    else:
        logger.warning(
            "checkpointer: MemorySaver —— DATABASE_URL 不是 postgres，"
            "会话只存在进程内存里，重启即失忆（见 ai/agent/checkpointer.py 的降级口径）"
        )


async def shutdown_checkpointer() -> None:
    if _BACKEND == "postgres":
        await _POOL.close()
