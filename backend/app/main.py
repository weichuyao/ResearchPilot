import asyncio
import logging
import sys

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from api.chat_routes import chat_router
from api.conversation_routes import conversation_router
from api.document_routes import document_router
from api.system_routes import system_router
from ai.agent.agents import agents, get_agent
from ai.agent.checkpointer import shutdown_checkpointer, startup_checkpointer
from core.logging_config import setup_logging
from db.database import create_db_and_tables

logger = logging.getLogger(__name__)

# Windows 上 psycopg 的异步模式只支持 SelectorEventLoop（aiosqlite / asyncpg 没有
# 这个限制，所以以前没暴露）。在这里设策略覆盖「导入 main 之后才创建循环」的路径
# （TestClient / 脚本）。⚠️ 直接 `python -m uvicorn` 起服务**救不了** —— uvicorn
# 先建循环、后导入应用，策略设上已经晚了；uvicorn 0.34 只在 --reload 模式自带
# Selector，所以本地 Windows 开发的启动命令要带 --reload（见 NOTES.md，容器是
# Linux 不受影响）。run_server.py 在 uvicorn.run 之前设了同一个策略。
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# 日志配置在导入时执行（幂等）：任何一条启动路径（uvicorn / 脚本 / 测试）只要
# 导入了 main 就有统一格式 + 文件落盘。在这之前它藏在 react_assistant 的
# 导入副作用里 —— 谁导入 agent 谁才有日志配置，见 core/logging_config.py 的
# 模块文档。
setup_logging()

app = FastAPI()


@app.on_event("startup")
async def initialize_database() -> None:
    await create_db_and_tables()
    # checkpointer 的连接与建表放在业务建表之后：
    # 两者都连同一个 postgres，先后无所谓，但日志顺序上先业务库再记忆库更好读。
    await startup_checkpointer()
    # 预热：惰性编译的图在这里全部编译（此刻事件循环在跑）。
    # 图的结构错误在启动时暴露，而不是留给第一个请求；/agents 列表也顺带变快。
    for agent_id in agents:
        get_agent(agent_id)
    logger.info("agents 预热完成：%s", ", ".join(agents))


@app.on_event("shutdown")
async def close_resources() -> None:
    await shutdown_checkpointer()


# ---- 统一异常层（改造 #7 的欠账，随 #5 一起补上）----
#
# 之前：未知 agent → 500 + 纯文本 "Internal Server Error"（SSE 路径连 JSON 都不是），
# 任何未捕获异常只有一行日志、没有 traceback，日志文件里查不到根因。
#
# 口径（延续 http-status-codes.md）：
#   4xx —— 我拒绝，但要告诉调用方为什么：detail 原样透传，那是特意写给人看的
#          （「文件超过 50 MB 上限」「未知的 agent: xxx」），包一层就丢了信息。
#   5xx —— 我坏了：detail 固定措辞，不给调用方暴露内部信息；
#          但日志里必须留下完整 traceback（#7 的文件日志在这里第一次派上真用场）。


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    if exc.status_code >= 500:
        logger.error("HTTP %s %s: %s", exc.status_code, request.url.path, exc.detail)
    else:
        logger.warning("HTTP %s %s: %s", exc.status_code, request.url.path, exc.detail)
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # 422 属于「我拒绝」：把校验失败的原样给出去，前端要靠它提示用户
    logger.warning("HTTP 422 %s: %s", request.url.path, str(exc.errors()[:3]))
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("未处理异常 %s %s: %r", request.method, request.url.path, exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})

# Cross-domain is allowed. For production environments,
# please change * to a specific domain name
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)
app.include_router(conversation_router)
app.include_router(document_router)
app.include_router(system_router)
