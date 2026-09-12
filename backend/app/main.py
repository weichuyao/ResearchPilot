from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.chat_routes import chat_router
from api.document_routes import document_router
from api.system_routes import system_router
from core.logging_config import setup_logging
from db.database import create_db_and_tables

# 日志配置在导入时执行（幂等）：任何一条启动路径（uvicorn / 脚本 / 测试）只要
# 导入了 main 就有统一格式 + 文件落盘。在这之前它藏在 react_assistant 的
# 导入副作用里 —— 谁导入 agent 谁才有日志配置，见 core/logging_config.py 的
# 模块文档。
setup_logging()

app = FastAPI()


@app.on_event("startup")
async def initialize_database() -> None:
    await create_db_and_tables()


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
app.include_router(document_router)
app.include_router(system_router)
