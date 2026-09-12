"""本地 Windows 开发的启动入口（代替 `python -m uvicorn ...`）。

## 为什么必须用它而不是 `python -m uvicorn`

Windows 上 psycopg 的异步模式只支持 SelectorEventLoop。策略必须在
**循环创建之前**设置，而 `python -m uvicorn` 这条路径是 uvicorn 先建循环、
后导入应用 —— main.py 里设的策略来不及生效。uvicorn 0.34 只在
`--reload` 的子进程路径里自带 Selector，所以裸启动会以
「psycopg_pool.PoolTimeout: couldn't get a connection after 30.00 sec」
告终（2026-09-12 实测三次，诊断见 app/main.py 的注释和 design-decisions.md）。

本文件在 `uvicorn.run()` **之前**设同一个策略 —— run() 内部才创建循环，
此时策略已经生效。容器（Linux）没有这个限制，两种方式都行。

用法（在 backend/ 下）：

    ./.venv-py311/Scripts/python.exe run_server.py
"""

import asyncio
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        app_dir="app",
        host="127.0.0.1",
        port=8002,
        log_level="warning",
    )
