"""统一的日志配置（改造 #7）。

## 为什么需要一个单独的模块

在这之前，根日志记录器的配置藏在 `react_assistant.py` 的导入副作用里 ——
谁导入那个模块，谁就顺手配置了全局日志。这个位置的三个问题：

1. **找得到才怪**：`/api/system_routes.py` 里打的日志，它的格式居然由
   `ai/agent/` 下某个 agent 文件决定。新来的人不可能想到去那里找。
2. **不导入就不配置**：任何一条不经过 agent 模块的代码路径，日志格式就退回
   Python 默认（裸 message，没时间戳没文件名），而且**不落盘**。
3. **改 agent 就改日志**：调整 agent 的导入顺序可能静默改变整个应用的日志行为。

## 为什么落文件

Docker 容器一重启，stdout 里的日志就全没了 —— `docker compose logs` 只能看
还在跑的进程的输出。而故障排查最需要的恰恰是**重启前**发生了什么（改造 #5
里那个"后端跑的是几小时前的旧代码"的事故，最后是靠手工比对进程启动时间才确认的）。

所以：根日志同时接 console（行为不变）+ RotatingFileHandler（新增）。
文件用轮转而不是无限追加 —— 一个 10 MB × 5 份的上界，防止磁盘被慢写满。

## uvicorn 的日志为什么不走根记录器

uvicorn 的 access / error 日志自带 handlers 且 `propagate=False`，默认到不了
根记录器，也就进不了文件。这里把文件 handler **追加**到它们的 logger 上，
console 保持 uvicorn 自己的格式（带颜色/对齐，动了反而难看）。
"""

import logging
import os
from logging.handlers import RotatingFileHandler

from core.config import settings

_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(filename)s[line:%(lineno)d] - %(funcName)s() - %(message)s"
_FILE_MAX_BYTES = 10 * 1024 * 1024
_FILE_BACKUP_COUNT = 5

# uvicorn 的 logger 树：`uvicorn` 与 `uvicorn.access` 各自带 handlers 且不向根传播；
# `uvicorn.error` **没有**自己的 handler，向上传播给 `uvicorn`。
# 所以文件 handler 只挂前两个 —— 给 `uvicorn.error` 也挂的话，
# 每条启动日志会经它自己写一次、再经 `uvicorn` 写一次，文件里重复两遍（实测踩过）。
UVICORN_LOGGERS = ("uvicorn", "uvicorn.access")

# 吵的库单独压到 WARNING。它们每次 HTTP 调用都会打完整请求体（含用户提问和
# 系统提示词）—— 既是噪音也是隐私问题。这条规则是 app 级策略，跟着日志配置走。
NOISY_LOGGERS = ("httpx", "httpcore", "openai", "urllib3", "chromadb")


def _log_dir() -> str:
    """日志目录，锚定到 backend/ 根。

    和 chromaClient.py 的 CHROMA_PATH 同一个教训：相对路径按进程 CWD 解析，
    从不同目录启动就会把文件写去不同的地方。
    """
    path = settings.LOG_DIR or "logs"
    if not os.path.isabs(path):
        # 本文件在 app/core/ 下：core -> app -> backend
        backend_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        path = os.path.join(backend_root, path)
    return path


def setup_logging() -> None:
    """配置根日志：console + 轮转文件。可重复调用（幂等）。

    幂等的实现是给文件 handler 打标（`app_file_handler` 属性）再查重，
    而不是清空所有 handlers —— 后者会顺手扔掉测试框架或 uvicorn 装的 handler。
    """
    root = logging.getLogger()
    root.setLevel(getattr(logging, (settings.LOG_LEVEL or "INFO").upper(), logging.INFO))

    if not any(getattr(h, "app_file_handler", False) for h in root.handlers):
        log_path = os.path.join(_log_dir(), "app.log")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        file_handler = RotatingFileHandler(
            log_path, maxBytes=_FILE_MAX_BYTES, backupCount=_FILE_BACKUP_COUNT, encoding="utf-8"
        )
        file_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        file_handler.app_file_handler = True
        root.addHandler(file_handler)

        for name in UVICORN_LOGGERS:
            logging.getLogger(name).addHandler(file_handler)

    # console 上的根格式只在 root 还没有任何 formatter 时补 —— 不覆盖
    # 别人（pytest / 脚本）已经配好的 console 行为。
    if root.handlers and not any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, RotatingFileHandler)
        for h in root.handlers
    ):
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter(_LOG_FORMAT))
        root.addHandler(console)

    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
