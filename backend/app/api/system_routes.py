"""系统状态接口（`/health`）。

## 为什么需要它

实测教训：排查问题时发现**后端进程跑的是几小时前的旧代码**，而没有任何地方能看出来。
`uvicorn` 没开 `--reload`，日志在隐藏窗口里，接口行为又"像是新的"（因为改动没生效，
表现是旧行为的延续，很容易被解释成"新代码没起作用"）。
最后是靠手工比对进程启动时间和构建产物时间才确认的。

**「我改的代码生效了吗」应该是一次请求就能回答的问题，而不是一次考古。**

所以 /health 回答三件事：

| 字段 | 回答的问题 |
|---|---|
| `started_at` / `git_rev` | 线上跑的是哪份代码 |
| `reranker` | 依赖的模型加载了没（重排模型缺失时系统会静默降级，不查看不出来） |
| `index` | 索引里到底有什么（几篇、多少块、有没有卡在失败/处理中的） |

`index.by_status` 里如果有 `indexing > 0`，说明**有导入任务卡住了** ——
后台任务跑在进程内，进程重启会让它丢失，记录就永远停在「处理中」。
这是知道「该重试哪一个」的唯一途径。
"""

from __future__ import annotations

import logging
import os
from datetime import datetime

from fastapi import APIRouter

from api.schema.documentSchema import HealthOut, status_text
from core.config import settings
from db.database import async_session_maker
from db.models.paper import STATUS_FAILED, STATUS_INDEXED, STATUS_INDEXING, STATUS_PENDING
from db.repository.paper_repo import PaperRepository

logger = logging.getLogger(__name__)

system_router = APIRouter(tags=["system"])

# 进程启动时刻。放在模块级：模块只被导入一次，所以它就是"这个进程是什么时候起来的"。
_STARTED_AT = datetime.now()


def _git_rev() -> str | None:
    """当前代码的提交号（短）。

    **优先读配置里的 GIT_REV** —— 容器里读不到 `.git`：
    `.git` 在仓库根目录，而镜像的构建上下文是 `backend/`，它不在里面。
    所以构建时用 `--build-arg GIT_REV=...` 传进来（见 backend/Dockerfile 与 compose）。

    本机跑的时候配置里没有这个值，就退回直接读 `.git` —— 不引入子进程依赖，
    也不需要 git 在 PATH 里。只处理 HEAD 指向分支/直接指向提交这两种情况，
    读不到就返回 None：它是个辅助信息，不该因为版本库布局特殊就让 /health 挂掉。
    """
    if settings.GIT_REV:
        return settings.GIT_REV
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        root = os.path.dirname(os.path.dirname(os.path.dirname(here)))
        git_dir = os.path.join(root, ".git")
        with open(os.path.join(git_dir, "HEAD"), encoding="utf-8") as fh:
            head = fh.read().strip()
        if head.startswith("ref: "):
            ref_path = os.path.join(git_dir, head[5:].strip())
            with open(ref_path, encoding="utf-8") as fh:
                return fh.read().strip()[:12]
        return head[:12]
    except Exception:
        return None


def _reranker_state() -> str:
    """重排模型是否可用。

    注意这一步会**真的触发加载**（第一次调用时读 266MB 的 ONNX，约 0.5 秒）。
    这是有意的：只报"文件在不在"是不够的 —— 模型文件在、但 onnxruntime 加载失败，
    系统会静默降级成纯混合检索，从回答质量上完全看不出来。
    宁可让 /health 慢半秒，也要报真实状态。
    """
    try:
        from ai.rag.rerank import available

        return "loaded" if available() else "unavailable"
    except Exception as exc:
        logger.warning("rerank 状态检查失败：%s", exc)
        return "error"


@system_router.get("/health", response_model=HealthOut)
async def health() -> HealthOut:
    async with async_session_maker() as session:
        papers = await PaperRepository.count(session=session)
        chunks = await PaperRepository.sum_chunks(session=session)
        by_status = await PaperRepository.count_by_status(session=session)

    statuses = {
        status_text(code): by_status.get(code, 0)
        for code in (STATUS_PENDING, STATUS_INDEXING, STATUS_INDEXED, STATUS_FAILED)
    }
    stuck = statuses.get("indexing", 0)
    if stuck:
        logger.warning("/health: 有 %d 篇卡在「处理中」—— 后台任务可能因进程重启丢失", stuck)

    return HealthOut(
        app=settings.APP_NAME,
        started_at=_STARTED_AT.strftime("%Y-%m-%d %H:%M:%S"),
        git_rev=_git_rev(),
        reranker=_reranker_state(),
        index={"papers": papers, "chunks": chunks, "by_status": statuses},
    )
