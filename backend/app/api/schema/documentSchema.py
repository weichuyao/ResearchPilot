"""文档接口的请求/响应模型。

状态用**两个字段**表达：`status`（整数，存库用）和 `status_text`（字符串，给人看）。
为什么都返回：整数是数据库和状态机用的，字符串是前端和日志用的。
只给整数，前端就得自己维护一份映射；只给字符串，前端就没法做 `if status == 2` 之外的判断。
"""

from datetime import datetime

from pydantic import BaseModel, Field

from db.models.paper import (
    STATUS_FAILED,
    STATUS_INDEXED,
    STATUS_INDEXING,
    STATUS_PENDING,
)

STATUS_TEXT = {
    STATUS_PENDING: "pending",
    STATUS_INDEXING: "indexing",
    STATUS_INDEXED: "indexed",
    STATUS_FAILED: "failed",
}


def status_text(status: int) -> str:
    return STATUS_TEXT.get(status, "unknown")


class DocumentOut(BaseModel):
    """一篇文档的完整状态。前端轮询 `GET /documents/{id}` 拿到的就是它。"""

    id: int
    title: str
    source_file: str
    status: int
    status_text: str
    chunk_count: int
    error: str = Field(default="", description="上次导入失败的原因，成功时为空")
    collection_id: int
    indexed_at: datetime | None = None


class UploadAccepted(BaseModel):
    """`POST /documents` 的 202 响应体。

    刻意带上 `poll`：客户端拿到 202 之后必须知道「下一步去哪查」。
    只返回一个 id 而不说怎么用，是把协议的一半留给调用方去猜。
    """

    id: int
    title: str
    status: int
    status_text: str
    poll: str = Field(description="查询进度的地址")


class DocumentList(BaseModel):
    total: int
    items: list[DocumentOut]


class HealthOut(BaseModel):
    """`GET /health`。

    存在的理由是实测教训：曾经有一个后端进程跑着**几小时前的旧代码**，
    而没有任何地方能看出来 —— 日志在隐藏窗口里，接口行为又"像是新的"。
    最后是靠手工比对进程启动时间才确认的。

    所以 /health 必须回答三个问题：
      · 线上跑的是哪份代码（started_at / git_rev）
      · 依赖的模型加载了没（reranker）
      · 索引里到底有什么（papers / chunks / by_status）

    改造 #11 之后多一个问题：**规划节点的本地改写模型是不是真的在干活**。
    它一直失败、每次都悄悄回落主力模型时，系统表现完全正常、只是没省下延迟 ——
    与 reranker 静默降级同一类问题，所以同样挂在 /health 上。
    """

    app: str
    started_at: str
    git_rev: str | None = Field(default=None, description="当前代码的 git 提交号")
    reranker: str = Field(description="loaded / unavailable（重排模型是否可用）")
    rewrite_model: dict = Field(
        default_factory=dict,
        description=(
            "规划节点（analyze/refine）的模型状态：configured=配置的模型名"
            "（空=与主力模型一致）、target=本地 Ollama tag、calls/fallbacks="
            "调用与回落次数、last_error=最近一次失败原因"
        ),
    )
    index: dict
