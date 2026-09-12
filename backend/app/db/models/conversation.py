"""会话索引（改造 #5 之二）。

**注意这不是消息表** —— 消息的 source of truth 是 checkpointer（LangGraph
按 thread_id 全量存，见 ai/agent/checkpointer.py）。这张表只存「会话列表
需要回答的元信息」，让侧边栏不用去碰 checkpoint 存储：

| 读什么 | 从哪 |
|---|---|
| 会话列表 / 标题 / 最后活跃时间 | 这张表 |
| 某会话的全部消息 | `agent.aget_state()` 读 checkpoint |

所以它坏了他可以随时从 checkpoint 重建 —— 和 paper 表可重建是同一条思路。
它存 `agent_id` 是因为读 checkpoint 的状态必须用**当时的那个图**去读
（不同图的 state schema 不同，thread 是绑定在图上的）。
"""

from sqlmodel import Field

from db.models.base import DBBaseModel


class Conversation(DBBaseModel, table=True):
    thread_id: str = Field(primary_key=True, title="LangGraph thread id")
    title: str = Field(title="标题（首条用户消息截断）")
    agent_id: str = Field(title="创建该会话的 agent key")
    last_message_at: str = Field(title="最后一条消息时间（ISO 字符串，按字典序即时间序）")
