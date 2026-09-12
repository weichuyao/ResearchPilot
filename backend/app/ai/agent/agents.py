from dataclasses import dataclass, field
from typing import Awaitable, Callable

from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field
from ai.agent.react_assistant import build_react_assistant
from ai.agent.research_workflow import build_research_workflow


# 默认 agent。
#
# ⚠️ 这个常量只在**客户端完全不传 agent_id** 时起作用（见 api/schema/chatSchema.py：
# 它是 UserInput.agent_id 的默认值）。网页前端在 /agents 列表加载完成前不传 agent_id
# （空值会被 useStreamChat 省略掉），加载完成后传的是用户选中的值。
#
# 现在是改造 #3 的 Corrective RAG 工作流：在 20 题评估集上与 ReAct 对照组质量打平，
# 上下文少一半；代价是慢约 2 秒（要点驱动的重试多花轮次）。
# 详见 reference/transformation-03-research-workflow-design.md。
DEFAULT_AGENT = "research-workflow"

class AgentInfo(BaseModel):
    """Info about an available agent."""

    key: str = Field(
        description="Agent key.",
        examples=["react-assistant"],
    )
    description: str = Field(
        description="Description of the agent.",
        examples=["ReAct free-loop baseline over the research tools"],
    )

@dataclass
class Agent:
    description: str
    # 图的编译是**惰性**的（改造 #5 之二）：AsyncPostgresSaver 的构造需要
    # 运行中的事件循环，而「导入本模块」的上下文（TestClient、脚本）没有循环。
    # 改造 #4 起工厂还是 async —— react-assistant 编译时要经 stdio 挂载 MCP 工具。
    # 首次编译发生在 startup 预热（循环内）或第一次请求。
    graph_factory: Callable[[], Awaitable[CompiledStateGraph]]
    _graph: CompiledStateGraph | None = field(default=None, repr=False)


agents: dict[str, Agent] = {
    "react-assistant": Agent(
        description="ReAct free-loop baseline over the research tools (no budget, no evidence assessment).",
        graph_factory=build_react_assistant,
    ),
    "research-workflow": Agent(
        description="Corrective-RAG research workflow: analyze, retrieve, assess evidence, refine, synthesize.",
        graph_factory=build_research_workflow,
    ),
}

## Get agent by agent_id
async def get_agent(agent_id: str) -> CompiledStateGraph:
    """取（并在首次调用时编译）agent 图。必须在事件循环内调用。"""
    agent = agents[agent_id]
    if agent._graph is None:
        agent._graph = await agent.graph_factory()
    return agent._graph


def get_all_agent_info() -> list[AgentInfo]:
    return [
        AgentInfo(key=agent_id, description=agent.description) for agent_id, agent in agents.items()
    ]
