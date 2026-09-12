from dataclasses import dataclass, field
from typing import Callable

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
    # 编译因此推迟到第一次 get_agent() —— 服务路径上一定在循环里（startup 事件
    # 会先把所有图预热一遍），评估脚本则在 asyncio.run 里调用。
    graph_factory: Callable[[], CompiledStateGraph]
    _graph: CompiledStateGraph | None = field(default=None, repr=False)

    @property
    def graph(self) -> CompiledStateGraph:
        if self._graph is None:
            self._graph = self.graph_factory()
        return self._graph


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
def get_agent(agent_id: str) -> CompiledStateGraph:
    """Get agent by agent_id"""
    return agents[agent_id].graph


def get_all_agent_info() -> list[AgentInfo]:
    return [
        AgentInfo(key=agent_id, description=agent.description) for agent_id, agent in agents.items()
    ]
