from dataclasses import dataclass

from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field
from ai.agent.oa_assistant import oa_assistant
from ai.agent.multi_agent import supervisor_agent
from ai.agent.research_workflow import research_workflow


# 默认 agent。
#
# ⚠️ 这个常量只在**客户端完全不传 agent_id** 时起作用（见 api/schema/chatSchema.py：
# 它是 UserInput.agent_id 的默认值）。网页前端每次都会显式传它选中的值，所以只改这里
# **不会**改变浏览器里的行为 —— 前端的初始值在 frontend/app/layout.tsx 的 useState，
# 可选项列表在 frontend/app/components/AgentSelector.tsx。
#
# 现在是改造 #3 的 Corrective RAG 工作流：在 20 题评估集上与原来的 ReAct 版质量打平
# 或反超（A 类四项全 1.0），上下文少一半；代价是慢约 1.3s（要点驱动的重试多花轮次）。
# 详见 reference/transformation-03-research-workflow-design.md。
DEFAULT_AGENT = "research-workflow"

class AgentInfo(BaseModel):
    """Info about an available agent."""

    key: str = Field(
        description="Agent key.",
        examples=["oa-assistant"],
    )
    description: str = Field(
        description="Description of the agent.",
        examples=["A oa assistant for company"],
    )

@dataclass
class Agent:
    description: str
    graph: CompiledStateGraph


agents: dict[str, Agent] = {
    "oa-assistant": Agent(description="A oa intelligent assistant.", graph=oa_assistant),
    "research-workflow": Agent(
        description="Corrective-RAG research workflow: analyze, retrieve, assess evidence, refine, synthesize.",
        graph=research_workflow,
    ),
    "multi-agent-supervisor": Agent(description="A supervisor for multi-agent assistant.", graph=supervisor_agent),
}

## Get agent by agent_id
def get_agent(agent_id: str) -> CompiledStateGraph:
    """Get agent by agent_id"""
    return agents[agent_id].graph


def get_all_agent_info() -> list[AgentInfo]:
    return [
        AgentInfo(key=agent_id, description=agent.description) for agent_id, agent in agents.items()
    ]
