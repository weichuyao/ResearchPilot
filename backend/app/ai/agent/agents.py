from dataclasses import dataclass

from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field
from ai.agent.oa_assistant import oa_assistant
from ai.agent.multi_agent import supervisor_agent
from ai.agent.research_workflow import research_workflow


# 默认 agent。
#
# 目前仍是 ReAct 版（model ⇄ tools 的自由循环）。research-workflow 是改造 #3 的产物，
# **刻意并存而不是替换** —— 两个图共用同一套工具和检索管线，可以用同一份评估集直接
# 对比。等 research-workflow 在评估集上证明自己不退步（含过程指标），再切默认。
DEFAULT_AGENT = "oa-assistant"

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
