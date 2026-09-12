"""ReAct 自由循环 agent（注册名 `react-assistant`）。

与 research_workflow（Corrective-RAG）并存，共用同一套工具与检索管线 ——
它是改造 #3 评估里的对照组：检索轮数由模型自由裁量，没有预算与三态判定。
原文件名 oa_assistant.py 是 OA 时期的遗留，随 OA 残留清理（改造 #1）改名。
"""

from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSerializable
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from ai.agent.checkpointer import get_checkpointer
from ai.llm import get_model, settings
from ai.tools.research_tools import list_papers, search_documents
from ai.tools.mcp_tools import get_mcp_tools


from langchain.globals import set_debug
from langchain.globals import set_verbose

# LangChain 的结构化调试输出（每次模型调用的完整 prompt / tool schema）。
# 开关交给 .env 的 DEBUG，别写死。
set_debug(settings.DEBUG)
set_verbose(False)

# 注意：这里**没有** stdlib 日志配置。根日志的配置（格式 / 文件落盘 / 噪音库压制）
# 统一在 core/logging_config.py，由 main.py 导入时执行 —— 之前它以 basicConfig 的
# 形式藏在这个文件的导入副作用里，等于「谁导入 agent 谁才有日志配置」。


class AgentState(MessagesState):
    """State of the agent."""

# 本地工具永远在场；MCP 外部工具（arXiv 等）在编译时尝试挂载（见 build）。
BASE_TOOLS = [list_papers, search_documents]

def wrap_model(model: BaseChatModel, tools) -> RunnableSerializable[AgentState, AIMessage]:
    model = model.bind_tools(tools)
    preprocessor = RunnableLambda(
        lambda state: [SystemMessage(content=instructions)] + state["messages"],
        name="StateModifier",
    )
    return preprocessor | model

instructions = """
    You are ResearchPilot, a research assistant for scientific papers and technical documents.
    Your job is to help users find, read and understand material in the document knowledge base,
    and to answer their research questions.

    Rules you must follow:

    1. Always look things up with your tools before answering a question about the documents.
       Base every factual statement on material retrieved through your tools. Never invent paper
       titles, authors, findings, figures or regulations.

    2. Pick the right tool. `list_papers` answers questions about WHICH papers exist, their
       authors, years and size — it is an exact lookup. `search_documents` finds passages by
       meaning. When the user names a specific paper, pass it to search_documents as the
       `paper` argument so that passages from the other papers cannot be returned.

    3. Whenever you state something that comes from the documents, cite where it came from,
       using the paper title and page number that the tool returned. The tool also returns a
       relevance score; use it to decide which passage to trust more when passages disagree.

    4. If a tool reports that no relevant documents were found, you may only say that the
       documents do not contain relevant information. Never turn "not found" into "does not
       exist", because a document that stays silent about something is not evidence that the
       thing is false.

    5. Answer the question the user actually asked, and keep the answer focused.
"""




# After "model", if there are tool calls, run "tools". Otherwise END.
def pending_tool_calls(state: AgentState) -> Literal["tools", "done"]:
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
        raise TypeError(f"Expected AIMessage, got {type(last_message)}")
    if last_message.tool_calls:
        return "tools"
    return "done"


async def build_react_assistant():
    """装配并编译图。惰性调用，两个原因：
    1. checkpointer 构造需要运行中的事件循环（ai/agent/checkpointer.py）；
    2. MCP 工具在这里经 stdio 挂载（ai/tools/mcp_tools.py）—— 拉不起来就
       WARNING + 只有本地工具，本 agent 的其余部分不受影响。"""
    tools = BASE_TOOLS + list(await get_mcp_tools())

    async def call_model(state: AgentState, config: RunnableConfig) -> AgentState:
        m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
        response = await wrap_model(m, tools).ainvoke(state, config)
        return {"messages": [response]}

    builder = StateGraph(AgentState)
    builder.add_node("model", call_model)
    builder.add_node("tools", ToolNode(tools=tools))
    builder.set_entry_point("model")
    builder.add_edge("tools", "model")
    builder.add_conditional_edges("model", pending_tool_calls, {"tools": "tools", "done": END})

    graph = builder.compile(checkpointer=get_checkpointer())
    graph.name = "react_assistant"
    return graph
