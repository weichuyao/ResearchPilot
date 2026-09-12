"""ReAct 自由循环 agent（注册名 `react-assistant`）。

与 research_workflow（Corrective-RAG）并存，共用同一套工具与检索管线 ——
它是改造 #3 评估里的对照组：检索轮数由模型自由裁量，没有预算与三态判定。
原文件名 oa_assistant.py 是 OA 时期的遗留，随 OA 残留清理（改造 #1）改名。
"""

from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSerializable
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from ai.llm import get_model, settings
from ai.tools.research_tools import list_papers, search_documents


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

tools = [list_papers, search_documents]

def wrap_model(model: BaseChatModel) -> RunnableSerializable[AgentState, AIMessage]:
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


async def call_model(state: AgentState, config: RunnableConfig) -> AgentState:
    """This node is to call llm model"""

    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_model(m)
    response = await model_runnable.ainvoke(state, config)

    return {"messages": [response]}



# After "model", if there are tool calls, run "tools". Otherwise END.
def pending_tool_calls(state: AgentState) -> Literal["tools", "done"]:
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
        raise TypeError(f"Expected AIMessage, got {type(last_message)}")
    if last_message.tool_calls:
        return "tools"
    return "done"


# Define the graph
agent = StateGraph(AgentState)
agent.add_node("model", call_model)
agent.add_node("tools", ToolNode(tools = tools))

agent.set_entry_point("model")
agent.add_edge("tools", "model")

agent.add_conditional_edges("model", pending_tool_calls, {"tools": "tools", "done": END})


react_assistant = agent.compile(
    checkpointer=MemorySaver(),
)
react_assistant.name = "react_assistant"

# Save the graph as a PNG
# graph_png = react_assistant.get_graph().draw_mermaid_png()

# with open("graph.png", "wb") as f:
#     f.write(graph_png)
