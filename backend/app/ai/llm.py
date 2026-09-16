from functools import cache
from typing import TypeAlias


from chromadb.auth import T
from langchain_community.chat_models import FakeListChatModel, tongyi
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from langchain_community.chat_models import ChatTongyi




from core.config import settings
from ai.models import (
    AllModelEnum,
    DeepseekModelName,
    FakeModelName,
    LocalModelName,
    OllamaModelName,
    OpenAIModelName,
    TongYiModelName,

)

_MODEL_TABLE = {
    OpenAIModelName.GPT_4O_MINI: "gpt-4o-mini",
    OpenAIModelName.GPT_4O: "gpt-4o",
    DeepseekModelName.DEEPSEEK_CHAT: "deepseek-chat",
    OllamaModelName.OLLAMA_GENERIC: "ollama",
    FakeModelName.FAKE: "fake",
    TongYiModelName.QWEN_PLUS: "qwen-plus",
    # 真实 tag 不在表里 —— 与 Ollama 分支同理，从 settings 读
    # （settings.OLLAMA_REWRITE_MODEL），这样换模型不用改代码。
    LocalModelName.LOCAL_REWRITER: "local-rewriter",
}


def _matches_model_family(model_name: str, model_family: type) -> bool:
    """Return whether a configured model value belongs to an enum family.

    Python 3.12+ accepts values in ``value in Enum``; Python 3.11 raises a
    TypeError.  Constructing the enum is compatible with both versions.
    """
    try:
        model_family(model_name)
        return True
    except ValueError:
        return False


class FakeToolModel(FakeListChatModel):
    def __init__(self, responses: list[str]):
        super().__init__(responses=responses)

    def bind_tools(self, tools):
        return self

ModelT: TypeAlias = (
    ChatOpenAI | ChatOllama | ChatDeepSeek | FakeToolModel | ChatTongyi
)



@cache
def get_model(model_name: AllModelEnum, /) -> ModelT:
    """
    Get model by model name.
    Args:
        model_name: Model name.
    Returns:
        Model instance.
    """

    
    api_model_name = _MODEL_TABLE.get(model_name)
    if not api_model_name:
        raise ValueError(f"Unsupported model: {model_name}")

    if _matches_model_family(model_name, OpenAIModelName):
        return ChatOpenAI(model=api_model_name, temperature=0.5, streaming=True)

   
    if _matches_model_family(model_name, DeepseekModelName):

        return ChatDeepSeek(
            model=api_model_name,
            temperature=0.5,
            streaming=True,
            api_key=settings.DEEPSEEK_API_KEY,
        )
    
    if _matches_model_family(model_name, OllamaModelName):
        if settings.OLLAMA_BASE_URL:
            chat_ollama = ChatOllama(
                model=settings.OLLAMA_MODEL, temperature=0.5, base_url=settings.OLLAMA_BASE_URL
            )
        else:
            chat_ollama = ChatOllama(model=settings.OLLAMA_MODEL, temperature=0.5)
        return chat_ollama
    if _matches_model_family(model_name, FakeModelName):
        return FakeToolModel(responses=["This is a test response from the fake model."])
    
    if _matches_model_family(model_name, TongYiModelName):
        return ChatTongyi(model=api_model_name, temperature=0.5, streaming=True)

    if _matches_model_family(model_name, LocalModelName):
        # 本地规划改写模型（改造 #11）：走 Ollama，因为项目已经用它跑 embedding，
        # OLLAMA_BASE_URL 配置现成，微调产物（GGUF）也是 ollama create 直接吃。
        #
        # 这里**刻意不设 format**：结构化输出由调用方用
        # `with_structured_output(schema, method="json_schema")` 指定 ——
        # Ollama 会把 JSON schema 交给运行时做**语法约束解码**，输出形状由语法
        # 保证，而不是靠提示词祈祷。实测（tmp/probe19.py）：
        #   · json_mode + schema 塞提示词 → 弱模型会把 schema 片段当输出吐回来
        #   · json_schema（语法约束）      → 结构 100% 合法
        # 结构合法不等于内容对：基座模型的查询仍是中文、facets 为空 ——
        # 那部分只能靠微调解决，也正是改造 #11 要度量的事。
        kwargs = {"model": settings.OLLAMA_REWRITE_MODEL, "temperature": 0.5}
        if settings.OLLAMA_BASE_URL:
            kwargs["base_url"] = settings.OLLAMA_BASE_URL
        return ChatOllama(**kwargs)
