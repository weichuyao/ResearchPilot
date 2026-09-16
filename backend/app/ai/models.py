from enum import StrEnum
from typing import TypeAlias

DEFAULT_MODEL = "deepseek-chat"


class OpenAIModelName(StrEnum):
    """https://platform.openai.com/docs/models/gpt-4o"""

    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4O = "gpt-4o"

class DeepseekModelName(StrEnum):
    """https://api-docs.deepseek.com/quick_start/pricing"""
    DEEPSEEK_CHAT = "deepseek-chat"

class OllamaModelName(StrEnum):
    """https://ollama.com/search"""

    OLLAMA_GENERIC = "ollama"

class FakeModelName(StrEnum):
    """Fake model for testing."""
    FAKE = "fake"
    
class TongYiModelName(StrEnum):
    """TongYi model"""
    QWEN_PLUS = "qwen-plus"
    QWEN_MAX = "qwen-max"


class LocalModelName(StrEnum):
    """本地部署的规划改写模型（改造 #11）。

    只有一个成员是刻意的：这个枚举的用途是**把「规划节点走本地模型」这件事
    变成一个可配置的名字**，而不是建立一套完整的本地模型注册表 ——
    具体用哪个 Ollama tag 由 settings.OLLAMA_REWRITE_MODEL 决定
    （与 OllamaModelName 用 settings.OLLAMA_MODEL 的做法一致）。
    """

    LOCAL_REWRITER = "local-rewriter"



AllModelEnum: TypeAlias = (
    OpenAIModelName
    | DeepseekModelName
    | OllamaModelName
    | FakeModelName
    | TongYiModelName
    | LocalModelName
)
