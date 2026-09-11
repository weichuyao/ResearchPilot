"""交叉编码器重排（cross-encoder rerank）。

## 为什么 RRF 之后还需要重排

RRF 是**排名融合**：它只看两个检索器给出的名次，从不把查询和段落放在一起看过。
它解决的是「把两路的好结果合并进同一个候选池」，而不是「池子里哪一条最相关」。

看实测数据（A2RNet 网络结构这个问题，10 条候选）：

    RRF #1   LSTKC++ [37] T-PAMI 2025  27.80 47.30 70.30 ...   <- 表格数字行
    RRF #3   the TAR module captures long-term temporal ...    <- 真正的答案
    RRF #9   一条 ViV-ReID 的片段                               <- 串了别的论文

RRF 的分数只由名次决定，所以一条表格数字行只要在向量和 BM25 两路都排得靠前，
就能压住真正描述网络结构的那一段。

交叉编码器（cross-encoder）把 (查询, 段落) **拼成一对**送进 Transformer，
靠交叉注意力直接读它们的相互作用 —— 这是「读了再判断」。
而向量检索是双编码器：查询和段落各自独立编码，从不互相看，最后只算一个余弦值。

重排之后，上例变成：

    重排 #1   the TAR module captures long-term temporal ...   <- 升到第一
    重排 #8   LSTKC++ [37] T-PAMI 2025 ...                     <- 表格行被降下去
    重排 #10  一条 ViV-ReID 的片段                               <- 压到最后

## 为什么它只能做「重排」，不能做「检索」

交叉编码器无法预计算索引 —— 每个 (查询, 段落) 对都要跑一次完整前向。
414 个段落就要跑 414 次，而向量检索是 1 次编码 + 414 次点积。
所以它的位置只能是：候选池已经很小（这里是 10 条）之后，用它精排。

二段式检索的标准结构就在这里：**召回求全，重排求精**。

## 为什么走 ONNX 而不是 sentence-transformers

环境里已经有 onnxruntime / tokenizers / numpy（随 Chroma 装进来的），**没有 torch**。
装 torch 要下 2.5 GB，还会牵动 lockfile 和两个 venv 的一致性
（见 lessons 里环境不一致那段）。ONNX int8 模型 266 MB，推理不需要任何额外依赖。

## 模型权重不进 git

266 MB 的二进制不该进版本库。权重放在 resource/models/ 下、已加 .gitignore，
用 scripts/fetch-reranker.ps1 下载。**因此全新 clone 出来的仓库是没有这个模型的** ——
所以这个模块必须能做到「模型不在就静默降级回纯混合检索」，而不是报错崩掉。
下面 _load() 的返回值设计就是为了这件事。
"""

from __future__ import annotations

import logging
import os
import threading

import numpy as np

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# 重排模型目录。默认指向 int8 版本；换模型只要改这个环境变量。
RERANK_MODEL_DIR = os.environ.get(
    "RERANK_MODEL_DIR", os.path.join("resource", "models", "bge-reranker-base-int8")
)

# 查询 + 段落拼起来的总长度上限。语料块是 800 字符（约 250 token），
# 加上查询也远不到 512，所以正常情况下不会触发截断。
MAX_LENGTH = 512

# ONNX 文件名按优先级找：int8 -> fp16 -> fp32。
_ONNX_CANDIDATES = ("model_int8.onnx", "model_quantized.onnx", "model_fp16.onnx", "model.onnx")

_lock = threading.Lock()
_state: dict = {"tokenizer": None, "session": None, "loaded": False}


def _onnx_path(model_dir: str) -> str | None:
    for name in _ONNX_CANDIDATES:
        candidate = os.path.join(model_dir, "onnx", name)
        if os.path.exists(candidate):
            return candidate
    return None


def _load():
    """惰性加载分词器和 ONNX 会话。模型缺失时返回 (None, None)，不抛异常。"""
    if _state["loaded"]:
        return _state["tokenizer"], _state["session"]

    with _lock:
        if _state["loaded"]:
            return _state["tokenizer"], _state["session"]

        _state["loaded"] = True  # 无论成功失败都只尝试一次

        tokenizer_path = os.path.join(RERANK_MODEL_DIR, "tokenizer.json")
        onnx_path = _onnx_path(RERANK_MODEL_DIR)

        if not os.path.exists(tokenizer_path) or onnx_path is None:
            logger.warning(
                "重排模型未就绪（%s），检索将只使用混合召回。"
                "运行 scripts/fetch-reranker.ps1 下载。",
                RERANK_MODEL_DIR,
            )
            return None, None

        try:
            from tokenizers import Tokenizer
            import onnxruntime as ort
        except ImportError as exc:
            logger.warning("重排依赖缺失（%s），检索将只使用混合召回。", exc)
            return None, None

        try:
            tokenizer = Tokenizer.from_file(tokenizer_path)
            tokenizer.enable_truncation(max_length=MAX_LENGTH)
            tokenizer.enable_padding()
            session = ort.InferenceSession(
                onnx_path, providers=["CPUExecutionProvider"]
            )
        except Exception as exc:
            logger.warning("重排模型加载失败（%s），检索将只使用混合召回。", exc)
            return None, None

        logger.info("重排模型已加载：%s", onnx_path)
        _state["tokenizer"] = tokenizer
        _state["session"] = session
        return tokenizer, session


def available() -> bool:
    tokenizer, session = _load()
    return tokenizer is not None and session is not None


def score_pairs(query: str, passages: list[str]) -> np.ndarray:
    """给每个 (查询, 段落) 对打一个分。返回原始 logit（越大越相关）。

    注意这里是**原始 logit**，不是概率。bge-reranker 系列训练时用单标签
    交叉熵，输出一个 logit，官方用法也是直接比较 logit 大小。
    小于 0 不代表「不相关」——实测里正确答案拿 5.48，同一问题下另一条
    确实相关的段落只有 -1.02，负数只是相对更低。
    """
    tokenizer, session = _load()
    if tokenizer is None or session is None:
        raise RuntimeError("rerank model not loaded")

    encoded = tokenizer.encode_batch([[query, passage] for passage in passages])
    feed = {
        "input_ids": np.array([e.ids for e in encoded], dtype=np.int64),
        "attention_mask": np.array([e.attention_mask for e in encoded], dtype=np.int64),
    }
    input_names = {i.name for i in session.get_inputs()}
    if "token_type_ids" in input_names:
        feed["token_type_ids"] = np.array([e.type_ids for e in encoded], dtype=np.int64)

    logits = session.run(None, feed)[0]
    return np.asarray(logits).reshape(-1)


def rerank_hits(query: str, hits: list[tuple[Document, str, float | None]],
                top_n: int) -> list[tuple[Document, str, float | None]]:
    """按交叉编码器打分重排，并截断到 top_n。

    hits 是 hybrid_search 的返回格式：[(Document, 来源标记, 向量分数或 None)]。
    返回同格式的新列表；重排分数不往外暴露，见下方「为什么不改输出格式」。

    模型不可用时**原样返回前 top_n 条**：降级成纯混合检索，而不是让整个检索失败。
    """
    if not hits:
        return hits

    tokenizer, session = _load()
    if tokenizer is None or session is None:
        return hits[:top_n]

    candidates = hits
    try:
        scores = score_pairs(query, [doc.page_content for doc, _o, _s in candidates])
    except Exception as exc:
        logger.warning("重排失败（%s），退回混合检索顺序。", exc)
        return hits[:top_n]

    order = sorted(range(len(candidates)), key=lambda i: -float(scores[i]))
    return [candidates[i] for i in order[:top_n]]


# ---- 为什么不把重排分数写进工具输出 ------------------------------------------
#
# 工具输出格式是评估脚本的解析对象。今天已经因为「改了格式但没改正则」导致
# retrieval_recall 静默从 0.917 掉到 0.25（见 run_eval.py 顶部注释）。
#
# 重排改变的是**顺序和条数**，不是「这条凭什么被返回」；后者已经由现有的
# relevance / exact terms 标记表达了。再塞一个 logit 进去，收益是模型多了
# 一个它用不上的数字，代价是测量链路上又多一个会失配的正则。
#
# 所以这里刻意**不**改输出格式。重排分数留在内部，只用于排序。
