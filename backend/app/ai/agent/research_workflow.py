"""改造 #3：Research Workflow —— Corrective RAG 变体。

设计（含为什么需要它、以及三条来自现有数据的证据）见
`reference/transformation-03-research-workflow-design.md`。这里只写实现要点。

## 图结构

    analyze ─→ retrieve ─→ assess ─┬─(还有 insufficient ∧ 预算未耗尽)─→ refine ─┐
                                   │                                            │
                                   │  ←─────────────────────────────────────────┘
                                   └─(全部已定论)──────────────────→ synthesize ─→ END

## 和原来的 ReAct agent 差在哪

原来的图只有 `model ⇄ tools` 两个节点，「要不要再查一次」完全交给模型，
实测同一批问题里轮数从 2 到 12 不等（B04 为了确认「没找到」连查 12 次、9 秒）。

这里把三件事从「模型的自由裁量」变成「图上的结构」：

  1. **检索轮数有上界**：`MAX_RETRIEVE_ROUNDS`，写在图上，不由模型决定。
  2. **「未找到」是判出来的，不是说的**：`assess` 节点按阈值给每个子问题定三态；
     只有在**预算真的耗尽**之后，才允许把 `insufficient` 降级成 `absent`。
     也就是说，系统只有在试够了之后才敢说「库里没有」。
  3. **查询改写成为必须**：首次检索不过阈值就一定会走 `refine`，而不是"模型碰巧会改写"。
     （实测：用评估题原文检索，12 道 A 类题只有 8 道能拿齐答案锚点；靠改写才到 12 道。
     这说明改写是真本事，但原来它没有任何保证。）

## 三态的含义（这条是整个设计里最重要的）

    sufficient    找到了支撑        → 可以正常回答 + 引用
    absent        查够了，语料里确实没有 → 只能说「当前论文库中未涉及」
    insufficient  这次没查够        → 内部状态，不给用户看，回去改写查询

`absent` **绝不能升级成「不存在」** —— 库里没有 ≠ 世上没有。这是从 baseline 的
「无根据的否定」缺陷一路继承下来的底线行为。

## ⚠️ 已知局限

**一、相关性判断不干净。** 交叉编码器在 20 题评估集上仍然错 2 题（A06 的 -0.314 与
B05 的 0.794 交叉）。这不是实现问题，是任务本身的性质：「这段材料能不能回答这个问题」
是个语义判断，一个 (问题, 段落) 打分函数做不干净。所以阈值被刻意取成保守值，
只用来触发「明显不相关就再查一轮」，**最终结论仍由 `synthesize` 读材料得出**。

**二、要点覆盖依赖 `analyze` 猜对措辞。** 覆盖判定是拿 `facets` 去**逐字比对**检索到的
文本，所以 `analyze` 如果猜的措辞和论文原文对不上（比如写 "labeling process" 而论文写
"annotation pipeline"），就会一直判缺失、白烧预算。预算上限挡住了无限循环，
但代价还是会付。缓解手段是 prompt 里明确要求「用论文自己的词汇、短名词短语」，
以及把要点数限制在 2~3 个。

**三、要点缺失不会导致拒答。** 这是刻意的：预算耗尽时只要材料**相关**就一定判
`sufficient`，让 synthesize 尽量答，答不全也比拒答强。反过来如果让缺失也判 `absent`，
系统就会变成「明明库里有却拒答」—— 比原来的 ReAct 版更糟。
"""

from __future__ import annotations

import logging
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from ai.agent.checkpointer import get_checkpointer
from ai.llm import get_model, settings
from ai.rag.pipeline import (
    EVIDENCE_RETRY_THRESHOLD,
    NO_HITS_MESSAGE,
    evidence_score,
    format_hits,
    retrieve,
)
from ai.rag.textnorm import norm_for_match
from ai.tools.research_tools import all_titles, list_papers, resolve_paper_sources

logger = logging.getLogger(__name__)


# ---- 硬约束 ------------------------------------------------------------------
# 检索轮数上限。**必须写在图上**：模型不知道"已经试过几次、该由谁负责"。
# 取 3 的理由：第一轮用 analyze 生成的查询，之后最多再改写两次；
# 实测正常的 A 类问题 1 轮就够，B 类（语料确实没有）才是真正需要多轮确认的。
MAX_RETRIEVE_ROUNDS = 3


# ---- 结构化输出：问题分析 -----------------------------------------------------
class SubQuestion(BaseModel):
    """一个待检索的子问题。"""

    question: str = Field(description="子问题，用中文复述要查什么")
    query: str = Field(
        description=(
            "第一次检索用的查询。写成**英文**，并尽量贴近论文的措辞"
            "（论文是英文写的，中英跨语检索实测不如直接给英文术语稳）。"
            "包含关键实体和同义词。"
        )
    )
    facets: list[str] = Field(
        default_factory=list,
        description=(
            "答案必须包含的要点，2~3 个，每个用**论文里的字面术语**（英文、短名词短语）。"
            "例：问「DPSM 怎么决定保留多少 token」→ ['first-order difference', 'kmin']；"
            "问「TAR 怎么组织历史特征」→ ['instance bank', 'identity-view pair']。"
            "这些词会拿去**逐字比对**检索到的材料，用来判断答案的几块齐不齐，"
            "所以必须是能在论文原文里原样出现的写法：不要写整句、不要写中文、"
            "不要同义改写（写 'labeling process' 就查不到 'annotation pipeline'）。"
            "不确定确切措辞时，给最有把握逐字出现的那几个词。"
        ),
    )
    paper: str | None = Field(
        default=None,
        description="用户明确点名了某篇论文时，填该论文标题的一部分；否则留空。",
    )


class ResearchPlan(BaseModel):
    sub_questions: list[SubQuestion] = Field(
        description="把用户问题拆成 1~4 个可独立检索的子问题"
    )
    needs_paper_list: bool = Field(
        default=False,
        description="问题问的是「库里有哪几篇论文 / 作者是谁 / 年份」这类元信息时填 true",
    )


ANALYZE_PROMPT = """You are the planning step of a research assistant.

Break the user's question into 1-4 sub-questions that can each be answered by searching a
knowledge base of four ship/person re-identification papers.

Rules:
- Most questions need only ONE sub-question. Do not split just to look thorough.
- Only split when the question genuinely asks several separate things
  (e.g. "how is the annotation pipeline built, and how many images does it yield?").
- Write each retrieval query in ENGLISH, close to how a paper would phrase it.
  A Chinese question is fine, but the query should use the paper's own vocabulary.
- For each sub-question also list 2-3 `facets`: the literal terms a complete answer must
  contain. These are matched as **substrings** against retrieved text, so they must be
  spelled the way the paper spells them — short English noun phrases, no sentences, no
  Chinese, no paraphrases. Prefer terms you are confident appear verbatim
  (module names, metric names, model names, dataset names).
  A good facet set for "how does DPSM decide how many tokens to keep" is
  ["first-order difference", "kmin"] — not ["the algorithm decides k dynamically"].
- If the user names a specific paper, set `paper` for the relevant sub-questions.
- Set `needs_paper_list` only when the question is about which papers exist / their
  authors / years — that is an exact lookup, not a similarity search."""


SYNTHESIZE_PROMPT = """You are the answer step of a research assistant working on a FIXED
corpus of four ship/person re-identification papers.

You are given the material that was retrieved, plus a verdict on how complete that
retrieval is. Write the answer.

Rules:

1. Base every factual statement on the retrieved material. Never invent paper titles,
   authors, findings, figures or numbers.

2. Cite as you go: paper title + page number, exactly as they appear in the material.
   When several passages disagree, trust the one with the higher relevance score.

3. Answer the question the user actually asked, and keep the answer focused.

4. The verdict below tells you how to phrase things:

   sufficient — answer normally, with citations.
   absent     — the search ran to its budget without finding relevant material. You may
                only say that the knowledge base does not cover this. You must NOT say the
                fact does not exist, and must NOT fall back on outside knowledge.
                A corpus that stays silent is not evidence that something is false.
   partial    — some sub-questions were answered and others were not. Say clearly which
                part you could answer and which part the corpus does not cover."""


class ResearchState(TypedDict):
    """图的状态。

    除了 messages，其余字段都是**审计用**的：它们让「系统查了几轮、每个子问题
    判成了什么、依据是哪个分数」变成可断言的东西，而不是只能从对话里猜。
    """

    messages: Annotated[list[AnyMessage], add_messages]
    plan: list[dict]           # 每个子问题的检索记录（含命中与分数）
    evidence: str              # sufficient | absent | insufficient | partial
    rounds: int
    listing: str               # list_papers 的输出（问题问元信息时才有）


def _model(config: RunnableConfig, temperature: float | None = None):
    """取模型。需要确定性时用 model_copy 覆盖温度。

    get_model 是 @cache 的，直接改实例会污染全局 —— run_eval 给评委降温时用的
    也是同一个做法。
    """
    model = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    if temperature is not None:
        model = model.model_copy(update={"temperature": temperature})
    return model


def _last_question(state: ResearchState) -> str:
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            content = message.content
            return content if isinstance(content, str) else str(content)
    return ""


# ---------------------------------------------------------------------------
# 节点：analyze —— 拆解问题（概率性，1 次 LLM 调用）
# ---------------------------------------------------------------------------
async def analyze(state: ResearchState, config: RunnableConfig) -> dict:
    question = _last_question(state)
    # 规划是**结构性**判断（拆几个子问题、查询怎么写），不需要措辞多样性，
    # 反而需要可复现 —— 所以 planning 类节点统统用 temperature=0。
    # 只有最后的 synthesize 保留默认温度：那一步是措辞，本来就不该完全确定。
    planner = _model(config, temperature=0.0).with_structured_output(ResearchPlan)
    plan: ResearchPlan = await planner.ainvoke(
        [SystemMessage(content=ANALYZE_PROMPT), HumanMessage(content=question)]
    )

    sub_questions = []
    for sub in plan.sub_questions:
        sub_questions.append(
            {
                "question": sub.question,
                "query": sub.query.strip(),
                "paper": (sub.paper or "").strip() or None,
                "facets": [f.strip() for f in (sub.facets or []) if f.strip()][:4],
                "facets_found": [],
                "facets_missing": [],
                "status": "pending",
                "top1": 0.0,
                "evidence": None,
                "attempts": 0,
                "queries": [],
                "formatted": "",
                "hits": [],
            }
        )
    if not sub_questions:   # 模型没拆出子问题时的兜底：拿原问题当唯一查询
        sub_questions.append(
            {
                "question": question, "query": question, "paper": None,
                "facets": [], "facets_found": [], "facets_missing": [],
                "status": "pending", "top1": 0.0, "evidence": None, "attempts": 0,
                "queries": [], "formatted": "", "hits": [],
            }
        )

    listing = ""
    if plan.needs_paper_list:
        listing = await list_papers.ainvoke({})

    logger.info("analyze: 拆出 %d 个子问题 %s",
                len(sub_questions), [s["query"] for s in sub_questions])
    return {"plan": sub_questions, "rounds": 0, "listing": listing, "evidence": "insufficient"}


# ---------------------------------------------------------------------------
# 节点：retrieve —— 执行检索（确定性，零 LLM 调用）
# ---------------------------------------------------------------------------
async def retrieve_node(state: ResearchState, config: RunnableConfig) -> dict:
    plan = state.get("plan") or []
    rounds = (state.get("rounds") or 0) + 1

    for sub in plan:
        if sub["status"] in ("sufficient", "absent"):
            continue

        allowed = None
        if sub.get("paper"):
            allowed = await resolve_paper_sources(sub["paper"])
            if allowed is None:
                logger.info("retrieve: 论文 %r 没有匹配，不做限定", sub["paper"])

        outcome = retrieve(sub["query"], allowed_sources=allowed)
        sub["attempts"] += 1
        sub["queries"].append(sub["query"])
        sub["top1"] = round(float(outcome.vector_top1), 4)
        # 判定用的是**子问题的自然语言原文**，不是改写后的检索查询 —— 理由见
        # ai/rag/pipeline.py 的 evidence_score()：改写过的查询往里面塞满了领域词汇，
        # 拿它当相关性判据必然虚高。
        score = evidence_score(sub["question"], outcome.hits)
        sub["evidence"] = None if score is None else round(score, 3)
        sub["formatted"] = NO_HITS_MESSAGE if outcome.rejected else format_hits(outcome.hits)

        # 要点覆盖：查答案要的几块齐不齐。**跨轮累加** —— 第 1 轮捞到的要点，
        # 第 2 轮不该因为这次没命中就重新算作缺失。
        text = norm_for_match("\n".join(doc.page_content for doc, _o, _s in outcome.hits))
        found = set(sub.get("facets_found") or [])
        for facet in sub.get("facets") or []:
            if norm_for_match(facet) in text:
                found.add(facet)
        sub["facets_found"] = sorted(found)
        sub["facets_missing"] = [f for f in (sub.get("facets") or []) if f not in found]
        sub["hits"] = [
            {
                "title": doc.metadata.get("paper_title"),
                "page": doc.metadata.get("page_label"),
                "score": None if score is None else round(float(score), 3),
                "origin": origin,
            }
            for doc, origin, score in outcome.hits
        ]
        logger.info("retrieve[%d]: %r -> top1=%.4f 命中 %d 条",
                    rounds, sub["query"], outcome.vector_top1, len(outcome.hits))

    return {"plan": plan, "rounds": rounds}


# ---------------------------------------------------------------------------
# 节点：assess —— 判证据充分性（确定性，零 LLM 调用）
# ---------------------------------------------------------------------------
def assess_node(state: ResearchState, config: RunnableConfig) -> dict:
    """三态判定（确定性，零 LLM 调用）。

    ## 判据是交叉编码器，不是向量余弦

    这里曾经用「向量 top-1 分数」判，**是错的**，而且是实测推翻的：

    | 判据 | 应回答（A 类）最低 | 应拒绝（B/C 类）最高 | 错几题 |
    |---|---|---|---|
    | 向量 top-1 | 0.353 | 0.470 | 6+ |
    | 交叉编码器（对**原始问题**打分） | **-0.314** | **0.794** | **2** |

    B 类是「语料有这个话题、但没有这个事实」（例如问论文是否用 Mamba）。
    向量余弦只回答「像不像」，所以任何关于 ReID 的句子都能拿高分。
    交叉编码器把问题和段落放在一起读，分得开得多 —— 但它**也不是干净的分界线**
    （上表那 2 题就是反例），所以阈值取得很保守，见 EVIDENCE_RETRY_THRESHOLD。

    ⚠️ 还有一条差点踩进去的坑：**换成向量 top-1 去判的时候，如果拿的是改写后的
    检索查询，判定会完全失效。** 实测：`refine` 把「这四篇论文是否采用 Mamba」
    改写成 "Mamba state space model SSM architecture for ship re-identification
    person re-identification" —— 里面塞满 re-ID 词汇，在 re-ID 语料上必然高分，
    哪怕根本没有 Mamba。**改写的目标（召回）和判定的目标（忠实）是冲突的**，
    所以打分一律用未经改写的子问题原文。

    ## 两个信号，缺一不可

    这里曾经只用「相关性」。实测**不够** —— A05 / A09 就是反例：

        ReAct 版 A05 用了 4 条查询，第 3 条是
            "similarity scores proxy token sorted Diff first-order derivative splitting point k"
        才捞到锚点 `first-order difference`；
        workflow 只用 1 条泛化查询，判了 sufficient 就停 —— 答案 GROUNDED、引用正确，
        但没捞到含精确术语的那一块。

    > **「材料相关」≠「我要的东西齐了」。**

    ReAct agent 隐含的停止规则是后者（模型觉得自己能答全了才停），
    而只判相关性的 assess 实现的是前者。所以现在两个信号分工：

    | 信号 | 问的问题 | 驱动什么 |
    |---|---|---|
    | 相关性（交叉编码器） | 材料**相不相关** | 要不要继续找；预算耗尽→决定 sufficient 还是 absent |
    | 覆盖（要点逐字比对） | 答案的几块**齐不齐** | 只决定要不要**再查一轮**，绝不导出 absent |

    要点缺失不导出 absent 是关键：否则会变成「明明库里有却拒答」，比 ReAct 版更糟。

    ## 为什么这里不叫 LLM

    两个信号都是确定性的：相关性来自交叉编码器，覆盖是**逐字比对**。
    「材料到底说没说这件事」的最终判断由 `synthesize` 做（它本来就要读一遍材料），
    再插一个 LLM 节点等于为同一个判断付两次钱。
    """
    plan = state.get("plan") or []
    rounds = state.get("rounds") or 0

    for sub in plan:
        if sub["status"] in ("sufficient", "absent"):
            continue
        score = sub.get("evidence")
        relevant = score is not None and score >= EVIDENCE_RETRY_THRESHOLD
        complete = not sub.get("facets_missing")

        if relevant and complete:
            sub["status"] = "sufficient"
        elif rounds >= MAX_RETRIEVE_ROUNDS:
            # 预算耗尽时，**要点缺失绝不能导致 absent**。
            #
            #   relevant（材料相关）→ sufficient：手里有相关材料，只是没凑齐，
            #       让 synthesize 用它答，答不全也比拒答强。
            #   不相关            → absent：换着法子查够了，确实没有。
            #
            # 这条区分是必须的：如果把「要点没凑齐」也判成 absent，系统就会变成
            # 「明明库里有却拒答」—— 比原来的 ReAct 版更糟。
            sub["status"] = "sufficient" if relevant else "absent"
        else:
            sub["status"] = "insufficient"

    statuses = {sub["status"] for sub in plan}
    if statuses == {"sufficient"}:
        overall = "sufficient"
    elif statuses == {"absent"}:
        overall = "absent"
    elif "insufficient" in statuses:
        overall = "insufficient"
    else:
        overall = "partial"      # 一部分 sufficient、一部分 absent

    logger.info("assess: 轮次 %d/%d 三态=%s",
                rounds, MAX_RETRIEVE_ROUNDS,
                {sub["query"][:40]: sub["status"] for sub in plan})
    return {"plan": plan, "evidence": overall}


# ---------------------------------------------------------------------------
# 节点：refine —— 改写没查到的查询（概率性，每轮 1 次 LLM 调用）
# ---------------------------------------------------------------------------
class RefinedQueries(BaseModel):
    queries: list[str] = Field(description="为每个待查子问题给出一条新查询，顺序一一对应")


REFINE_PROMPT = """The retrieval below was not good enough. Write a BETTER query for each
pending sub-question.

There are two different reasons a sub-question can be pending, and they call for different
fixes — read the "reason" line for each one:

  · LOW RELEVANCE   — nothing relevant was found at all. Rethink the wording from scratch.
  · MISSING FACETS  — relevant material WAS found, but a specific piece is still missing.
    Aim the new query **directly at the missing terms** rather than restating the question.
    Looking for one exact term is much easier than looking for a whole question.

Strategies that actually help on this corpus (four English re-identification papers):
- Use the paper's own vocabulary instead of the user's wording. The papers say
  "annotation pipeline", "pseudo-label", "instance bank", "ground truth" — not
  "labeling process", "auto labeling", "memory", "correct answer".
- Name the concrete entities: model names (Grounding DINO, Deep SORT, Segment Anything),
  dataset names, module abbreviations (SAP, TAR, SAM, DPSM, ROA).
- Try shorter and more literal. A long natural-language question is often a worse query
  than three or four technical terms.
- To chase a missing facet, put that term in the query together with one or two words of
  context — e.g. to find "first-order difference", query
  "similarity scores sorted first-order difference splitting point k".

Give exactly one new query per pending sub-question, in the same order."""


async def refine(state: ResearchState, config: RunnableConfig) -> dict:
    plan = state.get("plan") or []
    pending = [sub for sub in plan if sub["status"] == "insufficient"]
    if not pending:
        return {"plan": plan}

    lines = []
    for index, sub in enumerate(pending, start=1):
        missing = sub.get("facets_missing") or []
        reason = "MISSING FACETS: %s" % ", ".join(missing) if missing else "LOW RELEVANCE"
        lines.append(
            "%d. sub-question: %s\n"
            "   reason: %s\n"
            "   tried already: %s\n"
            "   relevance score: %s (retry threshold %.1f)"
            % (index, sub["question"], reason,
               " / ".join(sub["queries"]) or "(none)",
               sub.get("evidence"), EVIDENCE_RETRY_THRESHOLD)
        )

    result: RefinedQueries = await _model(config, temperature=0.0).with_structured_output(
        RefinedQueries
    ).ainvoke([SystemMessage(content=REFINE_PROMPT), HumanMessage(content="\n\n".join(lines))])

    for sub, query in zip(pending, result.queries):
        if query and query.strip():
            sub["query"] = query.strip()
            sub["status"] = "pending"

    logger.info("refine: 改写 %d 条查询 %s", len(pending), [s["query"] for s in pending])
    return {"plan": plan}


# ---------------------------------------------------------------------------
# 节点：synthesize —— 综合回答（概率性，1 次 LLM 调用）
# ---------------------------------------------------------------------------
def _context_block(state: ResearchState) -> str:
    parts = []
    listing = state.get("listing") or ""
    if listing:
        parts.append("=== Knowledge base listing (exact lookup) ===\n%s" % listing)
    for index, sub in enumerate(state.get("plan") or [], start=1):
        parts.append(
            "=== Sub-question %d: %s ===\nstatus: %s\nqueries tried: %s\n\n%s"
            % (index, sub["question"], sub["status"],
               " | ".join(sub["queries"]) or "(none)",
               sub["formatted"] or "(nothing retrieved)")
        )
    return "\n\n".join(parts)


async def synthesize(state: ResearchState, config: RunnableConfig) -> dict:
    verdict = state.get("evidence") or "insufficient"
    rounds = state.get("rounds") or 0
    body = (
        "User question:\n%s\n\n"
        "Retrieval verdict: %s\n"
        "Retrieval rounds used: %d of %d\n\n"
        "Retrieved material:\n\n%s"
        % (_last_question(state), verdict, rounds, MAX_RETRIEVE_ROUNDS, _context_block(state))
    )
    response = await _model(config).ainvoke(
        [SystemMessage(content=SYNTHESIZE_PROMPT), HumanMessage(content=body)]
    )
    content = response.content if isinstance(response.content, str) else str(response.content)
    return {"messages": [AIMessage(content=content)]}


# ---------------------------------------------------------------------------
# 路由
# ---------------------------------------------------------------------------
def after_assess(state: ResearchState) -> Literal["refine", "synthesize"]:
    """还有没查够的子问题、且预算没耗尽时，回去改写查询。"""
    plan = state.get("plan") or []
    rounds = state.get("rounds") or 0
    if any(sub["status"] == "insufficient" for sub in plan) and rounds < MAX_RETRIEVE_ROUNDS:
        return "refine"
    return "synthesize"


# ---------------------------------------------------------------------------
# 组图
# ---------------------------------------------------------------------------
graph = StateGraph(ResearchState)
graph.add_node("analyze", analyze)
graph.add_node("retrieve", retrieve_node)
graph.add_node("assess", assess_node)
graph.add_node("refine", refine)
graph.add_node("synthesize", synthesize)

graph.set_entry_point("analyze")
graph.add_edge("analyze", "retrieve")
graph.add_edge("retrieve", "assess")
graph.add_conditional_edges("assess", after_assess, {"refine": "refine", "synthesize": "synthesize"})
graph.add_edge("refine", "retrieve")
graph.add_edge("synthesize", END)

def build_research_workflow():
    """编译图。由 agents.py 惰性调用（理由同 react_assistant）。"""
    compiled = graph.compile(checkpointer=get_checkpointer())
    compiled.name = "research_workflow"
    return compiled
