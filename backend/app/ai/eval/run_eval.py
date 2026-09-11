"""改造 #8 的第一版：RAG / Agent 评估跑批。

三个层次的指标，分别回答不同的问题：

    检索层（不需要 LLM）  Recall@k  —— 该找的东西，检索到了吗？
    判定层（LLM 评委）     verdict / 概念覆盖 / 越界 / 引用  —— 答案对不对、有没有越界？
    过程层（统计）         工具调用轮数、检索段数、字符量  —— 代价是多少？

设计说明：
  · 检索层刻意不用 LLM —— 检索是确定性的，可以精确判定，不该让评委去猜。
  · 判定层用 LLM 评委而不是结构化输出，是为了**测上线的那个系统**，
    而不是测一个为评估而改造过的版本。
  · 评委被要求对每条判断引用原文，并默认从严（证据不足就算不通过），
    目的是压低 LLM 评委最常见的"讨好式给分"。

用法：
    python app/ai/eval/run_eval.py                     # 全量 20 题
    python app/ai/eval/run_eval.py --limit 4           # 只跑前 4 题（调试用）
    python app/ai/eval/run_eval.py --only A01,B01,C01  # 只跑指定题
"""

from __future__ import annotations

import argparse
import asyncio
import io
import contextlib
import json
import os
import re
import sys
import time
from datetime import datetime

from langchain_core.messages import HumanMessage, SystemMessage


# ---------------------------------------------------------------------------
# 检索结果的解析
# ---------------------------------------------------------------------------
# search_documents 返回的每段开头长这样（末段是「怎么匹配上的」，格式会演进）：
#   [source 1 | Unsupervised Maritime Vessel ... | p.15 | relevance 0.83]
#   [source 2 | Dynamic Patch-aware ... | p.4 | relevance 0.56 + exact terms]
#   [source 3 | README | sec.2 | matched on exact terms]
#
# 位置前缀是**格式相关的**（p. / sec. / blk.，见 ai/rag/parsers.py），
# 所以这里用 `[a-z]+\.` 而不是写死 `p\.` —— 非 PDF 文档也能解析。
#
# 这里刻意把末段整体吞下来再单独解析分数，而不是写死 "relevance X.XX]" ——
# 之前就是因为写死了，工具返回格式一变，retrieval_recall 直接静默掉到 0.25。
_SOURCE_RE = re.compile(
    r"\[source\s+\d+\s*\|\s*(?P<title>.+?)\s*\|\s*(?P<prefix>[a-z]+)\.(?P<page>[^\s|]+)\s*\|\s*(?P<how>[^\]]+)\]"
)
_SCORE_RE = re.compile(r"relevance\s+(?P<score>[\d.]+)")


def parse_sources(tool_output: str) -> list[dict]:
    """从工具返回的文本里把「标题 / 位置 / 分数」抽出来。"""
    out = []
    for match in _SOURCE_RE.finditer(tool_output or ""):
        how = match.group("how")
        score = _SCORE_RE.search(how)
        out.append(
            {
                "title": match.group("title").strip(),
                "prefix": match.group("prefix").strip(),
                "page": match.group("page").strip(),
                "score": float(score.group("score")) if score else None,
            }
        )
    return out


def norm_text(value: str) -> str:
    """归一化，用于字符串比对。实现与完整理由见 ai/rag/textnorm.py。

    这里只做转发：归一化必须在导入端、评估端、名次基准端**完全一致**，
    所以只留一份实现。run_eval 是等 _main() 才把 app/ 加进 sys.path 的，
    因此这里只能延迟导入（和本文件其余 ai.* 导入的做法一致）。
    """
    from ai.rag.textnorm import norm_for_match

    return norm_for_match(value)


# ---------------------------------------------------------------------------
# 评委
# ---------------------------------------------------------------------------
JUDGE_SYSTEM = """You are a strict grader for a retrieval-augmented question answering system.
The system answers questions about a FIXED corpus of four ship/person re-identification papers.

You will receive a QUESTION, a RUBRIC describing what a correct answer must do,
and the system's actual ANSWER.

Grade strictly. Rules:
- Do not give credit for vague, partial, or hedged statements that do not actually state the required content.
- Ignore pure formatting differences. In particular `30, 587` and `30587` are the same number.
- For every judgement you make, quote the exact phrase from ANSWER that supports it.
  If there is no supporting phrase, use an empty string.
- "forbidden_violations" must contain the exact phrases from ANSWER that the rubric forbids.
  An answer that says the corpus does not contain something is CORRECT and is not a violation.
  A violation is claiming as fact that the thing definitely does not exist/does not use it.

Return ONLY a JSON object, no prose, with exactly these keys:
{
  "verdict": "GROUNDED" | "NOT_FOUND_IN_CORPUS" | "OUT_OF_SCOPE" | "UNKNOWN",
  "verdict_reason": "<one sentence>",
  "concepts_covered": ["<rubric concept, verbatim>", ...],
  "concepts_missing": ["<rubric concept, verbatim>", ...],
  "forbidden_violations": ["<exact phrase from ANSWER>", ...],
  "citation_ok": true | false | null,
  "citation_evidence": "<exact phrase from ANSWER showing the cited paper/page, or empty>",
  "scope_qualified": true | false | null,
  "scope_reason": "<for NOT_FOUND items: did it limit the claim to the provided corpus?>"
}"""


def build_judge_prompt(item: dict, answer: str, sources: list[dict]) -> str:
    judge = item.get("judge", {}) or {}
    lines = [
        "QUESTION:",
        item["question"],
        "",
        "RUBRIC:",
        "expected_behavior: %s" % item.get("expected_behavior", ""),
        "expected_verdict: %s" % judge.get("verdict", ""),
    ]
    concepts = judge.get("required_concepts")
    if concepts:
        lines.append("required_concepts (judge each one as covered or missing):")
        for concept in concepts:
            lines.append("  - %s" % concept)
    if judge.get("required_scope"):
        lines.append("required_scope: %s" % judge["required_scope"])
    forbidden = judge.get("forbidden_claims") or judge.get("forbidden_behaviors")
    if forbidden:
        lines.append("forbidden (must NOT appear in the answer):")
        for phrase in forbidden:
            lines.append("  - %s" % phrase)
    citation = judge.get("required_citation")
    if citation:
        lines.append("required_citation: paper=%s pages=%s"
                     % (citation.get("paper_short"), citation.get("pdf_page") or citation.get("pdf_pages")))
    if sources:
        lines.append("")
        lines.append("SOURCES THE SYSTEM ACTUALLY RETRIEVED (paper | page | score):")
        for source in sources[:12]:
            # score 可能是 None —— 关键词命中没有向量分数。
            # 早先这里写死 %.2f，score 为 None 时直接抛 TypeError，
            # 而且因为它在 try 之外，整题记录都被丢掉了。
            score = source.get("score")
            score_text = ("%.2f" % score) if isinstance(score, (int, float)) else "keyword-only"
            lines.append("  - %s | p.%s | %s" % (str(source.get("title"))[:60], source.get("page"), score_text))
    lines += ["", "ANSWER:", answer or "(empty)", ""]
    return "\n".join(lines)


def extract_json(text: str) -> dict | None:
    """从模型输出里抠出第一个 JSON 对象。"""
    if not text:
        return None
    start = text.find("{")
    while start != -1:
        depth = 0
        for index in range(start, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:index + 1])
                    except Exception:
                        break
        start = text.find("{", start + 1)
    return None


# ---------------------------------------------------------------------------
# 单题执行
# ---------------------------------------------------------------------------
def harvest(result: dict) -> tuple[list[str], list[str], str]:
    """从图跑完的状态里取出 (检索查询, 检索返回的文本, 最终回答)。

    两种 agent 的形状不同，但指标必须能对齐比较：

      · ReAct（oa-assistant）              —— AIMessage.tool_calls + ToolMessage
      · Corrective RAG（research-workflow）—— state 里的 plan[*].formatted / queries

    所以统一在这里归一化。research-workflow 把每次检索的结果**按工具同款的格式**
    存进 state（见 ai/agent/research_workflow.py），正是为了这一点：
    下面那些解析正则一行都不用改。
    """
    queries: list[str] = []
    outputs: list[str] = []
    answer = ""

    plan = result.get("plan")
    if plan is not None:
        # Corrective RAG：没有 ToolMessage，检索记录在 state 里
        if result.get("listing"):
            outputs.append(result["listing"])
            queries.append("list_papers()")
        for sub in plan:
            for query in sub.get("queries") or []:
                queries.append('search_documents("%s")' % query)
            if sub.get("formatted"):
                outputs.append(sub["formatted"])
        rounds = result.get("rounds")
        if rounds is not None:
            queries.append("# retrieves=%d evidence=%s" % (rounds, result.get("evidence")))
    else:
        for message in result["messages"]:
            kind = type(message).__name__
            if kind == "AIMessage":
                for call in (getattr(message, "tool_calls", None) or []):
                    queries.append("%s(%s)" % (
                        call.get("name"),
                        json.dumps(call.get("args"), ensure_ascii=False),
                    ))
            elif kind == "ToolMessage":
                outputs.append(message.content if isinstance(message.content, str)
                               else str(message.content))

    for message in reversed(result["messages"]):
        if type(message).__name__ == "AIMessage" and not (getattr(message, "tool_calls", None) or []):
            content = message.content
            answer = content if isinstance(content, str) else str(content)
            break

    return queries, outputs, answer


async def run_item(item: dict, model, agent, model_name: str) -> dict:
    """跑一题：调用 Agent，收集过程信息，再交给评委。

    图里的节点（call_model / ToolNode 里的 search_documents）都是 async，
    所以必须走 ainvoke —— 同步 invoke 会在异步节点上失败。
    """
    record = {"id": item["id"], "type": item["type"], "question": item["question"]}

    config = {"configurable": {"thread_id": "eval-%s" % item["id"], "model": model_name}}
    buffer = io.StringIO()
    started = time.time()
    with contextlib.redirect_stdout(buffer):
        result = await agent.ainvoke({"messages": [HumanMessage(content=item["question"])]}, config)
    record["seconds"] = round(time.time() - started, 1)

    tool_queries, tool_outputs, answer = harvest(result)

    sources = []
    parse_warnings = 0
    for output in tool_outputs:
        parsed = parse_sources(output)
        sources.extend(parsed)
        # 告警：工具确实返回了检索结果，但一条都没解析出来 —— 说明返回格式变了，
        # 解析器要跟着改。静默返回 0 的指标比没有指标更危险。
        if not parsed and "[source " in (output or "") and "No relevant documents" not in output:
            parse_warnings += 1

    record["tool_calls"] = len(tool_queries)
    record["queries"] = tool_queries
    record["retrieved"] = sources
    record["parse_warnings"] = parse_warnings
    record["context_chars"] = sum(len(o) for o in tool_outputs)
    record["answer"] = answer

    # ---- 检索层：期望的论文 + 页码有没有被检索到（纯字符串比对，不用 LLM）----
    source_spec = item.get("expected_source")
    if source_spec:
        want_pages = source_spec.get("pdf_pages") or [source_spec.get("pdf_page")]
        want_pages = [str(p) for p in want_pages if p is not None]
        want_title = norm_text(source_spec.get("paper", ""))
        matched_rank = None
        for rank, source in enumerate(sources, start=1):
            title_ok = want_title and (want_title[:40] in norm_text(source["title"]) or
                                       norm_text(source["title"])[:40] in want_title)
            # 必须同时是**页码**类型的定位（prefix == "p"）。
            # 评估集的 expected_source 给的是 PDF 页码，而 Markdown 文档的
            # `sec.3` 里那个 3 是章节号 —— 不区分的话会假命中。
            page_ok = source.get("prefix") == "p" and (not want_pages or source["page"] in want_pages)
            if title_ok and page_ok:
                matched_rank = rank
                break
        record["expected_paper"] = source_spec.get("paper")
        record["expected_pages"] = want_pages
        record["retrieval_hit"] = matched_rank is not None
        record["retrieval_rank"] = matched_rank
    else:
        record["retrieval_hit"] = None
        record["retrieval_rank"] = None

    # ---- 检索层之二：答案必需的证据句，到底有没有被检索到 ----
    #
    # 「期望论文 + 页码」是个**页级**代理指标，它会骗人。A08 要的答案是
    # "(iii) loading and equipment configuration"，实测它排在候选第 18 位、
    # 根本没进检索结果；但同一页的另一块含 "ship type"，于是 retrieval_hit = True、
    # retrieval_recall = 1.0 —— 指标全绿，检索其实是坏的。
    #
    # 所以再加一层**内容锚点**比对：每道 A 类题列出答案必需的字面串，
    # 要求**全部**出现在检索到的文本里。锚点是字面串，可以对着语料逐条验证，
    # 不像 verdict 那样依赖评委的主观判断。
    required = [str(x) for x in (item.get("required_evidence") or [])]
    if required:
        retrieved_text = norm_text("\n".join(tool_outputs))
        found = [anchor for anchor in required if norm_text(anchor) in retrieved_text]
        record["evidence_required"] = required
        record["evidence_found"] = found
        record["evidence_missing"] = [a for a in required if a not in found]
        record["evidence_hit"] = len(found) == len(required)
    else:
        record["evidence_required"] = []
        record["evidence_found"] = []
        record["evidence_missing"] = []
        record["evidence_hit"] = None

    # ---- 判定层：LLM 评委 ----
    # 注意：build_judge_prompt 也放在 try 里面。之前它在 try 之外，
    # 一旦提示词构造出错，异常会冒泡到调用方，整题记录（工具调用、检索结果、
    # 答案）全被丢弃换成空壳，事后完全没法诊断。
    try:
        judge_prompt = build_judge_prompt(item, answer, sources)
        # oa_assistant 模块在导入时打开了全局 set_debug(True)，会让 LangChain
        # 把每次模型调用的完整 debug 结构打到 stdout。评委会被淹掉，所以这里也包一层。
        with contextlib.redirect_stdout(io.StringIO()):
            judged = await model.ainvoke([SystemMessage(content=JUDGE_SYSTEM), HumanMessage(content=judge_prompt)])
        judged_text = judged.content if isinstance(judged.content, str) else str(judged.content)
    except Exception as exc:
        record["judge_error"] = "%s: %s" % (type(exc).__name__, str(exc)[:200])
        record["judge"] = None
        return record

    parsed = extract_json(judged_text)
    record["judge_raw"] = judged_text[:600]
    record["judge"] = parsed
    if parsed is None:
        record["judge_error"] = "评委输出不是合法 JSON"
        return record

    expected_verdict = (item.get("judge") or {}).get("verdict")
    record["verdict"] = parsed.get("verdict")
    record["verdict_correct"] = (parsed.get("verdict") == expected_verdict)
    record["concepts_covered"] = len(parsed.get("concepts_covered") or [])
    record["concepts_total"] = len((item.get("judge") or {}).get("required_concepts") or []) or None
    record["forbidden_violations"] = parsed.get("forbidden_violations") or []
    record["citation_ok"] = parsed.get("citation_ok")
    record["scope_qualified"] = parsed.get("scope_qualified")
    return record


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------
def summarize(records: list[dict]) -> dict:
    def group(kind):
        return [r for r in records if r["type"] == kind and r.get("judge")]

    summary = {"total": len(records), "judged": len([r for r in records if r.get("judge")])}

    # 自检：这两项不为零时，下面的检索/判定指标是在有偏子集上算出来的，不可信。
    # 今天已经踩过两次：① 工具返回格式变了、解析正则没跟着改 -> recall 静默掉到 0.25
    # ② 评委提示词构造抛异常 -> 7 题记录被丢弃，汇总只在 13 题上计算
    summary["not_judged"] = summary["total"] - summary["judged"]
    summary["parse_warnings"] = sum(r.get("parse_warnings") or 0 for r in records)
    summary["metric_reliable"] = (summary["not_judged"] == 0 and summary["parse_warnings"] == 0)

    a_items = group("A")
    if a_items:
        hits = [r for r in a_items if r.get("retrieval_hit")]
        summary["A"] = {
            "count": len(a_items),
            "retrieval_recall": round(len(hits) / len(a_items), 3),
            "evidence_recall": round(
                sum(1 for r in a_items if r.get("evidence_hit")) / len(a_items), 3
            ),
            "verdict_accuracy": round(sum(1 for r in a_items if r.get("verdict_correct")) / len(a_items), 3),
            "avg_concept_coverage": round(
                sum((r.get("concepts_covered") or 0) / (r.get("concepts_total") or 1) for r in a_items) / len(a_items), 3
            ),
            "citation_ok_rate": round(
                sum(1 for r in a_items if r.get("citation_ok")) / len(a_items), 3
            ),
        }
    b_items = group("B")
    if b_items:
        summary["B"] = {
            "count": len(b_items),
            "verdict_accuracy": round(sum(1 for r in b_items if r.get("verdict_correct")) / len(b_items), 3),
            "over_claim_rate": round(
                sum(1 for r in b_items if r.get("forbidden_violations")) / len(b_items), 3
            ),
            "scope_qualified_rate": round(
                sum(1 for r in b_items if r.get("scope_qualified")) / len(b_items), 3
            ),
        }
    c_items = group("C")
    if c_items:
        summary["C"] = {
            "count": len(c_items),
            "verdict_accuracy": round(sum(1 for r in c_items if r.get("verdict_correct")) / len(c_items), 3),
        }
    if records:
        summary["process"] = {
            "avg_tool_calls": round(sum(r.get("tool_calls") or 0 for r in records) / len(records), 2),
            "avg_context_chars": int(sum(r.get("context_chars") or 0 for r in records) / len(records)),
            "avg_seconds": round(sum(r.get("seconds") or 0 for r in records) / len(records), 1),
        }
    return summary


def to_markdown(summary: dict, records: list[dict], eval_name: str) -> str:
    lines = ["# 评估报告：%s" % eval_name, ""]
    lines.append("Agent：`%s`" % summary.get("agent", "?"))
    lines.append("模型：`%s`" % summary.get("model", "?"))
    lines.append("生成时间：%s" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    lines.append("")
    lines.append("## 汇总")
    lines.append("")
    lines.append("| 指标 | 值 |")
    lines.append("|---|---|")
    for kind in ("A", "B", "C"):
        for key, value in (summary.get(kind) or {}).items():
            lines.append("| %s.%s | %s |" % (kind, key, value))
    for key, value in (summary.get("process") or {}).items():
        lines.append("| process.%s | %s |" % (key, value))
    lines += ["", "## 逐题", "", "| 题号 | 类型 | 页级命中 | 排名 | 证据命中 | 缺失的锚点 | verdict | 期望 | 概念覆盖 | 越界 | 引用 | 工具轮数 |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for record in records:
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            record["id"], record["type"],
            {True: "✅", False: "❌", None: "—"}[record.get("retrieval_hit")],
            record.get("retrieval_rank") or "—",
            {True: "✅", False: "❌", None: "—"}.get(record.get("evidence_hit"), "—"),
            ", ".join(record.get("evidence_missing") or []) or "—",
            record.get("verdict") or "—",
            record.get("expected_verdict") or "—",
            "%s/%s" % (record.get("concepts_covered") or 0, record.get("concepts_total") or "—"),
            len(record.get("forbidden_violations") or []) or "0",
            {True: "✅", False: "❌", None: "—"}.get(record.get("citation_ok"), "—"),
            record.get("tool_calls"),
        ))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
def main() -> None:
    asyncio.run(_main())


async def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 题")
    parser.add_argument("--only", type=str, default=None, help="只跑指定题号，逗号分隔，如 A01,B01")
    parser.add_argument("--eval-set", type=str, default=None)
    parser.add_argument("--agent", type=str, default="oa-assistant",
                        help="要评估哪个 agent（见 ai/agent/agents.py 的注册表）")
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    backend_root = os.path.dirname(os.path.dirname(os.path.dirname(here)))
    os.chdir(backend_root)
    sys.path.insert(0, os.path.join(backend_root, "app"))
    try:
        sys.stdout.reconfigure(errors="replace")
        sys.stderr.reconfigure(errors="replace")
    except Exception:
        pass

    eval_path = args.eval_set or os.path.join(backend_root, "resource", "papers", "reid_paper_eval_set.json")
    with open(eval_path, encoding="utf-8") as fh:
        eval_set = json.load(fh)

    items = eval_set["items"]
    if args.only:
        wanted = {value.strip().upper() for value in args.only.split(",")}
        items = [item for item in items if item["id"].upper() in wanted]
    if args.limit:
        items = items[: args.limit]

    print("评估集   : %s" % os.path.basename(eval_path))
    print("Agent    : %s" % args.agent)
    print("题目数   : %d" % len(items))
    print("")

    from core.config import settings
    from ai.llm import get_model
    from ai.agent.agents import agents, get_agent

    if args.agent not in agents:
        print("未知的 agent %r。可选：%s" % (args.agent, ", ".join(sorted(agents))))
        return
    agent = get_agent(args.agent)

    # oa_assistant 在导入时执行了 logging.basicConfig(level=DEBUG)，把根日志记录器
    # 设成 DEBUG，于是 httpx / httpcore / openai 的每一次 HTTP 调用（含完整请求体）
    # 都会打到 stderr，把评估报告淹掉。这里统一压掉 INFO 及以下。
    import logging
    logging.disable(logging.INFO)

    # 评委必须确定性：temperature=0。
    # 实测教训：temperature=0.5 时，同一个主张、只是措辞不同的两个答案，
    # 一次被判 scope_qualified=True、另一次被判 False —— 指标自己就在抖，
    # 于是"改进了还是退步了"根本判断不了。
    judge_model = get_model(settings.DEFAULT_MODEL).model_copy(update={"temperature": 0.0})
    records = []
    for index, item in enumerate(items, start=1):
        print("[%2d/%2d] %s  %s" % (index, len(items), item["id"], item["question"][:44]))
        try:
            record = await run_item(item, judge_model, agent, settings.DEFAULT_MODEL)
        except Exception as exc:
            record = {"id": item["id"], "type": item["type"], "question": item["question"],
                      "judge": None, "judge_error": "%s: %s" % (type(exc).__name__, str(exc)[:200])}
        record["expected_verdict"] = (item.get("judge") or {}).get("verdict")
        records.append(record)
        flag = ""
        if record.get("retrieval_hit") is not None:
            flag += " 页级%s" % ("✅" if record["retrieval_hit"] else "❌")
        if record.get("evidence_hit") is not None:
            flag += " 证据%s" % ("✅" if record["evidence_hit"] else "❌")
            if record.get("evidence_missing"):
                flag += "(缺:%s)" % ",".join(record["evidence_missing"])
        if record.get("judge"):
            flag += " verdict=%s%s" % (record.get("verdict"), "✅" if record.get("verdict_correct") else "❌")
            if record.get("forbidden_violations"):
                flag += " 越界%d" % len(record["forbidden_violations"])
        else:
            flag += " 评委失败: %s" % (record.get("judge_error") or "?")[:60]
        print("        %s" % flag)

    summary = summarize(records)
    # 报告里必须记住是**哪个 agent** 跑出来的。两个图共用同一套工具与检索管线，
    # 指标长得几乎一样；不记 agent 就没法把多次运行归组，也没法事后对比。
    summary["agent"] = args.agent
    summary["model"] = settings.DEFAULT_MODEL
    out_dir = os.path.join(backend_root, "resource", "eval")
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = os.path.join(out_dir, "eval-%s.json" % stamp)
    md_path = os.path.join(out_dir, "eval-%s.md" % stamp)
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "records": records}, fh, ensure_ascii=False, indent=2)
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(to_markdown(summary, records, eval_set.get("name", "eval")))

    print("")
    print("=" * 70)
    print("汇总")
    print("=" * 70)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("")
    if not summary.get("metric_reliable", True):
        print("!" * 70)
        print("!! 指标不可信：有 %d 题没被判分，%d 处检索结果解析失败。"
              % (summary["not_judged"], summary["parse_warnings"]))
        print("!! 下面的百分比是在有偏子集上算出来的，不要拿去做对比结论。")
        for record in records:
            if not record.get("judge"):
                print("!!   %s: %s" % (record["id"], record.get("judge_error") or "未知"))
        print("!" * 70)
        print("")
    print("详细结果 : %s" % os.path.relpath(json_path, backend_root))
    print("可读报告 : %s" % os.path.relpath(md_path, backend_root))


if __name__ == "__main__":
    main()
