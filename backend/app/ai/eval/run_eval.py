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
# search_documents 返回的每段开头长这样：
#   [source 1 | Unsupervised Maritime Vessel ... | p.15 | relevance 0.83]
_SOURCE_RE = re.compile(
    r"\[source\s+\d+\s*\|\s*(?P<title>.+?)\s*\|\s*p\.(?P<page>[^\s|]+)\s*\|\s*relevance\s+(?P<score>[\d.]+)\]"
)


def parse_sources(tool_output: str) -> list[dict]:
    """从工具返回的文本里把「论文 / 页码 / 分数」抽出来。"""
    out = []
    for match in _SOURCE_RE.finditer(tool_output or ""):
        out.append(
            {
                "title": match.group("title").strip(),
                "page": match.group("page").strip(),
                "score": float(match.group("score")),
            }
        )
    return out


def norm_text(value: str) -> str:
    """归一化，用于字符串比对。

    真实语料里的数字常常带逗号和空格（例如 PDF 两端对齐造成的 `30, 587`），
    所以比对前必须把数字里的分隔符去掉，否则会产生大量假失败。
    """
    text = str(value or "").lower()
    text = re.sub(r"(?<=\d)[,\s]+(?=\d)", "", text)   # 30, 587 -> 30587
    text = re.sub(r"[\s\u00a0]+", " ", text)
    return text.strip()


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
            lines.append("  - %s | p.%s | %.2f" % (source["title"][:60], source["page"], source["score"]))
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

    tool_queries, tool_outputs, answer = [], [], ""
    for message in result["messages"]:
        kind = type(message).__name__
        if kind == "AIMessage":
            for call in (getattr(message, "tool_calls", None) or []):
                tool_queries.append("%s(%s)" % (call.get("name"), json.dumps(call.get("args"), ensure_ascii=False)))
            if not (getattr(message, "tool_calls", None) or []):
                answer = message.content if isinstance(message.content, str) else str(message.content)
        elif kind == "ToolMessage":
            tool_outputs.append(message.content if isinstance(message.content, str) else str(message.content))

    sources = []
    for output in tool_outputs:
        sources.extend(parse_sources(output))

    record["tool_calls"] = len(tool_queries)
    record["queries"] = tool_queries
    record["retrieved"] = sources
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
            if title_ok and (not want_pages or source["page"] in want_pages):
                matched_rank = rank
                break
        record["expected_paper"] = source_spec.get("paper")
        record["expected_pages"] = want_pages
        record["retrieval_hit"] = matched_rank is not None
        record["retrieval_rank"] = matched_rank
    else:
        record["retrieval_hit"] = None
        record["retrieval_rank"] = None

    # ---- 判定层：LLM 评委 ----
    judge_prompt = build_judge_prompt(item, answer, sources)
    try:
        # oa_assistant 模块在导入时打开了全局 set_debug(True)，会让 LangChain
        # 把每次模型调用的完整 debug 结构打到 stdout。评委会被淹掉，所以这里也包一层。
        with contextlib.redirect_stdout(io.StringIO()):
            judged = await model.ainvoke([SystemMessage(content=JUDGE_SYSTEM), HumanMessage(content=judge_prompt)])
        judged_text = judged.content if isinstance(judged.content, str) else str(judged.content)
    except Exception as exc:
        record["judge_error"] = "%s: %s" % (type(exc).__name__, str(exc)[:150])
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

    a_items = group("A")
    if a_items:
        hits = [r for r in a_items if r.get("retrieval_hit")]
        summary["A"] = {
            "count": len(a_items),
            "retrieval_recall": round(len(hits) / len(a_items), 3),
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
    lines += ["", "## 逐题", "", "| 题号 | 类型 | 检索命中 | 排名 | verdict | 期望 | 概念覆盖 | 越界 | 引用 | 工具轮数 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for record in records:
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            record["id"], record["type"],
            {True: "✅", False: "❌", None: "—"}[record.get("retrieval_hit")],
            record.get("retrieval_rank") or "—",
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
    print("题目数   : %d" % len(items))
    print("")

    from core.config import settings
    from ai.llm import get_model
    from ai.agent.oa_assistant import oa_assistant

    # oa_assistant 在导入时执行了 logging.basicConfig(level=DEBUG)，把根日志记录器
    # 设成 DEBUG，于是 httpx / httpcore / openai 的每一次 HTTP 调用（含完整请求体）
    # 都会打到 stderr，把评估报告淹掉。这里统一压掉 INFO 及以下。
    import logging
    logging.disable(logging.INFO)

    judge_model = get_model(settings.DEFAULT_MODEL)
    records = []
    for index, item in enumerate(items, start=1):
        print("[%2d/%2d] %s  %s" % (index, len(items), item["id"], item["question"][:44]))
        try:
            record = await run_item(item, judge_model, oa_assistant, settings.DEFAULT_MODEL)
        except Exception as exc:
            record = {"id": item["id"], "type": item["type"], "question": item["question"],
                      "judge": None, "judge_error": "%s: %s" % (type(exc).__name__, str(exc)[:200])}
        record["expected_verdict"] = (item.get("judge") or {}).get("verdict")
        records.append(record)
        flag = ""
        if record.get("retrieval_hit") is not None:
            flag += " 检索%s" % ("✅" if record["retrieval_hit"] else "❌")
        if record.get("judge"):
            flag += " verdict=%s%s" % (record.get("verdict"), "✅" if record.get("verdict_correct") else "❌")
            if record.get("forbidden_violations"):
                flag += " 越界%d" % len(record["forbidden_violations"])
        else:
            flag += " 评委失败: %s" % (record.get("judge_error") or "?")[:60]
        print("        %s" % flag)

    summary = summarize(records)
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
    print("详细结果 : %s" % os.path.relpath(json_path, backend_root))
    print("可读报告 : %s" % os.path.relpath(md_path, backend_root))


if __name__ == "__main__":
    main()
