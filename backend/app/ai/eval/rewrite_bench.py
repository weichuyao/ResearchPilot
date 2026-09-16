"""确定性改写器基准：改写器到底有没有把「该找的东西」变得更好找？

## 为什么必须单独一个基准（而不是复用 rank_bench）

`rank_bench.py` 把**评估题原文**直接喂给 `hybrid_search`：

    hits, _ = hybrid_search(item["question"], top_n=top_k)

也就是说它测的是「混合检索 + 重排」这条链，**完全不经过改写器**。
拿它去比较微调前后的改写器模型，两组数字会一模一样 —— 而报告上看起来
像是「微调没有效果」，实际是**尺子测错了对象**。

这个项目在这一点上已经有前科：`retrieval_rank` 曾被当成排序指标用过，
后来发现它同时被查询改写与多轮拼接污染（design-decisions 决策六）。
所以这里的原则是：**被微调的组件，要有专门量它的尺子。**

## 它测什么

把改写器接回链路里，但只保留确定性部分：

    中文问题 → plan_question()（改写器）→ query
             → hybrid_search(query) → 命中列表
             → 期望论文+页码 的 1 基名次 / required_evidence 锚点是否齐全

全流程**不经任何 LLM 评委**，同一配置跑一百次结果一致。四个指标：

  · `structured_ok`  结构化输出通过校验的比例 —— **硬门槛**，不达标谈不上质量
  · `recall@1/5`     改写后命中期望论文+页码的比例（分母是全部参评题，含未命中）
  · `mean_rank`      命中者的平均名次
  · `anchor_complete` required_evidence 锚点全中的题数 —— 直接对应 A08 那类
                     「枚举被切块切断」的问题

`--baseline raw` 是个特殊档：不调改写器，直接用问题原文检索。它给出
**rank_bench 的口径**作为对照，让「改写器到底加了多少价值」变成两个可比的数。

用法：
    python app/ai/eval/rewrite_bench.py                    # 用当前配置的改写器
    python app/ai/eval/rewrite_bench.py --baseline raw     # 不调改写器（对照）
    python app/ai/eval/rewrite_bench.py --json -           # 写报告
"""

from __future__ import annotations

import argparse
import asyncio
import io
import contextlib
import json
import logging
import os
import sys
import time
from datetime import datetime

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# 复用 rank_bench 的取期望来源与名次计算 —— 两个基准的判据必须同源，
# 各写一份的话「同一个命中」在两个报告里可能算成不同的名次。
# 注意：**必须延迟导入**（在 _bootstrap() 把 app/ 加进 sys.path 之后）。
# 模块级导入会在 sys.path 还没配好时执行，直接 ImportError。


def _bootstrap() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    backend_root = os.path.dirname(os.path.dirname(os.path.dirname(here)))
    os.chdir(backend_root)
    sys.path.insert(0, os.path.join(backend_root, "app"))
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    return backend_root


def summarise(ranks: dict[str, int | None], n: int) -> dict:
    """与 rank_bench.summarise 同一套算法（分母是参评题数，含未命中）。"""
    found = [r for r in ranks.values() if r is not None]
    return {
        "hit": len(found),
        "recall@1": round(sum(1 for r in found if r <= 1) / n, 4) if n else 0.0,
        "recall@5": round(sum(1 for r in found if r <= 5) / n, 4) if n else 0.0,
        "mrr": round(sum(1.0 / r for r in found) / n, 4) if n else 0.0,
        "mean_rank": round(sum(found) / len(found), 3) if found else None,
    }


async def rewrite_one(question: str, config: dict) -> tuple[list[str], bool, str]:
    """调改写器拿查询列表，返回 (queries, 结构化是否可用, 失败原因)。"""
    from ai.agent.research_workflow import plan_question

    try:
        plan = await plan_question(question, config)
    except Exception as exc:
        return [], False, "%s: %s" % (type(exc).__name__, str(exc)[:120])

    queries = [sub.query.strip() for sub in plan.sub_questions if (sub.query or "").strip()]
    return queries, bool(queries), ""


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default=None, help="只跑指定题号，逗号分隔")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--eval-set", type=str, default=None)
    parser.add_argument("--baseline", choices=["raw"], default=None,
                        help="raw = 不调改写器，直接用问题原文检索（给对照用）")
    parser.add_argument("--json", type=str, default=None, help="写报告路径，- 为自动命名")
    args = parser.parse_args()

    backend_root = _bootstrap()

    from core.config import settings
    from ai.rag.hybrid import hybrid_search
    from ai.rag.textnorm import norm_for_match
    from ai.eval.rank_bench import expected_pages, first_match_rank

    # 评估与外部世界隔离（与 run_eval 同口径：MCP 是网络+子进程，不可复现）
    os.environ["MCP_ARXIV_ENABLED"] = "false"

    eval_path = args.eval_set or os.path.join(
        backend_root, "resource", "papers", "reid_paper_eval_set.json")
    with open(eval_path, encoding="utf-8") as fh:
        items = json.load(fh)["items"]
    if args.only:
        wanted = {v.strip().upper() for v in args.only.split(",")}
        items = [it for it in items if it["id"].upper() in wanted]
    if args.limit:
        items = items[: args.limit]

    if args.baseline == "raw":
        mode = "raw（不改写，问题原文直接检索）"
        model_desc = "—"
    else:
        mode = "rewrite（经改写器）"
        model_desc = settings.ANALYZE_MODEL or settings.DEFAULT_MODEL

    print("评估集   : %s" % os.path.basename(eval_path))
    print("模式     : %s" % mode)
    print("改写模型 : %s" % model_desc)
    print("题目数   : %d" % len(items))
    print("")

    logging.disable(logging.INFO)
    config = {"configurable": {"model": settings.DEFAULT_MODEL}}

    ranks: dict[str, int | None] = {}
    per_item: dict[str, dict] = {}
    structured_fail: list[str] = []
    anchor_ok = anchor_total = 0
    rewrite_seconds = 0.0

    for index, item in enumerate(items, start=1):
        spec = expected_pages(item)
        queries: list[str] = []
        ok = True
        reason = ""

        if args.baseline == "raw":
            queries = [item["question"]]
        else:
            started = time.time()
            # 规划会打日志/告警，基准自己打印进度，这里静音避免刷屏
            with contextlib.redirect_stdout(io.StringIO()):
                queries, ok, reason = await rewrite_one(item["question"], config)
            rewrite_seconds += time.time() - started
            if not ok:
                structured_fail.append(item["id"])
                print("[%2d/%2d] %-5s 改写失败：%s" % (index, len(items), item["id"], reason))

        # 多条查询各检索一次，合并去重后按最好名次计 —— 与 agent 实际行为一致
        # （agent 也是每个子问题各查一次）
        pool = []
        for query in queries:
            if not query:
                continue
            hits, _top1 = hybrid_search(query, top_n=args.top_k)
            pool.extend(hits)

        rank = first_match_rank(pool, *spec) if spec else None
        if spec:
            ranks[item["id"]] = rank

        # 锚点：改写后的检索结果里，答案必需的证据句齐不齐
        anchors = [str(a) for a in (item.get("required_evidence") or [])]
        if anchors:
            text = norm_for_match("\n".join(doc.page_content for doc, _o, _s in pool))
            missing = [a for a in anchors if norm_for_match(a) not in text]
            anchor_total += 1
            if not missing:
                anchor_ok += 1
            per_item[item["id"]] = {"rank": rank, "missing": missing,
                                    "queries": queries}
            if missing and (rank is None or rank > 5):
                print("[%2d/%2d] %-5s 名次=%s 缺锚点=%s"
                      % (index, len(items), item["id"], rank or "未命中", missing))
        else:
            per_item[item["id"]] = {"rank": rank, "missing": [], "queries": queries}

    n = len(ranks)
    metrics = summarise(ranks, n) if n else {}

    # 改写器的**回落率**才是这个基准的头号数字。
    #
    # 为什么不能只看「结构化输出失败」：_invoke_structured 内部会在本地模型
    # 失败时回落主力模型，于是 plan_question 依然返回一个合法计划 ——
    # 从调用方看不出区别。如果只报「失败 0 次」，就会得出「本地模型工作正常」
    # 的结论，而实际上每一题都是 DeepSeek 在干活、只是多绕了一圈。
    # （这个缺陷是本基准第一版实测时暴露的：A08 显示「失败 0/1」，
    # 而真实情况是本地模型输出了中文查询、被校验拒绝后回落。）
    status = {}
    if args.baseline != "raw":
        from ai.agent.research_workflow import rewrite_model_status

        status = rewrite_model_status()
    calls = int(status.get("calls") or 0)
    fallbacks = int(status.get("fallbacks") or 0)
    fallback_rate = (fallbacks / calls) if calls else 0.0

    print("")
    print("=" * 70)
    print("改写器名次基准（%s）" % mode)
    print("=" * 70)
    print("参评题数（有期望来源的 A 类）: %d" % n)
    if args.baseline != "raw":
        print("改写器调用 %d 次，回落主力模型 %d 次（回落率 %.0f%%）"
              % (calls, fallbacks, fallback_rate * 100))
        print("改写平均耗时: %.2fs/题" % (rewrite_seconds / max(len(items), 1)))
        print("结构化输出失败（本地与回落后都失败）: %d/%d" % (len(structured_fail), len(items)))
        if fallback_rate >= 0.5:
            print("!")
            print("! 回落率 %.0f%%：下面的指标**主要是主力模型（%s）的成绩**，不是本地模型的。"
                  % (fallback_rate * 100, settings.DEFAULT_MODEL))
            print("! 本地模型没有承担工作 —— 这正是微调要解决的事。")
            print("! 最近一次失败原因：%s" % (status.get("last_error") or "(无)"))
    print(json.dumps(metrics, ensure_ascii=False))
    print("锚点全中: %d/%d" % (anchor_ok, anchor_total))
    if structured_fail:
        print("! 结构化输出失败清单: %s" % ", ".join(structured_fail))
    print("")

    if args.json:
        from ai.agent.research_workflow import rewrite_model_status

        payload = {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "mode": args.baseline or "rewrite",
            "model": model_desc,
            "eval_set": os.path.basename(eval_path),
            "top_k": args.top_k,
            "graded_items": n,
            "metrics": metrics,
            "anchor": {"complete": anchor_ok, "total": anchor_total},
            "structured_failed": structured_fail,
            "rewrite_seconds_per_item": round(rewrite_seconds / max(len(items), 1), 3),
            "model_status": rewrite_model_status() if args.baseline != "raw" else {},
            "per_item": per_item,
        }
        if args.json == "-":
            out_dir = os.path.join(backend_root, "resource", "eval")
            os.makedirs(out_dir, exist_ok=True)
            args.json = os.path.join(
                out_dir, "rewrite-bench-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S"))
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        print("结果已写入 : %s" % os.path.relpath(args.json, backend_root))


if __name__ == "__main__":
    asyncio.run(main())
