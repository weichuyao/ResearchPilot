"""把多次评估运行汇总对比。

## 为什么需要它

`run_eval.py` 一次只跑一轮。但这份系统的单轮结果**不能单独用来下结论**：
Chroma 的 top-k 本身跨进程不可复现（实测两次运行前 20 条只重叠 17 条，
见 reference/design-decisions.md 决策六），而 20 题评估集上一题就是 8.3%。
所以任何"改了 X 之后指标从 A 变成 B"的说法，都必须建立在**重复运行**上。

这个脚本就是把 N 轮结果按 agent 分组，给出均值与极值，让"有没有变"变成可判断的。

## 用法

    python app/ai/eval/compare_runs.py                    # 汇总 resource/eval 下全部报告
    python app/ai/eval/compare_runs.py --last 6           # 只看最近 6 份
    python app/ai/eval/compare_runs.py --agent react-assistant

只统计带 `summary.agent` 字段的报告 —— 那个字段是后来才加的，早期报告没跑过哪个
agent 的记录，混进来只会误导。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as stats
import sys


def _bootstrap():
    here = os.path.dirname(os.path.abspath(__file__))
    backend_root = os.path.dirname(os.path.dirname(os.path.dirname(here)))
    os.chdir(backend_root)
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    return backend_root


# 关心哪些指标、以及"越大越好"还是"越小越好"。方向决定了极值怎么报。
METRICS = [
    ("A.retrieval_recall", "max"),
    ("A.evidence_recall", "max"),
    ("A.verdict_accuracy", "max"),
    ("A.avg_concept_coverage", "max"),
    ("A.citation_ok_rate", "max"),
    ("B.verdict_accuracy", "max"),
    ("B.over_claim_rate", "min"),
    ("B.scope_qualified_rate", "max"),
    ("C.verdict_accuracy", "max"),
    ("process.avg_context_chars", "min"),
    ("process.avg_tool_calls", "min"),
    ("process.avg_seconds", "min"),
]


def fmt_corpus(corpus) -> str | None:
    """把语料指纹格式化成一行；**没记录就返回 None**。

    为什么用 None 而不是一个占位字符串：**「没记录」和「记录到是同一份语料」
    是两件事。** 如果都当普通值，一组从头到尾都没记录的历史运行会被判成
    「语料一致」，而我们对它们其实一无所知。

    实测就是这样：一组 5 轮里前 3 轮跑在 4 篇语料上、后 2 轮跑在 7 篇上，
    因为没有字段，字符串比较看不出任何差异，警告不会响。
    """
    if not isinstance(corpus, dict) or corpus.get("papers") is None:
        return None
    return "%s 篇 / %s 块" % (corpus.get("papers"), corpus.get("chunks"))


def corpus_status(group: list[dict]) -> list[str]:
    """检查一组运行的语料是否可比，返回要打印的几行。

    三种情况分开说：
      · 记到了两个不同的语料            → 明确报「不一致」
      · 有的记了、有的没记              → 也报不一致（无法确认没记的那些）
      · 全都没记                        → 说明"无法确认"，而不是默认可比
    """
    known: dict[str, list[str]] = {}
    unknown: list[str] = []
    for run in group:
        value = fmt_corpus(run["summary"].get("corpus"))
        if value is None:
            unknown.append(run["stamp"])
        else:
            known.setdefault(value, []).append(run["stamp"])

    if len(known) <= 1 and not unknown:
        return ["  语料: %s" % next(iter(known))]

    lines = ["  ⚠️ 这组运行的**语料不可比**："]
    for value, stamps in sorted(known.items()):
        lines.append("      %s  <- %s" % (value, ", ".join(stamps)))
    if unknown:
        lines.append("      未记录 <- %s（语料字段是后来才加的，无法确认）"
                     % ", ".join(unknown))
    return lines


def pick(summary: dict, dotted: str):
    node = summary
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--last", type=int, default=None, help="只看最近 N 份报告")
    parser.add_argument("--agent", type=str, default=None, help="只统计某个 agent")
    parser.add_argument("--dir", type=str, default=None)
    args = parser.parse_args()

    backend_root = _bootstrap()
    out_dir = args.dir or os.path.join(backend_root, "resource", "eval")

    runs = []
    for path in sorted(glob.glob(os.path.join(out_dir, "eval-*.json"))):
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as exc:
            print("跳过 %s：%s" % (os.path.basename(path), exc))
            continue
        summary = data.get("summary") or {}
        if not summary.get("agent"):
            continue          # 早期报告没记 agent，无法归组
        if args.agent and summary["agent"] != args.agent:
            continue
        runs.append({
            "stamp": os.path.basename(path)[5:20],
            "agent": summary["agent"],
            "summary": summary,
        })

    if args.last:
        runs = runs[-args.last:]

    if not runs:
        print("没有可统计的报告（需要带 summary.agent 字段的运行）。")
        return

    by_agent: dict[str, list[dict]] = {}
    for run in runs:
        by_agent.setdefault(run["agent"], []).append(run)

    for agent, group in sorted(by_agent.items()):
        print("=" * 96)
        print("agent = %s    轮数 = %d    运行: %s" % (
            agent, len(group), ", ".join(r["stamp"] for r in group)))
        print("=" * 96)

        # ---- 语料指纹：不同语料上的指标不能直接比 ----
        #
        # 这是最容易犯、也最不容易发现的一类错误：往知识库里加了几篇文档之后
        # 重跑评估，然后拿新指标和旧基线比。评估集是**为特定语料写的固件**
        # （B 类题问"这四篇论文是否…"），语料一变，那些期望值就可能已经过期。
        # 所以这里必须显式报警，而不是让人自己去翻 JSON。
        for line in corpus_status(group):
            print(line)
        print()

        # ---- 题目数也要一致 ----
        #
        # 同一个坑的另一面：`--limit 2` 的调试运行和 20 题的完整运行混在一组里，
        # 均值会毫无意义（实测就是这样：2 题那轮的概念覆盖 0.75 被算进了均值，
        # 让人以为指标退化了）。**这一条是加语料指纹警告时顺带发现的。**
        totals = {r["summary"].get("total") for r in group}
        if len(totals) > 1:
            print("  ⚠️ 这 %d 轮的**题目数不一致**（%s），均值没有意义："
                  % (len(group), " / ".join(str(t) for t in sorted(totals, key=str))))
            for r in group:
                print("      %s  %s 题" % (r["stamp"], r["summary"].get("total")))
            print()

        unreliable = [r["stamp"] for r in group if not r["summary"].get("metric_reliable", True)]
        if unreliable:
            print("  ⚠️ 有 %d 轮 metric_reliable=false（判分缺失或解析失败），结论不可用: %s"
                  % (len(unreliable), ", ".join(unreliable)))
            print()

        print("  %-28s %-22s %-22s %s" % ("指标", "均值", "范围（最差~最好）", "单轮值"))
        print("  " + "-" * 92)
        for dotted, direction in METRICS:
            values = [pick(r["summary"], dotted) for r in group]
            values = [v for v in values if isinstance(v, (int, float))]
            if not values:
                continue
            mean = stats.mean(values)
            low, high = min(values), max(values)
            if direction == "max":
                worst, best = low, high
            else:
                worst, best = high, low
            spread = "" if abs(best - worst) < 1e-9 else "  ← 有抖动"
            print("  %-28s %-22s %-22s %s%s" % (
                dotted, ("%.3f" % mean) if isinstance(mean, float) else mean,
                "%.3f ~ %.3f" % (worst, best),
                " ".join("%.3f" % v if isinstance(v, float) else str(v) for v in values),
                spread))
        print()

    if len(by_agent) > 1:
        print("=" * 96)
        print("逐项对照（两个 agent 的均值）")
        print("=" * 96)

        # 跨 agent 对比时也要看语料是否一致 —— 否则比的是两件不同的事
        all_runs = [r for g in by_agent.values() for r in g]
        corpus_lines = corpus_status(all_runs)
        if corpus_lines and corpus_lines[0].startswith("  ⚠️"):
            for line in corpus_lines:
                print(line)
            print("      （下面的对照表因此不可信）")
            print()

        names = sorted(by_agent)
        print("  %-28s %s" % ("指标", "  ".join("%-18s" % n for n in names)))
        print("  " + "-" * 80)
        for dotted, direction in METRICS:
            cells = []
            for name in names:
                values = [pick(r["summary"], dotted) for r in by_agent[name]]
                values = [v for v in values if isinstance(v, (int, float))]
                cells.append("%.3f" % stats.mean(values) if values else "—")
            print("  %-28s %s" % (dotted, "  ".join("%-18s" % c for c in cells)))


if __name__ == "__main__":
    main()
