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
    python app/ai/eval/compare_runs.py --agent oa-assistant

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
