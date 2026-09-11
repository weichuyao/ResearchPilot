"""确定性排序基准：重排到底有没有把「该在的段落」往前挪？

## 为什么需要它

run_eval.py 走的是完整链路（Agent + 工具调用 + LLM 生成 + LLM 评委），
整条链上唯一不确定的就是 LLM。这在测「答案对不对」时是对的，但测**排序质量**
时会失灵，原因有两个：

  1. 评估集在 A 类已经饱和（12/12 全对），没有上升空间可测；
  2. 单题抖动就是 1/12 = 8.3%，一次运行的差异分不清是系统变了还是采样变了。

所以这个脚本把 LLM 整个摘掉，只问一个机械问题：

    给定评估题**原文**，期望来源（论文 + 页码）对应的段落，
    在返回列表里排第几？

向量编码是确定性的，BM25 是确定性的，重排是确定性的 —— 同一个问题跑一百次，
名次一模一样。这样哪怕只改善一名，也是可归因的。

## 用法

    python app/ai/eval/rank_bench.py
    python app/ai/eval/rank_bench.py --top-k 10 --rerank
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def _bootstrap():
    here = os.path.dirname(os.path.abspath(__file__))
    backend_root = os.path.dirname(os.path.dirname(os.path.dirname(here)))
    os.chdir(backend_root)
    sys.path.insert(0, os.path.join(backend_root, "app"))
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    return backend_root


def expected_pages(item) -> tuple[str, set[str]] | None:
    """从评估题里取出 (论文标题, 页码集合)。B/C 类没有期望来源，返回 None。"""
    source = item.get("expected_source")
    if not source:
        return None
    paper = source.get("paper")
    pages = source.get("pdf_pages") or ([source["pdf_page"]] if source.get("pdf_page") else [])
    if not paper or not pages:
        return None
    return paper, {str(p) for p in pages}


def first_match_rank(hits, paper: str, pages: set[str]) -> int | None:
    """返回列表里第一个「论文和页码都对上」的段落的 1 基名次。"""
    for rank, (doc, _origin, _score) in enumerate(hits, start=1):
        meta = doc.metadata or {}
        if meta.get("paper_title") != paper:
            continue
        if str(meta.get("page_label")) in pages:
            return rank
    return None


def summarise(ranks: dict[str, int | None], n: int) -> dict:
    found = [r for r in ranks.values() if r is not None]
    return {
        "hit": len(found),
        "recall@1": sum(1 for r in found if r <= 1) / n,
        "recall@3": sum(1 for r in found if r <= 3) / n,
        "recall@5": sum(1 for r in found if r <= 5) / n,
        "mrr": sum(1.0 / r for r in found) / n,
        "mean_rank": sum(found) / len(found) if found else None,
    }


def evidence_report(items, top_k: int, use_rerank: bool) -> dict | None:
    """每道 A 类题的「答案必需锚点」在检索结果里出现了几个。

    这是页级召回之外的第二个视角。页级指标只看「期望论文 + 页码有没有被碰到」——
    A08 要的答案是 `(iii) loading and equipment configuration`，实测它排在候选第 18 位、
    根本没进 top-10，但同一页另一块含 `ship type`，页级照样算命中、`retrieval_recall = 1.0`。

    锚点写在评估集的 `required_evidence` 字段里（已对着语料逐条验证存在）。
    归一化用 ai/rag/textnorm.py，与导入端、run_eval 完全同一份实现。
    """
    from ai.rag.hybrid import hybrid_search
    from ai.rag.rerank import rerank_hits
    from ai.rag.textnorm import norm_for_match

    graded = [item for item in items if item.get("required_evidence")]
    if not graded:
        return None

    def found(anchors, pool):
        text = norm_for_match("\n".join(doc.page_content for doc, _o, _s in pool))
        return [a for a in anchors if norm_for_match(a) in text]

    rows: dict[str, dict] = {}
    hybrid_complete = rerank_complete = 0
    for item in graded:
        anchors = [str(a) for a in item["required_evidence"]]
        hits, _vector_top1 = hybrid_search(item["question"], top_n=top_k)
        ordered = rerank_hits(item["question"], hits, top_n=top_k) if use_rerank else []
        in_hybrid = found(anchors, hits)
        in_rerank = found(anchors, ordered) if use_rerank else []
        if len(in_hybrid) == len(anchors):
            hybrid_complete += 1
        if use_rerank and len(in_rerank) == len(anchors):
            rerank_complete += 1
        missing = [a for a in anchors if a not in (in_rerank if use_rerank else in_hybrid)]
        rows[item["id"]] = {
            "required": anchors,
            "hybrid": "%d/%d" % (len(in_hybrid), len(anchors)),
            "rerank": "%d/%d" % (len(in_rerank), len(anchors)) if use_rerank else None,
            "missing": missing,
        }

    return {
        "count": len(graded),
        "top_k": top_k,
        "hybrid_complete": hybrid_complete,
        "rerank_complete": rerank_complete,
        "per_item": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=10, help="每次检索返回多少条")
    parser.add_argument("--rerank", action="store_true",
                        help="同时跑重排那一遍（需要 resource/models 里有模型）")
    parser.add_argument("--eval-set", type=str, default=None)
    parser.add_argument("--json", type=str, default=None,
                        help="把结果写成 JSON。给 - 则在 resource/eval/ 下自动命名")
    args = parser.parse_args()

    backend_root = _bootstrap()

    from ai.rag.hybrid import hybrid_search

    eval_path = args.eval_set or os.path.join(
        backend_root, "resource", "papers", "reid_paper_eval_set.json")
    with open(eval_path, encoding="utf-8") as fh:
        items = json.load(fh)["items"]

    if args.rerank:
        from ai.rag.rerank import available, rerank_hits
        if not available():
            print("重排模型不可用，--rerank 无法进行。先跑 scripts/fetch-reranker.ps1。")
            return

    before: dict[str, int | None] = {}
    after: dict[str, int | None] = {}
    header = "%-5s %-52s %s" % ("id", "question", "名次 混合 -> 重排")
    print(header)
    print("-" * len(header))

    graded = 0
    for item in items:
        spec = expected_pages(item)
        if spec is None:
            continue
        graded += 1
        paper, pages = spec

        hits, _vector_top1 = hybrid_search(item["question"], top_n=args.top_k)
        before[item["id"]] = first_match_rank(hits, paper, pages)

        if args.rerank:
            reordered = rerank_hits(item["question"], hits, top_n=args.top_k)
            after[item["id"]] = first_match_rank(reordered, paper, pages)

    for item in items:
        spec = expected_pages(item)
        if spec is None:
            continue
        b = before[item["id"]]
        line = "%-5s %-52s %s" % (
            item["id"], item["question"][:50],
            "%s" % (b if b is not None else "未命中"),
        )
        if args.rerank:
            a = after[item["id"]]
            arrow = ""
            if b is not None and a is not None:
                arrow = "   %s" % ("↑ 前进 %d" % (b - a) if a < b
                                   else "↓ 后退 %d" % (a - b) if a > b else "= 不变")
            line += " -> %-8s%s" % (a if a is not None else "未命中", arrow)
        print(line)

    print()
    print("题数 %d（只有 A 类有期望来源）" % graded)
    print()
    base = summarise(before, graded)
    print("%-12s %s" % ("混合检索", json.dumps(base, ensure_ascii=False)))
    reranked = None
    if args.rerank:
        rr = summarise(after, graded)
        reranked = rr
        print("%-12s %s" % ("+ 重排", json.dumps(rr, ensure_ascii=False)))
        print()
        for key in ("recall@1", "recall@3", "recall@5", "mrr"):
            delta = rr[key] - base[key]
            mark = "↑" if delta > 1e-9 else "↓" if delta < -1e-9 else "="
            print("  %-10s %6.3f -> %6.3f  %s %+.3f" % (key, base[key], rr[key], mark, delta))

    evidence = evidence_report(items, args.top_k, args.rerank)
    if evidence:
        print()
        rows = evidence["per_item"]
        header = "%-5s %-9s %-9s %s" % ("题号", "混合@k", "重排@k", "缺失的锚点（重排@k）")
        print(header)
        print("-" * len(header))
        for item_id, row in rows.items():
            print("%-5s %-9s %-9s %s" % (
                item_id, row["hybrid"], row["rerank"] or "—",
                ", ".join(row["missing"]) or "—"))
        print()
        print("锚点全中: 混合 %d/%d，重排 %d/%d" % (
            evidence["hybrid_complete"], evidence["count"],
            evidence["rerank_complete"], evidence["count"]))
        print("（注意：这里用的是评估题**原文**。Agent 会改写查询，所以端到端的"
              "evidence_recall 通常比这个高。）")

    if args.json:
        import datetime

        payload = {
            "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "eval_set": os.path.basename(eval_path),
            "top_k": args.top_k,
            "graded_items": graded,
            "hybrid": base,
            "rerank": reranked,
            "evidence": evidence,
            "per_item": {
                item["id"]: {"hybrid": before[item["id"]],
                             "rerank": after.get(item["id"]) if args.rerank else None}
                for item in items if expected_pages(item) is not None
            },
        }
        target = args.json
        if target == "-":
            stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            target = os.path.join("resource", "eval", "rank-bench-%s.json" % stamp)
        with open(target, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        print()
        print("结果已写入 : %s" % target)


if __name__ == "__main__":
    main()
