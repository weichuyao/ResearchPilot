"""锚点可达性天花板：把「索引召不到」和「规划问错了」分开算。

## 为什么需要它

`run_eval` 的 `evidence_recall` 要求一道 A 类题的**全部** `required_evidence`
锚点都出现在检索文本里。但锚点的性质差别极大：`mAP` 全库 1065 块，
`MultiSpectral Object ReID` 全库**只有 1 块**。同一个指标名下面，
一道题可以等价于"必须捞到那根针"，另一道题几乎不可能掉分 —— 分母不均匀。

所以先量一个与 Agent、LLM 完全无关的上界，**分三层**，因为"没召回到"有三个完全不同的原因：

  1. `pool_rank`  —— 候选池（混合检索，不过阀门不重排）里有没有？没有 = **索引/召回层的账**；
  2. `prod_rank`  —— 走生产链（过 0.35 阀门 + 重排）后排第几？没有但 1 里有 = **阀门或排序的账**；
  3. 端到端 `evidence_recall` —— 生产链能召回、Agent 却掉了 = **规划改写的账**（A20 就是这个）。

用锚点原样当查询是第 1、2 层的"最好情况"。注意它对第 2 层偏悲观：
`mAP`、`480`、`CMC` 这种裸词当查询本身就不像问题，阀门拒答是**预期行为**，
不能读成"索引坏了" —— 所以两层都要记，不能只记一层。

## 用法

    python app/ai/eval/anchor_ceiling.py
    python app/ai/eval/anchor_ceiling.py --json - --pool-k 30
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys


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


def corpus_counts(anchors: list[str]) -> dict[str, int]:
    """每个锚点在全库多少个块里出现（归一化后子串匹配，与 run_eval 同一份实现）。"""
    from ai.rag.chromaClient import document_vector_store
    from ai.rag.textnorm import norm_for_match

    needles = {anchor: norm_for_match(anchor) for anchor in anchors}
    counts = dict.fromkeys(anchors, 0)
    for doc in document_vector_store.get().get("documents") or []:
        text = norm_for_match(doc or "")
        for anchor, needle in needles.items():
            if needle in text:
                counts[anchor] += 1
    return counts


def measure(query: str, anchor: str, top_k: int, pool_k: int) -> dict:
    """同一个 anchor 在「候选池」与「生产链」两层各排第几。

    名次为空有两种完全不同的原因，混在一起会误导，所以两层都记：
      · 池子里就没有          -> 召回层的问题，重排和提示词都救不了；
      · 池子里有、生产链没有  -> 阀门（0.35）拒了这条查询，或重排把它排到 top_k 外。
    """
    from ai.rag.hybrid import hybrid_search
    from ai.rag.pipeline import retrieve
    from ai.rag.textnorm import norm_for_match

    needle = norm_for_match(anchor)

    def rank_of(hits):
        return next(
            (i for i, (doc, _o, _s) in enumerate(hits, start=1)
             if needle in norm_for_match(doc.page_content or "")),
            None,
        )

    pool, _vt = hybrid_search(query, top_n=pool_k)
    outcome = retrieve(query, candidates=top_k, top_n=top_k)
    return {
        "pool_rank": rank_of([(d, o, s) for d, o, s in pool]),
        "prod_rank": rank_of(outcome.hits),
        "rejected": outcome.rejected,
        "vector_top1": round(float(outcome.vector_top1), 4),
    }


def verdict_of(row: dict) -> str:
    """一个字符串判定，避免在终端里排中文字。"""
    if row["missing_in_pool"]:
        return "RECALL-MISS"
    if row["blocked_by_gate"]:
        return "GATE-BLOCKED"
    if row["unreachable_by_anchor"]:
        return "RANK-MISS"
    if row["unreachable_by_question"]:
        return "BRIDGE-GAP"
    return "REACHABLE"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=10, help="生产链返回条数（与生产一致）")
    parser.add_argument("--pool-k", type=int, default=30, help="候选池查看深度")
    parser.add_argument("--json", type=str, default=None,
                        help="结果写 JSON；给 - 则在 resource/eval/ 下自动命名")
    parser.add_argument("--eval-set", type=str, default=None)
    args = parser.parse_args()

    backend_root = _bootstrap()
    eval_path = args.eval_set or os.path.join(
        backend_root, "resource", "papers", "reid_paper_eval_set.json")
    with open(eval_path, encoding="utf-8") as fh:
        fixture = json.load(fh)
    items = [it for it in fixture["items"] if it.get("required_evidence")]

    all_anchors = [str(a) for it in items for a in it["required_evidence"]]
    counts = corpus_counts(sorted(set(all_anchors)))

    rows = []
    for item in items:
        anchors = [str(a) for a in item["required_evidence"]]
        per_anchor = {}
        for anchor in anchors:
            per_anchor[anchor] = {
                "corpus_chunks": counts[anchor],
                "by_anchor": measure(anchor, anchor, args.top_k, args.pool_k),
                "by_question": measure(item["question"], anchor, args.top_k, args.pool_k),
            }
        vals = per_anchor.values()
        row = {
            "id": item["id"],
            "question": item["question"],
            "anchors": per_anchor,
            "min_corpus_chunks": min(v["corpus_chunks"] for v in vals),
            "missing_in_pool": [a for a, v in per_anchor.items() if v["by_anchor"]["pool_rank"] is None],
            "blocked_by_gate": [a for a, v in per_anchor.items()
                                if v["by_anchor"]["prod_rank"] is None and v["by_anchor"]["rejected"]],
            "unreachable_by_anchor": [
                a for a, v in per_anchor.items()
                if v["by_anchor"]["prod_rank"] is None and not v["by_anchor"]["rejected"]],
            "unreachable_by_question": [a for a, v in per_anchor.items()
                                        if v["by_question"]["prod_rank"] is None],
        }
        row["verdict"] = verdict_of(row)
        rows.append(row)

    rows.sort(key=lambda r: (r["min_corpus_chunks"], r["id"]))
    print("%-5s %-6s %-14s %-14s %-13s %s" % (
        "item", "minN", "anchor-as-q", "question-as-q", "verdict", "anchors: chunks/prodRank"))
    print("-" * 100)
    for row in rows:
        detail = "; ".join(
            "%s: %dc/%s%s" % (
                a[:20], v["corpus_chunks"],
                v["by_anchor"]["prod_rank"] if v["by_anchor"]["prod_rank"] else "-",
                "(gate)" if v["by_anchor"]["rejected"] else "")
            for a, v in row["anchors"].items())
        print("%-5s %-6d %-14s %-14s %-13s %s" % (
            row["id"], row["min_corpus_chunks"],
            "all-in-pool" if not row["missing_in_pool"] else "MISS x%d" % len(row["missing_in_pool"]),
            "all-in-topk" if not row["unreachable_by_question"] else "MISS x%d" % len(row["unreachable_by_question"]),
            row["verdict"], detail))

    tally = {}
    for row in rows:
        tally[row["verdict"]] = tally.get(row["verdict"], 0) + 1
    pool_ok = sum(1 for r in rows if not r["missing_in_pool"])
    prod_ok = sum(1 for r in rows
                  if not r["missing_in_pool"] and not r["blocked_by_gate"]
                  and not r["unreachable_by_anchor"])
    q_ok = sum(1 for r in rows if not r["unreachable_by_question"])
    summary = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "corpus_fingerprint": fixture.get("verified_on"),
        "top_k": args.top_k,
        "pool_k": args.pool_k,
        "items_with_anchors": len(rows),
        "reachable_in_pool": pool_ok,
        "reachable_via_production": prod_ok,
        "reachable_by_question_text": q_ok,
        "verdict_tally": tally,
        "brittle_le3_chunks": sum(1 for r in rows if r["min_corpus_chunks"] <= 3),
        "unique_anchor_items": [r["id"] for r in rows if r["min_corpus_chunks"] == 1],
        "per_item": rows,
    }
    print()
    print("layer1 候选池里有(不过阀门) : %d/%d" % (pool_ok, len(rows)))
    print("layer2 生产链能召回         : %d/%d" % (prod_ok, len(rows)))
    print("layer3 用题目原文就能召回   : %d/%d   <- layer2-layer3 = 词汇桥接欠的账" % (q_ok, len(rows)))
    print("verdict tally: %s" % json.dumps(tally, ensure_ascii=False))
    print("min-anchor <=3 chunks items : %d ; unique(1 chunk): %s" % (
        summary["brittle_le3_chunks"], summary["unique_anchor_items"] or "none"))
    print("corpus fingerprint at run   : %s" % json.dumps(summary["corpus_fingerprint"], ensure_ascii=False))

    if args.json:
        target = args.json
        if target == "-":
            stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            target = os.path.join("resource", "eval", "anchor-ceiling-%s.json" % stamp)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=1)
        print("result: %s" % target)

    if args.json:
        target = args.json
        if target == "-":
            stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            target = os.path.join("resource", "eval", "anchor-ceiling-%s.json" % stamp)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=1)
        print("结果: %s" % target)


if __name__ == "__main__":
    main()
