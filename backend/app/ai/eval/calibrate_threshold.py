"""相关性阈值校准（design-decisions 决策一的流程化）。

阈值 0.35 是在 4 篇种子语料上人工校准的。语料变了（现在是 80 篇 / 7406 块），
分数分布整体移动 —— 「非空但不相关」的候选变多，老阈值必须重新校准。

**流程与当初完全一致，只是把算数自动化了：**

1. 准备一批带标注的查询（`resource/eval/threshold_queries.json`）：
   `{"q": 真实问法, "expect": true|false}` —— expect=true 表示库里有答案，
   false 表示话题不在库里（且要人工确认真的不在，这正是 B 类题复核的动作）。
2. 本脚本对每条查询跑**与线上一致的打分路径**（向量 top-1，k=20 同
   hybrid 的 VECTOR_TOP_K），输出分布、找空隙、给建议值。
3. **拍板是人做的**：建议值只是空隙中点，有没有空隙、空隙里混进了谁，
   要看打印出来的证据（命中的论文标题）再定。改完的值写进
   `pipeline.py` 的 `RELEVANCE_THRESHOLD`，并在 design-decisions.md 记录
   本轮校准的依据。

用法：
    python app/ai/eval/calibrate_threshold.py
    python app/ai/eval/calibrate_threshold.py --queries 某文件.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(_BACKEND_ROOT, "app"))
os.chdir(_BACKEND_ROOT)

DEFAULT_QUERIES = os.path.join("resource", "eval", "threshold_queries.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", default=DEFAULT_QUERIES)
    parser.add_argument("--top-k", type=int, default=20, help="与 hybrid 的 VECTOR_TOP_K 一致")
    parser.add_argument("--current", type=float, default=0.35, help="当前生效阈值（打印对照用）")
    args = parser.parse_args()

    from ai.rag.chromaClient import document_vector_store

    items = json.load(open(args.queries, encoding="utf-8"))
    print(f"标注查询 {len(items)} 条（应返回 {sum(1 for i in items if i['expect'])} / "
          f"应拒绝 {sum(1 for i in items if not i['expect'])}），top-k={args.top_k}\n")

    rows = []
    for item in items:
        hits = document_vector_store.similarity_search_with_relevance_scores(
            item["q"], k=args.top_k
        )
        top1 = float(hits[0][1]) if hits else 0.0
        title = hits[0][0].metadata.get("paper_title", "?") if hits else "—"
        rows.append((item["q"], bool(item["expect"]), top1, title))

    rows.sort(key=lambda r: -r[2])
    print(f"{'分数':>8}  期望   查询")
    for q, expect, score, title in rows:
        mark = "✅应返回" if expect else "⛔应拒绝"
        print(f"{score:8.4f}  {mark}  {q[:40]}")
        print(f"{'':>8}        └ 命中: {title[:70]}")

    yes = [r[2] for r in rows if r[1]]
    no = [r[2] for r in rows if not r[1]]
    if not yes or not no:
        print("\n⚠️ 两个组都必须有样本，没法校准")
        return

    yes_min, no_max = min(yes), max(no)
    print(f"\n应返回最低: {yes_min:.4f}   应拒绝最高: {no_max:.4f}")
    if yes_min > no_max:
        gap = yes_min - no_max
        suggest = (yes_min + no_max) / 2
        print(f"空隙 {gap:.4f}，建议阈值（空隙中点）: {suggest:.4f}")
        print(f"对照当前值 {args.current}: "
              + ("偏高，会误杀" if args.current > suggest else "偏低，会放进不相关材料")
              + " —— 确认证据后更新 pipeline.py")
    else:
        print("\n❌ 两组分数重叠 —— 没有干净的分界线。被穿过阈值的查询：")
        for q, expect, score, title in rows:
            if (expect and score <= no_max) or (not expect and score >= yes_min):
                print(f"  {score:.4f} expect={expect} {q[:44]}")
                print(f"           └ 命中: {title[:70]}")
        print("处理方式：优先怀疑『应拒绝』的标注（话题真的不在库里吗？），")
        print("删掉标错的再跑；剩下的重叠是这套阈值判据的真实极限，要记录下来。")


if __name__ == "__main__":
    main()
