"""改造 #11 阶段 3：构造规划节点的 SFT 数据。

## 这个脚本在做什么

把 `analyze` 节点这个窄任务变成训练样本：

    中文研究问题 → （教师：现有 DeepSeek plan_question）→ ResearchPlan JSON

教师蒸馏是主流做法，但**这个脚本真正的价值不在"调用了教师模型"**，
而在它做了四件容易糊弄过去的事：

### 1. 问题来源：从语料反向生成，而不是手写几十条

从 72 篇论文的标题 + 抽取到的正文片段生成中文研究型问题，覆盖改写器真实会遇到
的形态（点名论文的 / 问方法的 / 问数字的 / 问元信息的 / 带年份限定的）。

### 2. 硬案例注入：把项目文档里已记录的失败模式编进训练集

这是本项目特有的资产 —— `reference/design-decisions.md` 与
`reference/reid-paper-eval-set.md` 里记录了大量**实测失败模式**：
跨块枚举被切断（A08）、表格数字类查询向量检索弱、B 类"语料里有没有 X"
不能被编造成"有"、元信息问题要走 list_papers 而不是相似检索。
手写这些案例并让教师打标，等于把项目的经验固化成训练信号。

### 3. 清洗：用**与线上同一套**校验函数

`_reject_reason` 是线上回落判据（结构合法但内容违约也算失败）。
清洗复用它而不是另写一套 —— 否则会训出一个"通过训练脚本校验、
但上线就被回落"的模型。同类错误本项目在工具输出格式上踩过一次。

### 4. 训练/评估隔离：断言，不是自觉

脚本**断言** 30 道评估题从未进入训练集。泄漏的话，阶段 5 的对比全部作废，
而且从数字上看是"微调效果惊人"，很难事后发现。

## 输出

LLaMA-Factory `sharegpt` 格式 JSONL：
    {"messages":[{"role":"system",...},{"role":"user",...},{"role":"assistant",...}]}
外加 `dataset_info.json`（LLaMA-Factory 的数据集注册表）。

用法：
    # 1) 生成问题清单（会调 DeepSeek，先小批量试跑看质量与花费）
    python -m ai.finetune.build_sft_data gen-questions --per-paper 7 --limit 3

    # 2) 教师蒸馏 + 清洗 + 隔离断言 → 训练集
    python -m ai.finetune.build_sft_data distill --in questions.jsonl --out-dir finetune/data
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import random
import sys
from datetime import datetime

logger = logging.getLogger(__name__)

# 评估集里**绝不能**出现在训练数据中的题的 id（隔离断言的依据）
EVAL_SET = os.path.join("resource", "papers", "reid_paper_eval_set.json")


# ---------------------------------------------------------------------------
# 硬案例：项目文档里记录的失败模式
# ---------------------------------------------------------------------------
#
# 每条都是「用户会怎么问」，不是「我们想让模型输出什么」—— 标签一律由教师模型
# 产出，避免我把自己的期望硬编码进训练数据（那就成了自我实现的循环）。
HARD_CASES = [
    # ---- 枚举型：答案是一个枚举，facets 必须用字面术语 ----
    # ⚠️ 素材刻意选**评估集未覆盖**的论文。第一版这里用了 A²RNet/DPEFormer/ViV-ReID，
    # 结果与 A08/A04/A11 语义重合 —— 逐字断言没拦住（措辞不同），
    # 而这属于更隐蔽的泄漏：训了它，那几道题的测量就作废了。
    "Cooperative Vehicle Re-Identification 那篇用了哪几种多视图匹配策略？",
    "UD-Gaussian 的不确定性建模包含哪几个组成部分？",
    "PEFN 提出的 Patches Enhancement 用到了哪些层级的特征？",
    # ---- 数字 / 表格类：向量检索对表格数字很弱（决策五，108 个块标了 kind=table）----
    "MVReID-Former 在多视角数据集上的 Rank-1 和 mAP 分别是多少？",
    "PartFormer 的参数量和推理速度相比基线怎么样？",
    # ---- B 类：语料里没有该主题，查询**不能**被编造成"有" ----
    # 话题同样避开评估集用过的（AIS / LiDAR / GAN / 强化学习 / 联邦学习 / DeepSpeed）
    "这些论文里有用知识蒸馏来压缩模型的吗？",
    "知识库中的论文有没有用神经架构搜索（NAS）来设计 backbone？",
    "这些论文做过模型量化或 INT8 推理优化吗？",
    # ---- 元信息类：走精确查询（list_papers）而不是相似检索 ----
    "知识库里一共有哪几篇论文？",
    "这些论文分别是哪一年发表或投稿的？",
    # ---- 年份限定类：相对说法要换算成四位年份 ----
    "2025 年以来投稿的论文里，跨视角车辆重识别的方法有哪些？",
    "最近一年有没有关于可见光-红外 ReID 的新论文？",
    # ---- 点名论文 + 方法细节（paper 字段该被填上）----
    "UD-Gaussian 里 t 时刻的高斯分布是怎么参数化的？",
    "PEFN 的层级特征是怎么融合的？",
    "MVReID-Former 的多视角视觉 Transformer 是怎么组织的？",
    # ---- 短问题 / 口语化（真实用户不会像论文那样提问）----
    "PartFormer 是干嘛的？",
    "跨视角的车辆重识别一般用什么网络结构？",
    "遮挡 ReID 现在最好的是哪个方法？",
    # ---- 需要拆成多个子问题的复合问题 ----
    "MVReID-Former 用了多少个视角，这些视角的特征又是怎么融合的？",
    "这几年的车辆重识别，从姿态对齐到多视角融合，技术路线是怎么演变的？",
]


def _bootstrap() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    backend_root = os.path.dirname(os.path.dirname(os.path.dirname(here)))
    os.chdir(backend_root)
    sys.path.insert(0, os.path.join(backend_root, "app"))
    return backend_root


def norm_key(text: str) -> str:
    """与线上同一套归一化 —— 去重口径必须一致，不然「看起来不同」的两条
    在检索/比对层面是同一句。"""
    from ai.rag.textnorm import norm_for_match

    return norm_for_match(text)


def load_eval_questions(backend_root: str) -> list[tuple[str, str]]:
    """返回 [(题号, 问题原文)] —— 题号用于报错时指出撞了哪道题。"""
    path = os.path.join(backend_root, EVAL_SET)
    with open(path, encoding="utf-8") as fh:
        items = json.load(fh)["items"]
    return [(it["id"], it["question"]) for it in items]


def char_bigrams(text: str) -> set[str]:
    """字符二元组集合 —— 中文的近似重复检测用它比用词更稳。"""
    s = norm_key(text).replace(" ", "")
    return {s[i:i + 2] for i in range(len(s) - 1)} or {s}


def find_leaks(question: str, eval_questions: list[tuple[str, str]]) -> list[str]:
    """返回与该问题**逐字或近似**重合的评估题号。

    ## 为什么不能只比字面

    第一版只做归一化后逐字比对，结果放过了「A²RNet 的 SAP 模块从全局特征向量
    推断的三类语义属性是什么？」——它和 A08 措辞不同、语义完全相同。
    **训了它，A08 的测量就作废了**，而且指标会表现为"微调效果惊人"，极难事后
    发现。所以再加一层相似度：字符二元组 Jaccard ≥ 0.6 即判为泄漏。

    阈值取 0.6 的理由：中文问句的二元组重叠天然偏高（"什么""怎么"这类），
    同一主题、不同角度的问题大约在 0.3~0.5，只有近乎改写才会超过 0.6。
    """
    key = norm_key(question)
    bigrams = char_bigrams(question)
    hits = []
    for item_id, eq in eval_questions:
        if key == norm_key(eq):
            hits.append("%s(逐字)" % item_id)
            continue
        other = char_bigrams(eq)
        union = bigrams | other
        if union and len(bigrams & other) / len(union) >= 0.6:
            hits.append("%s(相似%.2f)" % (item_id, len(bigrams & other) / len(union)))
    return hits


# ---------------------------------------------------------------------------
# 1) 问题生成：从语料反向生成中文研究型问题
# ---------------------------------------------------------------------------
QUESTION_GEN_PROMPT = """你在为一个科研文献问答系统构造训练数据。

下面是某篇论文的标题和若干原文片段。请生成 {n} 个**中文**问题，要求：

- 像真实的科研人员会问的那样（口语化、有长有短），不要写成论文标题的复述
- 覆盖不同形态：问方法细节 / 问数据集与数字 / 问模块作用 / 问对比 / 带年份限定
- 只依据给定材料能回答的问题；不要编造材料里没有的概念
- 每行一个问题，不要编号、不要解释、不要加引号

论文标题：{title}

原文片段：
{snippets}"""


async def cmd_gen_questions(args) -> None:
    """从语料生成候选问题（这一步会调 DeepSeek）。"""
    from core.config import settings
    from ai.llm import get_model
    from langchain_core.messages import HumanMessage, SystemMessage
    from ai.rag.chromaClient import document_vector_store

    got = document_vector_store.get(include=["documents", "metadatas"])
    docs = got.get("documents") or []
    metas = got.get("metadatas") or []

    by_paper: dict[str, list[str]] = {}
    for text, meta in zip(docs, metas):
        title = (meta or {}).get("paper_title") or ""
        if title:
            by_paper.setdefault(title, []).append(text or "")

    titles = sorted(by_paper)
    if args.limit:
        titles = titles[: args.limit]

    model = get_model(settings.DEFAULT_MODEL)
    out_path = args.out or "finetune/data/questions.jsonl"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    total = 0
    with open(out_path, "w", encoding="utf-8") as fh:
        for index, title in enumerate(titles, start=1):
            chunks = by_paper[title]
            # 取若干片段做样本（首尾各取，覆盖摘要与方法/实验部分）
            picked = chunks[:2] + chunks[len(chunks) // 2: len(chunks) // 2 + 2] + chunks[-1:]
            snippets = "\n---\n".join(c[:700] for c in picked[:5])
            prompt = QUESTION_GEN_PROMPT.format(n=args.per_paper, title=title, snippets=snippets)
            try:
                resp = await model.ainvoke(
                    [SystemMessage(content="你只输出问题清单，每行一条。"),
                     HumanMessage(content=prompt)]
                )
                text = resp.content if isinstance(resp.content, str) else str(resp.content)
            except Exception as exc:
                logger.warning("论文《%s》问题生成失败：%s", title[:30], exc)
                continue

            questions = []
            for line in text.splitlines():
                q = line.strip().lstrip("0123456789.、) ").strip()
                if not q or len(q) < 6 or len(q) > 120:
                    continue
                if not any("\u4e00" <= ch <= "\u9fff" for ch in q):   # 必须是中文问题
                    continue
                # 质量闸①：必须真的是个问句。实测生成器会产出残句
                # （如「年发表的这篇MAPLE…」——模板没给年份，模型留了个空位），
                # 这种句子配出来的标签是垃圾，会污染训练集。
                if not any(mark in q for mark in
                           ("吗", "什么", "怎么", "如何", "哪些", "哪一", "哪个", "多少",
                            "是否", "为什么", "区别", "作用", "流程", "有哪些", "几")):
                    continue
                # 质量闸②：不得以悬空的连接词/量词开头
                if q[0] in "年的了的和与及在对于关于这那其":
                    continue
                questions.append(q)

            for q in questions:
                fh.write(json.dumps({"question": q, "source": "corpus", "paper": title},
                                    ensure_ascii=False) + "\n")
            fh.flush()
            total += len(questions)
            print("[%2d/%d] %-40s +%d 条（累计 %d）"
                  % (index, len(titles), title[:38], len(questions), total), flush=True)

    print("\n问题清单已写入 %s（共 %d 条）" % (out_path, total))


# ---------------------------------------------------------------------------
# 2) 教师蒸馏 + 清洗 + 隔离断言
# ---------------------------------------------------------------------------
async def cmd_distill(args) -> None:
    """教师打标 → 清洗 → 写 LLaMA-Factory 数据集。"""
    from core.config import settings
    from ai.agent.research_workflow import ANALYZE_PROMPT, plan_question, _reject_reason

    backend_root = _bootstrap()

    # ---- 问题集：生成的问题 + 硬案例 ----
    questions: list[dict] = []
    seen: set[str] = set()

    if args.in_path and os.path.isfile(args.in_path):
        with open(args.in_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                key = norm_key(row["question"])
                if key in seen:
                    continue
                seen.add(key)
                questions.append(row)
        print("从 %s 读入 %d 条（去重后）" % (args.in_path, len(questions)))

    for q in HARD_CASES:
        key = norm_key(q)
        if key in seen:
            continue
        seen.add(key)
        questions.append({"question": q, "source": "hard_case", "paper": ""})
    print("硬案例 %d 条已并入，总计 %d 条" % (len(HARD_CASES), len(questions)))

    # ---- 训练/评估隔离：断言，不是自觉 ----
    #
    # 这道断言在第一次运行时**真的抓到了泄漏**：我手写的 3 条硬案例与 A04/A14/B08
    # 逐字重合（因为参考的"已记录失败模式"本身就是评估题）；把检查加固成相似度
    # 之后又发现「SAP 模块的三类语义属性」与 A08 语义等同。
    # 如果不拦，阶段 5 的对比全部作废 —— 而且是表现为"微调效果惊人"的那种作废。
    eval_questions = load_eval_questions(backend_root)
    leaks: list[str] = []
    for row in questions:
        hits = find_leaks(row["question"], eval_questions)
        if hits:
            leaks.append("%s ← %s" % ("/".join(hits), row["question"][:50]))
    if leaks:
        print("!")
        print("! 训练/评估隔离被破坏，以下训练问题与评估集重合：")
        for line in leaks:
            print("!   %s" % line)
        print("! 继续下去会让阶段 5 的对比全部作废（且表现为『微调效果惊人』）。")
        print("! 请把这些硬案例改用评估集未覆盖的论文，再重跑。")
        raise SystemExit(2)
    print("隔离断言通过：%d 条训练问题与 %d 道评估题无逐字/近似重合"
          % (len(questions), len(eval_questions)))

    if args.limit:
        # 硬案例必须留下（它们是刻意注入的失败模式，不该被采样掉）
        hard = [r for r in questions if r["source"] == "hard_case"]
        rest = [r for r in questions if r["source"] != "hard_case"]
        random.Random(42).shuffle(rest)
        questions = hard + rest[: max(0, args.limit - len(hard))]
        print("采样到 %d 条（硬案例全部保留）" % len(questions))

    # ---- 教师蒸馏 ----
    config = {"configurable": {"model": settings.DEFAULT_MODEL}}
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    samples = []
    rejected = {"reject": 0, "exception": 0, "dup_plan": 0}
    seen_plans: set[str] = set()
    started = datetime.now()

    for index, row in enumerate(questions, start=1):
        question = row["question"]
        try:
            plan = await plan_question(question, config)
        except Exception as exc:
            rejected["exception"] += 1
            logger.warning("蒸馏失败（%s）：%s", type(exc).__name__, str(exc)[:80])
            continue

        # 清洗①：复用**线上同一个**校验函数（关键是这条，不是另写一套）
        reason = _reject_reason(plan)
        if reason:
            rejected["reject"] += 1
            logger.debug("清洗剔除：%s ← %s", reason, question[:40])
            continue

        target = plan.model_dump_json(exclude_none=True)
        # 清洗②：同一问题的输出去重（避免同一 plan 反复出现把数据集带偏）
        dup_key = hashlib.sha1((norm_key(question) + "|" + target).encode()).hexdigest()
        if dup_key in seen_plans:
            rejected["dup_plan"] += 1
            continue
        seen_plans.add(dup_key)

        samples.append({
            "messages": [
                {"role": "system", "content": ANALYZE_PROMPT},
                {"role": "user", "content": question},
                {"role": "assistant", "content": target},
            ],
            "_meta": {"source": row.get("source", ""), "paper": row.get("paper", "")},
        })
        if index % 25 == 0:
            print("  蒸馏 %d/%d（已收 %d）" % (index, len(questions), len(samples)), flush=True)

    # ---- 落盘（LLaMA-Factory sharegpt 格式）----
    random.Random(7).shuffle(samples)
    split = int(len(samples) * 0.95)
    train, holdout = samples[:split], samples[split:]

    def write_jsonl(path: str, rows: list[dict]) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps({"messages": r["messages"]}, ensure_ascii=False) + "\n")

    write_jsonl(os.path.join(out_dir, "train.jsonl"), train)
    write_jsonl(os.path.join(out_dir, "holdout.jsonl"), holdout)

    with open(os.path.join(out_dir, "dataset_info.json"), "w", encoding="utf-8") as fh:
        json.dump({
            "rewriter_train": {
                "file_name": "train.jsonl",
                "formatting": "sharegpt",
                "columns": {"messages": "messages"},
                "tags": {"role_tag": "role", "content_tag": "content",
                         "user_tag": "user", "assistant_tag": "assistant",
                         "system_tag": "system"},
            }
        }, fh, ensure_ascii=False, indent=2)

    # ---- 报告 ----
    by_source: dict[str, int] = {}
    for r in samples:
        by_source[r["_meta"]["source"]] = by_source.get(r["_meta"]["source"], 0) + 1

    stats = {
        "generated_at": started.strftime("%Y-%m-%d %H:%M:%S"),
        "seconds": round((datetime.now() - started).total_seconds(), 1),
        "candidates": len(questions),
        "kept": len(samples),
        "train": len(train),
        "holdout": len(holdout),
        "by_source": by_source,
        "rejected": rejected,
        "eval_isolation": {"checked": len(eval_questions), "leaked": 0, "method": "逐字 + 字符二元组 Jaccard>=0.6"},
    }
    with open(os.path.join(out_dir, "build_stats.json"), "w", encoding="utf-8") as fh:
        json.dump(stats, fh, ensure_ascii=False, indent=2)
    print("\n" + json.dumps(stats, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen-questions", help="从语料生成候选问题（调 DeepSeek）")
    g.add_argument("--per-paper", type=int, default=7)
    g.add_argument("--limit", type=int, default=None, help="只处理前 N 篇（试跑用）")
    g.add_argument("--out", type=str, default=None)
    g.set_defaults(func=cmd_gen_questions)

    d = sub.add_parser("distill", help="教师打标 + 清洗 + 隔离断言")
    d.add_argument("--in", dest="in_path", type=str, default=None)
    d.add_argument("--out-dir", type=str, default="finetune/data")
    d.add_argument("--limit", type=int, default=None)
    d.set_defaults(func=cmd_distill)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    asyncio.run(args.func(args))


if __name__ == "__main__":
    _bootstrap()
    main()
