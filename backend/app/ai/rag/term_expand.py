"""语料侧术语补全：把「模型背不出的字面术语」交给语料来提供。

> ## ⚠️ 先说结论：这个方案**实测无效**，默认关闭（`TERM_EXPAND_ENABLED=False`）
>
> 保留代码是为了留下证据、避免以后有人盲目重试。实测数据（`rewrite_bench`，
> 微调后的改写器，含重排，22 道参评题）：
>
> | 配置 | recall@1 | recall@5 | mrr | 锚点全中 |
> |---|---|---|---|---|
> | 微调后（不补全） | **0.3636** | 0.6364 | **0.4884** | **16/22** |
> | ＋补全 1 个术语 | 0.3182 | **0.7273** | 0.4797 | 13/22 |
> | ＋补全 2 个术语 | 0.2727 | 0.5909 | 0.4486 | 14/22 |
> | ＋补全 4 个术语 | 0.2727 | 0.6818 | 0.4520 | 15/22 |
>
> **稳定模式：补得越多越差。** 它用「recall@1 / 精度 / 锚点」换「recall@5 小幅提升」，
> 而锚点（证据完整性）恰恰是这个项目的核心指标。不带重排的 4 个配置也是同一模式，
> 说明**不是重排能纠正的问题**。
>
> 机制解释：追加的稀有词组会让 BM25 那一路主导 RRF 融合，检索被拉向
> 「含这些词组的块」，而不是「最能回答问题的块」—— 术语挖对了（都是目标论文的
> 真实术语），但"对论文的术语"不等于"这次问题需要的证据"。
>
> **如果还要继续试**，唯一没试过且机制上不同的是：
> **保留原查询结果 + 扩张结果做并集，再整体重排**（而不是把术语拼进原查询）。
> 那能保住原有的精度，同时让被漏掉的块有机会进来。

## 它想解决什么问题

微调后的 1.5B 改写器学会了「输出英文查询」这个格式（回落率 93%→0%），
但**背不出论文里的字面术语**。实测 A09：

    微调后    query: A²RNet temporal attention re-identification history features ...
              facets: ['A²RNet', 'historical features', 'TAR']        ← 泛化词
    正确的    含 'instance bank' / 'identity-view pair'               ← 论文原词

这不是「改写能力」缺失，是**领域知识**缺失：模型得先知道「A²RNet 用的是
instance bank 这个词」才能输出它。1.5B + 360 条样本装不进这种知识，
而 DeepSeek 靠预训练知识所以能。往数据里再强调、加训练轮数都不对路 ——
加轮数只会让它更牢地记住训练集里那 360 个问题的术语（过拟合），
还可能加重它已经出现的**术语幻觉**（实测它把 TAR 编造成
"Temporal Adaptive Representation"）。

## 所以：知识放到它能被验证的地方

模型只负责它已经学会的那部分（中文问题 → 英文查询骨架），
**术语从这个项目自己的 BM25 索引里现取** —— 语料里真实出现过才算数，
模型编不出来。术语有了语料出处，幻觉问题也一并消失。

## 怎么挑术语（打分信号是实测调出来的）

从「模型查询检出的块」里挖 2~3 元词组，三个约束缺一不可：

  1. **锚点约束**：只用含「查询里 IDF 最高的几个词」的块。不加这条时，
     ViV-ReID 的问题会挖出 `vessel reid`（另一篇船舶论文的术语）—— 跨论文噪声。
  2. **多块佐证**：词组至少出现在 2~3 个候选块里。只在一个块里出现的，
     多半是那一段的偶然措辞，不是这篇论文的术语。
  3. **按 IDF 和（稀有度）排序**，配停用词与数字过滤。第一版按「出现块数」排序，
     结果前 12 名全是 `of the` / `person re` 这类常见短语，目标术语一条都没进来。

## 它不做什么

**不试图猜「用户要的证据是哪个术语」** —— 那是做不到的：语料里有大量稀有术语，
判断哪两三个才是答案需要理解问题本身。这里只做一件事：把查询里缺的、
**这篇论文确实在用的**术语补进去，让检索落到正确的论文与页码上。
能不能答对仍然由 retrieval + synthesize 决定。

（上一条"把术语交给语料"的思路是对的、术语也挖对了；失败的是**用它的方式**
—— 追加到查询会破坏原有排序。见文件开头的后续建议。）
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# 数字一律排除：实测 A11 的候选词被 `7 165 tracklets` / `145 469 frames` 这类
# 纯数字词组占满（数字的 IDF 天然极高），它们对检索没有区分价值。
_HAS_DIGIT = re.compile(r"\d")

# 词组长度：2~3 元。单字太少（多是虚词或半截词），4 元以上几乎不重复出现。
_GRAM_MIN, _GRAM_MAX = 2, 3


def _grams(tokens: list[str]) -> list[tuple[str, ...]]:
    out = []
    for n in range(_GRAM_MIN, _GRAM_MAX + 1):
        for i in range(len(tokens) - n + 1):
            out.append(tuple(tokens[i:i + n]))
    return out


def mine_terms(
    query: str,
    *,
    top_chunks: int = 20,
    max_terms: int = 4,
    min_chunks: int = 3,
    anchor_count: int = 3,
) -> list[str]:
    """从语料里挖出与查询相关、且**稀有**的字面术语。返回按稀有度降序的词组。

    失败一律返回空列表（调用方按"没有补全"处理）—— 检索链路不能因为
    术语补全出问题就整体失败，这类降级必须静默但可见（见模块文档的调试入口）。
    """
    try:
        from collections import Counter

        from ai.rag.hybrid import STOPWORDS, get_index, hybrid_search, tokenize

        index, _documents, _keys = get_index()
        vocab, idf = index.vocab, index.idf

        # ---- 1. 锚点词：查询里 IDF 最高的几个（必须是语料词表里的）----
        q_tokens = [t for t in tokenize(query) if t in vocab and t not in STOPWORDS]
        q_tokens.sort(key=lambda t: -float(idf[vocab[t]]))
        anchors = set(q_tokens[:anchor_count])
        if not anchors:
            return []
        q_token_set = set(q_tokens)

        # ---- 2. 候选块：检索一遍，只要含锚点的 ----
        hits, _top1 = hybrid_search(query, top_n=top_chunks)
        counts: Counter = Counter()
        for doc, _origin, _score in hits:
            tokens = tokenize(doc.page_content)
            if not (anchors & set(tokens)):
                continue
            counts.update(_grams(tokens))

        # ---- 3. 打分与过滤 ----
        scored = []
        for gram, seen in counts.items():
            if seen < min_chunks:
                continue
            if any(t in STOPWORDS for t in gram):
                continue
            if any(_HAS_DIGIT.search(t) for t in gram):
                continue
            # 已经在查询里的词不占配额（补了也是重复）
            if set(gram) <= q_token_set:
                continue
            scored.append((gram, sum(float(idf[vocab[t]]) for t in gram)))
        scored.sort(key=lambda item: -item[1])

        # ---- 4. 去重叠：长词组优先（`association perception sap` 收下后，
        #        就不再收 `association perception`）----
        picked: list[tuple[str, ...]] = []
        for gram, _score in scored:
            gset = set(gram)
            if any(gset <= set(p) or set(p) <= gset for p in picked):
                continue
            picked.append(gram)
            if len(picked) >= max_terms:
                break
        return [" ".join(g) for g in picked]
    except Exception as exc:      # 术语补全不能拖垮检索链路
        logger.warning("术语补全失败（按无补全继续）：%s", exc)
        return []


def augment(query: str, facets: list[str] | None, terms: list[str]) -> tuple[str, list[str]]:
    """把挖到的术语接进查询与 facets。

    查询侧：直接追加 —— BM25 那一路按词匹配，追加稀有术语能把结果拉向
    该论文的细节页（这正是 A09 失败的地方）。
    facets 侧：补进去，让 assess 节点有更硬的判据（它按字面匹配检索文本）。
    """
    if not terms:
        return query, list(facets or [])
    new_query = (query + " " + " ".join(terms)).strip()
    merged = list(facets or [])
    for t in terms:
        if t not in merged:
            merged.append(t)
    return new_query, merged
