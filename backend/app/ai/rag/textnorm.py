"""文本归一化：全项目只有这一份实现。

## 为什么需要单独一个模块

同一套归一化要在三个地方用，而且必须**完全一致**：

    ai/rag/ingest.py      导入时把 PDF 文本规范化后写进 Chroma
    ai/eval/run_eval.py   比对「答案必需的锚点」有没有出现在检索结果里
    ai/eval/rank_bench.py 同上，确定性名次基准

写在三处就等于留了三份会各自漂移的副本。今天已经因为「工具返回格式改了、
评估脚本的解析正则没跟着改」导致 `retrieval_recall` 静默掉到 0.25 一次 ——
同一类错误不值得再犯第二遍。

## 归一化什么

**连字（ligature）。** PDF 抽取出来的不是普通 ASCII：实测这份语料里有 257 个连字，
而且都落在核心术语上 —— `identiﬁcation`（65 次）、`reidentiﬁcation`、`ﬁrst`、
`signi signiﬁcant`、`classiﬁcation`。

这不只是显示问题。BM25 的分词器是 `[a-z0-9]+`，遇到 `ﬁ`（U+FB01）不是单词字符，
会把 `identiﬁcation` 切成 `identi` + `cation` 两个垃圾词 —— 于是查询里的
"identification" 在关键词那一路**永远匹配不上**。向量那一路不受影响
（embedding 模型自己会处理），所以这个缺陷一直没被发现。

**破折号。** `association–forgetting` 用的是 en-dash（U+2013，304 次），
与用户输入的普通连字符不是同一个码位。em-dash（U+2014，14 次）故意不转 ——
它是行文里的破折号，不是构词连字符。

**数字分隔符。** PDF 两端对齐会产生 `30, 587` 这种带空格的分隔，比对前要去掉。
"""

from __future__ import annotations

import re

# PDF 连字 -> 普通字母
LIGATURES = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\ufb05": "ft",
    "\ufb06": "st",
}

# 各种破折号 -> 普通连字符。不含 em-dash（U+2014），理由见模块说明。
DASHES = ("\u2010", "\u2011", "\u2012", "\u2013", "\u2212")


def normalize_typography(text: str) -> str:
    """只做字符还原，不改大小写、不动空白。导入时用这个。"""
    for ligature, plain in LIGATURES.items():
        if ligature in text:
            text = text.replace(ligature, plain)
    for dash in DASHES:
        if dash in text:
            text = text.replace(dash, "-")
    return text


def norm_for_match(value) -> str:
    """比对用的归一化：字符还原 + 小写 + 去数字分隔符 + 折叠空白。"""
    text = normalize_typography(str(value or ""))
    text = text.lower()
    text = re.sub(r"(?<=\d)[,\s]+(?=\d)", "", text)   # 30, 587 -> 30587
    text = re.sub(r"[\s\u00a0]+", " ", text)
    return text.strip()


def content_key(text: str) -> str:
    """块级**内容指纹**：归一化文本的 sha1。

    使用方（必须共用这一个实现，否则「这里说不重、那里说重」）：
      · ai/rag/pipeline.py   检索层的跨副本去重（同一 PDF 上传两次，标题归一化
        都拦不住 `A²RNet / A RNet` 这种差异，只有内容指纹可信）
      · ai/rag/ingest.py     导入查重：新论文的块指纹与库中现有块比对，
        单篇来源匹配过半就拒绝入库（见 find_duplicate_owner）
    """
    import hashlib

    return hashlib.sha1(norm_for_match(text).encode("utf-8")).hexdigest()
