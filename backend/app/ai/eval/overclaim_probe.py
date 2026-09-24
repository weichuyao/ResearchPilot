"""B 类答案越界的可复现度量：只判「结论句的形态」，不判全文。

## 为什么不是把 forbidden_claims 改成字符串扫描

`over_claim_rate` 现在是 LLM 评委报的，实测同一道 B05 跑五轮，五轮答案**都**写了
夹具明令禁止的绝对否定，评委只报出一轮 —— 这条指标不能用于回归判定（决策十一）。

把它换成全文规则扫描也**不成立**，实测过：会把「这属于语料未覆盖，并不等于该做法
在文献中不存在」这种**拒绝**越界的句子判成越界，同时漏掉真正 5/5 都出现的
「**结论：不是。**」（那句里没有探针词）。全文自由文本的否定作用层次，规则判不了。

所以这里把判定单元缩到**结论句**，只分四种形态：

    BARE_NEGATIVE   裸否定结论 —— 夹具要求的行为之外的说法，必然越界
    QUALIFIED_EXCL  带语料范围的排除断言 —— 比裸否定好，但仍是"它不用 X"
    COVERAGE        覆盖式陈述 ——「知识库没有说明它用 X」，这才是夹具要的形式
    UNKNOWN         规则不敢判断，交给人

四种形态全部可复现、可单测（tests/eval/test_overclaim_classify.py 用的是真实落盘答案），
而且它不改现有的 `over_claim_rate`，只是**在旁边摆一把量得稳的尺子**。

## 用法

    python app/ai/eval/overclaim_probe.py                       # 扫 resource/eval 下全部报告
    python app/ai/eval/overclaim_probe.py --glob '../tmp/b05_*/*.json'
"""

from __future__ import annotations

import argparse
import collections
import glob
import io
import json
import os
import re
import sys

# 结论句里的「事实排除」断言：不是 / 并非 / 并未 / 未 / 没有 + 用/采用/由/通过/生成
_EXCLUSION = re.compile(
    r"(?:不是|并非|并未|未|没有|不用|不使用)\s*\*{0,2}\s*(?:是|由|用|采用|使用|借助|通过|生成|涉及|包含)")
# 覆盖式陈述：要么是「未/没有 + 提及|说明|检索到|证据…」，要么是「知识库/语料里没有…」
_COVERAGE = re.compile(
    r"(?:未|没有|无|没有任何)[^。；;，]{0,10}(?:提及|提到|说明|报道|记载|记录|检索到|找到|证据|覆盖|涉及)"
    r"|(?:知识库|语料|材料|文献)[^。；，]{0,8}(?:没有|未|无)")
# 明确的否定结论标记：「结论：不是」，或以「**不是。**」开头 —— 本身就是绝对否定
_BARE = re.compile(r"(?:结论\s*[:：]\s*\*{0,2}\s*|^\*{0,2}\s*)(?:不是|并非|未|没有|不需|不存在)")
_SCOPE = re.compile(r"知识库|语料|材料|文献|检索|文档中|已索引")

BARE_NEGATIVE = "BARE_NEGATIVE"
QUALIFIED_EXCL = "QUALIFIED_EXCLUSION"
COVERAGE = "COVERAGE"
UNKNOWN = "UNKNOWN"


def conclusion_sentence(answer: str) -> str:
    """取结论句：第一个非空行（跳过 markdown 的纯符号行）。"""
    for line in (answer or "").splitlines():
        text = line.strip()
        if len(text) >= 6:
            return text
    return (answer or "").strip()


def classify(answer: str) -> str:
    """把结论句归到四种形态之一。规则刻意保守：不确定就 UNKNOWN，交给人。

    顺序有意义：`结论：` 前缀的裸否定要优先于范围限定词（「结论：不是。根据知识库…」
    仍然是裸否定）；而「结论：知识库没有说明…」虽然也以「结论：」开头，否定的对象是
    **语料的覆盖情况**，属于合规形式，所以 COVERAGE 在通用裸否定之前判。
    """
    sentence = conclusion_sentence(answer)
    if not sentence:
        return UNKNOWN
    if re.match(r"^\*{0,2}\s*结论\s*[:：]\s*\*{0,2}\s*(?:不是|并非|未|没有|不需|不存在)", sentence):
        return BARE_NEGATIVE
    has_coverage = bool(_COVERAGE.search(sentence))
    has_exclusion = bool(_EXCLUSION.search(sentence))
    if has_coverage and not has_exclusion:
        return COVERAGE
    if _BARE.search(sentence):
        return BARE_NEGATIVE
    if has_exclusion:
        return QUALIFIED_EXCL if _SCOPE.search(sentence) else BARE_NEGATIVE
    if has_coverage:
        return COVERAGE
    return UNKNOWN


def _bootstrap() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    backend_root = os.path.dirname(os.path.dirname(os.path.dirname(here)))
    os.chdir(backend_root)
    sys.path.insert(0, os.path.join(backend_root, "app"))
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass


def scan(paths: list[str]) -> dict:
    per_item: dict[str, collections.Counter] = {}
    judge_gap: dict[str, int] = {}
    rows = []
    for path in paths:
        try:
            with io.open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            continue
        agent = (data.get("summary") or {}).get("agent") or "?"
        stamp = os.path.basename(path)
        for record in data.get("records") or []:
            if record.get("type") != "B":
                continue
            form = classify(record.get("answer") or "")
            judged = bool(record.get("forbidden_violations"))
            per_item.setdefault(record["id"], collections.Counter())[form] += 1
            # 规则判为越界形态、评委却没报 —— 这就是 over_claim_rate 的漏检面
            if form in {BARE_NEGATIVE, QUALIFIED_EXCL} and not judged:
                judge_gap[record["id"]] = judge_gap.get(record["id"], 0) + 1
            rows.append({
                "report": stamp, "agent": agent, "item": record["id"],
                "form": form, "judge_reported": judged,
                "verdict_correct": bool(record.get("verdict_correct")),
                "conclusion": conclusion_sentence(record.get("answer") or "")[:90],
            })
    return {
        "reports_scanned": len(paths),
        "b_answers_scanned": len(rows),
        "per_item_forms": {k: dict(v) for k, v in sorted(per_item.items())},
        "judge_missed_overclaim_forms": dict(sorted(judge_gap.items())),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", action="append", default=None,
                        help="要扫的报告 JSON 模式，可给多次")
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args()

    _bootstrap()
    patterns = args.glob or [os.path.join("resource", "eval", "eval-*.json")]
    paths = sorted({p for pattern in patterns for p in glob.glob(pattern)})
    if not paths:
        print("no reports matched")
        return

    result = scan(paths)
    print("scanned %d reports, %d B answers" % (
        result["reports_scanned"], result["b_answers_scanned"]))
    print()
    print("%-6s %-16s %-16s %-16s %s" % (
        "item", "BARE_NEGATIVE", "QUALIFIED_EXCL", "COVERAGE", "judge missed"))
    print("-" * 82)
    for item, forms in result["per_item_forms"].items():
        print("%-6s %-16s %-16s %-16s %s" % (
            item, forms.get(BARE_NEGATIVE, "-"), forms.get(QUALIFIED_EXCL, "-"),
            forms.get(COVERAGE, "-"), result["judge_missed_overclaim_forms"].get(item, "-")))
    print()
    print("form legend: BARE_NEGATIVE / QUALIFIED_EXCLUSION = 越界形态；"
          "COVERAGE = 夹具要求的形式；UNKNOWN = 规则不敢判，需人工")
    print("judge missed = 规则判为越界形态而评委没报的次数（over_claim_rate 的漏检面）")

    if args.json:
        with io.open(args.json, "w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False, indent=1)
        print("result: %s" % args.json)


if __name__ == "__main__":
    main()
