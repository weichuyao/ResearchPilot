"""B 类越界形态分类器的回归测试。

15 条样本全部是**真实落盘答案**的结论句，来自 2026-09-24 对 B05 的三轮各 5 次复跑
（`reference/design-decisions.md` 决策十一）。标签是逐条人工判读后再写死的 ——
这一步不能省：这条尺子的存在理由就是「模型评委量不稳」，如果它的判据又是模型给的，
等于没换。
"""

from __future__ import annotations

import os
import sys

import pytest

BACKEND = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(BACKEND, "app"))

from ai.eval.overclaim_probe import (  # noqa: E402
    BARE_NEGATIVE,
    COVERAGE,
    QUALIFIED_EXCL,
    UNKNOWN,
    classify,
)

# 修复前：工作流五轮
BEFORE_WORKFLOW = [
    "**结论：不是。** 根据知识库中的材料，DPEFormer 并未使用 GAN 来生成遮挡训练图像；"
    "它使用的是基于 SAM 的 **Realistic Occlusion Augmentation（ROA）** 策略。",
    "根据当前知识库中的材料，**DPEFormer 并不是用 GAN（生成对抗网络）生成遮挡训练图像的**。"
    "它使用的是另一种遮挡增强策略。",
    "**结论：不是。** 根据当前知识库中的材料，DPEFormer 的遮挡训练图像并非由 GAN 生成，"
    "而是由一种名为 **Realistic Occlusion Augmentation（ROA，真实遮挡增强）** 的策略生成。",
    "**结论：不是。** 根据当前知识库中的材料，DPEFormer 并未使用 GAN 来生成遮挡训练图像。",
    "**结论：不是。DPEFormer 的遮挡训练图像不是用 GAN 生成的，"
    "而是用基于 SAM 的 Realistic Occlusion Augmentation (ROA) 策略生成的。**",
]
BEFORE_WORKFLOW_FORMS = [BARE_NEGATIVE, QUALIFIED_EXCL, BARE_NEGATIVE, BARE_NEGATIVE, BARE_NEGATIVE]

# 对照组 agent 同题五轮：越界不是工作流特有的
CONTROL = [
    "**知识库中没有 DPEFormer 这篇论文。**",
    "**不是。** DPEFormer 生成遮挡训练图像用的是 **SAM（Segment Anything Model）**，不是 GAN。",
    "**不是。** 知识库中的 DPEFormer 论文没有使用 GAN 来生成遮挡训练图像。",
    "**不是。** 根据知识库中的论文，DPEFormer 生成遮挡训练图像用的是 SAM，而不是 GAN。",
    "**知识库中没有 DPEFormer 这篇论文。**",
]
CONTROL_FORMS = [COVERAGE, BARE_NEGATIVE, BARE_NEGATIVE, BARE_NEGATIVE, COVERAGE]

# 补上提示规则之后：五轮
AFTER = [
    "根据检索到的材料，DPEFormer 的遮挡训练图像生成方式**不是**通过 GAN（生成对抗网络）。",
    "根据当前知识库中的材料，DPEFormer 的遮挡训练图像**不是**用 GAN 生成的，"
    "而是通过 **Realistic Occlusion Augmentation (ROA)** 策略生成的。",
    "**结论：知识库中没有任何材料说明 DPEFormer 使用 GAN 生成遮挡训练图像。**",
    "**结论：语料库中没有说明 DPEFormer 使用 GAN 生成遮挡训练图像。**",
    "**结论：知识库没有说明 DPEFormer 使用 GAN 生成遮挡训练图像；"
    "它记载的是另一种机制——ROA（Realistic Occlusion Augmentation）。**",
]
AFTER_FORMS = [QUALIFIED_EXCL, QUALIFIED_EXCL, COVERAGE, COVERAGE, COVERAGE]

ALL_SAMPLES = (
    list(zip(BEFORE_WORKFLOW, BEFORE_WORKFLOW_FORMS))
    + list(zip(CONTROL, CONTROL_FORMS))
    + list(zip(AFTER, AFTER_FORMS))
)


@pytest.mark.parametrize("answer,expected", ALL_SAMPLES, ids=range(len(ALL_SAMPLES)))
def test_real_answers_are_classified_as_labelled_by_hand(answer: str, expected: str) -> None:
    assert classify(answer) == expected


def test_classifier_does_not_flag_a_correct_hedge_as_overclaim() -> None:
    """决策十一里那次全文扫描就是栽在这句：拒绝越界的话不能被判成越界。"""
    sentence = ("语料中没有任何一篇论文声明“将 AIS 信号作为 ReID 的输入特征”，"
                "但这属于**语料未覆盖**，并不等于该做法在文献中不存在。")
    assert classify(sentence) == COVERAGE


def test_ambiguous_conclusion_is_abstained_not_guessed() -> None:
    """规则不敢判时必须交给人，而不是猜一个形态出来污染统计。"""
    assert classify("DPEFormer 的遮挡增强值得单独一节来讲。") == UNKNOWN


def test_bare_negative_wins_over_a_later_scope_qualifier() -> None:
    """「结论：不是。根据知识库…」仍然是裸否定 —— 范围限定救不回结论句本身。"""
    assert classify("**结论：不是。** 根据知识库中的材料，该论文并未使用 GAN。") == BARE_NEGATIVE
