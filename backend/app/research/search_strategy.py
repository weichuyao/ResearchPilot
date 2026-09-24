"""Transparent V1 literature-query strategies.

The terms are deliberately explicit and deterministic.  They can be inspected,
tested and replaced later without pretending that query generation itself is
scientific assessment.
"""

from dataclasses import dataclass

from db.models.research import Hypothesis, ResearchQuestion
from research.enums import EvidenceRelationType, SearchIntent


@dataclass(frozen=True)
class SearchSpec:
    intent: SearchIntent
    query: str
    proposed_relation: EvidenceRelationType


CONTRADICTION_TERMS = (
    "failure limitation instability false negative overfitting "
    "negative transfer counterexample no improvement"
)
LIMITATION_TERMS = (
    "limitations boundary conditions sensitivity dataset bias confounder "
    "generalization reproducibility"
)


def build_search_specs(
    question: ResearchQuestion,
    hypothesis: Hypothesis,
    *,
    include_contradiction: bool = True,
    include_limitation: bool = True,
) -> list[SearchSpec]:
    base = f"{question.title}. {question.description}. Hypothesis: {hypothesis.statement}"
    specs = [SearchSpec(SearchIntent.PRIMARY, base, EvidenceRelationType.SUPPORT)]
    if include_contradiction:
        specs.append(
            SearchSpec(
                SearchIntent.CONTRADICTION,
                f"{base}. Contrary evidence: {CONTRADICTION_TERMS}",
                EvidenceRelationType.CONTRADICT,
            )
        )
    if include_limitation:
        specs.append(
            SearchSpec(
                SearchIntent.LIMITATION,
                f"{base}. Scope and limitations: {LIMITATION_TERMS}",
                EvidenceRelationType.LIMITATION,
            )
        )
    return specs


__all__ = ["SearchSpec", "build_search_specs"]
