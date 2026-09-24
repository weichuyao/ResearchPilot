"""Deterministic scientific-state validation.

No function in this module calls an LLM.  The same rules therefore apply to
API, controller, CLI and tests.
"""

from collections.abc import Mapping, Sequence
from enum import StrEnum

from db.models.research import Evidence, Experiment, Hypothesis, Observation, ResearchQuestion
from research.enums import EvidenceType


class ResearchValidationError(ValueError):
    """A requested research-state mutation violates a domain invariant."""


def require_text(value: str, field: str) -> None:
    if not value or not value.strip():
        raise ResearchValidationError(f"{field} must not be blank")


def require_non_empty(value: Sequence | Mapping, field: str) -> None:
    if not value:
        raise ResearchValidationError(f"{field} must not be empty")


def enum_value(value: str, enum_type: type[StrEnum], field: str) -> str:
    try:
        return enum_type(value).value
    except ValueError as exc:
        allowed = ", ".join(item.value for item in enum_type)
        raise ResearchValidationError(f"invalid {field}: {value!r}; expected one of {allowed}") from exc


def validate_question(question: ResearchQuestion) -> None:
    require_text(question.title, "question.title")
    require_text(question.description, "question.description")


def validate_hypothesis(hypothesis: Hypothesis) -> None:
    require_text(hypothesis.statement, "hypothesis.statement")
    require_text(hypothesis.prediction, "hypothesis.prediction")


def validate_evidence(evidence: Evidence) -> None:
    require_text(evidence.statement, "evidence.statement")
    evidence_type = EvidenceType(evidence.evidence_type)
    if evidence_type is EvidenceType.LITERATURE:
        for field, value in (
            ("evidence.excerpt", evidence.excerpt),
            ("evidence.source_id", evidence.source_id or ""),
            ("evidence.source_type", evidence.source_type or ""),
            ("evidence.chunk_id", evidence.chunk_id or ""),
        ):
            require_text(value, field)
        require_non_empty(evidence.provenance, "evidence.provenance")
        if evidence.locator_kind not in {"page", "section", "block"}:
            raise ResearchValidationError(
                "literature evidence locator_kind must be page, section or block"
            )
        if evidence.locator_kind == "page" and not evidence.page:
            raise ResearchValidationError("literature evidence with page locator requires page")
        if evidence.locator_kind in {"section", "block"} and not evidence.section:
            raise ResearchValidationError(
                f"literature evidence with {evidence.locator_kind} locator requires section"
            )
    elif evidence_type is EvidenceType.EXPERIMENTAL:
        require_text(evidence.source_id or "", "evidence.source_id")
        require_text(evidence.source_type or "", "evidence.source_type")
    elif evidence_type is EvidenceType.DERIVED_ANALYSIS:
        require_text(evidence.source_id or "", "evidence.source_id")
        require_non_empty(evidence.provenance, "evidence.provenance")


def validate_experiment(experiment: Experiment) -> None:
    for field, value in (
        ("experiment.purpose", experiment.purpose),
        ("experiment.independent_variable", experiment.independent_variable),
        ("experiment.control", experiment.control),
        ("experiment.treatment", experiment.treatment),
    ):
        require_text(value, field)
    require_non_empty(experiment.dependent_variables, "experiment.dependent_variables")
    require_non_empty(experiment.metrics, "experiment.metrics")
    require_non_empty(experiment.success_criteria, "experiment.success_criteria")


def validate_observation(observation: Observation) -> None:
    require_non_empty(observation.measured_results, "observation.measured_results")
    require_text(observation.description, "observation.description")


QUESTION_TRANSITIONS = {
    "OPEN": {"ACTIVE", "ARCHIVED"},
    "ACTIVE": {"ANSWERED", "ARCHIVED"},
    "ANSWERED": {"ACTIVE", "ARCHIVED"},
    "ARCHIVED": {"OPEN"},
}

HYPOTHESIS_TRANSITIONS = {
    "PROPOSED": {"TESTABLE", "INCONCLUSIVE"},
    "TESTABLE": {"UNDER_TEST", "WEAKENED", "REFUTED", "INCONCLUSIVE"},
    "UNDER_TEST": {"SUPPORTED", "PARTIALLY_SUPPORTED", "WEAKENED", "REFUTED", "INCONCLUSIVE"},
    "SUPPORTED": {"PARTIALLY_SUPPORTED", "WEAKENED", "REFUTED", "UNDER_TEST"},
    "PARTIALLY_SUPPORTED": {"SUPPORTED", "WEAKENED", "REFUTED", "UNDER_TEST", "INCONCLUSIVE"},
    "WEAKENED": {"PARTIALLY_SUPPORTED", "REFUTED", "UNDER_TEST", "INCONCLUSIVE"},
    "REFUTED": {"WEAKENED", "UNDER_TEST"},
    "INCONCLUSIVE": {"TESTABLE", "UNDER_TEST", "WEAKENED", "REFUTED"},
}

EXPERIMENT_TRANSITIONS = {
    "DRAFT": {"PLANNED", "CANCELLED"},
    "PLANNED": {"APPROVED", "DRAFT", "CANCELLED"},
    "APPROVED": {"RUNNING", "CANCELLED"},
    "RUNNING": {"COMPLETED", "FAILED", "CANCELLED"},
    "COMPLETED": set(),
    "FAILED": {"PLANNED", "CANCELLED"},
    "CANCELLED": {"DRAFT"},
}

RUN_TRANSITIONS = {
    "PENDING": {"RUNNING", "CANCELLED"},
    "RUNNING": {"COMPLETED", "FAILED", "CANCELLED"},
    "COMPLETED": set(),
    "FAILED": {"PENDING"},
    "CANCELLED": set(),
}

CONCLUSION_TRANSITIONS = {
    "DRAFT": {"PENDING_APPROVAL"},
    "PENDING_APPROVAL": {"APPROVED", "REJECTED", "DRAFT"},
    "APPROVED": {"DRAFT"},
    "REJECTED": {"DRAFT"},
}


def validate_transition(current: str, target: str, graph: dict[str, set[str]], entity: str) -> None:
    if target not in graph.get(current, set()):
        raise ResearchValidationError(f"illegal {entity} status transition: {current} -> {target}")
