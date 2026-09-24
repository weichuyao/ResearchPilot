"""Persistent object model for Scientific Research Harness V1.

These tables contain scientific state, not chat messages.  JSON columns hold
payload-shaped values (metrics/config/provenance); identity and traversable
relationships use foreign-key tables so provenance does not depend on parsing
JSON arrays.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import JSON, Column, Text, UniqueConstraint
from sqlmodel import Field, SQLModel

from research.enums import (
    ApprovalStatus,
    ConclusionStatus,
    ConfidenceLevel,
    EvidenceRelationType,
    EvidenceType,
    ExperimentStatus,
    HypothesisStatus,
    ObservationRelationType,
    QuestionStatus,
    ReviewStatus,
    RunStatus,
    SearchIntent,
    WorkflowStage,
)


def utc_now() -> datetime:
    # SQLModel currently emits TIMESTAMP WITHOUT TIME ZONE for these columns on
    # PostgreSQL.  Persist naive UTC consistently; asyncpg rejects an aware
    # datetime for a timezone-less column.  API serialization still documents
    # these values as UTC research-event timestamps.
    return datetime.now(timezone.utc).replace(tzinfo=None)


def prefixed_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex}"


class ResearchTimestamped(SQLModel):
    created_at: datetime = Field(default_factory=utc_now, nullable=False)
    updated_at: datetime = Field(default_factory=utc_now, nullable=False)


class ResearchQuestion(ResearchTimestamped, table=True):
    __tablename__ = "research_question"

    id: str = Field(default_factory=lambda: prefixed_id("RQ"), primary_key=True, max_length=40)
    title: str = Field(min_length=1, max_length=300, index=True)
    description: str = Field(sa_column=Column(Text, nullable=False))
    background: str = Field(default="", sa_column=Column(Text, nullable=False))
    scope: str = Field(default="", sa_column=Column(Text, nullable=False))
    out_of_scope: str = Field(default="", sa_column=Column(Text, nullable=False))
    status: str = Field(default=QuestionStatus.OPEN.value, max_length=32, index=True)


class Hypothesis(ResearchTimestamped, table=True):
    __tablename__ = "research_hypothesis"

    id: str = Field(default_factory=lambda: prefixed_id("H"), primary_key=True, max_length=40)
    research_question_id: str = Field(
        foreign_key="research_question.id", max_length=40, index=True
    )
    statement: str = Field(sa_column=Column(Text, nullable=False))
    rationale: str = Field(default="", sa_column=Column(Text, nullable=False))
    prediction: str = Field(sa_column=Column(Text, nullable=False))
    status: str = Field(default=HypothesisStatus.PROPOSED.value, max_length=32, index=True)
    parent_hypothesis_id: str | None = Field(
        default=None, foreign_key="research_hypothesis.id", max_length=40, index=True
    )


class Evidence(ResearchTimestamped, table=True):
    __tablename__ = "research_evidence"

    id: str = Field(default_factory=lambda: prefixed_id("E"), primary_key=True, max_length=40)
    research_question_id: str = Field(
        foreign_key="research_question.id", max_length=40, index=True
    )
    evidence_type: str = Field(max_length=32, index=True)
    statement: str = Field(sa_column=Column(Text, nullable=False))
    # Immutable source material.  It is intentionally separate from statement,
    # which may be a human/LLM interpretation of this excerpt.
    excerpt: str = Field(default="", sa_column=Column(Text, nullable=False))
    source_id: str | None = Field(default=None, max_length=300, index=True)
    source_type: str | None = Field(default=None, max_length=40)
    source_title: str | None = Field(default=None, max_length=500)
    page: str | None = Field(default=None, max_length=50)
    section: str | None = Field(default=None, max_length=300)
    locator_prefix: str | None = Field(default=None, max_length=20)
    locator_kind: str | None = Field(default=None, max_length=30)
    chunk_id: str | None = Field(default=None, max_length=160, index=True)
    search_intent: str = Field(default=SearchIntent.MANUAL.value, max_length=32)
    provenance: dict = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))


class EvidenceRelation(SQLModel, table=True):
    __tablename__ = "research_evidence_relation"
    __table_args__ = (
        UniqueConstraint("hypothesis_id", "evidence_id", "relation", name="uq_hyp_evidence_relation"),
    )

    id: str = Field(default_factory=lambda: prefixed_id("ER"), primary_key=True, max_length=40)
    hypothesis_id: str = Field(
        foreign_key="research_hypothesis.id", max_length=40, index=True
    )
    evidence_id: str = Field(foreign_key="research_evidence.id", max_length=40, index=True)
    relation: str = Field(max_length=24, index=True)
    review_status: str = Field(default=ReviewStatus.PROPOSED.value, max_length=24, index=True)
    rationale: str = Field(default="", sa_column=Column(Text, nullable=False))
    created_at: datetime = Field(default_factory=utc_now, nullable=False)
    reviewed_at: datetime | None = Field(default=None)
    reviewed_by: str | None = Field(default=None, max_length=200)


class EvidenceSearchAttempt(SQLModel, table=True):
    __tablename__ = "research_evidence_search_attempt"

    id: str = Field(default_factory=lambda: prefixed_id("SEA"), primary_key=True, max_length=40)
    research_question_id: str = Field(
        foreign_key="research_question.id", max_length=40, index=True
    )
    hypothesis_id: str | None = Field(
        default=None, foreign_key="research_hypothesis.id", max_length=40, index=True
    )
    search_intent: str = Field(max_length=32, index=True)
    query: str = Field(sa_column=Column(Text, nullable=False))
    search_mode: str = Field(default="balanced", max_length=24)
    rejected: bool = Field(default=False)
    ambiguous: bool = Field(default=False)
    vector_top1: float = Field(default=0.0)
    result_count: int = Field(default=0)
    created_at: datetime = Field(default_factory=utc_now, nullable=False, index=True)


class EvidenceSearchHit(SQLModel, table=True):
    __tablename__ = "research_evidence_search_hit"

    search_attempt_id: str = Field(
        foreign_key="research_evidence_search_attempt.id", primary_key=True, max_length=40
    )
    evidence_id: str = Field(foreign_key="research_evidence.id", primary_key=True, max_length=40)
    rank: int = Field(nullable=False)
    origin: str = Field(max_length=24)
    score: float | None = Field(default=None)


class Experiment(ResearchTimestamped, table=True):
    __tablename__ = "research_experiment"

    id: str = Field(default_factory=lambda: prefixed_id("EXP"), primary_key=True, max_length=40)
    research_question_id: str = Field(
        foreign_key="research_question.id", max_length=40, index=True
    )
    purpose: str = Field(sa_column=Column(Text, nullable=False))
    independent_variable: str = Field(sa_column=Column(Text, nullable=False))
    dependent_variables: list = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    control: str = Field(sa_column=Column(Text, nullable=False))
    treatment: str = Field(sa_column=Column(Text, nullable=False))
    controlled_variables: list = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    metrics: list = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    success_criteria: dict = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    status: str = Field(default=ExperimentStatus.DRAFT.value, max_length=24, index=True)


class ExperimentHypothesis(SQLModel, table=True):
    __tablename__ = "research_experiment_hypothesis"

    experiment_id: str = Field(
        foreign_key="research_experiment.id", primary_key=True, max_length=40
    )
    hypothesis_id: str = Field(
        foreign_key="research_hypothesis.id", primary_key=True, max_length=40
    )


class ResearchRun(ResearchTimestamped, table=True):
    __tablename__ = "research_run"

    id: str = Field(default_factory=lambda: prefixed_id("RUN"), primary_key=True, max_length=40)
    experiment_id: str = Field(
        foreign_key="research_experiment.id", max_length=40, index=True
    )
    config: dict = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    seed: int | None = Field(default=None)
    started_at: datetime | None = Field(default=None)
    finished_at: datetime | None = Field(default=None)
    artifact_paths: list = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    metrics: dict = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    status: str = Field(default=RunStatus.PENDING.value, max_length=24, index=True)
    environment: dict = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    raw_import: dict = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    import_hash: str | None = Field(default=None, max_length=128, index=True)


class Observation(ResearchTimestamped, table=True):
    __tablename__ = "research_observation"

    id: str = Field(default_factory=lambda: prefixed_id("O"), primary_key=True, max_length=40)
    experiment_id: str = Field(
        foreign_key="research_experiment.id", max_length=40, index=True
    )
    measured_results: dict = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    derived_statistics: dict = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    description: str = Field(sa_column=Column(Text, nullable=False))


class ObservationRun(SQLModel, table=True):
    __tablename__ = "research_observation_run"

    observation_id: str = Field(
        foreign_key="research_observation.id", primary_key=True, max_length=40
    )
    run_id: str = Field(foreign_key="research_run.id", primary_key=True, max_length=40)


class ObservationHypothesisRelation(SQLModel, table=True):
    __tablename__ = "research_observation_hypothesis"

    observation_id: str = Field(
        foreign_key="research_observation.id", primary_key=True, max_length=40
    )
    hypothesis_id: str = Field(
        foreign_key="research_hypothesis.id", primary_key=True, max_length=40
    )
    relation: str = Field(max_length=24)
    # An observation interpreting a hypothesis is a scientific judgement, not a
    # computation, so it carries the same reviewed/proposed state as an evidence
    # relation: an unreviewed link must never unlock a hypothesis status change.
    review_status: str = Field(default=ReviewStatus.PROPOSED.value, max_length=24, index=True)
    reviewed_by: str | None = Field(default=None, max_length=200)
    reviewed_at: datetime | None = Field(default=None)


class Conclusion(ResearchTimestamped, table=True):
    __tablename__ = "research_conclusion"

    id: str = Field(default_factory=lambda: prefixed_id("C"), primary_key=True, max_length=40)
    research_question_id: str = Field(
        foreign_key="research_question.id", max_length=40, index=True
    )
    statement: str = Field(sa_column=Column(Text, nullable=False))
    confidence: str = Field(default=ConfidenceLevel.LOW.value, max_length=20)
    limitations: list = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    unresolved_questions: list = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    status: str = Field(default=ConclusionStatus.DRAFT.value, max_length=24, index=True)


class ConclusionEvidence(SQLModel, table=True):
    __tablename__ = "research_conclusion_evidence"

    conclusion_id: str = Field(
        foreign_key="research_conclusion.id", primary_key=True, max_length=40
    )
    evidence_id: str = Field(foreign_key="research_evidence.id", primary_key=True, max_length=40)
    role: str = Field(primary_key=True, max_length=24)  # SUPPORTING | CONTRADICTING


class ConclusionObservation(SQLModel, table=True):
    __tablename__ = "research_conclusion_observation"

    conclusion_id: str = Field(
        foreign_key="research_conclusion.id", primary_key=True, max_length=40
    )
    observation_id: str = Field(
        foreign_key="research_observation.id", primary_key=True, max_length=40
    )


class ApprovalRequest(ResearchTimestamped, table=True):
    __tablename__ = "research_approval_request"

    id: str = Field(default_factory=lambda: prefixed_id("APR"), primary_key=True, max_length=40)
    entity_type: str = Field(max_length=40, index=True)
    entity_id: str = Field(max_length=40, index=True)
    action: str = Field(max_length=80, index=True)
    proposed_changes: dict = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    status: str = Field(default=ApprovalStatus.PENDING.value, max_length=24, index=True)
    requested_by: str = Field(default="system", max_length=200)
    reviewed_by: str | None = Field(default=None, max_length=200)
    review_reason: str = Field(default="", sa_column=Column(Text, nullable=False))
    reviewed_at: datetime | None = Field(default=None)


class ResearchEvent(SQLModel, table=True):
    __tablename__ = "research_event"

    id: str = Field(default_factory=lambda: prefixed_id("EVT"), primary_key=True, max_length=40)
    entity_type: str = Field(max_length=40, index=True)
    entity_id: str = Field(max_length=40, index=True)
    event_type: str = Field(max_length=80, index=True)
    actor: str = Field(default="system", max_length=200)
    before: dict = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    after: dict = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    approval_request_id: str | None = Field(
        default=None, foreign_key="research_approval_request.id", max_length=40, index=True
    )
    created_at: datetime = Field(default_factory=utc_now, nullable=False, index=True)


class ResearchWorkflowState(ResearchTimestamped, table=True):
    __tablename__ = "research_workflow_state"

    research_question_id: str = Field(
        foreign_key="research_question.id", primary_key=True, max_length=40
    )
    stage: str = Field(default=WorkflowStage.DEFINE_QUESTION.value, max_length=48, index=True)
    revision: int = Field(default=0, nullable=False)
    context: dict = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))


__all__ = [
    "ApprovalRequest",
    "Conclusion",
    "ConclusionEvidence",
    "ConclusionObservation",
    "Evidence",
    "EvidenceRelation",
    "EvidenceSearchAttempt",
    "EvidenceSearchHit",
    "Experiment",
    "ExperimentHypothesis",
    "Hypothesis",
    "Observation",
    "ObservationHypothesisRelation",
    "ObservationRun",
    "ResearchEvent",
    "ResearchQuestion",
    "ResearchRun",
    "ResearchWorkflowState",
]
