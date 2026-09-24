"""Scientific Research Harness domain package.

The package is deliberately independent from the chat agents.  Its persisted
objects are the authority; LangGraph state may only refer to their identifiers.
"""

from research.enums import (
    ApprovalStatus,
    ConclusionStatus,
    EvidenceRelationType,
    EvidenceType,
    ExperimentStatus,
    HypothesisStatus,
    QuestionStatus,
    ReviewStatus,
    RunStatus,
    WorkflowStage,
)

__all__ = [
    "ApprovalStatus",
    "ConclusionStatus",
    "EvidenceRelationType",
    "EvidenceType",
    "ExperimentStatus",
    "HypothesisStatus",
    "QuestionStatus",
    "ReviewStatus",
    "RunStatus",
    "WorkflowStage",
]
