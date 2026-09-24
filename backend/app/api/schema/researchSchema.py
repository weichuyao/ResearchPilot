"""HTTP payloads for the Scientific Research Harness API."""

from typing import Literal

from pydantic import BaseModel, Field

from research.schemas import HypothesisUpdateProposal


class QuestionCreate(BaseModel):
    title: str
    description: str
    background: str = ""
    scope: str = ""
    out_of_scope: str = ""


class HypothesisCreate(BaseModel):
    statement: str
    rationale: str = ""
    prediction: str
    parent_hypothesis_id: str | None = None


class EvidenceSearchIn(BaseModel):
    search_mode: Literal["balanced", "precise", "broad"] = "balanced"
    include_contradiction: bool = True
    include_limitation: bool = True


class EvidenceRelationReviewIn(BaseModel):
    decision: Literal["CONFIRMED", "REJECTED"]
    reviewer: str
    rationale: str = ""


class ExperimentCreate(BaseModel):
    research_question_id: str
    tested_hypotheses: list[str]
    purpose: str
    independent_variable: str
    dependent_variables: list[str]
    control: str
    treatment: str
    controlled_variables: list[str] = Field(default_factory=list)
    metrics: list[str]
    success_criteria: dict


class HumanDecisionIn(BaseModel):
    reviewer: str
    decision: Literal["APPROVED", "REJECTED"] = "APPROVED"
    reason: str = ""


class ActorIn(BaseModel):
    actor: str = "human"


class RunImportIn(BaseModel):
    source_format: Literal["json", "yaml", "yml", "csv"] = "json"
    content: str
    requested_by: str = "system"


class RunImportConfirmIn(RunImportIn):
    approval_request_id: str
    actor: str


class ConclusionCreate(BaseModel):
    statement: str
    confidence: Literal["LOW", "MEDIUM", "HIGH"] = "LOW"
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    contradicting_evidence_ids: list[str] = Field(default_factory=list)
    observation_ids: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)


class ObservationCreate(BaseModel):
    run_ids: list[str]
    hypothesis_relations: dict[str, Literal["SUPPORT", "CONTRADICT", "INCONCLUSIVE"]] = Field(
        default_factory=dict
    )
    actor: str = "human"


class WorkflowAdvanceIn(BaseModel):
    target: str
    actor: str = "human"
    context_update: dict = Field(default_factory=dict)


__all__ = [
    "ActorIn",
    "ConclusionCreate",
    "EvidenceRelationReviewIn",
    "EvidenceSearchIn",
    "ExperimentCreate",
    "HumanDecisionIn",
    "HypothesisCreate",
    "HypothesisUpdateProposal",
    "ObservationCreate",
    "QuestionCreate",
    "RunImportConfirmIn",
    "RunImportIn",
    "WorkflowAdvanceIn",
]
