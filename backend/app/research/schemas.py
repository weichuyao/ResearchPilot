"""Non-persistent payload schemas for the research domain."""

from pydantic import BaseModel, Field

from research.enums import HypothesisStatus


class HypothesisUpdateProposal(BaseModel):
    """A reasoner's proposal; creating this object never mutates a Hypothesis."""

    hypothesis_id: str
    proposed_status: HypothesisStatus
    supporting_evidence: list[str] = Field(default_factory=list)
    contradicting_evidence: list[str] = Field(default_factory=list)
    supporting_observations: list[str] = Field(default_factory=list)
    reasoning: str
    remaining_uncertainty: str
    suggested_next_experiment: str | None = None


class RunImportPayload(BaseModel):
    experiment_id: str
    run_id: str | None = None
    config: dict = Field(default_factory=dict)
    seed: int | None = None
    metrics: dict
    artifact_paths: list[str] = Field(default_factory=list)
    environment: dict = Field(default_factory=dict)

