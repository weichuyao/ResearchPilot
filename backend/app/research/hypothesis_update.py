"""Rule-validated hypothesis update proposals.

An LLM may construct HypothesisUpdateProposal, but only this service can turn a
reviewed proposal into a state transition.
"""

from sqlmodel import select

from db.models.research import (
    ApprovalRequest,
    Evidence,
    EvidenceRelation,
    Hypothesis,
    Observation,
    ObservationHypothesisRelation,
)
from db.repository.research_repository import ResearchRepository
from research.enums import (
    ApprovalStatus,
    EvidenceRelationType,
    HypothesisStatus,
    ObservationRelationType,
    ReviewStatus,
)
from research.schemas import HypothesisUpdateProposal
from research.validators import ResearchValidationError, require_text


class HypothesisUpdateService:
    def __init__(self, repository: ResearchRepository):
        self.repository = repository

    async def propose(
        self,
        proposal: HypothesisUpdateProposal,
        *,
        requested_by: str = "reasoner",
    ) -> ApprovalRequest:
        hypothesis = await self.repository._must_get(Hypothesis, proposal.hypothesis_id)
        require_text(proposal.reasoning, "proposal.reasoning")
        require_text(proposal.remaining_uncertainty, "proposal.remaining_uncertainty")
        await self._validate_sources(hypothesis, proposal)
        target = proposal.proposed_status.value
        if target in {HypothesisStatus.SUPPORTED.value, HypothesisStatus.PARTIALLY_SUPPORTED.value}:
            if not proposal.supporting_evidence and not proposal.supporting_observations:
                raise ResearchValidationError("supporting status requires supporting sources")
        if target in {HypothesisStatus.WEAKENED.value, HypothesisStatus.REFUTED.value}:
            if not proposal.contradicting_evidence:
                raise ResearchValidationError("weakening/refuting status requires contradictory evidence")
        return await self.repository.request_approval(
            entity_type="Hypothesis",
            entity_id=hypothesis.id,
            action=f"UPDATE_HYPOTHESIS_STATUS:{target}",
            proposed_changes=proposal.model_dump(mode="json"),
            requested_by=requested_by,
        )

    async def apply(
        self, approval_request_id: str, *, actor: str
    ) -> Hypothesis:
        approval = await self.repository._must_get(ApprovalRequest, approval_request_id)
        if approval.status != ApprovalStatus.APPROVED.value:
            raise ResearchValidationError("hypothesis update proposal is not approved")
        if not approval.action.startswith("UPDATE_HYPOTHESIS_STATUS:"):
            raise ResearchValidationError("approval is not a hypothesis status update")
        proposal = HypothesisUpdateProposal.model_validate(approval.proposed_changes)
        if proposal.hypothesis_id != approval.entity_id:
            raise ResearchValidationError("proposal hypothesis does not match approval entity")
        hypothesis = await self.repository._must_get(Hypothesis, proposal.hypothesis_id)
        await self._validate_sources(hypothesis, proposal)
        return await self.repository.transition_hypothesis(
            hypothesis.id,
            proposal.proposed_status.value,
            actor=actor,
            approval_request_id=approval.id,
        )

    async def _validate_sources(
        self, hypothesis: Hypothesis, proposal: HypothesisUpdateProposal
    ) -> None:
        for evidence_id in proposal.supporting_evidence:
            await self._require_confirmed_relation(
                hypothesis, evidence_id, EvidenceRelationType.SUPPORT.value
            )
        for evidence_id in proposal.contradicting_evidence:
            await self._require_confirmed_relation(
                hypothesis, evidence_id, EvidenceRelationType.CONTRADICT.value
            )
        for observation_id in proposal.supporting_observations:
            observation = await self.repository._must_get(Observation, observation_id)
            relation = (
                await self.repository.session.execute(
                    select(ObservationHypothesisRelation).where(
                        ObservationHypothesisRelation.observation_id == observation.id,
                        ObservationHypothesisRelation.hypothesis_id == hypothesis.id,
                        ObservationHypothesisRelation.relation == ObservationRelationType.SUPPORT.value,
                    )
                )
            ).scalars().first()
            if relation is None:
                raise ResearchValidationError(
                    f"observation {observation.id} is not linked as SUPPORT"
                )

    async def _require_confirmed_relation(
        self, hypothesis: Hypothesis, evidence_id: str, relation_type: str
    ) -> None:
        evidence = await self.repository._must_get(Evidence, evidence_id)
        if evidence.research_question_id != hypothesis.research_question_id:
            raise ResearchValidationError("proposal evidence belongs to another research question")
        relation = (
            await self.repository.session.execute(
                select(EvidenceRelation).where(
                    EvidenceRelation.hypothesis_id == hypothesis.id,
                    EvidenceRelation.evidence_id == evidence.id,
                    EvidenceRelation.relation == relation_type,
                    EvidenceRelation.review_status == ReviewStatus.CONFIRMED.value,
                )
            )
        ).scalars().first()
        if relation is None:
            raise ResearchValidationError(
                f"evidence {evidence.id} lacks confirmed {relation_type} relation"
            )


__all__ = ["HypothesisUpdateService"]

