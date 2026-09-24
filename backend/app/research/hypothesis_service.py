"""Hypothesis-centred evidence discovery and persistence."""

from dataclasses import dataclass, field

from db.models.research import (
    Evidence,
    EvidenceRelation,
    EvidenceSearchAttempt,
    Hypothesis,
    ResearchQuestion,
)
from db.repository.research_repository import ResearchRepository
from research.evidence_engine import LiteratureEvidenceEngine
from research.enums import ReviewStatus
from research.search_strategy import build_search_specs
from research.validators import ResearchValidationError


@dataclass
class HypothesisEvidenceSearchResult:
    attempts: list[EvidenceSearchAttempt] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    proposed_relations: list[EvidenceRelation] = field(default_factory=list)


class HypothesisEvidenceService:
    def __init__(self, repository: ResearchRepository, engine: LiteratureEvidenceEngine):
        self.repository = repository
        self.engine = engine

    async def search(
        self,
        question: ResearchQuestion,
        hypothesis: Hypothesis,
        *,
        include_contradiction: bool = True,
        include_limitation: bool = True,
        search_mode: str = "balanced",
        actor: str = "system",
    ) -> HypothesisEvidenceSearchResult:
        if hypothesis.research_question_id != question.id:
            raise ResearchValidationError(
                "hypothesis and question do not belong to the same research project"
            )
        result = HypothesisEvidenceSearchResult()
        seen_evidence: set[str] = set()
        for spec in build_search_specs(
            question,
            hypothesis,
            include_contradiction=include_contradiction,
            include_limitation=include_limitation,
        ):
            batch = await self.engine.search_query(
                question_id=question.id,
                query=spec.query,
                search_intent=spec.intent,
                search_mode=search_mode,
            )
            attempt = await self.repository.create_search_attempt(
                EvidenceSearchAttempt(
                    research_question_id=question.id,
                    hypothesis_id=hypothesis.id,
                    search_intent=spec.intent.value,
                    query=spec.query,
                    search_mode=search_mode,
                    rejected=batch.outcome.rejected,
                    ambiguous=batch.outcome.ambiguous,
                    vector_top1=batch.outcome.vector_top1,
                    result_count=len(batch.evidence),
                ),
                actor=actor,
            )
            result.attempts.append(attempt)
            for draft in batch.evidence:
                evidence = await self.repository.find_evidence_by_chunk(question.id, draft.chunk_id)
                if evidence is None:
                    evidence = await self.repository.create_evidence(draft, actor=actor)
                provenance = draft.provenance
                await self.repository.link_search_hit(
                    attempt_id=attempt.id,
                    evidence_id=evidence.id,
                    rank=int(provenance["rank"]),
                    origin=str(provenance["origin"]),
                    score=provenance.get("retrieval_score"),
                )
                if evidence.id not in seen_evidence:
                    result.evidence.append(evidence)
                    seen_evidence.add(evidence.id)
                relation = await self.repository.find_evidence_relation(
                    hypothesis.id, evidence.id, spec.proposed_relation.value
                )
                if relation is None:
                    relation = await self.repository.create_evidence_relation(
                        EvidenceRelation(
                            hypothesis_id=hypothesis.id,
                            evidence_id=evidence.id,
                            relation=spec.proposed_relation.value,
                            review_status=ReviewStatus.PROPOSED.value,
                            rationale=(
                                f"Candidate from {spec.intent.value.lower()} search; "
                                "requires scientific review."
                            ),
                        ),
                        actor=actor,
                    )
                result.proposed_relations.append(relation)
        return result


__all__ = ["HypothesisEvidenceSearchResult", "HypothesisEvidenceService"]
