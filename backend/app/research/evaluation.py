"""Deterministic evaluation of Harness integrity, not scientific novelty."""

from __future__ import annotations

import hashlib
import json

from sqlmodel import select

from db.models.research import (
    Conclusion,
    ConclusionEvidence,
    ConclusionObservation,
    Evidence,
    EvidenceSearchAttempt,
    Experiment,
    Hypothesis,
    Observation,
    ResearchQuestion,
    ResearchRun,
)
from db.repository.research_repository import ResearchRepository
from research.enums import (
    ConclusionStatus,
    EvidenceType,
    ExperimentStatus,
    HypothesisStatus,
    QuestionStatus,
    RunStatus,
    SearchIntent,
)


class HarnessEvaluator:
    def __init__(self, repository: ResearchRepository):
        self.repository = repository

    async def evaluate(self, research_question_id: str) -> dict:
        question = await self.repository._must_get(ResearchQuestion, research_question_id)
        conclusions = (
            await self.repository.session.execute(
                select(Conclusion).where(Conclusion.research_question_id == question.id)
            )
        ).scalars().all()
        hypotheses = (
            await self.repository.session.execute(
                select(Hypothesis).where(Hypothesis.research_question_id == question.id)
            )
        ).scalars().all()
        experiments = (
            await self.repository.session.execute(
                select(Experiment).where(Experiment.research_question_id == question.id)
            )
        ).scalars().all()
        experiment_ids = [item.id for item in experiments]
        runs = (
            await self.repository.session.execute(
                select(ResearchRun).where(
                    ResearchRun.experiment_id.in_(experiment_ids or ["__none__"])
                )
            )
        ).scalars().all()
        conclusion_ids = [item.id for item in conclusions]
        evidence_links = (
            await self.repository.session.execute(
                select(ConclusionEvidence).where(
                    ConclusionEvidence.conclusion_id.in_(conclusion_ids or ["__none__"])
                )
            )
        ).scalars().all()
        observation_links = (
            await self.repository.session.execute(
                select(ConclusionObservation).where(
                    ConclusionObservation.conclusion_id.in_(conclusion_ids or ["__none__"])
                )
            )
        ).scalars().all()
        link_counts = {item: 0 for item in conclusion_ids}
        for link in evidence_links:
            link_counts[link.conclusion_id] += 1
        for link in observation_links:
            link_counts[link.conclusion_id] += 1
        unsupported = [item for item, count in link_counts.items() if count == 0]

        used_evidence = []
        for evidence_id in dict.fromkeys(link.evidence_id for link in evidence_links):
            used_evidence.append(await self.repository._must_get(Evidence, evidence_id))
        traceable = [item for item in used_evidence if self._traceable(item)]
        literature = [item for item in used_evidence if item.evidence_type == EvidenceType.LITERATURE.value]
        citation_valid = [item for item in literature if self._citation_structurally_valid(item)]

        attempts = (
            await self.repository.session.execute(
                select(EvidenceSearchAttempt).where(
                    EvidenceSearchAttempt.research_question_id == question.id
                )
            )
        ).scalars().all()
        by_hypothesis: dict[str, set[str]] = {}
        for attempt in attempts:
            if attempt.hypothesis_id:
                by_hypothesis.setdefault(attempt.hypothesis_id, set()).add(attempt.search_intent)
        required = [
            hypothesis.id
            for hypothesis in hypotheses
            if SearchIntent.PRIMARY.value in by_hypothesis.get(hypothesis.id, set())
        ]
        covered = [
            item
            for item in required
            if SearchIntent.CONTRADICTION.value in by_hypothesis.get(item, set())
        ]

        invalid_states = []
        if question.status not in {item.value for item in QuestionStatus}:
            invalid_states.append(question.id)
        for hypothesis in hypotheses:
            if hypothesis.status not in {item.value for item in HypothesisStatus}:
                invalid_states.append(hypothesis.id)
        for experiment in experiments:
            if experiment.status not in {item.value for item in ExperimentStatus}:
                invalid_states.append(experiment.id)
        for run in runs:
            if run.status not in {item.value for item in RunStatus}:
                invalid_states.append(run.id)
        for conclusion in conclusions:
            if conclusion.status not in {item.value for item in ConclusionStatus}:
                invalid_states.append(conclusion.id)

        return {
            "research_question_id": question.id,
            "evidence_traceability": self._ratio(len(traceable), len(used_evidence)),
            # Structural integrity is deterministic offline.  Verifying that a
            # page still exists in the live vector index belongs to Phase-2/RAG
            # integration checks and is labelled rather than overstated here.
            "citation_integrity_structural": self._ratio(len(citation_valid), len(literature)),
            "state_consistency": {
                "invalid_count": len(invalid_states),
                "invalid_entity_ids": invalid_states,
            },
            "unsupported_conclusion_rate": self._ratio(len(unsupported), len(conclusions)),
            "unsupported_conclusion_ids": unsupported,
            "contradiction_coverage": self._ratio(len(covered), len(required)),
            "state_fingerprint": await self.state_fingerprint(question.id),
        }

    async def state_fingerprint(self, research_question_id: str) -> str:
        question = await self.repository._must_get(ResearchQuestion, research_question_id)
        hypotheses = (
            await self.repository.session.execute(
                select(Hypothesis).where(Hypothesis.research_question_id == research_question_id)
            )
        ).scalars().all()
        conclusions = (
            await self.repository.session.execute(
                select(Conclusion).where(Conclusion.research_question_id == research_question_id)
            )
        ).scalars().all()
        evidence = (
            await self.repository.session.execute(
                select(Evidence).where(Evidence.research_question_id == research_question_id)
            )
        ).scalars().all()
        experiments = (
            await self.repository.session.execute(
                select(Experiment).where(Experiment.research_question_id == research_question_id)
            )
        ).scalars().all()
        experiment_ids = [item.id for item in experiments]
        runs = (
            await self.repository.session.execute(
                select(ResearchRun).where(
                    ResearchRun.experiment_id.in_(experiment_ids or ["__none__"])
                )
            )
        ).scalars().all()
        observations = (
            await self.repository.session.execute(
                select(Observation).where(
                    Observation.experiment_id.in_(experiment_ids or ["__none__"])
                )
            )
        ).scalars().all()
        data = {
            "question": (question.id, question.status, question.title, question.description),
            "hypotheses": sorted(
                (item.id, item.status, item.parent_hypothesis_id) for item in hypotheses
            ),
            "evidence": sorted(
                (item.id, item.evidence_type, item.source_id, item.page, item.section, item.chunk_id)
                for item in evidence
            ),
            "experiments": sorted(
                (item.id, item.status, item.success_criteria) for item in experiments
            ),
            "runs": sorted(
                (item.id, item.experiment_id, item.status, item.metrics, item.import_hash)
                for item in runs
            ),
            "observations": sorted(
                (item.id, item.experiment_id, item.derived_statistics) for item in observations
            ),
            "conclusions": sorted((item.id, item.status, item.statement) for item in conclusions),
        }
        canonical = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @staticmethod
    def _traceable(evidence: Evidence) -> bool:
        if evidence.evidence_type == EvidenceType.LITERATURE.value:
            return bool(
                evidence.source_id
                and evidence.chunk_id
                and (evidence.page or evidence.section)
                and evidence.excerpt
            )
        return bool(evidence.source_id)

    @staticmethod
    def _citation_structurally_valid(evidence: Evidence) -> bool:
        return bool(
            evidence.source_id
            and evidence.chunk_id
            and evidence.excerpt
            and (evidence.page if evidence.locator_kind == "page" else evidence.section)
        )

    @staticmethod
    def _ratio(numerator: int, denominator: int) -> float:
        return 1.0 if denominator == 0 else round(numerator / denominator, 4)


__all__ = ["HarnessEvaluator"]
