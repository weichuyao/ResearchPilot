"""Portable, tamper-evident exports of one structured research project."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy

from sqlmodel import select

from db.models.research import (
    ApprovalRequest,
    Conclusion,
    ConclusionEvidence,
    ConclusionObservation,
    Evidence,
    EvidenceRelation,
    EvidenceSearchAttempt,
    EvidenceSearchHit,
    Experiment,
    ExperimentHypothesis,
    Hypothesis,
    Observation,
    ObservationHypothesisRelation,
    ObservationRun,
    ResearchEvent,
    ResearchQuestion,
    ResearchRun,
    ResearchWorkflowState,
    utc_now,
)
from db.repository.research_repository import ResearchRepository
from research.evaluation import HarnessEvaluator
from research.provenance import ProvenanceService
from research.report import ResearchReportService
from research.validators import ResearchValidationError


def _dump(item) -> dict:
    return item.model_dump(mode="json")


class ResearchArchiveService:
    SCHEMA_VERSION = "scientific-research-harness/v1"

    def __init__(self, repository: ResearchRepository):
        self.repository = repository
        self.session = repository.session

    async def export(self, research_question_id: str) -> dict:
        question = await self.repository._must_get(ResearchQuestion, research_question_id)
        hypotheses = await self._all(
            select(Hypothesis).where(Hypothesis.research_question_id == question.id), "id"
        )
        evidence = await self._all(
            select(Evidence).where(Evidence.research_question_id == question.id), "id"
        )
        attempts = await self._all(
            select(EvidenceSearchAttempt).where(
                EvidenceSearchAttempt.research_question_id == question.id
            ), "id"
        )
        hypothesis_ids = [item.id for item in hypotheses]
        evidence_ids = [item.id for item in evidence]
        attempt_ids = [item.id for item in attempts]
        evidence_relations = await self._all(
            select(EvidenceRelation).where(
                EvidenceRelation.hypothesis_id.in_(hypothesis_ids or ["__none__"])
            ), "id"
        )
        search_hits = await self._all(
            select(EvidenceSearchHit).where(
                EvidenceSearchHit.search_attempt_id.in_(attempt_ids or ["__none__"])
            ), "search_attempt_id", "rank"
        )
        experiments = await self._all(
            select(Experiment).where(Experiment.research_question_id == question.id), "id"
        )
        experiment_ids = [item.id for item in experiments]
        experiment_hypotheses = await self._all(
            select(ExperimentHypothesis).where(
                ExperimentHypothesis.experiment_id.in_(experiment_ids or ["__none__"])
            ), "experiment_id", "hypothesis_id"
        )
        runs = await self._all(
            select(ResearchRun).where(
                ResearchRun.experiment_id.in_(experiment_ids or ["__none__"])
            ), "id"
        )
        observations = await self._all(
            select(Observation).where(
                Observation.experiment_id.in_(experiment_ids or ["__none__"])
            ), "id"
        )
        observation_ids = [item.id for item in observations]
        observation_runs = await self._all(
            select(ObservationRun).where(
                ObservationRun.observation_id.in_(observation_ids or ["__none__"])
            ), "observation_id", "run_id"
        )
        observation_hypotheses = await self._all(
            select(ObservationHypothesisRelation).where(
                ObservationHypothesisRelation.observation_id.in_(observation_ids or ["__none__"])
            ), "observation_id", "hypothesis_id"
        )
        conclusions = await self._all(
            select(Conclusion).where(Conclusion.research_question_id == question.id), "id"
        )
        conclusion_ids = [item.id for item in conclusions]
        conclusion_evidence = await self._all(
            select(ConclusionEvidence).where(
                ConclusionEvidence.conclusion_id.in_(conclusion_ids or ["__none__"])
            ), "conclusion_id", "evidence_id", "role"
        )
        conclusion_observations = await self._all(
            select(ConclusionObservation).where(
                ConclusionObservation.conclusion_id.in_(conclusion_ids or ["__none__"])
            ), "conclusion_id", "observation_id"
        )
        entity_ids = [
            question.id, *hypothesis_ids, *evidence_ids, *experiment_ids,
            *[item.id for item in runs], *observation_ids, *conclusion_ids,
            *[item.id for item in evidence_relations], *attempt_ids,
        ]
        approvals = await self._all(
            select(ApprovalRequest).where(ApprovalRequest.entity_id.in_(entity_ids)), "id"
        )
        approval_ids = [item.id for item in approvals]
        events = await self._all(
            select(ResearchEvent).where(
                (ResearchEvent.entity_id.in_([*entity_ids, *approval_ids]))
                | (ResearchEvent.approval_request_id.in_(approval_ids or ["__none__"]))
            ), "created_at", "id"
        )
        workflow = await self.session.get(ResearchWorkflowState, question.id)
        evaluation = await HarnessEvaluator(self.repository).evaluate(question.id)
        provenance = {
            item.id: await ProvenanceService(self.repository).get_provenance(item.id)
            for item in conclusions
        }
        archive = {
            "schema_version": self.SCHEMA_VERSION,
            "exported_at": utc_now().isoformat() + "Z",
            "research_question_id": question.id,
            "state_fingerprint": evaluation["state_fingerprint"],
            "records": {
                "research_question": [_dump(question)],
                "hypotheses": [_dump(item) for item in hypotheses],
                "evidence": [_dump(item) for item in evidence],
                "evidence_relations": [_dump(item) for item in evidence_relations],
                "search_attempts": [_dump(item) for item in attempts],
                "search_hits": [_dump(item) for item in search_hits],
                "experiments": [_dump(item) for item in experiments],
                "experiment_hypotheses": [_dump(item) for item in experiment_hypotheses],
                "runs": [_dump(item) for item in runs],
                "observations": [_dump(item) for item in observations],
                "observation_runs": [_dump(item) for item in observation_runs],
                "observation_hypotheses": [_dump(item) for item in observation_hypotheses],
                "conclusions": [_dump(item) for item in conclusions],
                "conclusion_evidence": [_dump(item) for item in conclusion_evidence],
                "conclusion_observations": [_dump(item) for item in conclusion_observations],
                "approvals": [_dump(item) for item in approvals],
                "events": [_dump(item) for item in events],
                "workflow": [_dump(workflow)] if workflow else [],
            },
            "evaluation": evaluation,
            "conclusion_provenance": provenance,
            "report_markdown": await ResearchReportService(self.repository).generate(question.id),
        }
        archive["archive_sha256"] = self._digest(archive)
        return archive

    async def restore(self, archive: dict) -> dict:
        """Restore one verified archive without overwriting existing entities."""
        if archive.get("schema_version") != self.SCHEMA_VERSION:
            raise ResearchValidationError("unsupported research archive schema version")
        if not self.verify(archive):
            raise ResearchValidationError("research archive SHA-256 verification failed")
        records = archive.get("records")
        if not isinstance(records, dict):
            raise ResearchValidationError("research archive records are missing")
        questions = records.get("research_question") or []
        if len(questions) != 1:
            raise ResearchValidationError("research archive must contain exactly one question")
        question_id = questions[0].get("id")
        if question_id != archive.get("research_question_id"):
            raise ResearchValidationError("research archive question identity mismatch")
        self._validate_scope(records, question_id)
        if await self.session.get(ResearchQuestion, question_id) is not None:
            raise ResearchValidationError(f"research question already exists: {question_id}")

        await self._insert(ResearchQuestion, questions)
        await self._insert_hypotheses(records.get("hypotheses") or [], question_id)
        await self._insert(Evidence, records.get("evidence") or [])
        await self._insert(EvidenceSearchAttempt, records.get("search_attempts") or [])
        await self._insert(EvidenceRelation, records.get("evidence_relations") or [])
        await self._insert(EvidenceSearchHit, records.get("search_hits") or [])
        await self._insert(Experiment, records.get("experiments") or [])
        await self._insert(ExperimentHypothesis, records.get("experiment_hypotheses") or [])
        await self._insert(ResearchRun, records.get("runs") or [])
        await self._insert(Observation, records.get("observations") or [])
        await self._insert(ObservationRun, records.get("observation_runs") or [])
        await self._insert(
            ObservationHypothesisRelation, records.get("observation_hypotheses") or []
        )
        await self._insert(Conclusion, records.get("conclusions") or [])
        await self._insert(ConclusionEvidence, records.get("conclusion_evidence") or [])
        await self._insert(
            ConclusionObservation, records.get("conclusion_observations") or []
        )
        await self._insert(ApprovalRequest, records.get("approvals") or [])
        await self._insert(ResearchEvent, records.get("events") or [])
        await self._insert(ResearchWorkflowState, records.get("workflow") or [])

        fingerprint = await HarnessEvaluator(self.repository).state_fingerprint(question_id)
        if fingerprint != archive.get("state_fingerprint"):
            raise ResearchValidationError("restored state fingerprint does not match archive")
        return {
            "research_question_id": question_id,
            "state_fingerprint": fingerprint,
            "record_count": sum(len(rows) for rows in records.values()),
        }

    async def _all(self, statement, *sort_fields: str) -> list:
        rows = (await self.session.execute(statement)).scalars().all()
        return sorted(rows, key=lambda item: tuple(str(getattr(item, name)) for name in sort_fields))

    @staticmethod
    def _validate_scope(records: dict, question_id: str) -> None:
        """Reject internally re-signed archives containing cross-project records."""
        def rows(name: str) -> list[dict]:
            value = records.get(name) or []
            if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
                raise ResearchValidationError(f"archive record set is invalid: {name}")
            return value

        direct = ("hypotheses", "evidence", "search_attempts", "experiments", "conclusions")
        for name in direct:
            if any(row.get("research_question_id") != question_id for row in rows(name)):
                raise ResearchValidationError(
                    f"archive contains {name} from another research question"
                )
        workflow = rows("workflow")
        if len(workflow) > 1 or any(
            row.get("research_question_id") != question_id for row in workflow
        ):
            raise ResearchValidationError("archive workflow scope does not match question")

        ids = {
            "hypotheses": {row.get("id") for row in rows("hypotheses")},
            "evidence": {row.get("id") for row in rows("evidence")},
            "search_attempts": {row.get("id") for row in rows("search_attempts")},
            "experiments": {row.get("id") for row in rows("experiments")},
            "runs": {row.get("id") for row in rows("runs")},
            "observations": {row.get("id") for row in rows("observations")},
            "conclusions": {row.get("id") for row in rows("conclusions")},
            "evidence_relations": {row.get("id") for row in rows("evidence_relations")},
            "approvals": {row.get("id") for row in rows("approvals")},
        }
        checks = (
            ("evidence_relations", "hypothesis_id", "hypotheses"),
            ("evidence_relations", "evidence_id", "evidence"),
            ("search_hits", "search_attempt_id", "search_attempts"),
            ("search_hits", "evidence_id", "evidence"),
            ("experiment_hypotheses", "experiment_id", "experiments"),
            ("experiment_hypotheses", "hypothesis_id", "hypotheses"),
            ("runs", "experiment_id", "experiments"),
            ("observations", "experiment_id", "experiments"),
            ("observation_runs", "observation_id", "observations"),
            ("observation_runs", "run_id", "runs"),
            ("observation_hypotheses", "observation_id", "observations"),
            ("observation_hypotheses", "hypothesis_id", "hypotheses"),
            ("conclusion_evidence", "conclusion_id", "conclusions"),
            ("conclusion_evidence", "evidence_id", "evidence"),
            ("conclusion_observations", "conclusion_id", "conclusions"),
            ("conclusion_observations", "observation_id", "observations"),
        )
        for name, field, target in checks:
            if any(row.get(field) not in ids[target] for row in rows(name)):
                raise ResearchValidationError(
                    f"archive {name}.{field} references an out-of-scope entity"
                )
        for row in rows("hypotheses"):
            parent = row.get("parent_hypothesis_id")
            if parent and parent not in ids["hypotheses"]:
                raise ResearchValidationError("archive hypothesis parent is out of scope")

        scoped_entities = {
            question_id,
            *ids["hypotheses"], *ids["evidence"], *ids["search_attempts"],
            *ids["experiments"], *ids["runs"], *ids["observations"],
            *ids["conclusions"], *ids["evidence_relations"],
        }
        if any(row.get("entity_id") not in scoped_entities for row in rows("approvals")):
            raise ResearchValidationError("archive approval references an out-of-scope entity")
        event_entities = scoped_entities | ids["approvals"]
        for row in rows("events"):
            if row.get("entity_id") not in event_entities:
                raise ResearchValidationError("archive event references an out-of-scope entity")
            approval_id = row.get("approval_request_id")
            if approval_id and approval_id not in ids["approvals"]:
                raise ResearchValidationError(
                    "archive event references an out-of-scope approval"
                )

    async def _insert(self, model, rows: list[dict]) -> None:
        if not rows:
            return
        self.session.add_all([model.model_validate(row) for row in rows])
        await self.session.flush()

    async def _insert_hypotheses(self, rows: list[dict], question_id: str) -> None:
        pending = {row["id"]: row for row in rows}
        inserted: set[str] = set()
        while pending:
            ready = [
                row for row in pending.values()
                if not row.get("parent_hypothesis_id")
                or row.get("parent_hypothesis_id") in inserted
            ]
            if not ready:
                raise ResearchValidationError("hypothesis lineage is cyclic or incomplete")
            for row in ready:
                if row.get("research_question_id") != question_id:
                    raise ResearchValidationError("archive contains a hypothesis from another question")
                self.session.add(Hypothesis.model_validate(row))
                inserted.add(row["id"])
                pending.pop(row["id"])
            await self.session.flush()

    @classmethod
    def verify(cls, archive: dict) -> bool:
        expected = archive.get("archive_sha256")
        return isinstance(expected, str) and expected == cls._digest(archive)

    @staticmethod
    def _digest(archive: dict) -> str:
        payload = deepcopy(archive)
        payload.pop("archive_sha256", None)
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


__all__ = ["ResearchArchiveService"]
