"""Transactional repository for Scientific Research Harness state.

Methods flush but do not commit.  A caller can therefore compose several
mutations atomically and owns the transaction boundary.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone
from typing import TypeVar

from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import SQLModel, select

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
)
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
)
from research.validators import (
    CONCLUSION_TRANSITIONS,
    EXPERIMENT_TRANSITIONS,
    HYPOTHESIS_TRANSITIONS,
    QUESTION_TRANSITIONS,
    RUN_TRANSITIONS,
    ResearchValidationError,
    enum_value,
    require_text,
    validate_evidence,
    validate_experiment,
    validate_hypothesis,
    validate_observation,
    validate_question,
    validate_transition,
)


ModelT = TypeVar("ModelT", bound=SQLModel)


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _pending_review(status: str | None) -> bool:
    """NULL / 空串都按「未复核」处理。

    补列机制修好之前，给已有表新加的 `review_status` 对旧行写入的是 NULL 而不是
    `PROPOSED`（模型里的 `default=` 只在 ORM 插新行时生效）。如果这里用
    `!= PROPOSED` 判断，NULL 就会被当成"已经复核过了"，那条记录既不能确认也不能
    重判 —— 一个看不见的锁死状态。
    """
    return (status or ReviewStatus.PROPOSED.value) == ReviewStatus.PROPOSED.value


class ResearchNotFoundError(LookupError):
    pass


class ResearchRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def _must_get(self, model: type[ModelT], entity_id: str) -> ModelT:
        entity = await self.session.get(model, entity_id)
        if entity is None:
            raise ResearchNotFoundError(f"{model.__name__} not found: {entity_id}")
        return entity

    async def _event(
        self,
        entity: SQLModel,
        event_type: str,
        *,
        actor: str,
        before: dict | None = None,
        after: dict | None = None,
        approval_request_id: str | None = None,
    ) -> ResearchEvent:
        entity_id = getattr(entity, "id", None) or getattr(entity, "research_question_id", None)
        if not entity_id:
            raise ResearchValidationError(f"cannot audit {type(entity).__name__} without an entity id")
        event = ResearchEvent(
            entity_type=type(entity).__name__,
            entity_id=str(entity_id),
            event_type=event_type,
            actor=actor,
            before=before or {},
            after=after or {},
            approval_request_id=approval_request_id,
        )
        self.session.add(event)
        await self.session.flush()
        return event

    async def _require_approval(
        self, approval_request_id: str | None, *, entity_id: str, action: str
    ) -> ApprovalRequest:
        if not approval_request_id:
            raise ResearchValidationError(f"{action} requires an approved ApprovalRequest")
        approval = await self._must_get(ApprovalRequest, approval_request_id)
        if approval.entity_id != entity_id or approval.action != action:
            raise ResearchValidationError("approval request does not match entity/action")
        if approval.status != ApprovalStatus.APPROVED.value:
            raise ResearchValidationError("approval request is not approved")
        return approval

    async def _require_hypothesis_basis(self, hypothesis_id: str, target: str) -> None:
        """Major scientific status changes need reviewed evidence/observation.

        Human approval is necessary but not sufficient: approving a proposal
        with no scientific source would still create an unsupported state.
        """
        if target in {
            HypothesisStatus.SUPPORTED.value,
            HypothesisStatus.PARTIALLY_SUPPORTED.value,
        }:
            evidence_relation = EvidenceRelationType.SUPPORT.value
            observation_relation = ObservationRelationType.SUPPORT.value
        else:
            evidence_relation = EvidenceRelationType.CONTRADICT.value
            observation_relation = ObservationRelationType.CONTRADICT.value

        evidence_count = await self.session.scalar(
            select(func.count()).select_from(EvidenceRelation).where(
                EvidenceRelation.hypothesis_id == hypothesis_id,
                EvidenceRelation.relation == evidence_relation,
                EvidenceRelation.review_status == ReviewStatus.CONFIRMED.value,
            )
        )
        observation_count = await self.session.scalar(
            select(func.count()).select_from(ObservationHypothesisRelation).where(
                ObservationHypothesisRelation.hypothesis_id == hypothesis_id,
                ObservationHypothesisRelation.relation == observation_relation,
                ObservationHypothesisRelation.review_status == ReviewStatus.CONFIRMED.value,
            )
        )
        if not evidence_count and not observation_count:
            raise ResearchValidationError(
                f"{target} requires confirmed {evidence_relation.lower()} evidence or observation"
            )

    async def create_question(
        self, question: ResearchQuestion, *, actor: str = "human"
    ) -> ResearchQuestion:
        validate_question(question)
        if question.status != QuestionStatus.OPEN.value:
            raise ResearchValidationError("a new research question must start in OPEN")
        self.session.add(question)
        await self.session.flush()
        await self._event(question, "QUESTION_CREATED", actor=actor, after={"status": question.status})
        return question

    async def transition_question(
        self, question_id: str, target: str, *, actor: str = "human"
    ) -> ResearchQuestion:
        question = await self._must_get(ResearchQuestion, question_id)
        target = enum_value(target, QuestionStatus, "question.status")
        validate_transition(question.status, target, QUESTION_TRANSITIONS, "question")
        if target == QuestionStatus.ANSWERED.value:
            count = await self.session.scalar(
                select(func.count()).select_from(Conclusion).where(
                    Conclusion.research_question_id == question.id,
                    Conclusion.status == ConclusionStatus.APPROVED.value,
                )
            )
            if not count:
                raise ResearchValidationError("ANSWERED requires at least one approved conclusion")
        before = question.status
        question.status = target
        question.updated_at = _now()
        self.session.add(question)
        await self._event(
            question, "QUESTION_STATUS_CHANGED", actor=actor,
            before={"status": before}, after={"status": target},
        )
        return question

    async def create_hypothesis(
        self, hypothesis: Hypothesis, *, actor: str = "human"
    ) -> Hypothesis:
        validate_hypothesis(hypothesis)
        if hypothesis.status != HypothesisStatus.PROPOSED.value:
            raise ResearchValidationError("a new hypothesis must start in PROPOSED")
        await self._must_get(ResearchQuestion, hypothesis.research_question_id)
        if hypothesis.parent_hypothesis_id:
            parent = await self._must_get(Hypothesis, hypothesis.parent_hypothesis_id)
            if parent.research_question_id != hypothesis.research_question_id:
                raise ResearchValidationError("parent hypothesis belongs to another research question")
            if parent.id == hypothesis.id:
                raise ResearchValidationError("a hypothesis cannot be its own parent")
        self.session.add(hypothesis)
        await self.session.flush()
        await self._event(
            hypothesis, "HYPOTHESIS_PROPOSED", actor=actor, after={"status": hypothesis.status}
        )
        return hypothesis

    async def transition_hypothesis(
        self,
        hypothesis_id: str,
        target: str,
        *,
        actor: str = "human",
        approval_request_id: str | None = None,
    ) -> Hypothesis:
        hypothesis = await self._must_get(Hypothesis, hypothesis_id)
        target = enum_value(target, HypothesisStatus, "hypothesis.status")
        validate_transition(hypothesis.status, target, HYPOTHESIS_TRANSITIONS, "hypothesis")
        major = {
            HypothesisStatus.SUPPORTED.value,
            HypothesisStatus.PARTIALLY_SUPPORTED.value,
            HypothesisStatus.WEAKENED.value,
            HypothesisStatus.REFUTED.value,
        }
        if target in major:
            await self._require_approval(
                approval_request_id,
                entity_id=hypothesis.id,
                action=f"UPDATE_HYPOTHESIS_STATUS:{target}",
            )
            await self._require_hypothesis_basis(hypothesis.id, target)
        elif hypothesis.status == HypothesisStatus.PROPOSED.value and target == HypothesisStatus.TESTABLE.value:
            await self._require_approval(
                approval_request_id,
                entity_id=hypothesis.id,
                action="REGISTER_HYPOTHESIS",
            )
        before = hypothesis.status
        hypothesis.status = target
        hypothesis.updated_at = _now()
        self.session.add(hypothesis)
        await self._event(
            hypothesis, "HYPOTHESIS_STATUS_CHANGED", actor=actor,
            before={"status": before}, after={"status": target},
            approval_request_id=approval_request_id,
        )
        return hypothesis

    async def create_evidence(
        self, evidence: Evidence, *, actor: str = "human"
    ) -> Evidence:
        evidence.evidence_type = enum_value(evidence.evidence_type, EvidenceType, "evidence.type")
        evidence.search_intent = enum_value(evidence.search_intent, SearchIntent, "search_intent")
        validate_evidence(evidence)
        await self._must_get(ResearchQuestion, evidence.research_question_id)
        self.session.add(evidence)
        await self.session.flush()
        await self._event(evidence, "EVIDENCE_CREATED", actor=actor)
        return evidence

    async def create_evidence_relation(
        self, relation: EvidenceRelation, *, actor: str = "human"
    ) -> EvidenceRelation:
        relation.relation = enum_value(relation.relation, EvidenceRelationType, "evidence.relation")
        if relation.review_status != ReviewStatus.PROPOSED.value:
            raise ResearchValidationError("a new evidence relation must start in PROPOSED")
        hypothesis = await self._must_get(Hypothesis, relation.hypothesis_id)
        evidence = await self._must_get(Evidence, relation.evidence_id)
        if hypothesis.research_question_id != evidence.research_question_id:
            raise ResearchValidationError("hypothesis and evidence belong to different research questions")
        self.session.add(relation)
        await self.session.flush()
        await self._event(relation, "EVIDENCE_RELATION_PROPOSED", actor=actor)
        return relation

    async def find_evidence_by_chunk(
        self, research_question_id: str, chunk_id: str
    ) -> Evidence | None:
        return (
            await self.session.execute(
                select(Evidence).where(
                    Evidence.research_question_id == research_question_id,
                    Evidence.chunk_id == chunk_id,
                )
            )
        ).scalars().first()

    async def find_evidence_relation(
        self, hypothesis_id: str, evidence_id: str, relation: str
    ) -> EvidenceRelation | None:
        return (
            await self.session.execute(
                select(EvidenceRelation).where(
                    EvidenceRelation.hypothesis_id == hypothesis_id,
                    EvidenceRelation.evidence_id == evidence_id,
                    EvidenceRelation.relation == relation,
                )
            )
        ).scalars().first()

    async def create_search_attempt(
        self, attempt: EvidenceSearchAttempt, *, actor: str = "system"
    ) -> EvidenceSearchAttempt:
        attempt.search_intent = enum_value(
            attempt.search_intent, SearchIntent, "search_attempt.intent"
        )
        require_text(attempt.query, "search_attempt.query")
        await self._must_get(ResearchQuestion, attempt.research_question_id)
        if attempt.hypothesis_id:
            hypothesis = await self._must_get(Hypothesis, attempt.hypothesis_id)
            if hypothesis.research_question_id != attempt.research_question_id:
                raise ResearchValidationError(
                    "search attempt hypothesis belongs to another research question"
                )
        self.session.add(attempt)
        await self.session.flush()
        await self._event(attempt, "EVIDENCE_SEARCH_EXECUTED", actor=actor)
        return attempt

    async def link_search_hit(
        self,
        *,
        attempt_id: str,
        evidence_id: str,
        rank: int,
        origin: str,
        score: float | None,
    ) -> EvidenceSearchHit:
        attempt = await self._must_get(EvidenceSearchAttempt, attempt_id)
        evidence = await self._must_get(Evidence, evidence_id)
        if attempt.research_question_id != evidence.research_question_id:
            raise ResearchValidationError(
                "search attempt and evidence belong to different research questions"
            )
        if rank < 1:
            raise ResearchValidationError("search hit rank must be >= 1")
        hit = EvidenceSearchHit(
            search_attempt_id=attempt_id,
            evidence_id=evidence_id,
            rank=rank,
            origin=origin,
            score=score,
        )
        self.session.add(hit)
        await self.session.flush()
        return hit

    async def review_evidence_relation(
        self, relation_id: str, decision: str, *, reviewer: str, rationale: str = ""
    ) -> EvidenceRelation:
        relation = await self._must_get(EvidenceRelation, relation_id)
        decision = enum_value(decision, ReviewStatus, "relation.review_status")
        if decision == ReviewStatus.PROPOSED.value:
            raise ResearchValidationError("review decision must be CONFIRMED or REJECTED")
        if not _pending_review(relation.review_status):
            raise ResearchValidationError("evidence relation has already been reviewed")
        relation.review_status = decision
        relation.reviewed_by = reviewer
        relation.reviewed_at = _now()
        if rationale:
            relation.rationale = rationale
        self.session.add(relation)
        await self._event(relation, "EVIDENCE_RELATION_REVIEWED", actor=reviewer, after={"status": decision})
        return relation

    async def create_experiment(
        self,
        experiment: Experiment,
        hypothesis_ids: Iterable[str],
        *,
        actor: str = "human",
    ) -> Experiment:
        validate_experiment(experiment)
        if experiment.status != ExperimentStatus.DRAFT.value:
            raise ResearchValidationError("a new experiment must start in DRAFT")
        await self._must_get(ResearchQuestion, experiment.research_question_id)
        hypothesis_ids = list(dict.fromkeys(hypothesis_ids))
        if not hypothesis_ids:
            raise ResearchValidationError("experiment must test at least one hypothesis")
        hypotheses = [await self._must_get(Hypothesis, item) for item in hypothesis_ids]
        if any(item.research_question_id != experiment.research_question_id for item in hypotheses):
            raise ResearchValidationError("experiment and hypotheses belong to different research questions")
        if any(item.status == HypothesisStatus.PROPOSED.value for item in hypotheses):
            raise ResearchValidationError("experiment can only test registered hypotheses")
        self.session.add(experiment)
        await self.session.flush()
        self.session.add_all(
            [ExperimentHypothesis(experiment_id=experiment.id, hypothesis_id=item) for item in hypothesis_ids]
        )
        await self.session.flush()
        await self._event(experiment, "EXPERIMENT_CREATED", actor=actor)
        return experiment

    async def transition_experiment(
        self,
        experiment_id: str,
        target: str,
        *,
        actor: str = "human",
        approval_request_id: str | None = None,
    ) -> Experiment:
        experiment = await self._must_get(Experiment, experiment_id)
        target = enum_value(target, ExperimentStatus, "experiment.status")
        validate_transition(experiment.status, target, EXPERIMENT_TRANSITIONS, "experiment")
        if target == ExperimentStatus.APPROVED.value:
            await self._require_approval(
                approval_request_id, entity_id=experiment.id, action="APPROVE_EXPERIMENT"
            )
        before = experiment.status
        experiment.status = target
        experiment.updated_at = _now()
        self.session.add(experiment)
        await self._event(
            experiment, "EXPERIMENT_STATUS_CHANGED", actor=actor,
            before={"status": before}, after={"status": target},
            approval_request_id=approval_request_id,
        )
        return experiment

    async def create_run(self, run: ResearchRun, *, actor: str = "human") -> ResearchRun:
        experiment = await self._must_get(Experiment, run.experiment_id)
        if experiment.status not in {ExperimentStatus.APPROVED.value, ExperimentStatus.RUNNING.value}:
            raise ResearchValidationError("runs can only be added to an approved/running experiment")
        if run.status != RunStatus.PENDING.value:
            raise ResearchValidationError("a new run must start in PENDING")
        self.session.add(run)
        await self.session.flush()
        await self._event(run, "RUN_CREATED", actor=actor)
        return run

    async def transition_run(
        self, run_id: str, target: str, *, actor: str = "human"
    ) -> ResearchRun:
        run = await self._must_get(ResearchRun, run_id)
        target = enum_value(target, RunStatus, "run.status")
        validate_transition(run.status, target, RUN_TRANSITIONS, "run")
        before = run.status
        run.status = target
        run.updated_at = _now()
        if target == RunStatus.RUNNING.value and not run.started_at:
            run.started_at = _now()
        if target in {RunStatus.COMPLETED.value, RunStatus.FAILED.value, RunStatus.CANCELLED.value}:
            run.finished_at = _now()
        if target == RunStatus.COMPLETED.value and not run.metrics:
            raise ResearchValidationError("a completed run requires metrics")
        self.session.add(run)
        await self._event(
            run, "RUN_STATUS_CHANGED", actor=actor,
            before={"status": before}, after={"status": target},
        )
        return run

    async def create_observation(
        self,
        observation: Observation,
        run_ids: Iterable[str],
        hypothesis_relations: dict[str, str] | None = None,
        *,
        actor: str = "human",
    ) -> Observation:
        validate_observation(observation)
        experiment = await self._must_get(Experiment, observation.experiment_id)
        run_ids = list(dict.fromkeys(run_ids))
        if not run_ids:
            raise ResearchValidationError("observation must reference at least one run")
        runs = [await self._must_get(ResearchRun, item) for item in run_ids]
        if any(item.experiment_id != experiment.id for item in runs):
            raise ResearchValidationError("all observation runs must belong to its experiment")
        if any(item.status != RunStatus.COMPLETED.value for item in runs):
            raise ResearchValidationError("observation can only reference completed runs")
        links: list[ObservationHypothesisRelation] = []
        tested_hypothesis_ids = set(
            (
                await self.session.execute(
                    select(ExperimentHypothesis.hypothesis_id).where(
                        ExperimentHypothesis.experiment_id == experiment.id
                    )
                )
            ).scalars().all()
        )
        for hypothesis_id, relation in (hypothesis_relations or {}).items():
            hypothesis = await self._must_get(Hypothesis, hypothesis_id)
            if hypothesis.research_question_id != experiment.research_question_id:
                raise ResearchValidationError("observation hypothesis belongs to another research question")
            if hypothesis_id not in tested_hypothesis_ids:
                raise ResearchValidationError(
                    "observation can only interpret hypotheses tested by its experiment"
                )
            links.append(
                ObservationHypothesisRelation(
                    observation_id=observation.id,
                    hypothesis_id=hypothesis_id,
                    relation=enum_value(relation, ObservationRelationType, "observation.relation"),
                )
            )
        self.session.add(observation)
        await self.session.flush()
        self.session.add_all(
            [ObservationRun(observation_id=observation.id, run_id=item) for item in run_ids] + links
        )
        await self.session.flush()
        await self._event(observation, "OBSERVATION_CREATED", actor=actor)
        return observation

    async def list_observation_relations(
        self, observation_id: str
    ) -> list[ObservationHypothesisRelation]:
        return list(
            (
                await self.session.execute(
                    select(ObservationHypothesisRelation).where(
                        ObservationHypothesisRelation.observation_id == observation_id
                    )
                )
            ).scalars().all()
        )

    async def review_observation_relation(
        self,
        observation_id: str,
        hypothesis_id: str,
        decision: str,
        *,
        reviewer: str,
    ) -> ObservationHypothesisRelation:
        """同 `review_evidence_relation`：观察怎么解读假设是判断，不是计算结果。

        复合主键没有独立的 id，所以定位用 (observation, hypothesis) 这一对。
        """
        relation = await self.session.get(
            ObservationHypothesisRelation, {"observation_id": observation_id,
                                            "hypothesis_id": hypothesis_id}
        )
        if relation is None:
            raise ResearchNotFoundError(
                f"observation relation not found: {observation_id} -> {hypothesis_id}"
            )
        decision = enum_value(decision, ReviewStatus, "relation.review_status")
        if decision == ReviewStatus.PROPOSED.value:
            raise ResearchValidationError("review decision must be CONFIRMED or REJECTED")
        if not _pending_review(relation.review_status):
            raise ResearchValidationError("observation relation has already been reviewed")
        relation.review_status = decision
        relation.reviewed_by = reviewer
        relation.reviewed_at = _now()
        self.session.add(relation)
        observation = await self._must_get(Observation, observation_id)
        await self._event(
            observation,
            "OBSERVATION_RELATION_REVIEWED",
            actor=reviewer,
            after={"hypothesis_id": hypothesis_id, "status": decision},
        )
        return relation

    async def create_conclusion(
        self,
        conclusion: Conclusion,
        *,
        supporting_evidence_ids: Iterable[str] = (),
        contradicting_evidence_ids: Iterable[str] = (),
        observation_ids: Iterable[str] = (),
        actor: str = "human",
    ) -> Conclusion:
        require_text(conclusion.statement, "conclusion.statement")
        conclusion.confidence = enum_value(conclusion.confidence, ConfidenceLevel, "conclusion.confidence")
        if conclusion.status != ConclusionStatus.DRAFT.value:
            raise ResearchValidationError("a new conclusion must start in DRAFT")
        await self._must_get(ResearchQuestion, conclusion.research_question_id)
        supporting = list(dict.fromkeys(supporting_evidence_ids))
        contradicting = list(dict.fromkeys(contradicting_evidence_ids))
        observations = list(dict.fromkeys(observation_ids))
        if not supporting and not observations:
            raise ResearchValidationError(
                "conclusion requires supporting evidence or at least one observation"
            )
        evidence_rows = [
            (await self._must_get(Evidence, item), "SUPPORTING") for item in supporting
        ] + [(await self._must_get(Evidence, item), "CONTRADICTING") for item in contradicting]
        if any(row.research_question_id != conclusion.research_question_id for row, _ in evidence_rows):
            raise ResearchValidationError("conclusion evidence belongs to another research question")
        observation_rows = [await self._must_get(Observation, item) for item in observations]
        scientific_support_types = {
            EvidenceType.LITERATURE.value,
            EvidenceType.EXPERIMENTAL.value,
            EvidenceType.DERIVED_ANALYSIS.value,
        }
        if not observation_rows and not any(
            role == "SUPPORTING" and row.evidence_type in scientific_support_types
            for row, role in evidence_rows
        ):
            raise ResearchValidationError(
                "formal conclusion cannot be supported only by model hypotheses or human notes"
            )
        for observation in observation_rows:
            experiment = await self._must_get(Experiment, observation.experiment_id)
            if experiment.research_question_id != conclusion.research_question_id:
                raise ResearchValidationError("conclusion observation belongs to another research question")
        self.session.add(conclusion)
        await self.session.flush()
        self.session.add_all(
            [
                ConclusionEvidence(
                    conclusion_id=conclusion.id, evidence_id=row.id, role=role
                )
                for row, role in evidence_rows
            ]
            + [
                ConclusionObservation(conclusion_id=conclusion.id, observation_id=row.id)
                for row in observation_rows
            ]
        )
        await self.session.flush()
        await self._event(conclusion, "CONCLUSION_CREATED", actor=actor)
        return conclusion

    async def transition_conclusion(
        self,
        conclusion_id: str,
        target: str,
        *,
        actor: str = "human",
        approval_request_id: str | None = None,
    ) -> Conclusion:
        conclusion = await self._must_get(Conclusion, conclusion_id)
        target = enum_value(target, ConclusionStatus, "conclusion.status")
        validate_transition(conclusion.status, target, CONCLUSION_TRANSITIONS, "conclusion")
        if target == ConclusionStatus.APPROVED.value:
            await self._require_approval(
                approval_request_id, entity_id=conclusion.id, action="APPROVE_CONCLUSION"
            )
        before = conclusion.status
        conclusion.status = target
        conclusion.updated_at = _now()
        self.session.add(conclusion)
        await self._event(
            conclusion, "CONCLUSION_STATUS_CHANGED", actor=actor,
            before={"status": before}, after={"status": target},
            approval_request_id=approval_request_id,
        )
        return conclusion

    async def request_approval(
        self,
        *,
        entity_type: str,
        entity_id: str,
        action: str,
        proposed_changes: dict | None = None,
        requested_by: str = "system",
    ) -> ApprovalRequest:
        require_text(entity_type, "approval.entity_type")
        require_text(entity_id, "approval.entity_id")
        require_text(action, "approval.action")
        changes = proposed_changes or {}
        pending = (
            await self.session.execute(
                select(ApprovalRequest).where(
                    ApprovalRequest.entity_type == entity_type,
                    ApprovalRequest.entity_id == entity_id,
                    ApprovalRequest.action == action,
                    ApprovalRequest.status == ApprovalStatus.PENDING.value,
                )
            )
        ).scalars().first()
        if pending is not None:
            if pending.proposed_changes == changes:
                return pending
            raise ResearchValidationError(
                "a different pending approval already exists for this entity and action"
            )
        approval = ApprovalRequest(
            entity_type=entity_type,
            entity_id=entity_id,
            action=action,
            proposed_changes=changes,
            requested_by=requested_by,
        )
        self.session.add(approval)
        await self.session.flush()
        await self._event(approval, "APPROVAL_REQUESTED", actor=requested_by)
        return approval

    async def review_approval(
        self,
        approval_id: str,
        decision: str,
        *,
        reviewer: str,
        reason: str = "",
    ) -> ApprovalRequest:
        approval = await self._must_get(ApprovalRequest, approval_id)
        decision = enum_value(decision, ApprovalStatus, "approval.status")
        if decision not in {ApprovalStatus.APPROVED.value, ApprovalStatus.REJECTED.value}:
            raise ResearchValidationError("approval decision must be APPROVED or REJECTED")
        if approval.status != ApprovalStatus.PENDING.value:
            raise ResearchValidationError("approval request has already been decided")
        approval.status = decision
        approval.reviewed_by = reviewer
        approval.review_reason = reason
        approval.reviewed_at = _now()
        approval.updated_at = _now()
        self.session.add(approval)
        await self._event(approval, "APPROVAL_REVIEWED", actor=reviewer, after={"status": decision})
        return approval


__all__ = ["ResearchNotFoundError", "ResearchRepository"]
