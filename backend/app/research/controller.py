"""Persistent, deterministic controller for the V1 research workflow."""

from __future__ import annotations

from sqlalchemy import func
from sqlmodel import select

from db.models.research import (
    Conclusion,
    EvidenceSearchAttempt,
    Experiment,
    Hypothesis,
    Observation,
    ResearchEvent,
    ResearchQuestion,
    ResearchRun,
    ResearchWorkflowState,
)
from db.repository.research_repository import ResearchRepository
from research.enums import (
    ConclusionStatus,
    ExperimentStatus,
    HypothesisStatus,
    QuestionStatus,
    RunStatus,
    SearchIntent,
    WorkflowStage,
)
from research.validators import ResearchValidationError, enum_value, validate_transition


WORKFLOW_TRANSITIONS = {
    "DEFINE_QUESTION": {"LITERATURE_REVIEW"},
    "LITERATURE_REVIEW": {"GENERATE_HYPOTHESIS"},
    "GENERATE_HYPOTHESIS": {"SEARCH_SUPPORTING_EVIDENCE"},
    "SEARCH_SUPPORTING_EVIDENCE": {"SEARCH_CONTRADICTING_EVIDENCE"},
    "SEARCH_CONTRADICTING_EVIDENCE": {"ASSESS_EVIDENCE"},
    "ASSESS_EVIDENCE": {"SEARCH_SUPPORTING_EVIDENCE", "DESIGN_EXPERIMENT"},
    "DESIGN_EXPERIMENT": {"HUMAN_APPROVAL"},
    "HUMAN_APPROVAL": {"WAIT_FOR_RESULT", "DESIGN_EXPERIMENT"},
    "WAIT_FOR_RESULT": {"IMPORT_RESULT"},
    "IMPORT_RESULT": {"BUILD_OBSERVATION", "WAIT_FOR_RESULT"},
    "BUILD_OBSERVATION": {"UPDATE_HYPOTHESIS"},
    "UPDATE_HYPOTHESIS": {"DESIGN_EXPERIMENT", "GENERATE_RESEARCH_REPORT"},
    "GENERATE_RESEARCH_REPORT": {"COMPLETE", "UPDATE_HYPOTHESIS"},
    "COMPLETE": set(),
}


class ResearchController:
    def __init__(self, repository: ResearchRepository):
        self.repository = repository

    async def initialize(
        self, research_question_id: str, *, actor: str = "human"
    ) -> ResearchWorkflowState:
        question = await self.repository._must_get(ResearchQuestion, research_question_id)
        if question.status == QuestionStatus.OPEN.value:
            await self.repository.transition_question(
                question.id, QuestionStatus.ACTIVE.value, actor=actor
            )
        existing = await self.repository.session.get(ResearchWorkflowState, research_question_id)
        if existing:
            return existing
        state = ResearchWorkflowState(research_question_id=research_question_id)
        self.repository.session.add(state)
        await self.repository.session.flush()
        await self.repository._event(state, "WORKFLOW_INITIALIZED", actor=actor)
        return state

    async def advance(
        self,
        research_question_id: str,
        target: str,
        *,
        actor: str = "human",
        context_update: dict | None = None,
    ) -> ResearchWorkflowState:
        state = await self.repository.session.get(ResearchWorkflowState, research_question_id)
        if state is None:
            raise ResearchValidationError("research workflow has not been initialized")
        target = enum_value(target, WorkflowStage, "workflow.stage")
        validate_transition(state.stage, target, WORKFLOW_TRANSITIONS, "workflow")
        await self._check_precondition(research_question_id, target)
        before = state.stage
        state.stage = target
        state.revision += 1
        if context_update:
            state.context = {**state.context, **context_update}
        from db.models.research import utc_now
        state.updated_at = utc_now()
        self.repository.session.add(state)
        await self.repository._event(
            state,
            "WORKFLOW_STAGE_CHANGED",
            actor=actor,
            before={"stage": before},
            after={"stage": target, "revision": state.revision},
        )
        if target == WorkflowStage.COMPLETE.value:
            question = await self.repository._must_get(
                ResearchQuestion, research_question_id
            )
            if question.status == QuestionStatus.ACTIVE.value:
                await self.repository.transition_question(
                    question.id, QuestionStatus.ANSWERED.value, actor=actor
                )
        return state

    async def _count(self, statement) -> int:
        return int((await self.repository.session.scalar(statement)) or 0)

    async def _check_precondition(self, question_id: str, target: str) -> None:
        if target == WorkflowStage.SEARCH_SUPPORTING_EVIDENCE.value:
            count = await self._count(
                select(func.count()).select_from(Hypothesis).where(
                    Hypothesis.research_question_id == question_id,
                    Hypothesis.status != HypothesisStatus.PROPOSED.value,
                )
            )
            if not count:
                raise ResearchValidationError("evidence search requires a registered hypothesis")
        elif target == WorkflowStage.SEARCH_CONTRADICTING_EVIDENCE.value:
            count = await self._search_count(question_id, SearchIntent.PRIMARY.value)
            if not count:
                raise ResearchValidationError("contradiction search requires a primary search attempt")
        elif target == WorkflowStage.ASSESS_EVIDENCE.value:
            count = await self._search_count(question_id, SearchIntent.CONTRADICTION.value)
            if not count:
                raise ResearchValidationError("assessment requires a contradiction search attempt")
        elif target == WorkflowStage.HUMAN_APPROVAL.value:
            count = await self._count(
                select(func.count()).select_from(Experiment).where(
                    Experiment.research_question_id == question_id,
                    Experiment.status == ExperimentStatus.PLANNED.value,
                )
            )
            if not count:
                raise ResearchValidationError("human approval stage requires a planned experiment")
        elif target == WorkflowStage.WAIT_FOR_RESULT.value:
            count = await self._count(
                select(func.count()).select_from(Experiment).where(
                    Experiment.research_question_id == question_id,
                    Experiment.status.in_([
                        ExperimentStatus.APPROVED.value,
                        ExperimentStatus.RUNNING.value,
                    ]),
                )
            )
            if not count:
                raise ResearchValidationError("waiting for results requires an approved experiment")
        elif target == WorkflowStage.BUILD_OBSERVATION.value:
            count = await self._count(
                select(func.count()).select_from(ResearchRun).join(Experiment).where(
                    Experiment.research_question_id == question_id,
                    ResearchRun.status == RunStatus.COMPLETED.value,
                )
            )
            if not count:
                raise ResearchValidationError("observation stage requires a completed run")
        elif target == WorkflowStage.UPDATE_HYPOTHESIS.value:
            count = await self._count(
                select(func.count()).select_from(Observation).join(Experiment).where(
                    Experiment.research_question_id == question_id
                )
            )
            if not count:
                raise ResearchValidationError("hypothesis update requires an observation")
        elif target == WorkflowStage.GENERATE_RESEARCH_REPORT.value:
            count = await self._count(
                select(func.count()).select_from(Conclusion).where(
                    Conclusion.research_question_id == question_id,
                    Conclusion.status == ConclusionStatus.APPROVED.value,
                )
            )
            if not count:
                raise ResearchValidationError("formal report requires an approved conclusion")

    async def _search_count(self, question_id: str, intent: str) -> int:
        return await self._count(
            select(func.count()).select_from(EvidenceSearchAttempt).where(
                EvidenceSearchAttempt.research_question_id == question_id,
                EvidenceSearchAttempt.search_intent == intent,
            )
        )


__all__ = ["ResearchController", "WORKFLOW_TRANSITIONS"]
