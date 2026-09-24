"""Incremental API for structured research state."""

from fastapi import APIRouter, HTTPException
from sqlmodel import select

from api.schema.researchSchema import (
    ActorIn,
    ConclusionCreate,
    EvidenceRelationReviewIn,
    EvidenceSearchIn,
    ExperimentCreate,
    HumanDecisionIn,
    HypothesisCreate,
    HypothesisUpdateProposal,
    QuestionCreate,
    ObservationCreate,
    ObservationRelationReviewIn,
    RunImportConfirmIn,
    RunImportIn,
    WorkflowAdvanceIn,
)
from db.database import SessionDep
from db.models.research import (
    ApprovalRequest,
    Conclusion,
    Evidence,
    EvidenceRelation,
    EvidenceSearchAttempt,
    Experiment,
    ExperimentHypothesis,
    Hypothesis,
    Observation,
    ObservationHypothesisRelation,
    ObservationRun,
    ResearchQuestion,
    ResearchRun,
    ResearchWorkflowState,
)
from db.repository.research_repository import (
    ResearchNotFoundError,
    ResearchRepository,
)
from research.evidence_engine import LiteratureEvidenceEngine
from research.archive import ResearchArchiveService
from research.controller import ResearchController
from research.evaluation import HarnessEvaluator
from research.experiment_service import ExperimentService
from research.hypothesis_service import HypothesisEvidenceService
from research.hypothesis_update import HypothesisUpdateService
from research.observation_service import ObservationService
from research.provenance import ProvenanceService
from research.report import ResearchReportService
from research.schema_audit import inspect_research_schema
from research.validators import ResearchValidationError
from research.validators import HYPOTHESIS_TRANSITIONS


research_router = APIRouter(tags=["research"])


def _dump(value):
    return value.model_dump(mode="json")


def _http_error(exc: Exception):
    if isinstance(exc, ResearchNotFoundError):
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if isinstance(exc, ResearchValidationError):
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    raise exc


async def _apply_approval_decision(
    repository: ResearchRepository,
    approval: ApprovalRequest,
    *,
    decision: str,
    actor: str,
):
    """Apply both approval and rejection outcomes so no entity is stranded."""
    if approval.action == "APPROVE_EXPERIMENT":
        target = "APPROVED" if decision == "APPROVED" else "CANCELLED"
        experiment = await repository.transition_experiment(
            approval.entity_id,
            target,
            actor=actor,
            approval_request_id=approval.id if decision == "APPROVED" else None,
        )
        if decision == "APPROVED":
            links = (
                await repository.session.execute(
                    select(ExperimentHypothesis).where(
                        ExperimentHypothesis.experiment_id == experiment.id
                    )
                )
            ).scalars().all()
            for link in links:
                hypothesis = await repository._must_get(Hypothesis, link.hypothesis_id)
                if "UNDER_TEST" in HYPOTHESIS_TRANSITIONS.get(hypothesis.status, set()):
                    await repository.transition_hypothesis(
                        hypothesis.id, "UNDER_TEST", actor=actor
                    )
        return experiment
    if approval.action == "APPROVE_CONCLUSION":
        return await repository.transition_conclusion(
            approval.entity_id,
            "APPROVED" if decision == "APPROVED" else "REJECTED",
            actor=actor,
            approval_request_id=approval.id if decision == "APPROVED" else None,
        )
    if decision != "APPROVED":
        return None
    if approval.action == "REGISTER_HYPOTHESIS":
        return await repository.transition_hypothesis(
            approval.entity_id,
            "TESTABLE",
            actor=actor,
            approval_request_id=approval.id,
        )
    if approval.action.startswith("UPDATE_HYPOTHESIS_STATUS:"):
        return await HypothesisUpdateService(repository).apply(approval.id, actor=actor)
    return None


@research_router.get("/research/system/schema")
async def get_research_schema_status(session: SessionDep) -> dict:
    return await inspect_research_schema(session)


@research_router.post("/research/questions", status_code=201)
async def create_question(body: QuestionCreate, session: SessionDep) -> dict:
    repository = ResearchRepository(session)
    try:
        question = await repository.create_question(ResearchQuestion(**body.model_dump()))
        await session.commit()
        return _dump(question)
    except Exception as exc:
        await session.rollback()
        _http_error(exc)


@research_router.get("/research/questions")
async def list_questions(session: SessionDep) -> dict:
    questions = (
        await session.execute(select(ResearchQuestion).order_by(ResearchQuestion.updated_at.desc()))
    ).scalars().all()
    return {"total": len(questions), "items": [_dump(item) for item in questions]}


@research_router.get("/research/questions/{question_id}")
async def get_question(question_id: str, session: SessionDep) -> dict:
    try:
        return _dump(await ResearchRepository(session)._must_get(ResearchQuestion, question_id))
    except Exception as exc:
        _http_error(exc)


@research_router.get("/research/questions/{question_id}/state")
async def get_research_state(question_id: str, session: SessionDep) -> dict:
    """Return a read-only projection for the lightweight Research Workbench."""
    repository = ResearchRepository(session)
    try:
        question = await repository._must_get(ResearchQuestion, question_id)
        hypotheses = (
            await session.execute(
                select(Hypothesis)
                .where(Hypothesis.research_question_id == question_id)
                .order_by(Hypothesis.created_at)
            )
        ).scalars().all()
        hypothesis_ids = [item.id for item in hypotheses]
        evidence = (
            await session.execute(
                select(Evidence)
                .where(Evidence.research_question_id == question_id)
                .order_by(Evidence.created_at.desc())
            )
        ).scalars().all()
        relations = []
        attempts = []
        if hypothesis_ids:
            relations = (
                await session.execute(
                    select(EvidenceRelation).where(
                        EvidenceRelation.hypothesis_id.in_(hypothesis_ids)
                    )
                )
            ).scalars().all()
            attempts = (
                await session.execute(
                    select(EvidenceSearchAttempt)
                    .where(EvidenceSearchAttempt.hypothesis_id.in_(hypothesis_ids))
                    .order_by(EvidenceSearchAttempt.created_at.desc())
                )
            ).scalars().all()
        experiments = (
            await session.execute(
                select(Experiment)
                .where(Experiment.research_question_id == question_id)
                .order_by(Experiment.created_at)
            )
        ).scalars().all()
        experiment_ids = [item.id for item in experiments]
        experiment_hypotheses = []
        runs = []
        observations = []
        if experiment_ids:
            experiment_hypotheses = (
                await session.execute(
                    select(ExperimentHypothesis).where(
                        ExperimentHypothesis.experiment_id.in_(experiment_ids)
                    )
                )
            ).scalars().all()
            runs = (
                await session.execute(
                    select(ResearchRun).where(ResearchRun.experiment_id.in_(experiment_ids))
                )
            ).scalars().all()
            observations = (
                await session.execute(
                    select(Observation).where(Observation.experiment_id.in_(experiment_ids))
                )
            ).scalars().all()
        observation_ids = [item.id for item in observations]
        observation_runs = []
        observation_hypotheses = []
        if observation_ids:
            observation_runs = (
                await session.execute(
                    select(ObservationRun).where(
                        ObservationRun.observation_id.in_(observation_ids)
                    )
                )
            ).scalars().all()
            observation_hypotheses = (
                await session.execute(
                    select(ObservationHypothesisRelation).where(
                        ObservationHypothesisRelation.observation_id.in_(observation_ids)
                    )
                )
            ).scalars().all()
        conclusions = (
            await session.execute(
                select(Conclusion)
                .where(Conclusion.research_question_id == question_id)
                .order_by(Conclusion.created_at)
            )
        ).scalars().all()
        entity_ids = [
            question_id,
            *hypothesis_ids,
            *experiment_ids,
            *[item.id for item in conclusions],
        ]
        approvals = (
            await session.execute(
                select(ApprovalRequest)
                .where(ApprovalRequest.entity_id.in_(entity_ids))
                .order_by(ApprovalRequest.created_at.desc())
            )
        ).scalars().all()
        workflow = await session.get(ResearchWorkflowState, question_id)
        return {
            "question": _dump(question),
            "workflow": _dump(workflow) if workflow else None,
            "hypotheses": [_dump(item) for item in hypotheses],
            "evidence": [_dump(item) for item in evidence],
            "evidence_relations": [_dump(item) for item in relations],
            "search_attempts": [_dump(item) for item in attempts],
            "experiments": [_dump(item) for item in experiments],
            "experiment_hypotheses": [_dump(item) for item in experiment_hypotheses],
            "runs": [_dump(item) for item in runs],
            "observations": [_dump(item) for item in observations],
            "observation_runs": [_dump(item) for item in observation_runs],
            "observation_hypotheses": [_dump(item) for item in observation_hypotheses],
            "conclusions": [_dump(item) for item in conclusions],
            "approvals": [_dump(item) for item in approvals],
        }
    except Exception as exc:
        _http_error(exc)


@research_router.post("/research/questions/{question_id}/hypotheses", status_code=201)
async def create_hypothesis(question_id: str, body: HypothesisCreate, session: SessionDep) -> dict:
    repository = ResearchRepository(session)
    try:
        hypothesis = await repository.create_hypothesis(
            Hypothesis(research_question_id=question_id, **body.model_dump())
        )
        approval = await repository.request_approval(
            entity_type="Hypothesis",
            entity_id=hypothesis.id,
            action="REGISTER_HYPOTHESIS",
            proposed_changes={"status": "TESTABLE"},
        )
        await session.commit()
        return {"hypothesis": _dump(hypothesis), "registration_approval": _dump(approval)}
    except Exception as exc:
        await session.rollback()
        _http_error(exc)


@research_router.post("/hypotheses/{hypothesis_id}/evidence/search")
async def search_hypothesis_evidence(
    hypothesis_id: str, body: EvidenceSearchIn, session: SessionDep
) -> dict:
    repository = ResearchRepository(session)
    try:
        hypothesis = await repository._must_get(Hypothesis, hypothesis_id)
        question = await repository._must_get(ResearchQuestion, hypothesis.research_question_id)
        result = await HypothesisEvidenceService(
            repository, LiteratureEvidenceEngine(session)
        ).search(
            question,
            hypothesis,
            include_contradiction=body.include_contradiction,
            include_limitation=body.include_limitation,
            search_mode=body.search_mode,
        )
        await session.commit()
        return {
            "attempts": [_dump(item) for item in result.attempts],
            "evidence": [_dump(item) for item in result.evidence],
            "proposed_relations": [_dump(item) for item in result.proposed_relations],
        }
    except Exception as exc:
        await session.rollback()
        _http_error(exc)


@research_router.get("/hypotheses/{hypothesis_id}")
async def get_hypothesis(hypothesis_id: str, session: SessionDep) -> dict:
    try:
        return _dump(await ResearchRepository(session)._must_get(Hypothesis, hypothesis_id))
    except Exception as exc:
        _http_error(exc)


@research_router.get("/hypotheses/{hypothesis_id}/evidence")
async def get_hypothesis_evidence(hypothesis_id: str, session: SessionDep) -> dict:
    repository = ResearchRepository(session)
    try:
        await repository._must_get(Hypothesis, hypothesis_id)
        relations = (
            await session.execute(
                select(EvidenceRelation).where(EvidenceRelation.hypothesis_id == hypothesis_id)
            )
        ).scalars().all()
        rows = []
        for relation in relations:
            evidence = await repository._must_get(Evidence, relation.evidence_id)
            rows.append({"relation": _dump(relation), "evidence": _dump(evidence)})
        return {"items": rows}
    except Exception as exc:
        _http_error(exc)


@research_router.post("/evidence-relations/{relation_id}/review")
async def review_evidence_relation(
    relation_id: str, body: EvidenceRelationReviewIn, session: SessionDep
) -> dict:
    repository = ResearchRepository(session)
    try:
        relation = await repository.review_evidence_relation(
            relation_id, body.decision, reviewer=body.reviewer, rationale=body.rationale
        )
        await session.commit()
        return _dump(relation)
    except Exception as exc:
        await session.rollback()
        _http_error(exc)


@research_router.post("/experiments", status_code=201)
async def create_experiment(body: ExperimentCreate, session: SessionDep) -> dict:
    repository = ResearchRepository(session)
    data = body.model_dump(exclude={"tested_hypotheses"})
    try:
        experiment = await repository.create_experiment(
            Experiment(**data), body.tested_hypotheses
        )
        await repository.transition_experiment(experiment.id, "PLANNED")
        approval = await repository.request_approval(
            entity_type="Experiment",
            entity_id=experiment.id,
            action="APPROVE_EXPERIMENT",
            proposed_changes={"status": "APPROVED"},
        )
        await session.commit()
        return {"experiment": _dump(experiment), "approval": _dump(approval)}
    except Exception as exc:
        await session.rollback()
        _http_error(exc)


@research_router.post("/experiments/{experiment_id}/approve")
async def approve_experiment(
    experiment_id: str, body: HumanDecisionIn, session: SessionDep
) -> dict:
    repository = ResearchRepository(session)
    try:
        approval = (
            await session.execute(
                select(ApprovalRequest)
                .where(
                    ApprovalRequest.entity_id == experiment_id,
                    ApprovalRequest.action == "APPROVE_EXPERIMENT",
                    ApprovalRequest.status == "PENDING",
                )
                .order_by(ApprovalRequest.created_at.desc())
            )
        ).scalars().first()
        if approval is None:
            raise ResearchValidationError("no pending experiment approval")
        approval = await repository.review_approval(
            approval.id, body.decision, reviewer=body.reviewer, reason=body.reason
        )
        experiment = await _apply_approval_decision(
            repository, approval, decision=body.decision, actor=body.reviewer
        )
        await session.commit()
        return {"experiment": _dump(experiment), "approval": _dump(approval)}
    except Exception as exc:
        await session.rollback()
        _http_error(exc)


@research_router.post("/experiments/{experiment_id}/runs/import")
async def prepare_run_import(experiment_id: str, body: RunImportIn, session: SessionDep) -> dict:
    repository = ResearchRepository(session)
    service = ExperimentService(repository)
    try:
        prepared = service.prepare_run_import(body.content, body.source_format)
        if prepared.payload.experiment_id != experiment_id:
            raise ResearchValidationError("payload experiment_id does not match URL")
        approval = await service.request_import_approval(
            prepared, requested_by=body.requested_by
        )
        await session.commit()
        return {"payload_hash": prepared.payload_hash, "approval": _dump(approval)}
    except Exception as exc:
        await session.rollback()
        _http_error(exc)


@research_router.post("/experiments/{experiment_id}/runs/import/confirm")
async def confirm_run_import(
    experiment_id: str, body: RunImportConfirmIn, session: SessionDep
) -> dict:
    repository = ResearchRepository(session)
    service = ExperimentService(repository)
    try:
        prepared = service.prepare_run_import(body.content, body.source_format)
        if prepared.payload.experiment_id != experiment_id:
            raise ResearchValidationError("payload experiment_id does not match URL")
        run = await service.confirm_run_import(
            prepared,
            approval_request_id=body.approval_request_id,
            actor=body.actor,
        )
        await session.commit()
        return _dump(run)
    except Exception as exc:
        await session.rollback()
        _http_error(exc)


@research_router.get("/experiments/{experiment_id}")
async def get_experiment(experiment_id: str, session: SessionDep) -> dict:
    try:
        repository = ResearchRepository(session)
        experiment = await repository._must_get(Experiment, experiment_id)
        runs = (
            await session.execute(
                select(ResearchRun).where(ResearchRun.experiment_id == experiment_id)
            )
        ).scalars().all()
        hypothesis_links = (
            await session.execute(
                select(ExperimentHypothesis).where(
                    ExperimentHypothesis.experiment_id == experiment_id
                )
            )
        ).scalars().all()
        return {
            "experiment": _dump(experiment),
            "tested_hypotheses": [item.hypothesis_id for item in hypothesis_links],
            "runs": [_dump(item) for item in runs],
        }
    except Exception as exc:
        _http_error(exc)


@research_router.post("/experiments/{experiment_id}/observations", status_code=201)
async def create_observation(
    experiment_id: str, body: ObservationCreate, session: SessionDep
) -> dict:
    repository = ResearchRepository(session)
    try:
        observation = await ObservationService(repository).build_from_runs(
            experiment_id,
            body.run_ids,
            hypothesis_relations=body.hypothesis_relations,
            actor=body.actor,
        )
        await session.commit()
        return _dump(observation)
    except Exception as exc:
        await session.rollback()
        _http_error(exc)


@research_router.post("/observations/{observation_id}/relations/{hypothesis_id}/review")
async def review_observation_relation(
    observation_id: str, hypothesis_id: str, body: ObservationRelationReviewIn, session: SessionDep
) -> dict:
    """确认"这条观察确实支持/反驳这个假设"。

    观察的**数值**是确定性算出来的，但"它意味着什么"是判断，和证据关系同级，
    所以未确认的链接不得解锁假设状态跃迁（repository._require_hypothesis_basis）。
    """
    repository = ResearchRepository(session)
    try:
        relation = await repository.review_observation_relation(
            observation_id, hypothesis_id, body.decision, reviewer=body.reviewer
        )
        await session.commit()
        return _dump(relation)
    except Exception as exc:
        await session.rollback()
        _http_error(exc)


@research_router.post("/experiments/{experiment_id}/complete")
async def complete_experiment(
    experiment_id: str, body: ActorIn, session: SessionDep
) -> dict:
    repository = ResearchRepository(session)
    try:
        experiment = await ExperimentService(repository).complete_experiment(
            experiment_id, actor=body.actor
        )
        await session.commit()
        return _dump(experiment)
    except Exception as exc:
        await session.rollback()
        _http_error(exc)


@research_router.post("/hypotheses/{hypothesis_id}/evaluate")
async def evaluate_hypothesis(
    hypothesis_id: str, body: HypothesisUpdateProposal, session: SessionDep
) -> dict:
    if body.hypothesis_id != hypothesis_id:
        raise HTTPException(status_code=422, detail="proposal hypothesis_id does not match URL")
    repository = ResearchRepository(session)
    try:
        approval = await HypothesisUpdateService(repository).propose(body)
        await session.commit()
        return _dump(approval)
    except Exception as exc:
        await session.rollback()
        _http_error(exc)


@research_router.post("/approvals/{approval_id}/review")
async def review_approval(
    approval_id: str, body: HumanDecisionIn, session: SessionDep
) -> dict:
    repository = ResearchRepository(session)
    try:
        approval = await repository.review_approval(
            approval_id, body.decision, reviewer=body.reviewer, reason=body.reason
        )
        applied = await _apply_approval_decision(
            repository, approval, decision=body.decision, actor=body.reviewer
        )
        await session.commit()
        return {"approval": _dump(approval), "applied_entity": _dump(applied) if applied else None}
    except Exception as exc:
        await session.rollback()
        _http_error(exc)


@research_router.post("/research/questions/{question_id}/conclusions", status_code=201)
async def create_conclusion(question_id: str, body: ConclusionCreate, session: SessionDep) -> dict:
    repository = ResearchRepository(session)
    try:
        conclusion = await repository.create_conclusion(
            Conclusion(
                research_question_id=question_id,
                statement=body.statement,
                confidence=body.confidence,
                limitations=body.limitations,
                unresolved_questions=body.unresolved_questions,
            ),
            supporting_evidence_ids=body.supporting_evidence_ids,
            contradicting_evidence_ids=body.contradicting_evidence_ids,
            observation_ids=body.observation_ids,
        )
        await repository.transition_conclusion(conclusion.id, "PENDING_APPROVAL")
        approval = await repository.request_approval(
            entity_type="Conclusion",
            entity_id=conclusion.id,
            action="APPROVE_CONCLUSION",
        )
        await session.commit()
        return {"conclusion": _dump(conclusion), "approval": _dump(approval)}
    except Exception as exc:
        await session.rollback()
        _http_error(exc)


@research_router.get("/research/questions/{question_id}/report")
async def get_report(question_id: str, session: SessionDep) -> dict:
    try:
        report = await ResearchReportService(ResearchRepository(session)).generate(question_id)
        return {"research_question_id": question_id, "format": "markdown", "report": report}
    except Exception as exc:
        _http_error(exc)


@research_router.get("/research/questions/{question_id}/evaluation")
async def get_evaluation(question_id: str, session: SessionDep) -> dict:
    """Return deterministic integrity metrics for one research project."""
    try:
        return await HarnessEvaluator(ResearchRepository(session)).evaluate(question_id)
    except Exception as exc:
        _http_error(exc)


@research_router.get("/research/questions/{question_id}/export")
async def export_research_archive(question_id: str, session: SessionDep) -> dict:
    try:
        return await ResearchArchiveService(ResearchRepository(session)).export(question_id)
    except Exception as exc:
        _http_error(exc)


@research_router.get("/research/entities/{entity_id}/provenance")
async def get_provenance(entity_id: str, session: SessionDep) -> dict:
    try:
        return await ProvenanceService(ResearchRepository(session)).get_provenance(entity_id)
    except Exception as exc:
        _http_error(exc)


@research_router.post("/research/questions/{question_id}/workflow/initialize")
async def initialize_workflow(question_id: str, session: SessionDep) -> dict:
    repository = ResearchRepository(session)
    try:
        state = await ResearchController(repository).initialize(question_id)
        await session.commit()
        return _dump(state)
    except Exception as exc:
        await session.rollback()
        _http_error(exc)


@research_router.post("/research/questions/{question_id}/workflow/advance")
async def advance_workflow(
    question_id: str, body: WorkflowAdvanceIn, session: SessionDep
) -> dict:
    repository = ResearchRepository(session)
    try:
        state = await ResearchController(repository).advance(
            question_id,
            body.target,
            actor=body.actor,
            context_update=body.context_update,
        )
        await session.commit()
        return _dump(state)
    except Exception as exc:
        await session.rollback()
        _http_error(exc)
