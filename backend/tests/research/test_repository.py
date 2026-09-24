import asyncio

import pytest
from sqlmodel import select

from db.models.research import (
    ApprovalRequest,
    Conclusion,
    EvidenceRelation,
    Experiment,
    Hypothesis,
    Observation,
    ResearchEvent,
    ResearchQuestion,
    ResearchRun,
)
from db.repository.research_repository import ResearchRepository
from research.enums import (
    ApprovalStatus,
    ConclusionStatus,
    EvidenceRelationType,
    ExperimentStatus,
    HypothesisStatus,
    ObservationRelationType,
    ReviewStatus,
    RunStatus,
)
from research.validators import ResearchValidationError
from tests.research.helpers import isolated_session, literature_evidence


def _experiment(question_id: str) -> Experiment:
    return Experiment(
        research_question_id=question_id,
        purpose="Test whether the treatment improves identity discrimination.",
        independent_variable="hard-negative guidance enabled",
        dependent_variables=["mAP", "rank1"],
        control="baseline without guidance",
        treatment="baseline with guidance",
        controlled_variables=["dataset", "backbone", "optimizer", "seed set"],
        metrics=["mAP", "rank1"],
        success_criteria={"mAP_delta_min": 0.5, "minimum_runs": 3},
    )


def test_lineage_and_cross_question_relations_are_guarded(tmp_path):
    async def scenario():
        async with isolated_session(tmp_path / "relations.db") as session:
            repository = ResearchRepository(session)
            q1 = await repository.create_question(ResearchQuestion(title="RQ1", description="First"))
            q2 = await repository.create_question(ResearchQuestion(title="RQ2", description="Second"))
            parent = await repository.create_hypothesis(
                Hypothesis(
                    research_question_id=q1.id,
                    statement="H1",
                    prediction="mAP increases",
                )
            )
            child = await repository.create_hypothesis(
                Hypothesis(
                    research_question_id=q1.id,
                    parent_hypothesis_id=parent.id,
                    statement="H1.1",
                    prediction="the increase persists over seeds",
                )
            )
            assert child.parent_hypothesis_id == parent.id

            evidence = await repository.create_evidence(literature_evidence(q2.id))
            relation = EvidenceRelation(
                hypothesis_id=parent.id,
                evidence_id=evidence.id,
                relation=EvidenceRelationType.SUPPORT.value,
            )
            with pytest.raises(ResearchValidationError, match="different research questions"):
                await repository.create_evidence_relation(relation)

    asyncio.run(scenario())


def test_experiment_approval_run_observation_and_conclusion(tmp_path):
    async def scenario():
        database = tmp_path / "workflow.db"
        question_id = hypothesis_id = conclusion_id = ""
        async with isolated_session(database) as session:
            repository = ResearchRepository(session)
            question = await repository.create_question(
                ResearchQuestion(
                    title="Hard-negative guided local learning",
                    description="Does it improve fine-grained identity discrimination?",
                    background="ReID fixture",
                    scope="Fixed dataset and backbone",
                )
            )
            question_id = question.id
            await repository.transition_question(question.id, "ACTIVE")
            hypothesis = await repository.create_hypothesis(
                Hypothesis(
                    research_question_id=question.id,
                    statement="Hard-negative guidance improves local discrimination.",
                    rationale="It focuses learning on confusing identities.",
                    prediction="Mean mAP is at least 0.5 points above control across three runs.",
                )
            )
            hypothesis_id = hypothesis.id
            registration = await repository.request_approval(
                entity_type="Hypothesis",
                entity_id=hypothesis.id,
                action="REGISTER_HYPOTHESIS",
            )
            await repository.review_approval(
                registration.id, "APPROVED", reviewer="pi"
            )
            await repository.transition_hypothesis(
                hypothesis.id,
                HypothesisStatus.TESTABLE.value,
                approval_request_id=registration.id,
            )

            evidence = await repository.create_evidence(literature_evidence(question.id))
            relation = await repository.create_evidence_relation(
                EvidenceRelation(
                    hypothesis_id=hypothesis.id,
                    evidence_id=evidence.id,
                    relation=EvidenceRelationType.SUPPORT.value,
                    rationale="Directly reports the relevant comparison.",
                )
            )
            await repository.review_evidence_relation(
                relation.id, ReviewStatus.CONFIRMED.value, reviewer="reviewer-1"
            )

            experiment = await repository.create_experiment(
                _experiment(question.id), [hypothesis.id]
            )
            await repository.transition_experiment(experiment.id, ExperimentStatus.PLANNED.value)
            with pytest.raises(ResearchValidationError, match="requires an approved"):
                await repository.transition_experiment(experiment.id, ExperimentStatus.APPROVED.value)
            approval = await repository.request_approval(
                entity_type="Experiment",
                entity_id=experiment.id,
                action="APPROVE_EXPERIMENT",
                proposed_changes={"status": "APPROVED"},
            )
            await repository.review_approval(
                approval.id, ApprovalStatus.APPROVED.value, reviewer="pi"
            )
            await repository.transition_experiment(
                experiment.id,
                ExperimentStatus.APPROVED.value,
                approval_request_id=approval.id,
            )

            run = await repository.create_run(
                ResearchRun(experiment_id=experiment.id, config={"lr": 0.001}, seed=1)
            )
            await repository.transition_run(run.id, RunStatus.RUNNING.value)
            run.metrics = {"mAP": 88.21, "rank1": 94.32}
            await repository.transition_run(run.id, RunStatus.COMPLETED.value)

            observation = await repository.create_observation(
                Observation(
                    experiment_id=experiment.id,
                    measured_results={"treatment_mAP": 88.21, "control_mAP": 87.61},
                    derived_statistics={"delta_mAP": 0.60, "run_count": 1},
                    description="Treatment mAP exceeded control by 0.60 points in this run.",
                ),
                [run.id],
                {hypothesis.id: ObservationRelationType.SUPPORT.value},
            )

            with pytest.raises(ResearchValidationError, match="requires supporting evidence"):
                await repository.create_conclusion(
                    Conclusion(
                        research_question_id=question.id,
                        statement="An unsupported conclusion",
                    )
                )

            conclusion = await repository.create_conclusion(
                Conclusion(
                    research_question_id=question.id,
                    statement="The current evidence partially supports the hypothesis.",
                    confidence="MEDIUM",
                    limitations=["Only one completed run is available."],
                    unresolved_questions=["Does the effect persist across seeds?"],
                ),
                supporting_evidence_ids=[evidence.id],
                observation_ids=[observation.id],
            )
            conclusion_id = conclusion.id
            await repository.transition_conclusion(
                conclusion.id, ConclusionStatus.PENDING_APPROVAL.value
            )
            approval = await repository.request_approval(
                entity_type="Conclusion",
                entity_id=conclusion.id,
                action="APPROVE_CONCLUSION",
                proposed_changes={"status": "APPROVED"},
            )
            await repository.review_approval(
                approval.id, ApprovalStatus.APPROVED.value, reviewer="pi"
            )
            await repository.transition_conclusion(
                conclusion.id,
                ConclusionStatus.APPROVED.value,
                approval_request_id=approval.id,
            )
            await repository.transition_question(question.id, "ANSWERED")
            await session.commit()

            events = (await session.execute(select(ResearchEvent))).scalars().all()
            assert len(events) >= 15
            assert relation.review_status == ReviewStatus.CONFIRMED.value

        # A new engine/session proves that state is genuinely reloadable rather
        # than surviving only in the ORM identity map.
        async with isolated_session(database) as session:
            reloaded_question = await session.get(ResearchQuestion, question_id)
            reloaded_hypothesis = await session.get(Hypothesis, hypothesis_id)
            reloaded_conclusion = await session.get(Conclusion, conclusion_id)
            assert reloaded_question.status == "ANSWERED"
            assert reloaded_hypothesis.status == "TESTABLE"
            assert reloaded_conclusion.status == "APPROVED"

    asyncio.run(scenario())


def test_major_hypothesis_update_needs_matching_approval(tmp_path):
    async def scenario():
        async with isolated_session(tmp_path / "approval.db") as session:
            repository = ResearchRepository(session)
            question = await repository.create_question(ResearchQuestion(title="RQ", description="Question"))
            hypothesis = await repository.create_hypothesis(
                Hypothesis(
                    research_question_id=question.id,
                    statement="H1",
                    prediction="A measurable prediction",
                )
            )
            registration = await repository.request_approval(
                entity_type="Hypothesis",
                entity_id=hypothesis.id,
                action="REGISTER_HYPOTHESIS",
            )
            await repository.review_approval(registration.id, "APPROVED", reviewer="pi")
            await repository.transition_hypothesis(
                hypothesis.id, "TESTABLE", approval_request_id=registration.id
            )
            await repository.transition_hypothesis(hypothesis.id, "UNDER_TEST")
            with pytest.raises(ResearchValidationError, match="requires an approved"):
                await repository.transition_hypothesis(hypothesis.id, "SUPPORTED")
            approval = await repository.request_approval(
                entity_type="Hypothesis",
                entity_id=hypothesis.id,
                action="UPDATE_HYPOTHESIS_STATUS:PARTIALLY_SUPPORTED",
            )
            await repository.review_approval(approval.id, "APPROVED", reviewer="pi")
            with pytest.raises(ResearchValidationError, match="does not match"):
                await repository.transition_hypothesis(
                    hypothesis.id, "SUPPORTED", approval_request_id=approval.id
                )
            with pytest.raises(ResearchValidationError, match="requires confirmed support"):
                await repository.transition_hypothesis(
                    hypothesis.id, "PARTIALLY_SUPPORTED", approval_request_id=approval.id
                )
            evidence = await repository.create_evidence(literature_evidence(question.id))
            relation = await repository.create_evidence_relation(
                EvidenceRelation(
                    hypothesis_id=hypothesis.id,
                    evidence_id=evidence.id,
                    relation="SUPPORT",
                )
            )
            await repository.review_evidence_relation(
                relation.id, "CONFIRMED", reviewer="pi"
            )
            await repository.transition_hypothesis(
                hypothesis.id, "PARTIALLY_SUPPORTED", approval_request_id=approval.id
            )
            assert hypothesis.status == "PARTIALLY_SUPPORTED"

    asyncio.run(scenario())


def test_pending_approval_is_idempotent_but_cannot_change_payload(tmp_path):
    async def scenario():
        async with isolated_session(tmp_path / "approval-idempotency.db") as session:
            repository = ResearchRepository(session)
            question = await repository.create_question(
                ResearchQuestion(title="RQ", description="Question")
            )
            first = await repository.request_approval(
                entity_type="ResearchQuestion",
                entity_id=question.id,
                action="ARCHIVE",
                proposed_changes={"status": "ARCHIVED"},
            )
            repeated = await repository.request_approval(
                entity_type="ResearchQuestion",
                entity_id=question.id,
                action="ARCHIVE",
                proposed_changes={"status": "ARCHIVED"},
            )
            assert repeated.id == first.id
            with pytest.raises(ResearchValidationError, match="different pending approval"):
                await repository.request_approval(
                    entity_type="ResearchQuestion",
                    entity_id=question.id,
                    action="ARCHIVE",
                    proposed_changes={"status": "OPEN"},
                )

    asyncio.run(scenario())
