import asyncio

import pytest

from db.models.research import Evidence, Experiment, Hypothesis, Observation, ResearchQuestion, ResearchRun
from db.repository.research_repository import ResearchRepository
from research.enums import (
    ApprovalStatus,
    EvidenceType,
    ExperimentStatus,
    HypothesisStatus,
    QuestionStatus,
    RunStatus,
)
from research.validators import ResearchValidationError
from tests.research.helpers import isolated_session, literature_evidence


def test_new_question_must_start_open(tmp_path):
    async def scenario():
        async with isolated_session(tmp_path / "question.db") as session:
            repository = ResearchRepository(session)
            question = ResearchQuestion(
                title="A research question",
                description="A precise question",
                status=QuestionStatus.ANSWERED.value,
            )
            with pytest.raises(ResearchValidationError, match="must start in OPEN"):
                await repository.create_question(question)

    asyncio.run(scenario())


def test_literature_evidence_requires_original_locator(tmp_path):
    async def scenario():
        async with isolated_session(tmp_path / "evidence.db") as session:
            repository = ResearchRepository(session)
            question = await repository.create_question(
                ResearchQuestion(title="RQ", description="Does the treatment improve the metric?")
            )
            incomplete = Evidence(
                research_question_id=question.id,
                evidence_type=EvidenceType.LITERATURE.value,
                statement="A summary without original evidence",
            )
            with pytest.raises(ResearchValidationError, match="evidence.excerpt"):
                await repository.create_evidence(incomplete)

            complete = literature_evidence(question.id)
            await repository.create_evidence(complete)
            await session.commit()
            stored = await session.get(Evidence, complete.id)
            assert stored is not None
            assert stored.page == "7"
            assert stored.chunk_id == "chunk-fixture-001"
            assert stored.excerpt.startswith("The treatment")

    asyncio.run(scenario())


def test_illegal_question_transition_is_rejected(tmp_path):
    async def scenario():
        async with isolated_session(tmp_path / "transition.db") as session:
            repository = ResearchRepository(session)
            question = await repository.create_question(
                ResearchQuestion(title="RQ", description="A question")
            )
            with pytest.raises(ResearchValidationError, match="OPEN -> ANSWERED"):
                await repository.transition_question(question.id, QuestionStatus.ANSWERED.value)

    asyncio.run(scenario())


def test_experiment_requires_testable_design_fields(tmp_path):
    async def scenario():
        async with isolated_session(tmp_path / "experiment-validation.db") as session:
            repository = ResearchRepository(session)
            question = await repository.create_question(
                ResearchQuestion(title="RQ", description="A question")
            )
            hypothesis = await repository.create_hypothesis(
                Hypothesis(
                    research_question_id=question.id,
                    statement="H1",
                    prediction="A measurable change",
                )
            )
            incomplete = Experiment(
                research_question_id=question.id,
                purpose="Run it and see",
                independent_variable="",
                dependent_variables=[],
                control="",
                treatment="",
                metrics=[],
                success_criteria={},
            )
            with pytest.raises(ResearchValidationError, match="independent_variable"):
                await repository.create_experiment(incomplete, [hypothesis.id])

    asyncio.run(scenario())


def test_observation_rejects_unfinished_run(tmp_path):
    async def scenario():
        async with isolated_session(tmp_path / "observation-validation.db") as session:
            repository = ResearchRepository(session)
            question = await repository.create_question(
                ResearchQuestion(title="RQ", description="A question")
            )
            hypothesis = await repository.create_hypothesis(
                Hypothesis(
                    research_question_id=question.id,
                    statement="H1",
                    prediction="mAP changes",
                )
            )
            registration = await repository.request_approval(
                entity_type="Hypothesis",
                entity_id=hypothesis.id,
                action="REGISTER_HYPOTHESIS",
            )
            await repository.review_approval(
                registration.id, ApprovalStatus.APPROVED.value, reviewer="pi"
            )
            await repository.transition_hypothesis(
                hypothesis.id,
                HypothesisStatus.TESTABLE.value,
                approval_request_id=registration.id,
            )
            experiment = await repository.create_experiment(
                Experiment(
                    research_question_id=question.id,
                    purpose="Measure an effect",
                    independent_variable="treatment",
                    dependent_variables=["mAP"],
                    control="off",
                    treatment="on",
                    metrics=["mAP"],
                    success_criteria={"delta": 0.5},
                ),
                [hypothesis.id],
            )
            await repository.transition_experiment(experiment.id, ExperimentStatus.PLANNED.value)
            approval = await repository.request_approval(
                entity_type="Experiment", entity_id=experiment.id, action="APPROVE_EXPERIMENT"
            )
            await repository.review_approval(
                approval.id, ApprovalStatus.APPROVED.value, reviewer="pi"
            )
            await repository.transition_experiment(
                experiment.id, ExperimentStatus.APPROVED.value, approval_request_id=approval.id
            )
            run = await repository.create_run(ResearchRun(experiment_id=experiment.id))
            assert run.status == RunStatus.PENDING.value
            observation = Observation(
                experiment_id=experiment.id,
                measured_results={"mAP": 88.0},
                description="A raw measurement",
            )
            with pytest.raises(ResearchValidationError, match="completed runs"):
                await repository.create_observation(observation, [run.id])

    asyncio.run(scenario())


def test_model_generated_evidence_cannot_alone_support_conclusion(tmp_path):
    async def scenario():
        from db.models.research import Conclusion

        async with isolated_session(tmp_path / "model-only-conclusion.db") as session:
            repository = ResearchRepository(session)
            question = await repository.create_question(
                ResearchQuestion(title="RQ", description="Question")
            )
            model_claim = await repository.create_evidence(
                Evidence(
                    research_question_id=question.id,
                    evidence_type=EvidenceType.MODEL_HYPOTHESIS.value,
                    statement="A model-generated idea without source material.",
                )
            )
            with pytest.raises(ResearchValidationError, match="cannot be supported only"):
                await repository.create_conclusion(
                    Conclusion(
                        research_question_id=question.id,
                        statement="This must not become a formal conclusion.",
                    ),
                    supporting_evidence_ids=[model_claim.id],
                )

    asyncio.run(scenario())
