import asyncio

import pytest

from ai.rag.pipeline import Outcome
from db.models.research import Experiment, Hypothesis, ResearchQuestion, ResearchWorkflowState
from db.repository.research_repository import ResearchRepository
from research.controller import ResearchController
from research.evidence_engine import LiteratureEvidenceEngine
from research.enums import ApprovalStatus, ExperimentStatus, WorkflowStage
from research.hypothesis_service import HypothesisEvidenceService
from research.validators import ResearchValidationError
from tests.research.helpers import isolated_session


def test_controller_enforces_sequence_and_human_gate(tmp_path):
    async def scenario():
        database = tmp_path / "controller.db"
        question_id = ""
        async with isolated_session(database) as session:
            repository = ResearchRepository(session)
            question = await repository.create_question(
                ResearchQuestion(title="RQ", description="Question")
            )
            question_id = question.id
            await repository.transition_question(question.id, "ACTIVE")
            controller = ResearchController(repository)
            state = await controller.initialize(question.id)
            assert state.stage == WorkflowStage.DEFINE_QUESTION.value
            await controller.advance(question.id, "LITERATURE_REVIEW")
            await controller.advance(question.id, "GENERATE_HYPOTHESIS")
            with pytest.raises(ResearchValidationError, match="requires a registered hypothesis"):
                await controller.advance(question.id, "SEARCH_SUPPORTING_EVIDENCE")

            hypothesis = await repository.create_hypothesis(
                Hypothesis(
                    research_question_id=question.id,
                    statement="H1",
                    prediction="A measurable prediction",
                )
            )
            with pytest.raises(ResearchValidationError, match="registered hypothesis"):
                await controller.advance(question.id, "SEARCH_SUPPORTING_EVIDENCE")
            registration = await repository.request_approval(
                entity_type="Hypothesis",
                entity_id=hypothesis.id,
                action="REGISTER_HYPOTHESIS",
            )
            await repository.review_approval(registration.id, "APPROVED", reviewer="pi")
            await repository.transition_hypothesis(
                hypothesis.id, "TESTABLE", approval_request_id=registration.id
            )

            def empty_retrieve(query):
                return Outcome(query=query, hits=[], vector_top1=0.1, rejected=True)

            await HypothesisEvidenceService(
                repository,
                LiteratureEvidenceEngine(session, retrieve_fn=empty_retrieve),
            ).search(question, hypothesis)
            await controller.advance(question.id, "SEARCH_SUPPORTING_EVIDENCE")
            await controller.advance(question.id, "SEARCH_CONTRADICTING_EVIDENCE")
            await controller.advance(question.id, "ASSESS_EVIDENCE")
            await controller.advance(question.id, "DESIGN_EXPERIMENT")

            experiment = await repository.create_experiment(
                Experiment(
                    research_question_id=question.id,
                    purpose="Test H1",
                    independent_variable="treatment",
                    dependent_variables=["mAP"],
                    control="off",
                    treatment="on",
                    metrics=["mAP"],
                    success_criteria={"delta": 0.5},
                ),
                [hypothesis.id],
            )
            await repository.transition_experiment(experiment.id, "PLANNED")
            await controller.advance(question.id, "HUMAN_APPROVAL")
            with pytest.raises(ResearchValidationError, match="approved experiment"):
                await controller.advance(question.id, "WAIT_FOR_RESULT")

            approval = await repository.request_approval(
                entity_type="Experiment",
                entity_id=experiment.id,
                action="APPROVE_EXPERIMENT",
            )
            await repository.review_approval(
                approval.id, ApprovalStatus.APPROVED.value, reviewer="pi"
            )
            await repository.transition_experiment(
                experiment.id,
                ExperimentStatus.APPROVED.value,
                approval_request_id=approval.id,
            )
            await controller.advance(question.id, "WAIT_FOR_RESULT")
            assert state.stage == WorkflowStage.WAIT_FOR_RESULT.value
            assert state.revision == 8
            await session.commit()

        async with isolated_session(database) as session:
            reloaded = await session.get(ResearchWorkflowState, question_id)
            assert reloaded.stage == WorkflowStage.WAIT_FOR_RESULT.value
            assert reloaded.revision == 8

    asyncio.run(scenario())


def test_hypothesis_registration_requires_approval(tmp_path):
    async def scenario():
        async with isolated_session(tmp_path / "registration.db") as session:
            repository = ResearchRepository(session)
            question = await repository.create_question(
                ResearchQuestion(title="RQ", description="Question")
            )
            hypothesis = await repository.create_hypothesis(
                Hypothesis(
                    research_question_id=question.id,
                    statement="H1",
                    prediction="Prediction",
                )
            )
            with pytest.raises(ResearchValidationError, match="REGISTER_HYPOTHESIS"):
                await repository.transition_hypothesis(hypothesis.id, "TESTABLE")

    asyncio.run(scenario())
