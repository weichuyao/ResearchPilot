import asyncio

from ai.rag.pipeline import Outcome

from db.models.research import Conclusion, Experiment, Hypothesis, ResearchQuestion
from db.repository.research_repository import ResearchRepository
from research.evidence_engine import LiteratureEvidenceEngine
from research.evaluation import HarnessEvaluator
from research.hypothesis_service import HypothesisEvidenceService
from tests.research.helpers import isolated_session, literature_evidence


def test_harness_metrics_and_fingerprint_are_reloadable(tmp_path):
    async def scenario():
        database = tmp_path / "evaluation.db"
        question_id = fingerprint = ""
        async with isolated_session(database) as session:
            repository = ResearchRepository(session)
            question = await repository.create_question(
                ResearchQuestion(title="RQ", description="Question")
            )
            question_id = question.id
            hypothesis = await repository.create_hypothesis(
                Hypothesis(
                    research_question_id=question.id,
                    statement="H1",
                    prediction="Prediction",
                )
            )

            def empty_retrieve(query):
                return Outcome(query=query, hits=[], vector_top1=0.1, rejected=True)

            await HypothesisEvidenceService(
                repository,
                LiteratureEvidenceEngine(session, retrieve_fn=empty_retrieve),
            ).search(question, hypothesis)
            evidence = await repository.create_evidence(literature_evidence(question.id))
            await repository.create_conclusion(
                Conclusion(
                    research_question_id=question.id,
                    statement="A source-grounded draft conclusion.",
                ),
                supporting_evidence_ids=[evidence.id],
            )
            await session.commit()
            result = await HarnessEvaluator(repository).evaluate(question.id)
            fingerprint = result["state_fingerprint"]
            assert result["evidence_traceability"] == 1.0
            assert result["citation_integrity_structural"] == 1.0
            assert result["state_consistency"]["invalid_count"] == 0
            assert result["unsupported_conclusion_rate"] == 0.0
            assert result["contradiction_coverage"] == 1.0

        async with isolated_session(database) as session:
            reloaded = await HarnessEvaluator(ResearchRepository(session)).evaluate(question_id)
            assert reloaded["state_fingerprint"] == fingerprint

    asyncio.run(scenario())


def test_state_consistency_covers_experiments(tmp_path):
    async def scenario():
        async with isolated_session(tmp_path / "invalid-state.db") as session:
            repository = ResearchRepository(session)
            question = await repository.create_question(
                ResearchQuestion(title="RQ", description="Question")
            )
            invalid = Experiment(
                research_question_id=question.id,
                purpose="Deliberately malformed persisted fixture",
                independent_variable="x",
                dependent_variables=["y"],
                control="off",
                treatment="on",
                metrics=["y"],
                success_criteria={"y": 1},
                status="NOT_A_REAL_STATE",
            )
            session.add(invalid)
            await session.flush()
            result = await HarnessEvaluator(repository).evaluate(question.id)
            assert result["state_consistency"]["invalid_count"] == 1
            assert result["state_consistency"]["invalid_entity_ids"] == [invalid.id]

    asyncio.run(scenario())
