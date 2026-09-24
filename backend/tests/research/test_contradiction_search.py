import asyncio

from langchain_core.documents import Document
from sqlmodel import select

from ai.rag.pipeline import Outcome
from db.models.research import (
    Evidence,
    EvidenceRelation,
    EvidenceSearchAttempt,
    EvidenceSearchHit,
    Hypothesis,
    ResearchQuestion,
)
from db.repository.research_repository import ResearchRepository
from research.evidence_engine import LiteratureEvidenceEngine
from research.hypothesis_service import HypothesisEvidenceService
from research.enums import ReviewStatus
from tests.research.helpers import isolated_session


def test_three_channel_search_is_recorded_without_auto_confirmation(tmp_path):
    async def scenario():
        async with isolated_session(tmp_path / "contradiction.db") as session:
            repository = ResearchRepository(session)
            question = await repository.create_question(
                ResearchQuestion(
                    title="Hard-negative guided learning",
                    description="Does it improve local identity discrimination?",
                )
            )
            hypothesis = await repository.create_hypothesis(
                Hypothesis(
                    research_question_id=question.id,
                    statement="Global hard-negative relations improve local learning.",
                    prediction="mAP improves under a fixed protocol.",
                )
            )
            queries = []

            def fake_retrieve(query):
                queries.append(query)
                document = Document(
                    page_content="A shared passage returned by each deterministic search channel.",
                    metadata={
                        "source": "fixture.pdf",
                        "paper_title": "Fixture",
                        "page_label": "3",
                        "locator_prefix": "p",
                        "locator_kind": "page",
                        "chunk_id": "CH-shared",
                    },
                )
                return Outcome(
                    query=query,
                    hits=[(document, "hybrid", 0.8)],
                    vector_top1=0.8,
                    rejected=False,
                )

            service = HypothesisEvidenceService(
                repository,
                LiteratureEvidenceEngine(session, retrieve_fn=fake_retrieve),
            )
            result = await service.search(question, hypothesis)
            await session.commit()

            assert len(result.attempts) == 3
            assert {item.search_intent for item in result.attempts} == {
                "PRIMARY", "CONTRADICTION", "LIMITATION"
            }
            assert any("negative transfer" in query for query in queries)
            assert any("boundary conditions" in query for query in queries)

            # One source chunk is one Evidence entity even when all three search
            # channels find it; attempt-hit rows preserve all retrieval paths.
            assert len((await session.execute(select(Evidence))).scalars().all()) == 1
            assert len((await session.execute(select(EvidenceSearchHit))).scalars().all()) == 3
            relations = (await session.execute(select(EvidenceRelation))).scalars().all()
            assert {item.relation for item in relations} == {
                "SUPPORT", "CONTRADICT", "LIMITATION"
            }
            assert all(item.review_status == ReviewStatus.PROPOSED.value for item in relations)

    asyncio.run(scenario())


def test_contradiction_coverage_is_explicitly_configurable(tmp_path):
    async def scenario():
        async with isolated_session(tmp_path / "coverage.db") as session:
            repository = ResearchRepository(session)
            question = await repository.create_question(
                ResearchQuestion(title="RQ", description="Question")
            )
            hypothesis = await repository.create_hypothesis(
                Hypothesis(
                    research_question_id=question.id,
                    statement="H1",
                    prediction="A measurable prediction",
                )
            )

            def empty_retrieve(query):
                return Outcome(query=query, hits=[], vector_top1=0.1, rejected=True)

            service = HypothesisEvidenceService(
                repository,
                LiteratureEvidenceEngine(session, retrieve_fn=empty_retrieve),
            )
            result = await service.search(
                question, hypothesis, include_contradiction=False
            )
            assert [item.search_intent for item in result.attempts] == ["PRIMARY", "LIMITATION"]
            attempts = (
                await session.execute(select(EvidenceSearchAttempt))
            ).scalars().all()
            assert all(item.rejected for item in attempts)
            assert all(item.result_count == 0 for item in attempts)

    asyncio.run(scenario())
