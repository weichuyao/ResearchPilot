import asyncio

import pytest
from langchain_core.documents import Document

from ai.rag.ingest import chunk_document
from ai.rag.pipeline import Outcome, format_hits
from db.models.paper import Paper, STATUS_INDEXED
from db.models.research import Hypothesis, ResearchQuestion
from db.repository.research_repository import ResearchRepository
from research.evidence_engine import LiteratureEvidenceEngine
from research.provenance_ids import chunk_id_from_document
from research.validators import ResearchValidationError
from tests.research.helpers import isolated_session


def _document(with_chunk_id: bool = True) -> Document:
    metadata = {
        "source": "fixture-paper.pdf",
        "paper_title": "Fixture Paper",
        "page": 6,
        "page_label": "7",
        "locator_prefix": "p",
        "locator_kind": "page",
        "kind": "text",
    }
    if with_chunk_id:
        metadata.update({"chunk_id": "CH-explicit", "chunk_index": 0})
    return Document(
        page_content="The treatment improves mAP by 0.60 points over the control.",
        metadata=metadata,
    )


def test_engine_builds_persistable_literature_evidence(tmp_path):
    async def scenario():
        async with isolated_session(tmp_path / "engine.db") as session:
            repository = ResearchRepository(session)
            question = await repository.create_question(
                ResearchQuestion(
                    title="Hard-negative guidance",
                    description="Does it improve fine-grained identity discrimination?",
                )
            )
            hypothesis = await repository.create_hypothesis(
                Hypothesis(
                    research_question_id=question.id,
                    statement="Hard-negative guidance improves local discrimination.",
                    prediction="mAP improves over the fixed control.",
                )
            )
            paper = Paper(
                source_file="fixture-paper.pdf",
                title="Fixture Paper",
                status=STATUS_INDEXED,
                chunk_count=1,
            )
            session.add(paper)
            await session.flush()

            captured = {}

            def fake_retrieve(query):
                captured["query"] = query
                return Outcome(
                    query=query,
                    hits=[(_document(), "hybrid", 0.82)],
                    vector_top1=0.82,
                    rejected=False,
                )

            engine = LiteratureEvidenceEngine(session, retrieve_fn=fake_retrieve)
            drafts = await engine.search_evidence(question, hypothesis)
            assert len(drafts) == 1
            evidence = drafts[0]
            assert evidence.source_id == str(paper.id)
            assert evidence.page == "7"
            assert evidence.chunk_id == "CH-explicit"
            assert evidence.excerpt == _document().page_content
            assert evidence.provenance["rank"] == 1
            assert hypothesis.statement in captured["query"]
            await repository.create_evidence(evidence)
            await session.commit()

    asyncio.run(scenario())


def test_engine_rejected_search_and_mode_validation(tmp_path):
    async def scenario():
        async with isolated_session(tmp_path / "engine-reject.db") as session:
            question = ResearchQuestion(title="RQ", description="Question")

            def rejected(query):
                return Outcome(query=query, hits=[], vector_top1=0.1, rejected=True)

            engine = LiteratureEvidenceEngine(session, retrieve_fn=rejected)
            assert await engine.search_evidence(question) == []
            with pytest.raises(ResearchValidationError, match="invalid search_mode"):
                await engine.search_evidence(question, search_mode="magic")

    asyncio.run(scenario())


def test_engine_drops_unlocatable_literature_hits(tmp_path):
    async def scenario():
        async with isolated_session(tmp_path / "engine-locator.db") as session:
            document = _document()
            document.metadata.pop("locator_kind")
            document.metadata.pop("page_label")
            document.metadata.pop("page")

            def retrieve(query):
                return Outcome(
                    query=query,
                    hits=[(document, "vector", 0.9)],
                    vector_top1=0.9,
                    rejected=False,
                )

            engine = LiteratureEvidenceEngine(session, retrieve_fn=retrieve)
            assert await engine.search_evidence("A research question") == []

    asyncio.run(scenario())


def test_legacy_chunk_id_is_deterministic():
    legacy = _document(with_chunk_id=False)
    first = chunk_id_from_document(legacy)
    second = chunk_id_from_document(legacy)
    assert first == second
    assert first.startswith("CH-")
    assert first != chunk_id_from_document(
        Document(page_content=legacy.page_content + " changed", metadata=legacy.metadata)
    )


def test_new_ingest_adds_chunk_provenance_without_changing_citation_format(tmp_path):
    path = tmp_path / "fixture.txt"
    path.write_text(
        "A sufficiently long fixture paragraph for deterministic chunking. "
        "It contains enough source text to pass the minimum section length and be indexed.",
        encoding="utf-8",
    )
    chunks, _report = chunk_document(
        str(path), source_file="fixture.txt", title="Fixture Text"
    )
    assert chunks
    assert all(chunk.metadata["chunk_id"].startswith("CH-") for chunk in chunks)
    assert [chunk.metadata["chunk_index"] for chunk in chunks] == list(range(len(chunks)))

    # The established text protocol is consumed by the eval regex and chat UI.
    # New provenance metadata must not alter it.
    rendered = format_hits([(chunks[0], "vector", 0.75)])
    assert rendered.startswith("[source 1 | Fixture Text | blk.1 | relevance 0.75]\n")
    assert "chunk_id" not in rendered
