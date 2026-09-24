"""Adapter from the existing RAG pipeline to persistent Evidence drafts."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from ai.rag.pipeline import Outcome, retrieve
from db.models.paper import Paper
from db.models.research import Evidence, Hypothesis, ResearchQuestion
from research.enums import EvidenceType, SearchIntent
from research.provenance_ids import chunk_id_from_document
from research.validators import ResearchValidationError, require_text


RetrieveFn = Callable[..., Outcome]


@dataclass(frozen=True)
class LiteratureSearchBatch:
    query: str
    search_intent: SearchIntent
    search_mode: str
    outcome: Outcome
    evidence: list[Evidence]


class LiteratureEvidenceEngine:
    """Reuse retrieval while returning source-grounded domain objects.

    Returned Evidence instances are drafts and are not automatically persisted.
    The caller must pass them through ResearchRepository.create_evidence(), which
    applies the final provenance invariants and transaction boundary.
    """

    SEARCH_MODES = {"balanced", "precise", "broad"}

    def __init__(
        self,
        session: AsyncSession | None = None,
        *,
        retrieve_fn: RetrieveFn = retrieve,
    ):
        self.session = session
        self.retrieve_fn = retrieve_fn

    async def search_evidence(
        self,
        research_question: ResearchQuestion | str,
        hypothesis: Hypothesis | str | None = None,
        search_mode: str = "balanced",
    ) -> list[Evidence]:
        if search_mode not in self.SEARCH_MODES:
            raise ResearchValidationError(
                f"invalid search_mode: {search_mode!r}; expected balanced, precise or broad"
            )
        question_id, question_text = self._question_parts(research_question)
        hypothesis_text = self._hypothesis_text(hypothesis)
        query = question_text if not hypothesis_text else f"{question_text}\nHypothesis: {hypothesis_text}"
        require_text(query, "literature query")

        # The existing embedding/retrieval/rerank stack is synchronous and CPU/
        # IO bound.  Do not block FastAPI's event loop when this adapter is used.
        batch = await self.search_query(
            question_id=question_id,
            query=query,
            search_intent=SearchIntent.PRIMARY,
            search_mode=search_mode,
        )
        return batch.evidence

    async def search_query(
        self,
        *,
        question_id: str,
        query: str,
        search_intent: SearchIntent,
        search_mode: str = "balanced",
    ) -> LiteratureSearchBatch:
        if search_mode not in self.SEARCH_MODES:
            raise ResearchValidationError(
                f"invalid search_mode: {search_mode!r}; expected balanced, precise or broad"
            )
        require_text(query, "literature query")
        outcome = await asyncio.to_thread(self.retrieve_fn, query)
        if outcome.rejected:
            return LiteratureSearchBatch(query, search_intent, search_mode, outcome, [])

        papers = await self._resolve_papers(outcome)
        drafts: list[Evidence] = []
        for rank, (document, origin, score) in enumerate(outcome.hits, start=1):
            metadata = document.metadata or {}
            source_file = str(metadata.get("source") or "unknown")
            paper = papers.get(source_file)
            locator_kind = str(metadata.get("locator_kind") or "").strip().lower()
            raw_label = metadata.get("page_label") or metadata.get("page")
            if locator_kind not in {"page", "section", "block"} or raw_label in (None, ""):
                # An unlocatable excerpt cannot support a verifiable citation.
                continue
            label = str(raw_label)
            excerpt = document.page_content.strip()
            if not excerpt:
                continue
            drafts.append(
                Evidence(
                    research_question_id=question_id,
                    evidence_type=EvidenceType.LITERATURE.value,
                    # No LLM summarization happens in the engine.  Until a human
                    # or reasoner supplies an interpretation, statement mirrors
                    # the immutable excerpt rather than inventing a claim.
                    statement=excerpt,
                    excerpt=excerpt,
                    source_id=str(paper.id) if paper and paper.id is not None else source_file,
                    source_type="paper",
                    source_title=(paper.title if paper else metadata.get("paper_title")),
                    page=label if locator_kind == "page" else None,
                    section=label if locator_kind in {"section", "block"} else None,
                    locator_prefix=str(metadata.get("locator_prefix") or "p"),
                    locator_kind=locator_kind,
                    chunk_id=chunk_id_from_document(document),
                    search_intent=search_intent.value,
                    provenance={
                        "source_file": source_file,
                        "retrieval_query": query,
                        "search_mode": search_mode,
                        "rank": rank,
                        "origin": origin,
                        "retrieval_score": score,
                        "vector_top1": outcome.vector_top1,
                        "ambiguous": outcome.ambiguous,
                    },
                )
            )
        return LiteratureSearchBatch(query, search_intent, search_mode, outcome, drafts)

    @staticmethod
    def _question_parts(question: ResearchQuestion | str) -> tuple[str, str]:
        if isinstance(question, ResearchQuestion):
            text = "\n".join(
                part.strip() for part in (question.title, question.description) if part and part.strip()
            )
            return question.id, text
        require_text(question, "research_question")
        # A raw string is useful for preview searches, but persisted drafts need
        # a real question ID and will be rejected by the repository if it does
        # not exist.
        return question, question

    @staticmethod
    def _hypothesis_text(hypothesis: Hypothesis | str | None) -> str:
        if hypothesis is None:
            return ""
        return hypothesis.statement if isinstance(hypothesis, Hypothesis) else hypothesis.strip()

    async def _resolve_papers(self, outcome: Outcome) -> dict[str, Paper]:
        if self.session is None:
            return {}
        sources = {
            str(document.metadata.get("source"))
            for document, _origin, _score in outcome.hits
            if document.metadata.get("source")
        }
        if not sources:
            return {}
        rows = (
            await self.session.execute(select(Paper).where(Paper.source_file.in_(sources)))
        ).scalars().all()
        return {row.source_file: row for row in rows}


__all__ = ["LiteratureEvidenceEngine", "LiteratureSearchBatch"]
