"""Deterministic Markdown research report assembled from authoritative state."""

from sqlmodel import select

from db.models.research import (
    Conclusion,
    Evidence,
    EvidenceRelation,
    Experiment,
    ExperimentHypothesis,
    Hypothesis,
    Observation,
    ObservationHypothesisRelation,
    ResearchQuestion,
)
from db.repository.research_repository import ResearchRepository
from research.enums import ConclusionStatus, ReviewStatus
from research.provenance import ProvenanceService


class ResearchReportService:
    def __init__(self, repository: ResearchRepository):
        self.repository = repository

    async def generate(self, research_question_id: str) -> str:
        question = await self.repository._must_get(ResearchQuestion, research_question_id)
        hypotheses = (
            await self.repository.session.execute(
                select(Hypothesis)
                .where(Hypothesis.research_question_id == question.id)
                .order_by(Hypothesis.created_at, Hypothesis.id)
            )
        ).scalars().all()
        conclusions = (
            await self.repository.session.execute(
                select(Conclusion).where(
                    Conclusion.research_question_id == question.id,
                    Conclusion.status == ConclusionStatus.APPROVED.value,
                )
            )
        ).scalars().all()
        lines = [
            "# Research Question",
            "",
            f"**{question.title}**",
            "",
            question.description,
            "",
            "# Current Hypotheses",
            "",
        ]
        for hypothesis in hypotheses:
            lines.extend(await self._hypothesis_section(hypothesis))
        lines.extend(["# Overall Conclusions", ""])
        if conclusions:
            for conclusion in conclusions:
                lines.extend(
                    [
                        f"- [{conclusion.id}] {conclusion.statement} "
                        f"(confidence: {conclusion.confidence})"
                    ]
                )
        else:
            lines.append("No approved conclusion.")
        lines.extend(["", "# Unresolved Questions", ""])
        unresolved = [item for conclusion in conclusions for item in conclusion.unresolved_questions]
        lines.extend([f"- {item}" for item in unresolved] or ["- None recorded."])
        lines.extend(["", "# Evidence Provenance", ""])
        provenance = ProvenanceService(self.repository)
        for conclusion in conclusions:
            graph = await provenance.get_provenance(conclusion.id)
            lines.append(f"## {conclusion.id}")
            lines.append("")
            for edge in graph["edges"]:
                lines.append(f"- `{edge['from']}` --{edge['type']}--> `{edge['to']}`")
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    async def _hypothesis_section(self, hypothesis: Hypothesis) -> list[str]:
        relations = (
            await self.repository.session.execute(
                select(EvidenceRelation).where(
                    EvidenceRelation.hypothesis_id == hypothesis.id,
                    EvidenceRelation.review_status == ReviewStatus.CONFIRMED.value,
                )
            )
        ).scalars().all()
        grouped: dict[str, list[Evidence]] = {key: [] for key in ("SUPPORT", "CONTRADICT", "LIMITATION")}
        for relation in relations:
            if relation.relation in grouped:
                grouped[relation.relation].append(
                    await self.repository._must_get(Evidence, relation.evidence_id)
                )
        experiment_links = (
            await self.repository.session.execute(
                select(ExperimentHypothesis).where(
                    ExperimentHypothesis.hypothesis_id == hypothesis.id
                )
            )
        ).scalars().all()
        experiments = [
            await self.repository._must_get(Experiment, link.experiment_id)
            for link in experiment_links
        ]
        observation_links = (
            await self.repository.session.execute(
                select(ObservationHypothesisRelation).where(
                    ObservationHypothesisRelation.hypothesis_id == hypothesis.id,
                    ObservationHypothesisRelation.review_status == ReviewStatus.CONFIRMED.value,
                )
            )
        ).scalars().all()
        observations = [
            await self.repository._must_get(Observation, link.observation_id)
            for link in observation_links
        ]
        lines = [
            f"## {hypothesis.id}",
            "",
            f"Status: {hypothesis.status}",
            "",
            f"Prediction: {hypothesis.prediction}",
            "",
        ]
        for heading, key in (
            ("Supporting Evidence", "SUPPORT"),
            ("Contradictory Evidence", "CONTRADICT"),
            ("Limitations", "LIMITATION"),
        ):
            lines.extend([f"### {heading}", ""])
            lines.extend(self._evidence_lines(grouped[key]) or ["- None confirmed."])
            lines.append("")
        lines.extend(["### Experiments", ""])
        lines.extend(
            [f"- [{item.id}] {item.purpose} — {item.status}" for item in experiments]
            or ["- None recorded."]
        )
        lines.extend(["", "### Observations", ""])
        lines.extend(
            [f"- [{item.id}] {item.description}" for item in observations]
            or ["- None recorded."]
        )
        lines.extend(
            [
                "",
                "### Current Interpretation",
                "",
                f"Current reviewed status: {hypothesis.status}.",
                "",
                "### Remaining Uncertainty",
                "",
                "See limitations and unresolved questions; no additional inference was generated.",
                "",
                "### Suggested Next Experiment",
                "",
                "- No automatic recommendation recorded." if not experiments else f"- Continue/review {experiments[-1].id}.",
                "",
            ]
        )
        return lines

    @staticmethod
    def _evidence_lines(evidence: list[Evidence]) -> list[str]:
        lines = []
        for item in evidence:
            location = (
                f"p.{item.page}" if item.page else f"sec.{item.section}" if item.section else "unknown locator"
            )
            lines.append(
                f"- [{item.id}] {item.statement} — {item.source_title or item.source_id}, {location}, chunk `{item.chunk_id}`"
            )
        return lines


__all__ = ["ResearchReportService"]
