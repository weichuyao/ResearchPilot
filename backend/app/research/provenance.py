"""Deterministic provenance traversal over persisted research relations."""

from sqlmodel import select

from db.models.research import (
    Conclusion,
    ConclusionEvidence,
    ConclusionObservation,
    Evidence,
    Experiment,
    Observation,
    ObservationRun,
    ResearchRun,
)
from db.repository.research_repository import ResearchNotFoundError, ResearchRepository


class ProvenanceService:
    def __init__(self, repository: ResearchRepository):
        self.repository = repository

    async def get_provenance(self, entity_id: str) -> dict:
        conclusion = await self.repository.session.get(Conclusion, entity_id)
        if conclusion is not None:
            return await self._conclusion_graph(conclusion)
        evidence = await self.repository.session.get(Evidence, entity_id)
        if evidence is not None:
            return {"root": evidence.id, "nodes": [self._evidence_node(evidence)], "edges": []}
        observation = await self.repository.session.get(Observation, entity_id)
        if observation is not None:
            nodes, edges = await self._observation_branch(observation)
            return {"root": observation.id, "nodes": nodes, "edges": edges}
        raise ResearchNotFoundError(f"provenance entity not found: {entity_id}")

    async def _conclusion_graph(self, conclusion: Conclusion) -> dict:
        nodes = [{"id": conclusion.id, "type": "Conclusion", "statement": conclusion.statement}]
        edges = []
        evidence_links = (
            await self.repository.session.execute(
                select(ConclusionEvidence).where(ConclusionEvidence.conclusion_id == conclusion.id)
            )
        ).scalars().all()
        for link in evidence_links:
            evidence = await self.repository._must_get(Evidence, link.evidence_id)
            nodes.append(self._evidence_node(evidence))
            edges.append({"from": conclusion.id, "to": evidence.id, "type": link.role})
        observation_links = (
            await self.repository.session.execute(
                select(ConclusionObservation).where(
                    ConclusionObservation.conclusion_id == conclusion.id
                )
            )
        ).scalars().all()
        for link in observation_links:
            observation = await self.repository._must_get(Observation, link.observation_id)
            branch_nodes, branch_edges = await self._observation_branch(observation)
            nodes.extend(branch_nodes)
            edges.append({"from": conclusion.id, "to": observation.id, "type": "OBSERVATION"})
            edges.extend(branch_edges)
        # A graph can converge on the same Experiment/Run.  Keep first node in
        # stable traversal order while preserving every typed edge.
        unique = {node["id"]: node for node in nodes}
        return {"root": conclusion.id, "nodes": list(unique.values()), "edges": edges}

    async def _observation_branch(self, observation: Observation) -> tuple[list[dict], list[dict]]:
        experiment = await self.repository._must_get(Experiment, observation.experiment_id)
        nodes = [
            {
                "id": observation.id,
                "type": "Observation",
                "description": observation.description,
                "measured_results": observation.measured_results,
                "derived_statistics": observation.derived_statistics,
            },
            {"id": experiment.id, "type": "Experiment", "purpose": experiment.purpose},
        ]
        edges = [{"from": observation.id, "to": experiment.id, "type": "FROM_EXPERIMENT"}]
        links = (
            await self.repository.session.execute(
                select(ObservationRun).where(ObservationRun.observation_id == observation.id)
            )
        ).scalars().all()
        for link in links:
            run = await self.repository._must_get(ResearchRun, link.run_id)
            nodes.append(
                {
                    "id": run.id,
                    "type": "Run",
                    "metrics": run.metrics,
                    "artifacts": run.artifact_paths,
                    "import_hash": run.import_hash,
                }
            )
            edges.append({"from": experiment.id, "to": run.id, "type": "HAS_RUN"})
        return nodes, edges

    @staticmethod
    def _evidence_node(evidence: Evidence) -> dict:
        return {
            "id": evidence.id,
            "type": "Evidence",
            "evidence_type": evidence.evidence_type,
            "statement": evidence.statement,
            "excerpt": evidence.excerpt,
            "source_id": evidence.source_id,
            "source_title": evidence.source_title,
            "page": evidence.page,
            "section": evidence.section,
            "chunk_id": evidence.chunk_id,
        }


__all__ = ["ProvenanceService"]

