"""Build reproducible observations from completed runs without LLM inference."""

from __future__ import annotations

from statistics import mean

from sqlmodel import select

from db.models.research import Experiment, Observation, ResearchRun
from db.repository.research_repository import ResearchRepository
from research.enums import RunStatus
from research.validators import ResearchValidationError


class ObservationService:
    def __init__(self, repository: ResearchRepository):
        self.repository = repository

    async def build_from_runs(
        self,
        experiment_id: str,
        run_ids: list[str],
        *,
        hypothesis_relations: dict[str, str] | None = None,
        actor: str = "human",
    ) -> Observation:
        experiment = await self.repository._must_get(Experiment, experiment_id)
        unique_ids = list(dict.fromkeys(run_ids))
        if not unique_ids:
            raise ResearchValidationError("observation requires at least one run")
        runs = (
            await self.repository.session.execute(
                select(ResearchRun).where(ResearchRun.id.in_(unique_ids))
            )
        ).scalars().all()
        by_id = {run.id: run for run in runs}
        if set(by_id) != set(unique_ids):
            missing = sorted(set(unique_ids) - set(by_id))
            raise ResearchValidationError(f"observation run not found: {missing}")
        ordered = [by_id[item] for item in unique_ids]
        if any(run.experiment_id != experiment.id for run in ordered):
            raise ResearchValidationError("all runs must belong to the selected experiment")
        if any(run.status != RunStatus.COMPLETED.value for run in ordered):
            raise ResearchValidationError("all runs must be completed before observation")

        numeric: dict[str, list[float]] = {}
        groups: dict[str, dict[str, list[float]]] = {}
        for run in ordered:
            group = str(run.config.get("group") or "ungrouped")
            for metric, value in run.metrics.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    numeric.setdefault(metric, []).append(float(value))
                    groups.setdefault(group, {}).setdefault(metric, []).append(float(value))
        if not numeric:
            raise ResearchValidationError("runs do not contain numeric metrics")
        metric_means = {key: mean(values) for key, values in numeric.items()}
        group_means = {
            group: {key: mean(values) for key, values in metrics.items()}
            for group, metrics in groups.items()
        }
        delta = {}
        if "treatment" in group_means and "control" in group_means:
            common = set(group_means["treatment"]) & set(group_means["control"])
            delta = {
                key: group_means["treatment"][key] - group_means["control"][key]
                for key in sorted(common)
            }
        measured = {"runs": {run.id: run.metrics for run in ordered}}
        statistics = {
            "run_count": len(ordered),
            "metric_means": metric_means,
            "group_means": group_means,
            "treatment_minus_control": delta,
        }
        description = f"Computed deterministic summary for {len(ordered)} completed run(s)."
        observation = await self.repository.create_observation(
            Observation(
                experiment_id=experiment.id,
                measured_results=measured,
                derived_statistics=statistics,
                description=description,
            ),
            unique_ids,
            hypothesis_relations=hypothesis_relations,
            actor=actor,
        )
        return observation


__all__ = ["ObservationService"]
