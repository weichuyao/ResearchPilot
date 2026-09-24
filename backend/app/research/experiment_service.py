"""Standard, reviewable import protocol for externally executed experiments."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import dataclass

import yaml
from sqlmodel import select

from db.models.research import ApprovalRequest, Experiment, Observation, ResearchRun
from db.repository.research_repository import ResearchNotFoundError, ResearchRepository
from research.enums import ApprovalStatus, ExperimentStatus, RunStatus
from research.schemas import RunImportPayload
from research.validators import ResearchValidationError, require_text


@dataclass(frozen=True)
class PreparedRunImport:
    payload: RunImportPayload
    raw_payload: dict
    payload_hash: str
    source_format: str


def import_approval_action(payload_hash: str) -> str:
    """Build a compact approval key while the full digest stays in the payload."""
    return f"CONFIRM_RUN_IMPORT:{payload_hash[:32]}"


class ExperimentService:
    FORMATS = {"json", "yaml", "yml", "csv"}

    def __init__(self, repository: ResearchRepository):
        self.repository = repository

    def prepare_run_import(self, content: str | bytes, source_format: str) -> PreparedRunImport:
        source_format = source_format.lower().lstrip(".")
        if source_format not in self.FORMATS:
            raise ResearchValidationError("run import format must be json, yaml or csv")
        text = content.decode("utf-8") if isinstance(content, bytes) else content
        require_text(text, "run import content")
        try:
            if source_format == "json":
                raw = json.loads(text)
            elif source_format in {"yaml", "yml"}:
                raw = yaml.safe_load(text)
            else:
                raw = self._parse_csv(text)
            if not isinstance(raw, dict):
                raise ValueError("top-level payload must be an object")
            payload = RunImportPayload.model_validate(raw)
        except Exception as exc:
            raise ResearchValidationError(f"invalid {source_format} run import: {exc}") from exc
        if not payload.metrics:
            raise ResearchValidationError("run import metrics must not be empty")
        canonical = json.dumps(raw, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        payload_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return PreparedRunImport(payload, raw, payload_hash, source_format)

    @staticmethod
    def _parse_csv(text: str) -> dict:
        rows = list(csv.DictReader(io.StringIO(text)))
        if len(rows) != 1:
            raise ValueError("CSV import must contain exactly one data row")
        row = rows[0]
        raw: dict = {
            "experiment_id": row.pop("experiment_id", ""),
            "run_id": row.pop("run_id", "") or None,
            "seed": int(row.pop("seed")) if row.get("seed") else None,
            "metrics": {},
            "config": {},
            "environment": {},
            "artifact_paths": [],
        }
        artifacts = row.pop("artifact_paths", "")
        if artifacts:
            raw["artifact_paths"] = [item.strip() for item in artifacts.split(";") if item.strip()]
        for key, value in row.items():
            if value in (None, ""):
                continue
            if "." not in key:
                raise ValueError(f"unsupported CSV column: {key}")
            namespace, name = key.split(".", 1)
            if namespace not in {"metrics", "config", "environment"}:
                raise ValueError(f"unsupported CSV namespace: {namespace}")
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                parsed = value
            raw[namespace][name] = parsed
        return raw

    async def request_import_approval(
        self, prepared: PreparedRunImport, *, requested_by: str = "system"
    ) -> ApprovalRequest:
        experiment = await self.repository._must_get(  # repository owns identity lookup
            Experiment, prepared.payload.experiment_id
        )
        if experiment.status not in {ExperimentStatus.APPROVED.value, ExperimentStatus.RUNNING.value}:
            raise ResearchValidationError("run results can only be imported for an approved experiment")
        return await self.repository.request_approval(
            entity_type="Experiment",
            entity_id=experiment.id,
            action=import_approval_action(prepared.payload_hash),
            proposed_changes={
                "run_id": prepared.payload.run_id,
                "metrics": prepared.payload.metrics,
                "payload_hash": prepared.payload_hash,
                "source_format": prepared.source_format,
            },
            requested_by=requested_by,
        )

    async def confirm_run_import(
        self,
        prepared: PreparedRunImport,
        *,
        approval_request_id: str,
        actor: str,
    ) -> ResearchRun:
        try:
            approval = await self.repository._must_get(ApprovalRequest, approval_request_id)
        except ResearchNotFoundError as exc:
            raise ResearchValidationError("run import requires its matching approved request") from exc
        expected_action = import_approval_action(prepared.payload_hash)
        if (
            approval.entity_id != prepared.payload.experiment_id
            or approval.action != expected_action
            or approval.proposed_changes.get("payload_hash") != prepared.payload_hash
            or approval.status != ApprovalStatus.APPROVED.value
        ):
            raise ResearchValidationError("run import requires its matching approved request")
        duplicate = (
            await self.repository.session.execute(
                select(ResearchRun).where(ResearchRun.import_hash == prepared.payload_hash)
            )
        ).scalars().first()
        if duplicate is not None:
            raise ResearchValidationError(f"run import payload already exists as {duplicate.id}")
        experiment = await self.repository._must_get(
            Experiment, prepared.payload.experiment_id
        )
        if experiment.status == ExperimentStatus.APPROVED.value:
            await self.repository.transition_experiment(
                experiment.id, ExperimentStatus.RUNNING.value, actor=actor
            )
        run = ResearchRun(
            id=prepared.payload.run_id or None,
            experiment_id=prepared.payload.experiment_id,
            config=prepared.payload.config,
            seed=prepared.payload.seed,
            artifact_paths=prepared.payload.artifact_paths,
            metrics=prepared.payload.metrics,
            environment=prepared.payload.environment,
            raw_import={"format": prepared.source_format, "payload": prepared.raw_payload},
            import_hash=prepared.payload_hash,
        )
        # Passing None explicitly bypasses SQLModel's default factory.
        if run.id is None:
            from db.models.research import prefixed_id
            run.id = prefixed_id("RUN")
        await self.repository.create_run(run, actor=actor)
        await self.repository.transition_run(run.id, RunStatus.RUNNING.value, actor=actor)
        await self.repository.transition_run(run.id, RunStatus.COMPLETED.value, actor=actor)
        return run

    async def complete_experiment(
        self, experiment_id: str, *, actor: str = "human"
    ) -> Experiment:
        experiment = await self.repository._must_get(Experiment, experiment_id)
        if experiment.status != ExperimentStatus.RUNNING.value:
            raise ResearchValidationError("only a running experiment can be completed")
        runs = (
            await self.repository.session.execute(
                select(ResearchRun).where(ResearchRun.experiment_id == experiment_id)
            )
        ).scalars().all()
        if not runs or any(run.status != RunStatus.COMPLETED.value for run in runs):
            raise ResearchValidationError(
                "experiment completion requires all runs to be completed"
            )
        observation = (
            await self.repository.session.execute(
                select(Observation).where(Observation.experiment_id == experiment_id)
            )
        ).scalars().first()
        if observation is None:
            raise ResearchValidationError(
                "experiment completion requires at least one observation"
            )
        return await self.repository.transition_experiment(
            experiment.id, ExperimentStatus.COMPLETED.value, actor=actor
        )


__all__ = ["ExperimentService", "PreparedRunImport", "import_approval_action"]
