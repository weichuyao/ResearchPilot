import asyncio
import json

import pytest

from db.models.research import Experiment, Hypothesis, ResearchQuestion
from db.repository.research_repository import ResearchRepository
from research.enums import ApprovalStatus, ExperimentStatus, ObservationRelationType
from research.experiment_service import ExperimentService
from research.observation_service import ObservationService
from research.validators import ResearchValidationError
from tests.research.helpers import isolated_session


async def _approved_experiment(repository):
    question = await repository.create_question(
        ResearchQuestion(title="RQ", description="Does treatment improve mAP?")
    )
    hypothesis = await repository.create_hypothesis(
        Hypothesis(
            research_question_id=question.id,
            statement="Treatment improves mAP.",
            prediction="Treatment mean exceeds control mean.",
        )
    )
    registration = await repository.request_approval(
        entity_type="Hypothesis",
        entity_id=hypothesis.id,
        action="REGISTER_HYPOTHESIS",
    )
    await repository.review_approval(registration.id, "APPROVED", reviewer="pi")
    await repository.transition_hypothesis(
        hypothesis.id, "TESTABLE", approval_request_id=registration.id
    )
    experiment = await repository.create_experiment(
        Experiment(
            research_question_id=question.id,
            purpose="Compare treatment and control",
            independent_variable="guidance",
            dependent_variables=["mAP", "rank1"],
            control="guidance off",
            treatment="guidance on",
            controlled_variables=["dataset", "backbone"],
            metrics=["mAP", "rank1"],
            success_criteria={"mAP_delta_min": 0.5},
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
    return question, hypothesis, experiment


async def _import(service, repository, raw, fmt="json"):
    content = json.dumps(raw) if fmt == "json" else raw
    prepared = service.prepare_run_import(content, fmt)
    approval = await service.request_import_approval(prepared, requested_by="agent")
    await repository.review_approval(
        approval.id, ApprovalStatus.APPROVED.value, reviewer="human-reviewer"
    )
    return await service.confirm_run_import(
        prepared, approval_request_id=approval.id, actor="human-reviewer"
    ), prepared


def test_json_import_requires_confirmation_and_rejects_duplicate(tmp_path):
    async def scenario():
        async with isolated_session(tmp_path / "import.db") as session:
            repository = ResearchRepository(session)
            _question, _hypothesis, experiment = await _approved_experiment(repository)
            service = ExperimentService(repository)
            raw = {
                "experiment_id": experiment.id,
                "run_id": "RUN-001",
                "config": {"group": "treatment"},
                "seed": 1,
                "metrics": {"mAP": 88.21, "rank1": 94.32},
                "artifact_paths": ["results/run-001/metrics.json"],
                "environment": {"torch": "2.x"},
            }
            prepared = service.prepare_run_import(json.dumps(raw), "json")
            with pytest.raises(ResearchValidationError, match="matching approved"):
                await service.confirm_run_import(
                    prepared, approval_request_id="APR-missing", actor="human"
                )
            approval = await service.request_import_approval(prepared)
            assert len(approval.action) <= 80
            approval.proposed_changes = {
                **approval.proposed_changes,
                "payload_hash": "0" * 64,
            }
            await repository.review_approval(approval.id, "APPROVED", reviewer="human")
            with pytest.raises(ResearchValidationError, match="matching approved"):
                await service.confirm_run_import(
                    prepared, approval_request_id=approval.id, actor="human"
                )
            run, _ = await _import(service, repository, raw)
            assert run.status == "COMPLETED"
            assert run.metrics["mAP"] == 88.21
            assert run.raw_import["format"] == "json"
            await session.refresh(experiment)
            assert experiment.status == "RUNNING"
            approval = await service.request_import_approval(prepared)
            await repository.review_approval(approval.id, "APPROVED", reviewer="human")
            with pytest.raises(ResearchValidationError, match="already exists"):
                await service.confirm_run_import(
                    prepared, approval_request_id=approval.id, actor="human"
                )

    asyncio.run(scenario())


def test_yaml_and_csv_protocols_are_explicit():
    repository = object()  # parsing is pure; no database needed
    service = ExperimentService(repository)  # type: ignore[arg-type]
    yaml_prepared = service.prepare_run_import(
        """
experiment_id: EXP-1
run_id: RUN-YAML
config: {group: treatment}
seed: 2
metrics: {mAP: 88.4}
""",
        "yaml",
    )
    assert yaml_prepared.payload.metrics == {"mAP": 88.4}

    csv_prepared = service.prepare_run_import(
        "experiment_id,run_id,seed,metrics.mAP,config.group,artifact_paths\n"
        "EXP-1,RUN-CSV,3,87.6,control,metrics.json;config.yaml\n",
        "csv",
    )
    assert csv_prepared.payload.metrics == {"mAP": 87.6}
    assert csv_prepared.payload.config == {"group": "control"}
    assert len(csv_prepared.payload.artifact_paths) == 2


def test_observation_is_computed_from_runs_not_interpreted(tmp_path):
    async def scenario():
        async with isolated_session(tmp_path / "observation-service.db") as session:
            repository = ResearchRepository(session)
            _question, hypothesis, experiment = await _approved_experiment(repository)
            import_service = ExperimentService(repository)
            control, _ = await _import(
                import_service,
                repository,
                {
                    "experiment_id": experiment.id,
                    "run_id": "RUN-CONTROL",
                    "config": {"group": "control"},
                    "seed": 1,
                    "metrics": {"mAP": 87.61, "rank1": 93.8},
                },
            )
            treatment, _ = await _import(
                import_service,
                repository,
                {
                    "experiment_id": experiment.id,
                    "run_id": "RUN-TREATMENT",
                    "config": {"group": "treatment"},
                    "seed": 1,
                    "metrics": {"mAP": 88.21, "rank1": 94.3},
                },
            )
            observation = await ObservationService(repository).build_from_runs(
                experiment.id,
                [control.id, treatment.id],
                hypothesis_relations={
                    hypothesis.id: ObservationRelationType.SUPPORT.value
                },
            )
            await session.commit()
            await session.refresh(experiment)
            assert experiment.status == "RUNNING"
            stats = observation.derived_statistics
            assert stats["run_count"] == 2
            assert stats["metric_means"]["mAP"] == pytest.approx(87.91)
            assert stats["treatment_minus_control"]["mAP"] == pytest.approx(0.60)
            assert "supports" not in observation.description.lower()
            await import_service.complete_experiment(experiment.id, actor="pi")
            assert experiment.status == "COMPLETED"

    asyncio.run(scenario())
