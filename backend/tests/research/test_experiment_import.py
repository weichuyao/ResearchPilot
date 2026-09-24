import asyncio
import json

import pytest

from db.models.research import Experiment, Hypothesis, ResearchQuestion
from db.repository.research_repository import ResearchNotFoundError, ResearchRepository
from research.enums import (
    ApprovalStatus,
    ExperimentStatus,
    HypothesisStatus,
    ObservationRelationType,
    ReviewStatus,
)
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


def test_observation_link_needs_confirmation_before_it_moves_a_hypothesis(tmp_path):
    """观察的**数值**是算出来的，但它对假设的**解读**是判断 —— 未确认不得解锁跃迁。

    这条测试是这条约束的全部理由：`_require_hypothesis_basis` 只数已确认的观察关系。
    没有它，"人工批准后才有 SUPPORTED" 会被一条自动生成的观察悄悄绕过。
    """
    async def scenario():
        async with isolated_session(tmp_path / "obs-confirm.db") as session:
            repository = ResearchRepository(session)
            _question, hypothesis, experiment = await _approved_experiment(repository)
            await repository.transition_hypothesis(
                hypothesis.id, HypothesisStatus.UNDER_TEST.value
            )
            import_service = ExperimentService(repository)
            control, _ = await _import(import_service, repository, {
                "experiment_id": experiment.id, "run_id": "RUN-C",
                "config": {"group": "control"}, "metrics": {"mAP": 87.61}})
            treatment, _ = await _import(import_service, repository, {
                "experiment_id": experiment.id, "run_id": "RUN-T",
                "config": {"group": "treatment"}, "metrics": {"mAP": 88.21}})
            observation = await ObservationService(repository).build_from_runs(
                experiment.id, [control.id, treatment.id],
                hypothesis_relations={hypothesis.id: ObservationRelationType.SUPPORT.value},
            )
            # rollback 会让 ORM 实例过期，之后只碰这些纯字符串，不碰 ORM 对象
            observation_id = observation.id
            hypothesis_id = hypothesis.id
            await session.commit()

            links = await repository.list_observation_relations(observation_id)
            assert [link.review_status for link in links] == [ReviewStatus.PROPOSED.value]

            approval = await repository.request_approval(
                entity_type="Hypothesis", entity_id=hypothesis_id,
                action="UPDATE_HYPOTHESIS_STATUS:SUPPORTED",
            )
            await repository.review_approval(approval.id, "APPROVED", reviewer="pi")
            approval_id = approval.id
            await session.commit()

            # 人工批准了，但观察关系还没确认 —— 两者都是必要的，都不是充分的
            with pytest.raises(ResearchValidationError, match="confirmed support"):
                await repository.transition_hypothesis(
                    hypothesis_id, HypothesisStatus.SUPPORTED.value,
                    approval_request_id=approval_id, actor="pi",
                )
            await session.rollback()

            reviewed = await repository.review_observation_relation(
                observation_id, hypothesis_id, "CONFIRMED", reviewer="pi"
            )
            assert reviewed.review_status == ReviewStatus.CONFIRMED.value
            assert reviewed.reviewed_by == "pi"
            assert reviewed.reviewed_at is not None
            await session.commit()

            moved = await repository.transition_hypothesis(
                hypothesis_id, HypothesisStatus.SUPPORTED.value,
                approval_request_id=approval_id, actor="pi",
            )
            assert moved.status == HypothesisStatus.SUPPORTED.value

            with pytest.raises(ResearchValidationError, match="already been reviewed"):
                await repository.review_observation_relation(
                    observation_id, hypothesis_id, "REJECTED", reviewer="someone-else"
                )
            with pytest.raises(ResearchValidationError, match="must be CONFIRMED or REJECTED"):
                await repository.review_observation_relation(
                    observation_id, hypothesis_id, "PROPOSED", reviewer="pi"
                )
            # 不存在的链接是「查不到」不是「违反不变量」—— 路由据此分 404 / 422
            with pytest.raises(ResearchNotFoundError):
                await repository.review_observation_relation(
                    observation_id, "H-does-not-exist", "CONFIRMED", reviewer="pi"
                )

    asyncio.run(scenario())
