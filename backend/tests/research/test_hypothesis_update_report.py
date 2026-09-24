import asyncio

import pytest

from db.models.research import (
    Conclusion,
    EvidenceRelation,
    Experiment,
    Hypothesis,
    Observation,
    ResearchQuestion,
    ResearchRun,
)
from db.repository.research_repository import ResearchRepository
from research.enums import ApprovalStatus, ConclusionStatus, ObservationRelationType, RunStatus
from research.hypothesis_update import HypothesisUpdateService
from research.provenance import ProvenanceService
from research.report import ResearchReportService
from research.schemas import HypothesisUpdateProposal
from research.validators import ResearchValidationError
from tests.research.helpers import isolated_session, literature_evidence


async def _registered_hypothesis(repository, question):
    hypothesis = await repository.create_hypothesis(
        Hypothesis(
            research_question_id=question.id,
            statement="Hard-negative guidance improves local discrimination.",
            prediction="Treatment mAP exceeds control mAP.",
        )
    )
    approval = await repository.request_approval(
        entity_type="Hypothesis", entity_id=hypothesis.id, action="REGISTER_HYPOTHESIS"
    )
    await repository.review_approval(approval.id, "APPROVED", reviewer="pi")
    await repository.transition_hypothesis(
        hypothesis.id, "TESTABLE", approval_request_id=approval.id
    )
    await repository.transition_hypothesis(hypothesis.id, "UNDER_TEST")
    return hypothesis


async def _approved_experiment(repository, question, hypothesis):
    experiment = await repository.create_experiment(
        Experiment(
            research_question_id=question.id,
            purpose="Controlled comparison",
            independent_variable="guidance",
            dependent_variables=["mAP"],
            control="off",
            treatment="on",
            metrics=["mAP"],
            success_criteria={"delta": 0.5},
        ),
        [hypothesis.id],
    )
    await repository.transition_experiment(experiment.id, "PLANNED")
    approval = await repository.request_approval(
        entity_type="Experiment", entity_id=experiment.id, action="APPROVE_EXPERIMENT"
    )
    await repository.review_approval(approval.id, "APPROVED", reviewer="pi")
    await repository.transition_experiment(
        experiment.id, "APPROVED", approval_request_id=approval.id
    )
    return experiment


def test_hypothesis_proposal_is_validated_then_human_applied(tmp_path):
    async def scenario():
        async with isolated_session(tmp_path / "hypothesis-update.db") as session:
            repository = ResearchRepository(session)
            question = await repository.create_question(
                ResearchQuestion(title="RQ", description="Question")
            )
            hypothesis = await _registered_hypothesis(repository, question)
            evidence = await repository.create_evidence(literature_evidence(question.id))
            service = HypothesisUpdateService(repository)
            proposal = HypothesisUpdateProposal(
                hypothesis_id=hypothesis.id,
                proposed_status="PARTIALLY_SUPPORTED",
                supporting_evidence=[evidence.id],
                reasoning="The paper reports a relevant improvement.",
                remaining_uncertainty="Independent experimental replication is pending.",
            )
            with pytest.raises(ResearchValidationError, match="lacks confirmed SUPPORT"):
                await service.propose(proposal)

            relation = await repository.create_evidence_relation(
                EvidenceRelation(
                    hypothesis_id=hypothesis.id,
                    evidence_id=evidence.id,
                    relation="SUPPORT",
                )
            )
            await repository.review_evidence_relation(
                relation.id, "CONFIRMED", reviewer="scientist"
            )
            approval = await service.propose(proposal)
            assert hypothesis.status == "UNDER_TEST"
            with pytest.raises(ResearchValidationError, match="not approved"):
                await service.apply(approval.id, actor="scientist")
            await repository.review_approval(approval.id, "APPROVED", reviewer="pi")
            updated = await service.apply(approval.id, actor="pi")
            assert updated.status == "PARTIALLY_SUPPORTED"

    asyncio.run(scenario())


def test_report_and_provenance_follow_both_source_branches(tmp_path):
    async def scenario():
        async with isolated_session(tmp_path / "report.db") as session:
            repository = ResearchRepository(session)
            question = await repository.create_question(
                ResearchQuestion(
                    title="Hard-negative guided local learning",
                    description="Does it improve identity discrimination?",
                )
            )
            hypothesis = await _registered_hypothesis(repository, question)
            evidence = await repository.create_evidence(literature_evidence(question.id))
            relation = await repository.create_evidence_relation(
                EvidenceRelation(
                    hypothesis_id=hypothesis.id,
                    evidence_id=evidence.id,
                    relation="SUPPORT",
                )
            )
            await repository.review_evidence_relation(relation.id, "CONFIRMED", reviewer="pi")
            experiment = await _approved_experiment(repository, question, hypothesis)
            run = await repository.create_run(
                ResearchRun(
                    id="RUN-REPORT",
                    experiment_id=experiment.id,
                    metrics={"mAP": 88.21},
                    artifact_paths=["results/metrics.json"],
                )
            )
            await repository.transition_run(run.id, RunStatus.RUNNING.value)
            await repository.transition_run(run.id, RunStatus.COMPLETED.value)
            observation = await repository.create_observation(
                Observation(
                    experiment_id=experiment.id,
                    measured_results={"mAP": 88.21},
                    derived_statistics={"delta_mAP": 0.60},
                    description="Treatment exceeded control by 0.60 mAP.",
                ),
                [run.id],
                {hypothesis.id: ObservationRelationType.SUPPORT.value},
            )
            conclusion = await repository.create_conclusion(
                Conclusion(
                    research_question_id=question.id,
                    statement="Current evidence supports a limited positive effect.",
                    confidence="MEDIUM",
                    limitations=["One run"],
                    unresolved_questions=["Does it persist across seeds?"],
                ),
                supporting_evidence_ids=[evidence.id],
                observation_ids=[observation.id],
            )
            await repository.transition_conclusion(conclusion.id, "PENDING_APPROVAL")
            approval = await repository.request_approval(
                entity_type="Conclusion",
                entity_id=conclusion.id,
                action="APPROVE_CONCLUSION",
            )
            await repository.review_approval(
                approval.id, ApprovalStatus.APPROVED.value, reviewer="pi"
            )
            await repository.transition_conclusion(
                conclusion.id,
                ConclusionStatus.APPROVED.value,
                approval_request_id=approval.id,
            )

            graph = await ProvenanceService(repository).get_provenance(conclusion.id)
            types = {node["type"] for node in graph["nodes"]}
            assert {"Conclusion", "Evidence", "Observation", "Experiment", "Run"} <= types
            evidence_node = next(node for node in graph["nodes"] if node["type"] == "Evidence")
            assert evidence_node["page"] == "7"
            assert evidence_node["chunk_id"] == "chunk-fixture-001"
            assert {edge["type"] for edge in graph["edges"]} >= {
                "SUPPORTING", "OBSERVATION", "FROM_EXPERIMENT", "HAS_RUN"
            }

            report = await ResearchReportService(repository).generate(question.id)
            assert "# Research Question" in report
            assert "### Contradictory Evidence" in report
            assert "# Evidence Provenance" in report
            assert "p.7" in report
            assert evidence.id in report
            assert run.id in report
            assert "Does it persist across seeds?" in report

    asyncio.run(scenario())
