"""Fully offline Scientific Research Harness V1 demonstration.

Run from backend/:
    .venv-py311/Scripts/python.exe scripts/scientific_harness_demo.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile

from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel


BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BACKEND_ROOT, "app"))

import db.models  # noqa: E402,F401
from ai.rag.pipeline import Outcome  # noqa: E402
from db.models.research import Conclusion, EvidenceRelation, Experiment, Hypothesis, ResearchQuestion  # noqa: E402
from db.repository.research_repository import ResearchRepository  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from research.controller import ResearchController  # noqa: E402
from research.archive import ResearchArchiveService  # noqa: E402
from research.evidence_engine import LiteratureEvidenceEngine  # noqa: E402
from research.experiment_service import ExperimentService  # noqa: E402
from research.hypothesis_service import HypothesisEvidenceService  # noqa: E402
from research.hypothesis_update import HypothesisUpdateService  # noqa: E402
from research.observation_service import ObservationService  # noqa: E402
from research.provenance import ProvenanceService  # noqa: E402
from research.report import ResearchReportService  # noqa: E402
from research.schemas import HypothesisUpdateProposal  # noqa: E402


def fixture_retrieve(query: str) -> Outcome:
    lower = query.lower()
    if "contrary evidence" in lower:
        content, page, chunk = (
            "Performance becomes unstable when hard negatives contain false-negative identities.",
            "8",
            "CH-DEMO-CONTRADICT",
        )
    elif "scope and limitations" in lower:
        content, page, chunk = (
            "The reported gain is limited to one dataset and one backbone configuration.",
            "11",
            "CH-DEMO-LIMITATION",
        )
    else:
        content, page, chunk = (
            "Hard-negative guided local learning improves fine-grained ReID discrimination.",
            "5",
            "CH-DEMO-SUPPORT",
        )
    document = Document(
        page_content=content,
        metadata={
            "source": "offline-reid-fixture.pdf",
            "paper_title": "Offline ReID Fixture",
            "page_label": page,
            "locator_prefix": "p",
            "locator_kind": "page",
            "chunk_id": chunk,
        },
    )
    return Outcome(query=query, hits=[(document, "hybrid", 0.86)], vector_top1=0.86, rejected=False)


async def approve(repository, *, entity_type, entity_id, action, changes=None):
    request = await repository.request_approval(
        entity_type=entity_type,
        entity_id=entity_id,
        action=action,
        proposed_changes=changes or {},
    )
    await repository.review_approval(request.id, "APPROVED", reviewer="demo-scientist")
    return request


async def main() -> None:
    with tempfile.TemporaryDirectory(prefix="research-harness-demo-") as directory:
        path = os.path.join(directory, "research.db")
        engine = create_async_engine(f"sqlite+aiosqlite:///{path}")

        @event.listens_for(engine.sync_engine, "connect")
        def _foreign_keys(connection, _record):
            cursor = connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

        async with engine.begin() as connection:
            await connection.run_sync(SQLModel.metadata.create_all)
        maker = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
        async with maker() as session:
            repository = ResearchRepository(session)
            controller = ResearchController(repository)

            question = await repository.create_question(
                ResearchQuestion(
                    title="Hard-negative guided local representation learning",
                    description="Does it improve fine-grained identity discrimination?",
                    scope="Fixed ReID dataset, backbone and optimizer.",
                    out_of_scope="Claims of universal superiority.",
                )
            )
            await controller.initialize(question.id)
            await controller.advance(question.id, "LITERATURE_REVIEW")
            await controller.advance(question.id, "GENERATE_HYPOTHESIS")

            hypothesis = await repository.create_hypothesis(
                Hypothesis(
                    research_question_id=question.id,
                    statement="Global hard-negative relations improve local discriminative learning.",
                    rationale="Confusing identities should provide a stronger local learning signal.",
                    prediction="Treatment mean mAP exceeds control by at least 0.5 points.",
                )
            )
            registration = await approve(
                repository,
                entity_type="Hypothesis",
                entity_id=hypothesis.id,
                action="REGISTER_HYPOTHESIS",
            )
            await repository.transition_hypothesis(
                hypothesis.id, "TESTABLE", approval_request_id=registration.id
            )
            await repository.transition_hypothesis(hypothesis.id, "UNDER_TEST")

            search = await HypothesisEvidenceService(
                repository,
                LiteratureEvidenceEngine(session, retrieve_fn=fixture_retrieve),
            ).search(question, hypothesis)
            for relation in search.proposed_relations:
                await repository.review_evidence_relation(
                    relation.id, "CONFIRMED", reviewer="demo-scientist"
                )
            await controller.advance(question.id, "SEARCH_SUPPORTING_EVIDENCE")
            await controller.advance(question.id, "SEARCH_CONTRADICTING_EVIDENCE")
            await controller.advance(question.id, "ASSESS_EVIDENCE")
            await controller.advance(question.id, "DESIGN_EXPERIMENT")

            experiment = await repository.create_experiment(
                Experiment(
                    research_question_id=question.id,
                    purpose="Compare fixed control and hard-negative treatment across seeds.",
                    independent_variable="hard-negative guidance",
                    dependent_variables=["mAP", "rank1"],
                    control="guidance disabled",
                    treatment="guidance enabled",
                    controlled_variables=["dataset", "backbone", "optimizer"],
                    metrics=["mAP", "rank1"],
                    success_criteria={"mAP_delta_min": 0.5, "minimum_runs": 3},
                ),
                [hypothesis.id],
            )
            await repository.transition_experiment(experiment.id, "PLANNED")
            await controller.advance(question.id, "HUMAN_APPROVAL")
            experiment_approval = await approve(
                repository,
                entity_type="Experiment",
                entity_id=experiment.id,
                action="APPROVE_EXPERIMENT",
            )
            await repository.transition_experiment(
                experiment.id, "APPROVED", approval_request_id=experiment_approval.id
            )
            await controller.advance(question.id, "WAIT_FOR_RESULT")
            await controller.advance(question.id, "IMPORT_RESULT")

            experiment_service = ExperimentService(repository)
            runs = []
            for run_id, seed, group, m_ap, rank1 in (
                ("RUN-001", 1, "control", 87.61, 93.80),
                ("RUN-002", 1, "treatment", 88.21, 94.32),
                ("RUN-003", 2, "treatment", 88.31, 94.40),
            ):
                raw = {
                    "experiment_id": experiment.id,
                    "run_id": run_id,
                    "seed": seed,
                    "config": {"group": group},
                    "metrics": {"mAP": m_ap, "rank1": rank1},
                    "artifact_paths": [f"demo/{run_id}/metrics.json"],
                    "environment": {"fixture": True},
                }
                prepared = experiment_service.prepare_run_import(json.dumps(raw), "json")
                import_approval = await experiment_service.request_import_approval(prepared)
                await repository.review_approval(
                    import_approval.id, "APPROVED", reviewer="demo-scientist"
                )
                runs.append(
                    await experiment_service.confirm_run_import(
                        prepared,
                        approval_request_id=import_approval.id,
                        actor="demo-scientist",
                    )
                )
            await controller.advance(question.id, "BUILD_OBSERVATION")
            observation = await ObservationService(repository).build_from_runs(
                experiment.id,
                [run.id for run in runs],
                hypothesis_relations={hypothesis.id: "SUPPORT"},
            )
            # 数值是算出来的，「这条观察支持该假设」却是判断 —— 同样要过确认。
            await repository.review_observation_relation(
                observation.id, hypothesis.id, "CONFIRMED", reviewer="demo-scientist"
            )
            await experiment_service.complete_experiment(
                experiment.id, actor="demo-scientist"
            )
            await controller.advance(question.id, "UPDATE_HYPOTHESIS")

            support = next(
                relation for relation in search.proposed_relations if relation.relation == "SUPPORT"
            )
            contradiction = next(
                relation for relation in search.proposed_relations if relation.relation == "CONTRADICT"
            )
            update = HypothesisUpdateProposal(
                hypothesis_id=hypothesis.id,
                proposed_status="PARTIALLY_SUPPORTED",
                supporting_evidence=[support.evidence_id],
                contradicting_evidence=[contradiction.evidence_id],
                supporting_observations=[observation.id],
                reasoning="The deterministic observation supports the prediction, while literature records instability risk.",
                remaining_uncertainty="Only the fixed demo protocol has been evaluated.",
            )
            update_service = HypothesisUpdateService(repository)
            update_approval = await update_service.propose(update)
            await repository.review_approval(
                update_approval.id, "APPROVED", reviewer="demo-scientist"
            )
            await update_service.apply(update_approval.id, actor="demo-scientist")

            conclusion = await repository.create_conclusion(
                Conclusion(
                    research_question_id=question.id,
                    statement="Under the fixed demo protocol, the treatment shows a limited positive effect.",
                    confidence="MEDIUM",
                    limitations=["Fixed offline fixture", "Two treatment seeds"],
                    unresolved_questions=["Does the effect generalize to other datasets?"],
                ),
                supporting_evidence_ids=[support.evidence_id],
                contradicting_evidence_ids=[contradiction.evidence_id],
                observation_ids=[observation.id],
            )
            await repository.transition_conclusion(conclusion.id, "PENDING_APPROVAL")
            conclusion_approval = await approve(
                repository,
                entity_type="Conclusion",
                entity_id=conclusion.id,
                action="APPROVE_CONCLUSION",
            )
            await repository.transition_conclusion(
                conclusion.id, "APPROVED", approval_request_id=conclusion_approval.id
            )
            await controller.advance(question.id, "GENERATE_RESEARCH_REPORT")
            await controller.advance(question.id, "COMPLETE")
            await session.commit()

            print(await ResearchReportService(repository).generate(question.id))
            print("\n# Provenance JSON\n")
            print(
                json.dumps(
                    await ProvenanceService(repository).get_provenance(conclusion.id),
                    ensure_ascii=False,
                    indent=2,
                )
            )
            archive = await ResearchArchiveService(repository).export(question.id)
            assert ResearchArchiveService.verify(archive)
            print("\n# Research Archive\n")
            print(json.dumps({
                "schema_version": archive["schema_version"],
                "state_fingerprint": archive["state_fingerprint"],
                "archive_sha256": archive["archive_sha256"],
                "record_count": sum(len(rows) for rows in archive["records"].values()),
                "verified": True,
            }, ensure_ascii=False, indent=2))
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
