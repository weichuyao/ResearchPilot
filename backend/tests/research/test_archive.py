import asyncio
import json
import sys
from copy import deepcopy

import pytest

from db.models.research import Conclusion, ResearchQuestion
from db.repository.research_repository import ResearchRepository
from research.archive import ResearchArchiveService
from research.validators import ResearchValidationError
from scripts.verify_research_archive import main as verify_archive_main
from tests.research.helpers import isolated_session, literature_evidence


def test_archive_is_complete_deterministic_and_tamper_evident(tmp_path):
    async def scenario():
        async with isolated_session(tmp_path / "archive.db") as session:
            repository = ResearchRepository(session)
            question = await repository.create_question(
                ResearchQuestion(title="Portable RQ", description="Can this state be exported?")
            )
            evidence = await repository.create_evidence(literature_evidence(question.id))
            conclusion = await repository.create_conclusion(
                Conclusion(
                    research_question_id=question.id,
                    statement="The archive retains source-grounded state.",
                ),
                supporting_evidence_ids=[evidence.id],
            )
            await repository.request_approval(
                entity_type="Conclusion",
                entity_id=conclusion.id,
                action="APPROVE_CONCLUSION",
            )
            await session.commit()

            service = ResearchArchiveService(repository)
            archive = await service.export(question.id)
            assert archive["schema_version"] == "scientific-research-harness/v1"
            assert archive["records"]["research_question"][0]["id"] == question.id
            assert archive["records"]["evidence"][0]["chunk_id"] == evidence.chunk_id
            assert conclusion.id in archive["conclusion_provenance"]
            assert any(
                event["entity_type"] == "ApprovalRequest"
                for event in archive["records"]["events"]
            )
            assert service.verify(archive)

            archive["records"]["evidence"][0]["page"] = "tampered"
            assert not service.verify(archive)

    asyncio.run(scenario())


def test_archive_rejects_resigned_cross_question_records(tmp_path):
    async def scenario():
        async with isolated_session(tmp_path / "scope-source.db") as session:
            repository = ResearchRepository(session)
            question = await repository.create_question(
                ResearchQuestion(title="Scoped RQ", description="One project only")
            )
            archive = await ResearchArchiveService(repository).export(question.id)

        tampered = deepcopy(archive)
        tampered["records"]["evidence"] = [
            literature_evidence("RQ-foreign").model_dump(mode="json")
        ]
        tampered["archive_sha256"] = ResearchArchiveService._digest(tampered)
        async with isolated_session(tmp_path / "scope-target.db") as session:
            service = ResearchArchiveService(ResearchRepository(session))
            with pytest.raises(ResearchValidationError, match="another research question"):
                await service.restore(tampered)

    asyncio.run(scenario())


def test_archive_verifier_cli_rejects_tampering(tmp_path, monkeypatch, capsys):
    archive = {
        "schema_version": ResearchArchiveService.SCHEMA_VERSION,
        "research_question_id": "RQ-test",
        "state_fingerprint": "fingerprint",
        "records": {"research_question": []},
    }
    archive["archive_sha256"] = ResearchArchiveService._digest(archive)
    path = tmp_path / "archive.json"
    path.write_text(json.dumps(archive), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["verify_research_archive.py", str(path)])
    assert verify_archive_main() == 0
    assert '"valid": true' in capsys.readouterr().out

    archive["research_question_id"] = "RQ-tampered"
    path.write_text(json.dumps(archive), encoding="utf-8")
    assert verify_archive_main() == 1
    assert "does not match" in capsys.readouterr().err


def test_archive_rejects_internally_resigned_stale_fingerprint(tmp_path):
    """改完内容再把档案哈希重算一遍，让它**自我一致**——这一层只能靠指纹拦。

    `test_archive_rejects_resigned_cross_question_records` 防的是夹带别的项目；
    这里防的是同项目内的内容篡改：所有外键都合法，作用域校验全过，
    只有重新计算的状态指纹会对不上。断言里必须连"什么都没落库"一起验，
    否则只证明了报错、没证明回滚。
    """
    async def scenario():
        async with isolated_session(tmp_path / "fp-source.db") as session:
            repository = ResearchRepository(session)
            question = await repository.create_question(
                ResearchQuestion(title="Fingerprint RQ", description="Stale fingerprint must roll back")
            )
            evidence = await repository.create_evidence(literature_evidence(question.id))
            conclusion = await repository.create_conclusion(
                Conclusion(research_question_id=question.id, statement="Approved wording"),
                supporting_evidence_ids=[evidence.id],
            )
            await session.commit()
            archive = await ResearchArchiveService(repository).export(question.id)

        tampered = deepcopy(archive)
        for row in tampered["records"]["conclusions"]:
            if row["id"] == conclusion.id:
                row["statement"] = "Silently rewritten wording"
        tampered["archive_sha256"] = ResearchArchiveService._digest(tampered)
        # 档案自身是自洽的：verify() 一定放行，所以它必须死在指纹这一关
        assert ResearchArchiveService.verify(tampered)
        assert tampered["state_fingerprint"] == archive["state_fingerprint"]

        async with isolated_session(tmp_path / "fp-target.db") as session:
            service = ResearchArchiveService(ResearchRepository(session))
            with pytest.raises(ResearchValidationError, match="fingerprint does not match"):
                await service.restore(tampered)
            await session.rollback()
            assert await session.get(ResearchQuestion, question.id) is None
            assert await session.get(Conclusion, conclusion.id) is None

    asyncio.run(scenario())


def test_archive_restores_into_empty_store_without_overwrite(tmp_path):
    async def scenario():
        source_path = tmp_path / "source.db"
        target_path = tmp_path / "target.db"
        async with isolated_session(source_path) as session:
            repository = ResearchRepository(session)
            question = await repository.create_question(
                ResearchQuestion(title="Restorable RQ", description="Portable structured state")
            )
            evidence = await repository.create_evidence(literature_evidence(question.id))
            await repository.create_conclusion(
                Conclusion(research_question_id=question.id, statement="Restorable conclusion"),
                supporting_evidence_ids=[evidence.id],
            )
            await session.commit()
            archive = await ResearchArchiveService(repository).export(question.id)

        async with isolated_session(target_path) as session:
            repository = ResearchRepository(session)
            service = ResearchArchiveService(repository)
            restored = await service.restore(archive)
            await session.commit()
            assert restored["research_question_id"] == question.id
            assert restored["state_fingerprint"] == archive["state_fingerprint"]
            assert await repository._must_get(ResearchQuestion, question.id)
            with pytest.raises(ResearchValidationError, match="already exists"):
                await service.restore(archive)

    asyncio.run(scenario())
