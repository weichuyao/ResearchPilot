"""审批被**拒绝**时的实体状态 —— 这些路径此前一次都没被测过。

`docs/final_audit.md` §12 记了一次反向路径审计修复：通用审批端点与实验专用端点
都要应用状态变化，实验/结论被拒绝时分别进入 `CANCELLED`/`REJECTED`，不能留下
"审批已决定、实体还停在旧状态"的悬挂状态。修复当时只靠人工实测确认，没有测试 ——
而 `grep CANCELLED tests/` 是 0 命中，正是这类"改完没锁住"的窗口。

副作用住在路由里（`_apply_approval_decision`），直接调仓储测不到它，所以全部走 HTTP。
"""

from __future__ import annotations

import asyncio

from db.models.research import Conclusion, Experiment, ResearchQuestion
from db.repository.research_repository import ResearchRepository
from tests.research.helpers import api_client, literature_evidence


def _create_question_and_hypothesis(client):
    question_id = client.post(
        "/research/questions",
        json={"title": "Rejection path", "description": "What happens on a no?"},
    ).json()["id"]
    client.post(f"/research/questions/{question_id}/workflow/initialize")
    created = client.post(
        f"/research/questions/{question_id}/hypotheses",
        json={"statement": "H1", "prediction": "A measurable prediction"},
    ).json()
    client.post(
        f"/approvals/{created['registration_approval']['id']}/review",
        json={"reviewer": "pi", "decision": "APPROVED"},
    )
    return question_id, created["hypothesis"]["id"]


def test_rejected_experiment_becomes_cancelled_not_stranded(tmp_path):
    with api_client(tmp_path / "reject-exp.db") as (client, _maker):
        question_id, hypothesis_id = _create_question_and_hypothesis(client)
        created = client.post(
            "/experiments",
            json={
                "research_question_id": question_id,
                "tested_hypotheses": [hypothesis_id],
                "purpose": "Test H1",
                "independent_variable": "treatment",
                "dependent_variables": ["mAP"],
                "control": "off",
                "treatment": "on",
                "metrics": ["mAP"],
                "success_criteria": {"delta": 0.5},
            },
        ).json()
        experiment_id = created["experiment"]["id"]

        response = client.post(
            f"/experiments/{experiment_id}/approve",
            json={"reviewer": "pi", "decision": "REJECTED", "reason": "对照设置不可比"},
        )
        assert response.status_code == 200, response.text
        assert response.json()["experiment"]["status"] == "CANCELLED"
        assert response.json()["approval"]["status"] == "REJECTED"
        assert response.json()["approval"]["review_reason"] == "对照设置不可比"


def test_rejected_conclusion_lands_in_rejected_state(tmp_path):
    """结论被拒后必须是 REJECTED，而不是停在 PENDING_APPROVAL 等一个永远不会来的批准。"""
    path = tmp_path / "reject-conclusion.db"

    async def seed(maker):
        async with maker() as session:
            repository = ResearchRepository(session)
            question = await repository.create_question(
                ResearchQuestion(title="Seeded RQ", description="Conclusion rejection")
            )
            await repository.create_evidence(literature_evidence(question.id))
            await session.commit()
            return question.id

    with api_client(path) as (client, maker):
        question_id = asyncio.run(seed(maker))
        evidence_id = client.get(f"/research/questions/{question_id}/state").json()["evidence"][0]["id"]
        created = client.post(
            f"/research/questions/{question_id}/conclusions",
            json={
                "statement": "A claim that will not be accepted.",
                "confidence": "LOW",
                "supporting_evidence_ids": [evidence_id],
            },
        ).json()
        assert created["conclusion"]["status"] == "PENDING_APPROVAL"

        response = client.post(
            f"/approvals/{created['approval']['id']}/review",
            json={"reviewer": "pi", "decision": "REJECTED", "reason": "证据不支撑该表述"},
        )
        assert response.status_code == 200, response.text
        assert response.json()["approval"]["status"] == "REJECTED"
        assert response.json()["applied_entity"]["status"] == "REJECTED"


def test_rejecting_a_hypothesis_registration_does_not_advance_it(tmp_path):
    """注册被拒就还是 PROPOSED —— 拒绝不能有任何"顺手推进一点"的副作用。"""
    with api_client(tmp_path / "reject-registration.db") as (client, _maker):
        question_id = client.post(
            "/research/questions",
            json={"title": "RQ", "description": "Registration rejection"},
        ).json()["id"]
        created = client.post(
            f"/research/questions/{question_id}/hypotheses",
            json={"statement": "H-nope", "prediction": "Not testable as written"},
        ).json()
        response = client.post(
            f"/approvals/{created['registration_approval']['id']}/review",
            json={"reviewer": "pi", "decision": "REJECTED"},
        )
        assert response.status_code == 200, response.text
        assert response.json()["applied_entity"] is None
        assert client.get(f"/hypotheses/{created['hypothesis']['id']}").json()["status"] == "PROPOSED"


def test_replayed_rejection_is_refused(tmp_path):
    """同一个审批不能二次决定，否则第二次的措辞会盖掉第一次的审查记录。"""
    with api_client(tmp_path / "double-reject.db") as (client, _maker):
        question_id, hypothesis_id = _create_question_and_hypothesis(client)
        created = client.post(
            "/experiments",
            json={
                "research_question_id": question_id,
                "tested_hypotheses": [hypothesis_id],
                "purpose": "Test H1",
                "independent_variable": "treatment",
                "dependent_variables": ["mAP"],
                "control": "off",
                "treatment": "on",
                "metrics": ["mAP"],
                "success_criteria": {"delta": 0.5},
            },
        ).json()
        approval_id = created["approval"]["id"]
        assert client.post(
            f"/approvals/{approval_id}/review",
            json={"reviewer": "pi", "decision": "REJECTED"},
        ).status_code == 200
        second = client.post(
            f"/approvals/{approval_id}/review",
            json={"reviewer": "other", "decision": "APPROVED"},
        )
        assert second.status_code == 422, second.text
        assert "already been decided" in second.text
