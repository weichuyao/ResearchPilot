"""HTTP 层的契约测试。

原来这里是一条 100 行的巨型测试串完所有路由。它能证明"链路通"，但一旦断在中间，
报错位置说明不了是哪一层坏的 —— 而这条链上每一段的失败形状几乎一样（分层是本项目
反复踩过的同一课）。所以按契约拆开，每个测试只守一条约定。
"""

from __future__ import annotations

from tests.research.helpers import api_client


EXPERIMENT_BODY = {
    "purpose": "Test H1",
    "independent_variable": "treatment",
    "dependent_variables": ["mAP"],
    "control": "off",
    "treatment": "on",
    "metrics": ["mAP"],
    "success_criteria": {"delta": 0.5},
}


def _question(client, title: str = "RQ") -> str:
    response = client.post(
        "/research/questions", json={"title": title, "description": "A precise question"}
    )
    assert response.status_code == 201, response.text
    return response.json()["id"]


def _registered_hypothesis(client, question_id: str) -> str:
    created = client.post(
        f"/research/questions/{question_id}/hypotheses",
        json={"statement": "H1", "prediction": "A measurable prediction"},
    ).json()
    client.post(
        f"/approvals/{created['registration_approval']['id']}/review",
        json={"reviewer": "pi", "decision": "APPROVED"},
    )
    return created["hypothesis"]["id"]


def _planned_experiment(client, question_id: str, hypothesis_id: str) -> dict:
    body = dict(EXPERIMENT_BODY, research_question_id=question_id, tested_hypotheses=[hypothesis_id])
    response = client.post("/experiments", json=body)
    assert response.status_code == 201, response.text
    return response.json()


def test_schema_endpoint_reports_the_expected_table_count(tmp_path):
    with api_client(tmp_path / "api-schema.db") as (client, _maker):
        response = client.get("/research/system/schema")
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["compatible"] is True
        assert body["expected_table_count"] == 18


def test_initializing_workflow_activates_the_question(tmp_path):
    """OPEN → ACTIVE 是 initialize 的副作用，不是客户端可以自己声明的。"""
    with api_client(tmp_path / "api-init.db") as (client, _maker):
        question_id = _question(client)
        assert client.get(f"/research/questions/{question_id}").json()["status"] == "OPEN"
        assert client.post(f"/research/questions/{question_id}/workflow/initialize").status_code == 200
        assert client.get(f"/research/questions/{question_id}").json()["status"] == "ACTIVE"
        assert client.get("/research/questions").json()["items"][0]["id"] == question_id


def test_registering_a_hypothesis_needs_an_approval_before_it_becomes_testable(tmp_path):
    with api_client(tmp_path / "api-register.db") as (client, _maker):
        question_id = _question(client)
        created = client.post(
            f"/research/questions/{question_id}/hypotheses",
            json={"statement": "H1", "prediction": "A measurable prediction"},
        )
        assert created.status_code == 201, created.text
        body = created.json()
        assert body["hypothesis"]["status"] == "PROPOSED"

        review = client.post(
            f"/approvals/{body['registration_approval']['id']}/review",
            json={"reviewer": "pi", "decision": "APPROVED"},
        )
        assert review.status_code == 200, review.text
        assert review.json()["applied_entity"]["status"] == "TESTABLE"


def test_approving_an_experiment_moves_the_hypothesis_it_tests(tmp_path):
    with api_client(tmp_path / "api-exp.db") as (client, _maker):
        question_id = _question(client)
        hypothesis_id = _registered_hypothesis(client, question_id)
        created = _planned_experiment(client, question_id, hypothesis_id)
        assert created["experiment"]["status"] == "PLANNED"

        review = client.post(
            f"/experiments/{created['experiment']['id']}/approve",
            json={"reviewer": "pi", "decision": "APPROVED"},
        )
        assert review.status_code == 200, review.text
        assert review.json()["experiment"]["status"] == "APPROVED"
        assert client.get(f"/hypotheses/{hypothesis_id}").json()["status"] == "UNDER_TEST"


def test_state_projection_aggregates_without_a_second_store(tmp_path):
    """UI 只读这一份投影：它必须能从同一批表里聚出来，不引入第二套状态。"""
    with api_client(tmp_path / "api-state.db") as (client, _maker):
        question_id = _question(client)
        hypothesis_id = _registered_hypothesis(client, question_id)
        experiment_id = _planned_experiment(client, question_id, hypothesis_id)["experiment"]["id"]

        state = client.get(f"/research/questions/{question_id}/state")
        assert state.status_code == 200, state.text
        body = state.json()
        assert body["question"]["id"] == question_id
        assert body["hypotheses"][0]["id"] == hypothesis_id
        assert body["experiments"][0]["id"] == experiment_id
        assert body["experiment_hypotheses"] == [
            {"experiment_id": experiment_id, "hypothesis_id": hypothesis_id}
        ]


def test_evaluation_counts_invalid_states_from_the_stored_rows(tmp_path):
    with api_client(tmp_path / "api-eval.db") as (client, _maker):
        question_id = _question(client)
        _registered_hypothesis(client, question_id)
        evaluation = client.get(f"/research/questions/{question_id}/evaluation")
        assert evaluation.status_code == 200
        body = evaluation.json()
        assert body["research_question_id"] == question_id
        assert body["state_consistency"]["invalid_count"] == 0
        assert len(body["state_fingerprint"]) == 64


def test_export_seals_the_archive_without_touching_state(tmp_path):
    """导出可反复核验，但**整档摘要按设计每次不同**。

    `archive_sha256` 覆盖 `exported_at`，所以它是"某一次导出的快照"摘要，不是内容摘要。
    跨次比较请比 `state_fingerprint`（只覆盖科研状态，不含导出时间），
    否则会写出一个必然失败的断言 —— 这里就是先踩过一次才写下来的。
    """
    from research.archive import ResearchArchiveService

    with api_client(tmp_path / "api-export.db") as (client, _maker):
        question_id = _question(client)
        first = client.get(f"/research/questions/{question_id}/export").json()
        second = client.get(f"/research/questions/{question_id}/export").json()
        assert len(first["archive_sha256"]) == 64
        assert ResearchArchiveService.verify(first)
        assert ResearchArchiveService.verify(second)
        assert first["state_fingerprint"] == second["state_fingerprint"]
        # 状态没动，就不该多出任何记录
        assert {k: len(v) for k, v in first["records"].items()} == \
               {k: len(v) for k, v in second["records"].items()}
