import asyncio

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

import db.models  # noqa: F401
from api.research_routes import research_router
from db.database import get_async_session


def test_research_api_core_mutations_use_service_rules(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'api.db'}")
    maker = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    async def create_schema():
        async with engine.begin() as connection:
            await connection.run_sync(SQLModel.metadata.create_all)

    asyncio.run(create_schema())

    async def session_override():
        async with maker() as session:
            yield session

    app = FastAPI()
    app.include_router(research_router)
    app.dependency_overrides[get_async_session] = session_override
    with TestClient(app) as client:
        schema = client.get("/research/system/schema")
        assert schema.status_code == 200, schema.text
        assert schema.json()["compatible"] is True
        assert schema.json()["expected_table_count"] == 18

        response = client.post(
            "/research/questions",
            json={"title": "RQ", "description": "A precise question"},
        )
        assert response.status_code == 201, response.text
        question_id = response.json()["id"]
        response = client.post(f"/research/questions/{question_id}/workflow/initialize")
        assert response.status_code == 200, response.text
        assert client.get(f"/research/questions/{question_id}").json()["status"] == "ACTIVE"
        listing = client.get("/research/questions")
        assert listing.status_code == 200
        assert listing.json()["items"][0]["id"] == question_id

        response = client.post(
            f"/research/questions/{question_id}/hypotheses",
            json={"statement": "H1", "prediction": "A measurable prediction"},
        )
        assert response.status_code == 201, response.text
        data = response.json()
        hypothesis_id = data["hypothesis"]["id"]
        approval_id = data["registration_approval"]["id"]
        assert data["hypothesis"]["status"] == "PROPOSED"

        response = client.post(
            f"/approvals/{approval_id}/review",
            json={"reviewer": "pi", "decision": "APPROVED"},
        )
        assert response.status_code == 200, response.text
        assert response.json()["applied_entity"]["status"] == "TESTABLE"

        response = client.post(
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
        )
        assert response.status_code == 201, response.text
        experiment_data = response.json()
        experiment_id = experiment_data["experiment"]["id"]
        experiment_approval_id = experiment_data["approval"]["id"]
        assert experiment_data["experiment"]["status"] == "PLANNED"
        response = client.post(
            f"/approvals/{experiment_approval_id}/review",
            json={"reviewer": "pi", "decision": "APPROVED"},
        )
        assert response.status_code == 200, response.text
        assert response.json()["applied_entity"]["status"] == "APPROVED"
        hypothesis_response = client.get(f"/hypotheses/{hypothesis_id}")
        assert hypothesis_response.status_code == 200
        assert hypothesis_response.json()["status"] == "UNDER_TEST"

        state = client.get(f"/research/questions/{question_id}/state")
        assert state.status_code == 200, state.text
        assert state.json()["question"]["id"] == question_id
        assert state.json()["hypotheses"][0]["id"] == hypothesis_id
        assert state.json()["experiments"][0]["id"] == experiment_id
        assert state.json()["experiment_hypotheses"] == [
            {"experiment_id": experiment_id, "hypothesis_id": hypothesis_id}
        ]

        evaluation = client.get(f"/research/questions/{question_id}/evaluation")
        assert evaluation.status_code == 200
        assert evaluation.json()["research_question_id"] == question_id
        assert evaluation.json()["state_consistency"]["invalid_count"] == 0

        archive = client.get(f"/research/questions/{question_id}/export")
        assert archive.status_code == 200, archive.text
        assert archive.json()["research_question_id"] == question_id
        assert len(archive.json()["archive_sha256"]) == 64

    asyncio.run(engine.dispose())
