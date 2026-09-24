import pytest

from scripts import research_harness_smoke as smoke


def test_read_only_smoke_checks_route_contract(monkeypatch):
    def fake_fetch(_base, path):
        if path == "/health":
            return {"app": "ResearchPilot", "index": {"papers": 2, "chunks": 10}}
        if path == "/openapi.json":
            return {"paths": {item: {} for item in smoke.REQUIRED_ROUTES}}
        if path == "/research/questions":
            return {"total": 0, "items": []}
        if path == "/research/system/schema":
            return {"compatible": True, "dialect": "postgresql", "existing_table_count": 18}
        raise AssertionError(f"unexpected path: {path}")

    monkeypatch.setattr(smoke, "fetch_json", fake_fetch)
    result = smoke.run("http://example.test")
    assert result["ok"] is True
    assert result["required_routes"] == len(smoke.REQUIRED_ROUTES)
    assert result["research_projects"] == 0
    assert result["schema"] == {"dialect": "postgresql", "tables": 18, "compatible": True}


def test_read_only_smoke_fails_when_route_is_missing(monkeypatch):
    monkeypatch.setattr(smoke, "fetch_json", lambda _base, path: (
        {"paths": {}} if path == "/openapi.json" else
        {"app": "ResearchPilot", "index": {}}
    ))
    with pytest.raises(RuntimeError, match="missing Harness routes"):
        smoke.run("http://example.test")
