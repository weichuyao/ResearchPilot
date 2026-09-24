"""Read-only deployment smoke test for Scientific Research Harness.

Usage:
    python scripts/research_harness_smoke.py --base-url http://127.0.0.1:8002
"""

from __future__ import annotations

import argparse
import hashlib
import json
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


REQUIRED_ROUTES = {
    "/research/questions",
    "/research/questions/{question_id}/state",
    "/research/questions/{question_id}/evaluation",
    "/research/questions/{question_id}/export",
    "/research/questions/{question_id}/report",
    "/research/system/schema",
    "/hypotheses/{hypothesis_id}/evidence/search",
    "/experiments/{experiment_id}/runs/import",
    "/research/entities/{entity_id}/provenance",
}


def fetch_json(base_url: str, path: str) -> dict:
    with urlopen(base_url.rstrip("/") + path, timeout=20) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def archive_digest(archive: dict) -> str:
    payload = dict(archive)
    payload.pop("archive_sha256", None)
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def run(base_url: str) -> dict:
    health = fetch_json(base_url, "/health")
    openapi = fetch_json(base_url, "/openapi.json")
    missing = sorted(REQUIRED_ROUTES - set(openapi.get("paths", {})))
    if missing:
        raise RuntimeError(f"missing Harness routes: {missing}")
    listing = fetch_json(base_url, "/research/questions")
    schema = fetch_json(base_url, "/research/system/schema")
    if not schema.get("compatible"):
        raise RuntimeError(f"incompatible Harness schema: {schema}")
    result = {
        "ok": True,
        "base_url": base_url,
        "app": health.get("app"),
        "papers": health.get("index", {}).get("papers"),
        "chunks": health.get("index", {}).get("chunks"),
        "required_routes": len(REQUIRED_ROUTES),
        "schema": {
            "dialect": schema.get("dialect"),
            "tables": schema.get("existing_table_count"),
            "compatible": True,
        },
        "research_projects": listing.get("total", 0),
        "sample_project": None,
    }
    items = listing.get("items") or []
    if items:
        question_id = items[0]["id"]
        state = fetch_json(base_url, f"/research/questions/{question_id}/state")
        evaluation = fetch_json(base_url, f"/research/questions/{question_id}/evaluation")
        archive = fetch_json(base_url, f"/research/questions/{question_id}/export")
        if archive.get("archive_sha256") != archive_digest(archive):
            raise RuntimeError("sample project archive failed SHA-256 verification")
        if state.get("question", {}).get("id") != question_id:
            raise RuntimeError("state projection returned the wrong research question")
        result["sample_project"] = {
            "id": question_id,
            "workflow": (state.get("workflow") or {}).get("stage"),
            "invalid_states": evaluation.get("state_consistency", {}).get("invalid_count"),
            "archive_verified": True,
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8002")
    args = parser.parse_args()
    try:
        # ASCII-safe output survives the default Windows console code page.
        print(json.dumps(run(args.base_url), indent=2))
        return 0
    except (HTTPError, URLError, TimeoutError, RuntimeError, json.JSONDecodeError) as exc:
        print(json.dumps({"ok": False, "base_url": args.base_url, "error": str(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
