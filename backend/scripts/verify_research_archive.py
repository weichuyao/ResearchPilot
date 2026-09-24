"""Verify a Scientific Research Harness archive without changing any state.

Usage:
    .venv-py311/Scripts/python.exe scripts/verify_research_archive.py archive.json
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BACKEND_ROOT, "app"))

from research.archive import ResearchArchiveService  # noqa: E402


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: verify_research_archive.py <archive.json>", file=sys.stderr)
        return 2
    path = Path(sys.argv[1])
    try:
        archive = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"INVALID: cannot read archive: {exc}", file=sys.stderr)
        return 1
    if archive.get("schema_version") != ResearchArchiveService.SCHEMA_VERSION:
        print(f"INVALID: unsupported schema_version={archive.get('schema_version')!r}", file=sys.stderr)
        return 1
    if not ResearchArchiveService.verify(archive):
        print("INVALID: archive SHA-256 does not match its contents", file=sys.stderr)
        return 1
    counts = {name: len(rows) for name, rows in archive.get("records", {}).items()}
    print(json.dumps({
        "valid": True,
        "research_question_id": archive.get("research_question_id"),
        "state_fingerprint": archive.get("state_fingerprint"),
        "archive_sha256": archive.get("archive_sha256"),
        "record_counts": counts,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
