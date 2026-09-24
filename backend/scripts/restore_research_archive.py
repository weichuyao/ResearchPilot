"""Restore a verified archive into the configured database without overwriting.

The explicit question-id confirmation prevents restoring the wrong file by typo.

Usage:
    python scripts/restore_research_archive.py archive.json \
        --confirm-question-id RQ-...
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path


BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BACKEND_ROOT, "app"))

from db.database import async_engine, async_session_maker  # noqa: E402
from db.repository.research_repository import ResearchRepository  # noqa: E402
from research.archive import ResearchArchiveService  # noqa: E402


async def restore(path: Path, confirmed_id: str) -> dict:
    try:
        archive = json.loads(path.read_text(encoding="utf-8"))
        question_id = archive.get("research_question_id")
        if question_id != confirmed_id:
            raise ValueError(
                f"confirmation mismatch: archive contains {question_id!r}, got {confirmed_id!r}"
            )
        async with async_session_maker() as session:
            try:
                result = await ResearchArchiveService(ResearchRepository(session)).restore(archive)
                await session.commit()
                return result
            except Exception:
                await session.rollback()
                raise
    finally:
        # Dispose on the same event loop that used asyncpg connections.
        await async_engine.dispose()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--confirm-question-id", required=True)
    args = parser.parse_args()
    try:
        result = asyncio.run(restore(args.archive, args.confirm_question_id))
        print(json.dumps({"restored": True, **result}, indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({"restored": False, "error": str(exc)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
