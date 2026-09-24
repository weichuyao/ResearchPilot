"""Stable identifiers for source fragments used in provenance links."""

from __future__ import annotations

import hashlib

from ai.rag.textnorm import content_key


def stable_chunk_id(
    *,
    source: str,
    locator_prefix: str,
    locator_label: str,
    content: str,
    chunk_index: int | None = None,
) -> str:
    """Build an ID from source location and normalized content.

    `chunk_index` is present on newly ingested data.  Legacy vector entries do
    not have it, so the same function provides a deterministic content-based
    fallback without requiring an immediate corpus rebuild.
    """
    fields = [source, locator_prefix, locator_label]
    if chunk_index is not None:
        fields.append(str(chunk_index))
    fields.append(content_key(content))
    digest = hashlib.sha256("\x1f".join(fields).encode("utf-8")).hexdigest()[:32]
    return f"CH-{digest}"


def chunk_id_from_document(document) -> str:
    metadata = document.metadata or {}
    existing = metadata.get("chunk_id")
    if existing:
        return str(existing)
    raw_index = metadata.get("chunk_index")
    try:
        chunk_index = int(raw_index) if raw_index is not None else None
    except (TypeError, ValueError):
        chunk_index = None
    return stable_chunk_id(
        source=str(metadata.get("source") or "unknown"),
        locator_prefix=str(metadata.get("locator_prefix") or "p"),
        locator_label=str(metadata.get("page_label") or metadata.get("page") or "?"),
        content=document.page_content,
        chunk_index=chunk_index,
    )

