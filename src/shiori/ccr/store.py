"""
CcrStore — in-process store for original content before lossy compression.

Purpose:
    Preserve originals so the LLM can retrieve them via the shiori_retrieve tool
    if the compressed version omits a needed detail.

Inputs / Outputs:
    put(content) → content_id (UUID string)
    get(content_id) → original content string, or None if absent/expired

Invariants:
    - TTL is enforced lazily on get(); no background cleanup task
    - put() is O(1); get() is O(1)
    - Thread safety: not required — FastAPI async handlers run on a single event loop
    - content_id is a UUID4 string; collisions are astronomically unlikely
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass


@dataclass(slots=True)
class _Entry:
    content: str
    expires_at: float


class CcrStore:
    def __init__(self, ttl_seconds: int = 600) -> None:
        self._ttl = ttl_seconds
        self._store: dict[str, _Entry] = {}

    def put(self, content: str) -> str:
        """Store content and return a unique content_id."""
        content_id = str(uuid.uuid4())
        self._store[content_id] = _Entry(
            content=content,
            expires_at=time.monotonic() + self._ttl,
        )
        return content_id

    def get(self, content_id: str) -> str | None:
        """Return stored content, or None if absent or TTL-expired."""
        entry = self._store.get(content_id)
        if entry is None:
            return None
        if time.monotonic() > entry.expires_at:
            del self._store[content_id]
            return None
        return entry.content
