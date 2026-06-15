"""
CCR retrieval endpoint.

GET /v1/shiori/retrieve/{content_id}

Returns the original content stored before lossy compression, or 404 if the
entry is absent or TTL-expired.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from shiori.ccr.store import CcrStore


def build_ccr_router(ccr_store: CcrStore) -> APIRouter:
    router = APIRouter()

    @router.get("/v1/shiori/retrieve/{content_id}")
    async def retrieve(content_id: str):
        content = ccr_store.get(content_id)
        if content is None:
            raise HTTPException(status_code=404, detail="Content not found or expired")
        return {"content_id": content_id, "content": content}

    return router
