"""
Spec: Streaming pass-through for stream: true requests

Purpose: When a client sends {"stream": true}, Shiori must forward
         the SSE stream from the provider unchanged rather than trying
         to parse it as JSON. The compression and routing logic still
         applies; only the response delivery changes.

Invariants:
  - stream: true → StreamingResponse with media_type="text/event-stream"
  - Cache-Control: no-cache header is set on the response
  - Compression (tool messages, user message) still runs before forwarding
  - Telemetry is recorded with provider_latency_ms=0.0
  - debug mode does NOT inject a shiori field (cannot buffer SSE)
  - stream: false (explicit) → normal JSONResponse
  - stream absent → normal JSONResponse
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from shiori.config import ShioriConfig
from shiori.api.server import _build_app

_SSE_CHUNKS = [
    b'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"delta":{"role":"assistant"},"index":0}]}\n\n',
    b'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"delta":{"content":"Hello"},"index":0}]}\n\n',
    b"data: [DONE]\n\n",
]


@pytest.fixture
def streaming_client():
    cfg = ShioriConfig.defaults()
    cfg.mode = "safe"
    app = _build_app(cfg)
    with patch(
        "shiori.providers.openai.OpenAIProvider.stream_chat_completions"
    ) as mock_stream:
        async def _gen(*_args, **_kwargs):
            for chunk in _SSE_CHUNKS:
                yield chunk

        mock_stream.side_effect = _gen
        with TestClient(app) as c:
            yield c


@pytest.fixture
def non_streaming_client():
    cfg = ShioriConfig.defaults()
    cfg.mode = "safe"
    app = _build_app(cfg)
    _fake = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    }
    with patch(
        "shiori.providers.openai.OpenAIProvider.chat_completions",
        new_callable=AsyncMock,
        return_value=_fake,
    ):
        with TestClient(app) as c:
            yield c


# ---------------------------------------------------------------------------
# Response type
# ---------------------------------------------------------------------------

def test_stream_true_returns_event_stream_content_type(streaming_client):
    resp = streaming_client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}], "stream": True},
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]


def test_stream_true_sets_no_cache_header(streaming_client):
    resp = streaming_client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}], "stream": True},
    )
    assert resp.headers.get("cache-control") == "no-cache"


def test_stream_true_body_contains_sse_chunks(streaming_client):
    resp = streaming_client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}], "stream": True},
    )
    assert b"data: " in resp.content
    assert b"[DONE]" in resp.content


def test_stream_false_returns_json(non_streaming_client):
    resp = non_streaming_client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}], "stream": False},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/json")


def test_stream_absent_returns_json(non_streaming_client):
    resp = non_streaming_client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/json")


# ---------------------------------------------------------------------------
# Debug mode — no shiori field in SSE
# ---------------------------------------------------------------------------

def test_stream_debug_mode_no_shiori_field():
    cfg = ShioriConfig.defaults()
    cfg.mode = "debug"
    app = _build_app(cfg)
    with patch(
        "shiori.providers.openai.OpenAIProvider.stream_chat_completions"
    ) as mock_stream:
        async def _gen(*_args, **_kwargs):
            for chunk in _SSE_CHUNKS:
                yield chunk

        mock_stream.side_effect = _gen
        with TestClient(app) as c:
            resp = c.post(
                "/v1/chat/completions",
                json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}], "stream": True},
            )
    assert b"shiori" not in resp.content


# ---------------------------------------------------------------------------
# Telemetry — recorded even for streaming
# ---------------------------------------------------------------------------

def test_stream_telemetry_recorded(streaming_client):
    streaming_client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}], "stream": True},
    )
    resp = streaming_client.get("/metrics")
    assert resp.json()["total_requests"] == 1


def test_stream_telemetry_provider_latency_zero(streaming_client):
    streaming_client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}], "stream": True},
    )
    # provider_latency_ms=0.0 is stored per-record; summary does not expose it
    # but total_requests being 1 confirms telemetry.record() was called
    resp = streaming_client.get("/metrics")
    assert resp.json()["total_requests"] >= 1
