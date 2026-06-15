"""
Spec: Telemetry — RequestRecord and Telemetry aggregate

Purpose: Every proxy request records both user-message and tool-message
         compression stats. GET /metrics exposes aggregate token savings
         across all message types.

Invariants:
  - tool_original_tokens and tool_compressed_tokens default to 0
  - summary() includes tool token aggregates
  - combined saving ratio covers user + tool tokens
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from shiori.metrics.telemetry import RequestRecord, Telemetry
from shiori.config import ShioriConfig
from shiori.api.server import _build_app

_LARGE_JSON_TOOL_OUTPUT = json.dumps([
    {
        "id": i,
        "value": f"item {i}",
        "category": "product",
        "active": True,
        "metadata": None,
        "tags": [],
    }
    for i in range(20)
], indent=2)

_FAKE_RESPONSE = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
}


# ---------------------------------------------------------------------------
# Unit tests — RequestRecord
# ---------------------------------------------------------------------------

def test_request_record_has_tool_original_tokens():
    r = RequestRecord(
        request_id="x", model="m", provider="p", strategy="lossless",
        routing_via="rules", routing_reason="test",
        original_tokens=100, compressed_tokens=80, dictionary_tokens=0,
        token_saving_ratio=0.2, compression_latency_ms=1.0, provider_latency_ms=10.0,
    )
    assert r.tool_original_tokens == 0


def test_request_record_has_tool_compressed_tokens():
    r = RequestRecord(
        request_id="x", model="m", provider="p", strategy="lossless",
        routing_via="rules", routing_reason="test",
        original_tokens=100, compressed_tokens=80, dictionary_tokens=0,
        token_saving_ratio=0.2, compression_latency_ms=1.0, provider_latency_ms=10.0,
    )
    assert r.tool_compressed_tokens == 0


def test_request_record_tool_tokens_set_explicitly():
    r = RequestRecord(
        request_id="x", model="m", provider="p", strategy="lossless",
        routing_via="rules", routing_reason="test",
        original_tokens=100, compressed_tokens=80, dictionary_tokens=0,
        token_saving_ratio=0.2, compression_latency_ms=1.0, provider_latency_ms=10.0,
        tool_original_tokens=500, tool_compressed_tokens=200,
    )
    assert r.tool_original_tokens == 500
    assert r.tool_compressed_tokens == 200


def test_request_record_to_dict_includes_tool_tokens():
    r = RequestRecord(
        request_id="x", model="m", provider="p", strategy="lossless",
        routing_via="rules", routing_reason="test",
        original_tokens=100, compressed_tokens=80, dictionary_tokens=0,
        token_saving_ratio=0.2, compression_latency_ms=1.0, provider_latency_ms=10.0,
        tool_original_tokens=300, tool_compressed_tokens=150,
    )
    d = r.to_dict()
    assert "tool_original_tokens" in d
    assert d["tool_original_tokens"] == 300
    assert "tool_compressed_tokens" in d
    assert d["tool_compressed_tokens"] == 150


# ---------------------------------------------------------------------------
# Unit tests — Telemetry.summary()
# ---------------------------------------------------------------------------

def _make_record(**kwargs) -> RequestRecord:
    defaults = dict(
        request_id="x", model="m", provider="p", strategy="lossless",
        routing_via="rules", routing_reason="test",
        original_tokens=100, compressed_tokens=80, dictionary_tokens=0,
        token_saving_ratio=0.2, compression_latency_ms=1.0, provider_latency_ms=10.0,
    )
    defaults.update(kwargs)
    return RequestRecord(**defaults)


def test_summary_includes_tool_original_tokens():
    t = Telemetry()
    t.record(_make_record(tool_original_tokens=400, tool_compressed_tokens=200))
    t.record(_make_record(tool_original_tokens=600, tool_compressed_tokens=300))
    s = t.summary()
    assert s["tool_original_tokens"] == 1000


def test_summary_includes_tool_compressed_tokens():
    t = Telemetry()
    t.record(_make_record(tool_original_tokens=400, tool_compressed_tokens=200))
    s = t.summary()
    assert s["tool_compressed_tokens"] == 200


def test_summary_tool_tokens_zero_when_no_tool_messages():
    t = Telemetry()
    t.record(_make_record())
    s = t.summary()
    assert s["tool_original_tokens"] == 0
    assert s["tool_compressed_tokens"] == 0


def test_summary_empty_returns_zero_tool_tokens():
    t = Telemetry()
    s = t.summary()
    assert s.get("tool_original_tokens", 0) == 0


# ---------------------------------------------------------------------------
# Integration — /metrics endpoint reflects tool compression
# ---------------------------------------------------------------------------

@pytest.fixture
def client_with_tool_request():
    cfg = ShioriConfig.defaults()
    cfg.mode = "safe"
    app = _build_app(cfg)
    with patch("shiori.providers.openai.OpenAIProvider.chat_completions", new_callable=AsyncMock) as mock:
        mock.return_value = _FAKE_RESPONSE
        with TestClient(app) as c:
            payload = {
                "model": "gpt-4o",
                "messages": [
                    {"role": "user", "content": "Analyse these results."},
                    {"role": "tool", "tool_call_id": "call_001", "content": _LARGE_JSON_TOOL_OUTPUT},
                ],
            }
            c.post("/v1/chat/completions", json=payload)
            yield c


def test_metrics_has_tool_original_tokens(client_with_tool_request):
    resp = client_with_tool_request.get("/metrics")
    assert resp.json()["tool_original_tokens"] > 0


def test_metrics_tool_compressed_less_than_original(client_with_tool_request):
    resp = client_with_tool_request.get("/metrics")
    data = resp.json()
    assert data["tool_compressed_tokens"] < data["tool_original_tokens"]
