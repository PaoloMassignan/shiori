"""
Spec: OpenAI-compatible proxy endpoint

Purpose: POST /v1/chat/completions receives an OpenAI-style request,
         compresses the last user message, and returns an OpenAI-style response.

Test strategy: mock the provider so no real API key is needed.

Invariants:
  - Response is OpenAI-compatible (has "choices", "usage" fields)
  - Last user message content is modified when compression fires
  - System message receives the dictionary when dictionary is non-empty
  - GET /health returns 200
  - GET /metrics returns aggregate counters
  - No real network call is ever made (provider is mocked)
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from shiori.config import ShioriConfig
from shiori.api.server import _build_app

_FAKE_RESPONSE = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "Paris"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}


@pytest.fixture
def client():
    cfg = ShioriConfig.defaults()
    cfg.mode = "safe"
    app = _build_app(cfg)

    with patch("shiori.providers.openai.OpenAIProvider.chat_completions", new_callable=AsyncMock) as mock_provider:
        mock_provider.return_value = _FAKE_RESPONSE
        with TestClient(app) as c:
            yield c, mock_provider


def test_health_endpoint(client):
    c, _ = client
    resp = c.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_metrics_endpoint(client):
    c, _ = client
    resp = c.get("/metrics")
    assert resp.status_code == 200
    assert "total_requests" in resp.json()


def test_chat_completions_returns_openai_response(client):
    c, mock = client
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
    }
    resp = c.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "choices" in data
    assert data["choices"][0]["message"]["content"] == "Paris"


def test_provider_is_called_once(client):
    c, mock = client
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "Hello."}],
    }
    c.post("/v1/chat/completions", json=payload)
    mock.assert_called_once()


def test_metrics_updated_after_request(client):
    c, _ = client
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "Hello world."}],
    }
    c.post("/v1/chat/completions", json=payload)
    metrics = c.get("/metrics").json()
    assert metrics["total_requests"] == 1
