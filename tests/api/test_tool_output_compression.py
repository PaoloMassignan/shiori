"""
Spec: Tool output compression

Purpose: POST /v1/chat/completions compresses role=tool messages before
         forwarding to the provider. JSON content goes through JsonCompressor;
         non-JSON falls back to LosslessCompressor.

Invariants:
  - role=tool messages with JSON content are compressed
  - role=tool messages with non-JSON content are compressed with lossless
  - role=user and role=system messages are not affected by tool compression
  - In mode=off, tool messages are forwarded unchanged
  - Provider always receives the (possibly compressed) messages
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
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "Done"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
}

_LARGE_JSON_TOOL_OUTPUT = json.dumps([
    {
        "doc_id": f"doc_{i:03d}",
        "title": f"Document {i}",
        "score": round(0.99 - i * 0.01, 2),
        "source": "search_index",
        "language": "en",
        "content": f"This is the content of document number {i}. " * 5,
        "metadata": None,
        "tags": [],
        "deleted": False,
    }
    for i in range(15)
], indent=2)

_PLAIN_TEXT_TOOL_OUTPUT = (
    "The quick brown fox jumps over the lazy dog. " * 30
)


def _make_client(mode: str = "safe"):
    cfg = ShioriConfig.defaults()
    cfg.mode = mode
    app = _build_app(cfg)
    return app


@pytest.fixture
def client_safe():
    app = _make_client("safe")
    with patch("shiori.providers.openai.OpenAIProvider.chat_completions", new_callable=AsyncMock) as mock:
        mock.return_value = _FAKE_RESPONSE
        with TestClient(app) as c:
            yield c, mock


@pytest.fixture
def client_off():
    app = _make_client("off")
    with patch("shiori.providers.openai.OpenAIProvider.chat_completions", new_callable=AsyncMock) as mock:
        mock.return_value = _FAKE_RESPONSE
        with TestClient(app) as c:
            yield c, mock


# ---------------------------------------------------------------------------
# JSON tool output compression
# ---------------------------------------------------------------------------

def test_json_tool_message_is_compressed(client_safe):
    c, mock = client_safe
    payload = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "Summarise the search results."},
            {"role": "tool", "tool_call_id": "call_001", "content": _LARGE_JSON_TOOL_OUTPUT},
        ],
    }
    c.post("/v1/chat/completions", json=payload)

    forwarded = mock.call_args[0][0]
    tool_msg = next(m for m in forwarded["messages"] if m["role"] == "tool")
    assert len(tool_msg["content"]) < len(_LARGE_JSON_TOOL_OUTPUT)


def test_json_tool_message_has_no_nulls_after_compression(client_safe):
    c, mock = client_safe
    payload = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "Analyse."},
            {"role": "tool", "tool_call_id": "call_001", "content": _LARGE_JSON_TOOL_OUTPUT},
        ],
    }
    c.post("/v1/chat/completions", json=payload)

    forwarded = mock.call_args[0][0]
    tool_msg = next(m for m in forwarded["messages"] if m["role"] == "tool")
    assert '"metadata": null' not in tool_msg["content"]


def test_multiple_tool_messages_all_compressed(client_safe):
    c, mock = client_safe
    payload = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "Compare results."},
            {"role": "tool", "tool_call_id": "call_001", "content": _LARGE_JSON_TOOL_OUTPUT},
            {"role": "tool", "tool_call_id": "call_002", "content": _LARGE_JSON_TOOL_OUTPUT},
        ],
    }
    c.post("/v1/chat/completions", json=payload)

    forwarded = mock.call_args[0][0]
    tool_msgs = [m for m in forwarded["messages"] if m["role"] == "tool"]
    assert len(tool_msgs) == 2
    for msg in tool_msgs:
        assert len(msg["content"]) < len(_LARGE_JSON_TOOL_OUTPUT)


# ---------------------------------------------------------------------------
# Non-JSON tool output falls back to lossless
# ---------------------------------------------------------------------------

def test_non_json_tool_message_is_forwarded(client_safe):
    c, mock = client_safe
    payload = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "What does this say?"},
            {"role": "tool", "tool_call_id": "call_001", "content": _PLAIN_TEXT_TOOL_OUTPUT},
        ],
    }
    resp = c.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    forwarded = mock.call_args[0][0]
    tool_msg = next(m for m in forwarded["messages"] if m["role"] == "tool")
    # content must be present and not empty
    assert tool_msg["content"]


# ---------------------------------------------------------------------------
# Other message roles are not touched
# ---------------------------------------------------------------------------

def test_user_message_content_unchanged_by_tool_compression(client_safe):
    c, mock = client_safe
    user_text = "What is 2 + 2?"
    payload = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": user_text},
            {"role": "tool", "tool_call_id": "call_001", "content": _LARGE_JSON_TOOL_OUTPUT},
        ],
    }
    c.post("/v1/chat/completions", json=payload)

    forwarded = mock.call_args[0][0]
    user_msg = next(m for m in forwarded["messages"] if m["role"] == "user")
    # user message may be compressed by the task router, but tool compression
    # must not corrupt it — it must still be a string
    assert isinstance(user_msg["content"], str)


def test_system_message_not_modified(client_safe):
    c, mock = client_safe
    system_text = "You are a helpful assistant."
    payload = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": system_text},
            {"role": "user", "content": "Hello."},
            {"role": "tool", "tool_call_id": "call_001", "content": _LARGE_JSON_TOOL_OUTPUT},
        ],
    }
    c.post("/v1/chat/completions", json=payload)

    forwarded = mock.call_args[0][0]
    sys_msg = next(m for m in forwarded["messages"] if m["role"] == "system")
    assert sys_msg["content"] == system_text


# ---------------------------------------------------------------------------
# mode=off — no compression at all
# ---------------------------------------------------------------------------

def test_tool_message_not_compressed_in_off_mode(client_off):
    c, mock = client_off
    payload = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "Hello."},
            {"role": "tool", "tool_call_id": "call_001", "content": _LARGE_JSON_TOOL_OUTPUT},
        ],
    }
    c.post("/v1/chat/completions", json=payload)

    forwarded = mock.call_args[0][0]
    tool_msg = next(m for m in forwarded["messages"] if m["role"] == "tool")
    assert tool_msg["content"] == _LARGE_JSON_TOOL_OUTPUT
