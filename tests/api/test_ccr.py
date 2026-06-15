"""
Spec: CCR — Compressed Content Retrieval

Purpose: When strategy=llmlingua fires and the payload has a tools key, Shiori
         stores the original text, prepends a [SHIORI_CCR id=...] marker to the
         compressed text, and injects the shiori_retrieve tool into the payload.
         The original is retrievable via GET /v1/shiori/retrieve/{content_id}.

Test strategy:
  - Unit tests on CcrStore (put/get, TTL, unknown ID)
  - Integration tests via FastAPI TestClient:
      - CCR fires when strategy=llmlingua + tools in payload
      - CCR does not fire when strategy=lossless
      - CCR does not fire when tools absent from payload
      - Retrieval endpoint returns original content
      - Retrieval endpoint returns 404 for unknown/expired IDs

Invariants:
  - No real API calls; provider and llmlingua compressor are mocked
  - CcrStore TTL is tested by patching time.monotonic
"""
from __future__ import annotations

import re
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from shiori.ccr.store import CcrStore
from shiori.compressors.base import CompressionResult
from shiori.config import ShioriConfig
from shiori.api.server import _build_app
from shiori.router.decision import RoutingDecision
from shiori.router.features import PromptFeatures

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

_FAKE_RESPONSE = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "Done"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
}

_ORIGINAL_TEXT = (
    "This is the full original document with every detail preserved. "
    "It contains important information that the LLM may need. " * 10
)

_COMPRESSED_TEXT = "condensed version of the document"

_FAKE_FEATURES = PromptFeatures(
    prompt_length_tokens=500,
    has_diff=False,
    has_code_fence=False,
    has_log_lines=False,
    has_stack_trace=False,
    has_passphrase_query=False,
    is_long_prose=True,
    has_summarize_instruction=True,
)

_LOSSY_DECISION = RoutingDecision(
    strategy="llmlingua",
    reason="long prose with summarization instruction",
    via="fallback",
    features=_FAKE_FEATURES,
)

_LOSSLESS_DECISION = RoutingDecision(
    strategy="lossless",
    reason="no signal for lossy compression",
    via="fallback",
    features=_FAKE_FEATURES,
)

_LOSSY_RESULT = CompressionResult(
    original_text=_ORIGINAL_TEXT,
    compressed_text=_COMPRESSED_TEXT,
    original_tokens=200,
    compressed_tokens=10,
    content_tokens=10,
)

_LOSSLESS_RESULT = CompressionResult(
    original_text=_ORIGINAL_TEXT,
    compressed_text=_ORIGINAL_TEXT,  # no change
    original_tokens=200,
    compressed_tokens=200,
    content_tokens=200,
)


def _make_client(mode: str = "aggressive"):
    cfg = ShioriConfig.defaults()
    cfg.mode = mode
    return _build_app(cfg)


# ---------------------------------------------------------------------------
# Unit tests — CcrStore
# ---------------------------------------------------------------------------

def test_store_put_returns_string_id():
    store = CcrStore()
    content_id = store.put("hello")
    assert isinstance(content_id, str)
    assert len(content_id) == 36  # UUID4 format


def test_store_get_returns_stored_content():
    store = CcrStore()
    content_id = store.put("original content")
    assert store.get(content_id) == "original content"


def test_store_get_unknown_returns_none():
    store = CcrStore()
    assert store.get("nonexistent-id") is None


def test_store_get_expired_returns_none():
    store = CcrStore(ttl_seconds=10)
    content_id = store.put("will expire")
    with patch("shiori.ccr.store.time") as mock_time:
        mock_time.monotonic.return_value = time.monotonic() + 20
        assert store.get(content_id) is None


def test_store_get_not_expired_returns_content():
    store = CcrStore(ttl_seconds=600)
    content_id = store.put("still valid")
    assert store.get(content_id) == "still valid"


def test_store_put_ids_are_unique():
    store = CcrStore()
    ids = {store.put("x") for _ in range(100)}
    assert len(ids) == 100


# ---------------------------------------------------------------------------
# Integration — CCR injection fires when strategy=llmlingua + tools present
# ---------------------------------------------------------------------------

@pytest.fixture
def client_lossy():
    app = _make_client("aggressive")
    mock_compressor = MagicMock()
    mock_compressor.compress.return_value = _LOSSY_RESULT
    mock_compressor.name = "llmlingua"

    with patch("shiori.providers.openai.OpenAIProvider.chat_completions", new_callable=AsyncMock) as mock_prov:
        mock_prov.return_value = _FAKE_RESPONSE
        with patch("shiori.api.openai_routes.route", return_value=_LOSSY_DECISION):
            with patch("shiori.compressors.llmlingua.LLMLinguaCompressor", return_value=mock_compressor):
                with TestClient(app) as c:
                    yield c, mock_prov


def test_ccr_marker_prepended_to_compressed_text(client_lossy):
    c, mock = client_lossy
    payload = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": _ORIGINAL_TEXT}],
        "tools": [],
    }
    c.post("/v1/chat/completions", json=payload)

    forwarded = mock.call_args[0][0]
    user_msg = next(m for m in forwarded["messages"] if m["role"] == "user")
    assert user_msg["content"].startswith("[SHIORI_CCR id=")
    assert _COMPRESSED_TEXT in user_msg["content"]


def test_ccr_tool_injected_into_tools(client_lossy):
    c, mock = client_lossy
    payload = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": _ORIGINAL_TEXT}],
        "tools": [],
    }
    c.post("/v1/chat/completions", json=payload)

    forwarded = mock.call_args[0][0]
    tool_names = [t["function"]["name"] for t in forwarded["tools"]]
    assert "shiori_retrieve" in tool_names


def test_ccr_original_retrievable_via_endpoint(client_lossy):
    c, mock = client_lossy
    payload = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": _ORIGINAL_TEXT}],
        "tools": [],
    }
    c.post("/v1/chat/completions", json=payload)

    forwarded = mock.call_args[0][0]
    user_msg = next(m for m in forwarded["messages"] if m["role"] == "user")
    match = re.search(r"\[SHIORI_CCR id=([^\]]+)\]", user_msg["content"])
    content_id = match.group(1)

    resp = c.get(f"/v1/shiori/retrieve/{content_id}")
    assert resp.status_code == 200
    assert resp.json()["content"] == _ORIGINAL_TEXT


# ---------------------------------------------------------------------------
# Integration — CCR does NOT fire when strategy=lossless
# ---------------------------------------------------------------------------

@pytest.fixture
def client_lossless():
    app = _make_client("aggressive")
    mock_compressor = MagicMock()
    mock_compressor.compress.return_value = _LOSSLESS_RESULT
    mock_compressor.name = "lossless"

    with patch("shiori.providers.openai.OpenAIProvider.chat_completions", new_callable=AsyncMock) as mock_prov:
        mock_prov.return_value = _FAKE_RESPONSE
        with patch("shiori.api.openai_routes.route", return_value=_LOSSLESS_DECISION):
            with TestClient(app) as c:
                yield c, mock_prov


def test_no_ccr_marker_when_strategy_lossless(client_lossless):
    c, mock = client_lossless
    payload = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": _ORIGINAL_TEXT}],
        "tools": [],
    }
    c.post("/v1/chat/completions", json=payload)

    forwarded = mock.call_args[0][0]
    user_msg = next(m for m in forwarded["messages"] if m["role"] == "user")
    assert "[SHIORI_CCR" not in user_msg["content"]


def test_no_ccr_tool_when_strategy_lossless(client_lossless):
    c, mock = client_lossless
    payload = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": _ORIGINAL_TEXT}],
        "tools": [],
    }
    c.post("/v1/chat/completions", json=payload)

    forwarded = mock.call_args[0][0]
    tool_names = [t["function"]["name"] for t in forwarded.get("tools", [])]
    assert "shiori_retrieve" not in tool_names


# ---------------------------------------------------------------------------
# Integration — CCR disabled when payload has no tools key
# ---------------------------------------------------------------------------

def test_no_ccr_when_no_tools_in_payload(client_lossy):
    c, mock = client_lossy
    payload = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": _ORIGINAL_TEXT}],
        # no "tools" key
    }
    c.post("/v1/chat/completions", json=payload)

    forwarded = mock.call_args[0][0]
    assert "tools" not in forwarded
    user_msg = next(m for m in forwarded["messages"] if m["role"] == "user")
    assert "[SHIORI_CCR" not in user_msg["content"]


# ---------------------------------------------------------------------------
# Retrieval endpoint — 404 for unknown / expired
# ---------------------------------------------------------------------------

def test_retrieve_unknown_id_returns_404():
    app = _make_client()
    with TestClient(app) as c:
        resp = c.get("/v1/shiori/retrieve/nonexistent-id")
    assert resp.status_code == 404
