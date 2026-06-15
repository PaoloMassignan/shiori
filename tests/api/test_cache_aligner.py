"""
Spec: CacheAligner

Purpose: apply_cache_align() converts role=system content to Anthropic content-block
         format with cache_control so KV cache hits fire on repeated system prompts.

Test strategy:
  - Unit tests on apply_cache_align() directly (correctness, all edge cases)
  - Provider integration test via respx HTTP mock (confirms the provider sends the
    right payload to Anthropic)

Invariants:
  - Only role=system messages are modified
  - String content is rstrip()-normalised before conversion
  - Content already in block format gets cache_control on the last block
  - Non-Anthropic providers are unaffected (tested via the provider integration test)
  - cache_align=False disables the behaviour
"""
from __future__ import annotations

import json

import pytest
import respx
from httpx import Response

from shiori.api.cache_aligner import apply_cache_align
from shiori.providers.openai import OpenAIProvider


# ---------------------------------------------------------------------------
# Unit tests on apply_cache_align()
# ---------------------------------------------------------------------------

def test_string_content_converted_to_blocks():
    messages = [{"role": "system", "content": "You are helpful."}]
    result = apply_cache_align(messages)
    content = result[0]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert content[0]["text"] == "You are helpful."
    assert content[0]["cache_control"] == {"type": "ephemeral"}


def test_trailing_whitespace_stripped():
    messages = [{"role": "system", "content": "You are helpful.   \n  "}]
    result = apply_cache_align(messages)
    assert result[0]["content"][0]["text"] == "You are helpful."


def test_non_system_messages_not_touched():
    messages = [
        {"role": "system", "content": "Sys."},
        {"role": "user", "content": "Hello."},
        {"role": "assistant", "content": "Hi."},
    ]
    result = apply_cache_align(messages)
    assert result[1]["content"] == "Hello."
    assert result[2]["content"] == "Hi."


def test_existing_block_content_gets_cache_control_on_last_block():
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "First block."},
                {"type": "text", "text": "Second block."},
            ],
        }
    ]
    result = apply_cache_align(messages)
    blocks = result[0]["content"]
    assert len(blocks) == 2
    assert "cache_control" not in blocks[0]
    assert blocks[1]["cache_control"] == {"type": "ephemeral"}


def test_existing_block_content_first_block_unchanged():
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "Only block."}],
        }
    ]
    result = apply_cache_align(messages)
    assert result[0]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert result[0]["content"][0]["text"] == "Only block."


def test_empty_messages_returns_empty():
    assert apply_cache_align([]) == []


def test_no_system_message_returns_unchanged():
    messages = [{"role": "user", "content": "Hi."}]
    result = apply_cache_align(messages)
    assert result == messages


def test_multiple_system_messages_all_get_cache_control():
    messages = [
        {"role": "system", "content": "First."},
        {"role": "user", "content": "Hello."},
        {"role": "system", "content": "Second."},
    ]
    result = apply_cache_align(messages)
    assert result[0]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert result[2]["content"][0]["cache_control"] == {"type": "ephemeral"}


def test_original_messages_list_not_mutated():
    messages = [{"role": "system", "content": "Sys."}]
    apply_cache_align(messages)
    assert messages[0]["content"] == "Sys."


# ---------------------------------------------------------------------------
# Provider integration — Anthropic (cache_align=True)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@respx.mock
async def test_provider_adds_cache_control_for_anthropic():
    fake_response = {
        "id": "msg_test",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
    }
    route = respx.post("https://api.anthropic.com/v1/chat/completions").mock(
        return_value=Response(200, json=fake_response)
    )

    provider = OpenAIProvider(
        base_url="https://api.anthropic.com/v1",
        api_key="test-key",
        cache_align=True,
    )
    payload = {
        "model": "claude-3-5-sonnet-latest",
        "messages": [
            {"role": "system", "content": "You are a coding assistant."},
            {"role": "user", "content": "Hello."},
        ],
    }
    await provider.chat_completions(payload, {})

    sent_body = json.loads(route.calls[0].request.content)
    system_msg = next(m for m in sent_body["messages"] if m["role"] == "system")
    assert isinstance(system_msg["content"], list)
    assert system_msg["content"][0]["cache_control"] == {"type": "ephemeral"}


@pytest.mark.asyncio
@respx.mock
async def test_provider_no_cache_control_for_openai():
    fake_response = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
    }
    route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=Response(200, json=fake_response)
    )

    provider = OpenAIProvider(
        base_url="https://api.openai.com/v1",
        api_key="test-key",
        cache_align=True,
    )
    payload = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello."},
        ],
    }
    await provider.chat_completions(payload, {})

    sent_body = json.loads(route.calls[0].request.content)
    system_msg = next(m for m in sent_body["messages"] if m["role"] == "system")
    assert isinstance(system_msg["content"], str)


@pytest.mark.asyncio
@respx.mock
async def test_provider_cache_align_false_disables_behaviour():
    fake_response = {
        "id": "msg_test",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
    }
    route = respx.post("https://api.anthropic.com/v1/chat/completions").mock(
        return_value=Response(200, json=fake_response)
    )

    provider = OpenAIProvider(
        base_url="https://api.anthropic.com/v1",
        api_key="test-key",
        cache_align=False,
    )
    payload = {
        "model": "claude-3-5-sonnet-latest",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello."},
        ],
    }
    await provider.chat_completions(payload, {})

    sent_body = json.loads(route.calls[0].request.content)
    system_msg = next(m for m in sent_body["messages"] if m["role"] == "system")
    assert isinstance(system_msg["content"], str)
