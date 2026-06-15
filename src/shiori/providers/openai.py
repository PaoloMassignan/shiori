"""
OpenAI provider — forwards requests to the OpenAI (or OpenAI-compatible) API.

Handles:
  - system prompt injection for dictionary-based compression
  - transparent pass-through of all other request fields
  - streaming via stream_chat_completions() async generator

Failure modes:
  - provider unreachable: raises httpx.ConnectError
  - API key missing/invalid: raises httpx.HTTPStatusError (401)
"""
from __future__ import annotations

import json
from typing import Any, AsyncIterator

import httpx

from shiori.api.cache_aligner import apply_cache_align


class OpenAIProvider:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: float = 120.0,
        cache_align: bool = True,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._cache_align = cache_align and "anthropic.com" in base_url

    def _prepare_messages(
        self,
        payload: dict[str, Any],
        dictionary: dict[str, str],
    ) -> list[dict]:
        """Inject dictionary system prompt and apply cache alignment."""
        messages = list(payload.get("messages", []))
        if dictionary:
            dict_lines = ["[LZP_DICTIONARY]"] + [f"{s} = {p}" for s, p in dictionary.items()]
            dict_block = "\n".join(dict_lines)

            if messages and messages[0].get("role") == "system":
                messages[0] = {
                    **messages[0],
                    "content": dict_block + "\n\n" + messages[0]["content"],
                }
            else:
                messages.insert(0, {"role": "system", "content": dict_block})

        if self._cache_align:
            messages = apply_cache_align(messages)

        return messages

    async def chat_completions(
        self,
        payload: dict[str, Any],
        dictionary: dict[str, str],
    ) -> dict[str, Any]:
        """
        Forward a chat/completions request, return parsed JSON response.

        If dictionary is non-empty, inject it into the system prompt so the LLM
        can reconstruct compressed phrases (§A → original phrase).
        """
        messages = self._prepare_messages(payload, dictionary)
        forward_payload = {**payload, "messages": messages}

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                content=json.dumps(forward_payload),
            )
            response.raise_for_status()
            return response.json()

    async def stream_chat_completions(
        self,
        payload: dict[str, Any],
        dictionary: dict[str, str],
    ) -> AsyncIterator[bytes]:
        """
        Forward a streaming chat/completions request, yielding raw SSE bytes.

        Uses httpx.AsyncClient.stream() to avoid buffering the full response.
        The caller must consume the generator inside an async context.
        """
        messages = self._prepare_messages(payload, dictionary)
        forward_payload = {**payload, "messages": messages}

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                content=json.dumps(forward_payload),
            ) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes():
                    yield chunk
