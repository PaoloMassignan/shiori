"""
OpenAI provider — forwards requests to the OpenAI (or OpenAI-compatible) API.

Handles:
  - system prompt injection for dictionary-based compression
  - transparent pass-through of all other request fields
  - streaming (forwarded as-is)

Failure modes:
  - provider unreachable: raises httpx.ConnectError
  - API key missing/invalid: raises httpx.HTTPStatusError (401)
"""
from __future__ import annotations

import json
from typing import Any

import httpx


class OpenAIProvider:
    def __init__(self, base_url: str, api_key: str, timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    async def chat_completions(
        self,
        payload: dict[str, Any],
        dictionary: dict[str, str],
    ) -> dict[str, Any]:
        """
        Forward a chat/completions request.

        If dictionary is non-empty, inject it into the system prompt so the LLM
        can reconstruct compressed phrases (§A → original phrase).
        """
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
