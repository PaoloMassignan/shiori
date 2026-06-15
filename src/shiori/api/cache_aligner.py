"""
CacheAligner — convert role=system content to Anthropic content-block format
with cache_control markers so KV cache hits fire on repeated system prompts.

Purpose:
    Stabilise the system-prompt prefix so Anthropic's prompt caching activates.
    For OpenAI providers, caching is automatic; this function is a no-op there.

Inputs / Outputs:
    Input:  list of OpenAI-format message dicts
    Output: new list with role=system messages converted to block format

Invariants:
    - Only role=system messages are modified
    - String content is rstrip()-normalised before conversion
    - Content already in block format gets cache_control on the last block only
    - Input list is never mutated
    - Pure function: no I/O, no side effects, O(n_messages)
"""
from __future__ import annotations


def apply_cache_align(messages: list[dict]) -> list[dict]:
    """Add cache_control to every role=system message in the list."""
    result = list(messages)
    for i, msg in enumerate(result):
        if msg.get("role") != "system":
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            result[i] = {
                **msg,
                "content": [
                    {
                        "type": "text",
                        "text": content.rstrip(),
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            }
        elif isinstance(content, list) and content:
            blocks = list(content)
            blocks[-1] = {**blocks[-1], "cache_control": {"type": "ephemeral"}}
            result[i] = {**msg, "content": blocks}
    return result
