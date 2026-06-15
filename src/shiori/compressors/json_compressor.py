"""
JsonCompressor — compress JSON tool outputs and structured data.

Purpose:
    Reduce token cost of JSON payloads (tool outputs, RAG results, API responses)
    without losing information visible to the LLM.

Inputs / Outputs:
    Input:  any string. Non-JSON input is returned unchanged.
    Output: CompressionResult with compact JSON and optional CONST header.

Strategy (three passes, in order):
    1. Null/empty elimination — drop keys whose value is null, "", [], or {}.
    2. Compact serialisation — json.dumps(separators=(',',':')) removes whitespace.
    3. Constant hoisting (arrays of dicts, len >= 2) — key-value pairs that are
       identical and non-empty across every object are extracted to a single
       "[CONST: k=v, ...]" header line and removed from each object.

Invariants:
    - compressed_tokens <= original_tokens (falls back to identity if not shorter)
    - dictionary is always {} (CONST header is self-describing, no symbol table needed)
    - Non-JSON input never raises; returns identity
    - Deterministic
"""
from __future__ import annotations

import json

import tiktoken

from shiori.compressors.base import CompressionResult

_ENCODING = tiktoken.get_encoding("cl100k_base")


def _count(text: str) -> int:
    return len(_ENCODING.encode(text))


def _is_empty(value: object) -> bool:
    return value is None or value == "" or value == [] or value == {}


def _strip_empty(value: object) -> object:
    if isinstance(value, dict):
        return {k: _strip_empty(v) for k, v in value.items() if not _is_empty(v)}
    if isinstance(value, list):
        return [_strip_empty(item) for item in value]
    return value


def _find_constants(objects: list[dict]) -> dict:
    """Return keys whose value is identical and non-empty across all objects."""
    if len(objects) < 2:
        return {}
    common_keys = set(objects[0].keys())
    for obj in objects[1:]:
        common_keys &= set(obj.keys())
    constants: dict = {}
    for key in sorted(common_keys):
        first = objects[0][key]
        if not _is_empty(first) and all(obj[key] == first for obj in objects[1:]):
            constants[key] = first
    return constants


def _identity(text: str, original_tokens: int) -> CompressionResult:
    return CompressionResult(
        original_text=text,
        compressed_text=text,
        original_tokens=original_tokens,
        compressed_tokens=original_tokens,
        content_tokens=original_tokens,
    )


class JsonCompressor:
    name = "json"

    def compress(self, text: str) -> CompressionResult:
        original_tokens = _count(text)

        try:
            data = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return _identity(text, original_tokens)

        cleaned = _strip_empty(data)

        if (
            isinstance(cleaned, list)
            and len(cleaned) >= 2
            and all(isinstance(obj, dict) for obj in cleaned)
        ):
            constants = _find_constants(cleaned)
            if constants:
                body_objects = [
                    {k: v for k, v in obj.items() if k not in constants}
                    for obj in cleaned
                ]
                const_str = ", ".join(
                    f"{k}={json.dumps(v, separators=(',', ':'))}"
                    for k, v in constants.items()
                )
                header = f"[CONST: {const_str}]"
                body = json.dumps(body_objects, separators=(",", ":"))
                compressed_text = header + "\n" + body
            else:
                compressed_text = json.dumps(cleaned, separators=(",", ":"))
        else:
            compressed_text = json.dumps(cleaned, separators=(",", ":"))

        compressed_tokens = _count(compressed_text)
        if compressed_tokens >= original_tokens:
            return _identity(text, original_tokens)

        return CompressionResult(
            original_text=text,
            compressed_text=compressed_text,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            content_tokens=compressed_tokens,
        )
