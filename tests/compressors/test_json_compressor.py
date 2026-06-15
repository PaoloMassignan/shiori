"""
Spec: JsonCompressor

Purpose: Compress JSON tool outputs by eliminating null/empty values,
         removing whitespace, and hoisting constant keys in homogeneous arrays.

Invariants:
  - compressed_tokens <= original_tokens always
  - dictionary is always empty (no symbol table needed)
  - Non-JSON input is returned unchanged without raising
  - Deterministic
"""
import json

import pytest

from shiori.compressors.json_compressor import JsonCompressor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_ARRAY_OF_OBJECTS = json.dumps([
    {"id": 1, "name": "Alice", "status": "active", "metadata": None, "tags": []},
    {"id": 2, "name": "Bob",   "status": "active", "metadata": None, "tags": []},
    {"id": 3, "name": "Carol", "status": "active", "metadata": None, "tags": []},
], indent=2)

_SINGLE_OBJECT = json.dumps({
    "id": 1,
    "name": "Alice",
    "email": "alice@example.com",
    "deleted_at": None,
    "notes": "",
    "children": [],
}, indent=2)

_SEARCH_RESULTS = json.dumps([
    {
        "doc_id": f"doc_{i:03d}",
        "score": round(0.9 - i * 0.01, 2),
        "source": "web",
        "language": "en",
        "content": f"Result content for document {i} with some text.",
        "metadata": None,
        "tags": [],
    }
    for i in range(20)
], indent=2)


# ---------------------------------------------------------------------------
# Null / empty elimination
# ---------------------------------------------------------------------------

def test_strips_null_values():
    result = JsonCompressor().compress(_SINGLE_OBJECT)
    assert "null" not in result.compressed_text
    assert "deleted_at" not in result.compressed_text


def test_strips_empty_string_values():
    result = JsonCompressor().compress(_SINGLE_OBJECT)
    assert '"notes"' not in result.compressed_text


def test_strips_empty_list_values():
    result = JsonCompressor().compress(_ARRAY_OF_OBJECTS)
    assert '"tags"' not in result.compressed_text


def test_preserves_non_empty_values():
    result = JsonCompressor().compress(_SINGLE_OBJECT)
    assert "Alice" in result.compressed_text
    assert "alice@example.com" in result.compressed_text


# ---------------------------------------------------------------------------
# Constant hoisting for homogeneous arrays
# ---------------------------------------------------------------------------

def test_hoists_constant_keys_to_header():
    result = JsonCompressor().compress(_ARRAY_OF_OBJECTS)
    assert "[CONST:" in result.compressed_text
    assert 'status="active"' in result.compressed_text


def test_hoisted_keys_removed_from_objects():
    result = JsonCompressor().compress(_SEARCH_RESULTS)
    # source and language are constant — should appear once in header, not in each object
    body_lines = result.compressed_text.split("\n", 1)
    body = body_lines[1] if len(body_lines) > 1 else result.compressed_text
    parsed_body = json.loads(body)
    for obj in parsed_body:
        assert "source" not in obj
        assert "language" not in obj


def test_hoisting_achieves_significant_savings_on_search_results():
    result = JsonCompressor().compress(_SEARCH_RESULTS)
    assert result.token_saving_ratio >= 0.30


def test_no_hoisting_for_non_constant_keys():
    data = json.dumps([
        {"id": 1, "status": "active"},
        {"id": 2, "status": "inactive"},
    ])
    result = JsonCompressor().compress(data)
    # status varies — must NOT be hoisted
    assert "status" not in result.compressed_text or "[CONST:" not in result.compressed_text or "status" in result.compressed_text.split("\n", 1)[-1]


def test_no_hoisting_for_single_element_array():
    data = json.dumps([{"id": 1, "status": "active"}])
    result = JsonCompressor().compress(data)
    assert "[CONST:" not in result.compressed_text


# ---------------------------------------------------------------------------
# Non-JSON passthrough
# ---------------------------------------------------------------------------

def test_non_json_returns_unchanged():
    text = "This is plain text, not JSON."
    result = JsonCompressor().compress(text)
    assert result.compressed_text == text
    assert result.changed is False


def test_partial_json_returns_unchanged():
    text = '{"broken": '
    result = JsonCompressor().compress(text)
    assert result.compressed_text == text


# ---------------------------------------------------------------------------
# Protocol invariants
# ---------------------------------------------------------------------------

def test_safety_never_expands():
    result = JsonCompressor().compress(_ARRAY_OF_OBJECTS)
    assert result.compressed_tokens <= result.original_tokens


def test_safety_never_expands_on_tiny_input():
    result = JsonCompressor().compress('{"x":1}')
    assert result.compressed_tokens <= result.original_tokens


def test_dictionary_always_empty():
    result = JsonCompressor().compress(_ARRAY_OF_OBJECTS)
    assert result.dictionary == {}


def test_deterministic():
    r1 = JsonCompressor().compress(_ARRAY_OF_OBJECTS)
    r2 = JsonCompressor().compress(_ARRAY_OF_OBJECTS)
    assert r1.compressed_text == r2.compressed_text


def test_name():
    assert JsonCompressor().name == "json"


# ---------------------------------------------------------------------------
# Compact serialisation
# ---------------------------------------------------------------------------

def test_output_has_no_unnecessary_whitespace():
    result = JsonCompressor().compress(_SINGLE_OBJECT)
    # Compact JSON should not have ": " or ",\n"
    assert ": " not in result.compressed_text
    assert ",\n" not in result.compressed_text
