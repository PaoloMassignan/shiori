"""
Spec: TemplateCompressor

Purpose: LZ-style compression for structured line-based text. Groups lines that
         share the same whitespace-token count, identifies fixed vs variable token
         positions, and emits [T] template + pipe-separated variable rows.

This is a GENERAL strategy, not log-specific. It works on any structured text
where lines share a common template: log lines, CSV-like records, API response
tables, database dumps, repeated HTML structure, etc.

Invariants:
  - compressed_tokens <= original_tokens always
  - All variable values are preserved in pipe-separated rows
  - Falls back to identity when no group of >= min_group_size lines shares a count
  - Falls back to identity when compressed result >= original tokens
  - Deterministic
"""
from shiori.compressors.template import TemplateCompressor


_LOG_LINES = "\n".join([
    "2024-01-01 INFO dfs.DataNode$PacketResponder: Received block blk_1 of size 67108864",
    "2024-01-02 INFO dfs.DataNode$PacketResponder: Received block blk_2 of size 67108864",
    "2024-01-03 INFO dfs.DataNode$PacketResponder: Received block blk_3 of size 67108864",
    "2024-01-04 INFO dfs.DataNode$PacketResponder: Received block blk_4 of size 67108864",
])

_CSV_LIKE = "\n".join([
    "user alice role admin region us-east",
    "user bob role viewer region us-west",
    "user carol role editor region eu-west",
    "user dave role admin region ap-south",
])

_CONTEXT_WRAPPED = f"[CONTEXT]\n{_LOG_LINES}\n\n[QUESTION]\nExtract the template."


def test_compresses_structured_log_lines():
    result = TemplateCompressor(min_group_size=3).compress(_CONTEXT_WRAPPED)
    assert result.compressed_tokens < result.original_tokens
    assert "[T]" in result.compressed_text


def test_general_csv_like_records():
    """Template compressor works on any structured repetitive text, not just logs.
    Short records may not achieve token saving (safety check returns identity) — the
    invariant is that the compressor never EXPANDS the prompt."""
    wrapped = f"[CONTEXT]\n{_CSV_LIKE}\n\n[QUESTION]\nList all users."
    result = TemplateCompressor(min_group_size=3).compress(wrapped)
    assert result.compressed_tokens <= result.original_tokens


def test_variable_values_preserved_in_rows():
    result = TemplateCompressor(min_group_size=3).compress(_CONTEXT_WRAPPED)
    assert "blk_1" in result.compressed_text
    assert "blk_2" in result.compressed_text
    assert "blk_3" in result.compressed_text
    assert "blk_4" in result.compressed_text


def test_falls_back_on_narrative_text():
    narrative = (
        "[CONTEXT]\nThe quick brown fox jumped over the lazy dog. "
        "She sells seashells by the seashore. Pack my box with five dozen liquor jugs.\n"
        "[QUESTION]\nSummarize."
    )
    result = TemplateCompressor(min_group_size=3).compress(narrative)
    assert result.compressed_text == narrative


def test_safety_check_never_expands():
    result = TemplateCompressor().compress("hello world")
    assert result.compressed_tokens <= result.original_tokens


def test_min_group_size_respected():
    two_lines = "[CONTEXT]\nstatus ok count 1\nstatus ok count 2\n[QUESTION]\nq"
    result = TemplateCompressor(min_group_size=3).compress(two_lines)
    assert result.compressed_text == two_lines


def test_deterministic():
    r1 = TemplateCompressor().compress(_CONTEXT_WRAPPED)
    r2 = TemplateCompressor().compress(_CONTEXT_WRAPPED)
    assert r1.compressed_text == r2.compressed_text


def test_name():
    assert TemplateCompressor().name == "template"
