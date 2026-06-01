"""
Spec: CavemanCompressor

Purpose: Remove articles (a/an/the) and weak discourse connectives from the
         context section of a prompt, without touching the question or instructions.

Inputs:  A prompt string (may or may not contain [QUESTION] marker)
Outputs: CompressionResult with compressed_tokens <= original_tokens

Invariants:
  - compressed_tokens <= original_tokens always
  - [QUESTION] and everything after it is never modified
  - If no saving is achieved, returns original text unchanged
  - Result is deterministic (same input → same output)
"""
from shiori.compressors.caveman import CavemanCompressor


def test_removes_articles_from_context():
    prompt = "The dog sat on the mat. A cat watched the scene."
    result = CavemanCompressor().compress(prompt)
    assert "The dog" not in result.compressed_text
    assert "the mat" not in result.compressed_text
    assert "A cat" not in result.compressed_text


def test_does_not_touch_question():
    prompt = "The quick brown fox.\n[QUESTION]\nWhat is the answer?"
    result = CavemanCompressor().compress(prompt)
    assert "[QUESTION]" in result.compressed_text
    assert "What is the answer?" in result.compressed_text


def test_safety_check_never_expands():
    prompt = "x"
    result = CavemanCompressor().compress(prompt)
    assert result.compressed_tokens <= result.original_tokens


def test_empty_string():
    result = CavemanCompressor().compress("")
    assert result.compressed_text == ""
    assert result.compressed_tokens == 0


def test_deterministic():
    prompt = "The cat sat on a mat. The dog watched."
    r1 = CavemanCompressor().compress(prompt)
    r2 = CavemanCompressor().compress(prompt)
    assert r1.compressed_text == r2.compressed_text


def test_result_has_correct_token_counts():
    prompt = "The quick brown fox jumped over the lazy dog."
    result = CavemanCompressor().compress(prompt)
    assert result.original_tokens > 0
    assert result.compressed_tokens <= result.original_tokens
    assert result.dictionary == {}
    assert result.dictionary_tokens == 0


def test_removes_connectives_at_line_start():
    prompt = "First sentence.\nAdditionally, this is another sentence.\nThird sentence."
    result = CavemanCompressor().compress(prompt)
    assert "Additionally," not in result.compressed_text


def test_name():
    assert CavemanCompressor().name == "caveman"
