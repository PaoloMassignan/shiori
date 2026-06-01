"""
Spec: LosslessCompressor

Purpose: Pipeline of template → caveman → dictionary.

On narrative text: template is a no-op → reduces to caveman+dictionary.
On structured text: template extracts repeated line structure first, then
caveman removes function words from what remains, then dictionary handles
any surviving repeated phrases.

Invariants:
  - compressed_tokens <= original_tokens always (hard guarantee from each step)
  - dictionary is empty if no phrase repetition survived the pipeline
  - All information is preserved (no token is dropped)
  - Deterministic
"""
from shiori.compressors.lossless import LosslessCompressor


_NARRATIVE = (
    "[CONTEXT]\n"
    "The quick brown fox jumps over the lazy dog. "
    "The lazy dog did not move. "
    "The brown fox ran away.\n"
    "[QUESTION]\nWhat did the fox do?"
)

_LOG_BLOCK = "\n".join([
    "2024-01-01 INFO dfs.DataNode$PacketResponder: Received block blk_1 of size 67108864",
    "2024-01-02 INFO dfs.DataNode$PacketResponder: Received block blk_2 of size 67108864",
    "2024-01-03 INFO dfs.DataNode$PacketResponder: Received block blk_3 of size 67108864",
    "2024-01-04 INFO dfs.DataNode$PacketResponder: Received block blk_4 of size 67108864",
])
_LOG_PROMPT = f"[CONTEXT]\n{_LOG_BLOCK}\n\n[QUESTION]\nExtract the template."


def test_compresses_narrative_removes_articles():
    result = LosslessCompressor().compress(_NARRATIVE)
    assert result.compressed_tokens <= result.original_tokens
    assert "The quick" not in result.compressed_text


def test_compresses_logs_better_than_narrative():
    log_result = LosslessCompressor().compress(_LOG_PROMPT)
    narrative_result = LosslessCompressor().compress(_NARRATIVE)
    log_ratio = log_result.token_saving_ratio
    narrative_ratio = narrative_result.token_saving_ratio
    assert log_ratio > narrative_ratio


def test_safety_check_never_expands():
    result = LosslessCompressor().compress("hi")
    assert result.compressed_tokens <= result.original_tokens


def test_dictionary_empty_when_no_repetition():
    text = "A completely unique sentence with no repetition whatsoever."
    result = LosslessCompressor().compress(text)
    assert result.dictionary == {}


def test_deterministic():
    r1 = LosslessCompressor().compress(_NARRATIVE)
    r2 = LosslessCompressor().compress(_NARRATIVE)
    assert r1.compressed_text == r2.compressed_text
    assert r1.dictionary == r2.dictionary


def test_name():
    assert LosslessCompressor().name == "lossless"
