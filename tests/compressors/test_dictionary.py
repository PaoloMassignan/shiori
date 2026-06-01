"""
Spec: DictionaryCompressor

Purpose: Replace repeated long phrases (>= min_chars, >= min_occurrences) with
         §A §B symbols. The dictionary is returned in CompressionResult for injection
         into the LLM system prompt.

Invariants:
  - compressed_tokens <= original_tokens always
  - Every substituted symbol appears in the dictionary
  - Phrases shorter than min_chars are never substituted
  - Phrases occurring fewer than min_occurrences times are never substituted
  - Numbers, dates, URLs, quoted strings are never substituted
"""
from shiori.compressors.dictionary import DictionaryCompressor


# Dictionary compression only pays off when occurrences * saving_per_occurrence > header_overhead.
# [LZP_DICTIONARY] header costs ~25 tokens. Each substitution saves (phrase_tokens - 2) tokens.
# With a 9-token phrase, need ~4+ occurrences to break even on header overhead.
_LONG_REPEAT = "the Secretary of State for Health and Social Care"  # ~9 tokens


def test_substitutes_repeated_long_phrase():
    phrase = _LONG_REPEAT
    # 5 occurrences × 7 tokens saved = 35 saved; header costs ~25 → net +10
    text = " ".join([
        f"{phrase} announced.",
        f"{phrase} confirmed.",
        f"{phrase} denied.",
        f"{phrase} refused.",
        f"{phrase} accepted the proposal.",
    ])
    result = DictionaryCompressor(min_chars=12, min_occurrences=2).compress(text)
    assert len(result.dictionary) >= 1
    # phrase appears in dictionary section (definition) but not in content section
    assert any(phrase == v for v in result.dictionary.values())
    content_section = result.compressed_text.split("[LZP_CONTENT]", 1)[-1]
    assert phrase not in content_section


def test_does_not_substitute_short_phrase():
    text = "cat cat cat cat cat"
    result = DictionaryCompressor(min_chars=12, min_occurrences=2).compress(text)
    assert result.dictionary == {}
    assert result.compressed_text == text


def test_does_not_substitute_single_occurrence():
    phrase = "UserAuthenticationService"
    result = DictionaryCompressor(min_chars=12, min_occurrences=2).compress(phrase)
    assert result.dictionary == {}


def test_safety_check_never_expands():
    text = "hello world"
    result = DictionaryCompressor().compress(text)
    assert result.compressed_tokens <= result.original_tokens


def test_dictionary_symbols_start_with_section():
    phrase = _LONG_REPEAT
    text = f"{phrase} one. {phrase} two. {phrase} three."
    result = DictionaryCompressor(min_chars=12, min_occurrences=2).compress(text)
    for sym in result.dictionary:
        assert sym.startswith("§")


def test_deterministic():
    phrase = _LONG_REPEAT
    text = f"{phrase} one. {phrase} two. {phrase} three."
    r1 = DictionaryCompressor(min_chars=12, min_occurrences=2).compress(text)
    r2 = DictionaryCompressor(min_chars=12, min_occurrences=2).compress(text)
    assert r1.compressed_text == r2.compressed_text
    assert r1.dictionary == r2.dictionary


def test_name():
    assert DictionaryCompressor().name == "dictionary"
