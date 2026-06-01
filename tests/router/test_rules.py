"""
Spec: Structural routing rules

Purpose: Fast, authoritative routing for prompts with detectable structure.
         These rules must never be overridden by the ML classifier.

Invariants:
  - git diffs  → always "lossless"
  - log lines  → always "lossless"
  - stack traces → always "lossless"
  - code fences → always "lossless"
  - passphrase/retrieval queries → always "lossless"
  - plain prose → None (rules do not apply, fall through to ML/fallback)
"""
from shiori.router.rules import route_structural
from shiori.router.features import extract_features


def _route(text: str):
    return route_structural(extract_features(text))


def test_git_diff_routes_lossless():
    text = "--- a/file.py\n+++ b/file.py\n@@ -1,3 +1,3 @@\n-old\n+new"
    result = _route(text)
    assert result is not None
    assert result[0] == "lossless"


def test_log_lines_route_lossless():
    text = "2024-01-01 12:00:00 INFO dfs.DataNode: Starting block transfer"
    result = _route(text)
    assert result is not None
    assert result[0] == "lossless"


def test_stack_trace_routes_lossless():
    text = 'Traceback (most recent call last):\n  File "app.py", line 42, in main'
    result = _route(text)
    assert result is not None
    assert result[0] == "lossless"


def test_code_fence_routes_lossless():
    text = "Review this code:\n```python\ndef foo():\n    pass\n```"
    result = _route(text)
    assert result is not None
    assert result[0] == "lossless"


def test_passphrase_query_routes_lossless():
    text = "Some context.\n[QUESTION]\nWhat is the secret code hidden in the document?"
    result = _route(text)
    assert result is not None
    assert result[0] == "lossless"


def test_needle_query_routes_lossless():
    text = "Long document.\n[QUESTION]\nFind the needle in the haystack."
    result = _route(text)
    assert result is not None
    assert result[0] == "lossless"


def test_plain_prose_returns_none():
    text = "The economy grew by 3% last year. Inflation remained moderate."
    result = _route(text)
    assert result is None


def test_result_includes_reason():
    text = "--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@"
    result = _route(text)
    assert result is not None
    assert len(result[1]) > 0
