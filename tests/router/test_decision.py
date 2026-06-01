"""
Spec: route() — top-level routing function

Purpose: Integrate structural rules, ML classifier, and fallback heuristics into
         a single RoutingDecision. Proxy mode (safe/aggressive/off) controls
         which layers are consulted.

Invariants:
  - "off" mode always returns strategy="none"
  - "safe" mode never returns strategy="llmlingua"
  - "aggressive" mode may return either "lossless" or "llmlingua"
  - Structural signals override ML and fallback in all modes except "off"
  - Decision is deterministic for same (prompt, mode) pair
  - features are always populated in the decision
"""
from shiori.router import route


_DIFF_PROMPT = "--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-old\n+new"
_SHORT_QA = "What is the capital of France?\n[QUESTION]\nAnswer: Paris"
_LONG_NARRATIVE = "The economy " + "grew steadily. " * 400  # >2000 tokens


def test_off_mode_always_none():
    d = route(_DIFF_PROMPT, proxy_mode="off")
    assert d.strategy == "none"
    assert d.via == "off"


def test_off_mode_no_compression_for_any_content():
    for text in [_DIFF_PROMPT, _SHORT_QA, _LONG_NARRATIVE]:
        d = route(text, proxy_mode="off")
        assert d.strategy == "none"


def test_safe_mode_never_lossy():
    for text in [_DIFF_PROMPT, _SHORT_QA, _LONG_NARRATIVE]:
        d = route(text, proxy_mode="safe")
        assert d.strategy != "llmlingua"


def test_structural_signal_overrides_in_aggressive():
    d = route(_DIFF_PROMPT, proxy_mode="aggressive")
    assert d.strategy == "lossless"
    assert d.via == "rules"


def test_safe_mode_diff_routes_lossless():
    d = route(_DIFF_PROMPT, proxy_mode="safe")
    assert d.strategy == "lossless"
    assert d.via == "rules"


def test_decision_has_features():
    d = route(_SHORT_QA, proxy_mode="safe")
    assert d.features is not None
    assert d.features.prompt_length_tokens > 0


def test_decision_has_reason():
    d = route(_DIFF_PROMPT, proxy_mode="aggressive")
    assert len(d.reason) > 0


def test_deterministic():
    d1 = route(_SHORT_QA, proxy_mode="aggressive")
    d2 = route(_SHORT_QA, proxy_mode="aggressive")
    assert d1.strategy == d2.strategy
    assert d1.via == d2.via
