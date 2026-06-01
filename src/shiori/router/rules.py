"""
Rule-based structural router.

These rules are authoritative: when a structural signal is detected,
the routing decision is made immediately without consulting the ML classifier.

Priority (highest first):
  1. git diff                   → lossless
  2. log lines                  → lossless
  3. stack trace                → lossless
  4. code fence                 → lossless
  5. passphrase / retrieval     → lossless

Falls through to None when no structural signal is detected.
"""
from __future__ import annotations

from shiori.router.features import PromptFeatures


def route_structural(features: PromptFeatures) -> tuple[str, str] | None:
    """Return (strategy, reason) or None if no structural rule applies."""
    if features.has_diff:
        return "lossless", "git diff detected — lossy compression would corrupt patch"
    if features.has_log_lines:
        return "lossless", "log lines detected — template+caveman preserves variable tokens"
    if features.has_stack_trace:
        return "lossless", "stack trace detected — class names and line numbers must be preserved"
    if features.has_code_fence:
        return "lossless", "code fence detected — identifiers and syntax must be preserved"
    if features.has_passphrase_query:
        return "lossless", "retrieval / passphrase query detected — exact token match required"
    return None
