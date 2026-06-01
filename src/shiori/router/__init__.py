"""
Shiori router — dispatches each prompt to the safest compression strategy.

Routing layers (highest priority first):
  1. Structural rules (always fast, authoritative for structured content)
  2. ML classifier (MiniLM binary lossless/lossy, if model is available)
  3. Rule-based fallback heuristics
  4. Proxy mode override (safe caps to lossless; off skips compression)

Usage:
    from shiori.router import route
    decision = route(prompt, proxy_mode="aggressive", ml_model_path="models/shiori_router")
"""
from __future__ import annotations

from shiori.router.decision import RoutingDecision
from shiori.router.features import PromptFeatures, extract_features
from shiori.router.rules import route_structural
from shiori.router.classifier import classify

_LONG_PROSE_LOSSY_THRESHOLD = 4000


def route(
    prompt: str,
    proxy_mode: str = "aggressive",
    ml_model_path: str | None = None,
) -> RoutingDecision:
    """
    Route a prompt to a compression strategy.

    proxy_mode:
      "off"        → always "none"
      "safe"       → structural rules only; ML and fallback both cap to "lossless"
      "aggressive" → full routing: rules → ML → fallback
      "debug"      → same as aggressive (caller handles extended logging)
    """
    features = extract_features(prompt)

    if proxy_mode == "off":
        return RoutingDecision(
            strategy="none",
            reason="proxy mode is off",
            via="off",
            features=features,
        )

    # Layer 1 — structural rules (authoritative)
    rule_result = route_structural(features)
    if rule_result is not None:
        strategy, reason = rule_result
        return RoutingDecision(strategy=strategy, reason=reason, via="rules", features=features)

    # In safe mode, skip ML and fallback — always lossless
    if proxy_mode == "safe":
        return RoutingDecision(
            strategy="lossless",
            reason="safe mode: no structural signal detected, defaulting to lossless",
            via="safe",
            features=features,
        )

    # Layer 2 — ML classifier
    ml_result = classify(prompt, model_path=ml_model_path)
    if ml_result is not None:
        strategy = "llmlingua" if ml_result == "lossy" else "lossless"
        return RoutingDecision(
            strategy=strategy,
            reason=f"ML classifier: {ml_result}",
            via="ml",
            features=features,
        )

    # Layer 3 — rule-based fallback
    if features.is_long_prose and features.has_summarize_instruction:
        return RoutingDecision(
            strategy="llmlingua",
            reason="long prose with summarization instruction",
            via="fallback",
            features=features,
        )
    if features.is_long_prose and features.prompt_length_tokens > _LONG_PROSE_LOSSY_THRESHOLD:
        return RoutingDecision(
            strategy="llmlingua",
            reason=f"long prose ({features.prompt_length_tokens} tokens > {_LONG_PROSE_LOSSY_THRESHOLD} threshold)",
            via="fallback",
            features=features,
        )

    return RoutingDecision(
        strategy="lossless",
        reason="no signal for lossy compression — defaulting to lossless",
        via="fallback",
        features=features,
    )
