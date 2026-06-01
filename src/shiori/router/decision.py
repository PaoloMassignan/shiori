"""
RoutingDecision — the output of the Shiori router.

Immutable. Contains the chosen strategy, the reason for the decision,
which routing layer made the decision, and the extracted features.
"""
from __future__ import annotations

from dataclasses import dataclass

from shiori.router.features import PromptFeatures


@dataclass(frozen=True)
class RoutingDecision:
    """
    strategy: the compression mode to apply
    reason:   human-readable explanation of the routing decision
    via:      which routing layer made the decision
               "rules"    — structural rules (authoritative)
               "ml"       — ML classifier
               "fallback" — rule-based fallback heuristics
               "off"      — proxy mode is 'off'
               "safe"     — proxy mode is 'safe' (caps strategy to lossless)
    features: the PromptFeatures that were computed
    """
    strategy: str
    reason: str
    via: str
    features: PromptFeatures
