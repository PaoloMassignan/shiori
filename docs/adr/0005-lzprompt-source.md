# ADR-005: LZPrompt is the source of truth for routing decisions

## Status
Accepted

## Context
Routing rules in Shiori encode empirical knowledge: "LLMLingua hurts QA by -0.071; lossless helps QA by +0.050." These numbers must come from a reproducible benchmark, not intuition.

## Decision
Every routing rule or ML training decision in Shiori must be traceable to a committed benchmark result in LZPrompt. The README documents the benchmark evidence for each routing decision. New routing heuristics require a benchmark run in LZPrompt before being added to Shiori.

## Consequences
- Routing changes require a LZPrompt benchmark run first
- This creates a feedback loop: LZPrompt discoveries → Shiori routing updates
- Out-of-distribution tasks (not yet benchmarked) fall back to the lossless default (ADR-004)
