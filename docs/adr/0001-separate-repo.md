# ADR-001: Shiori as a separate repository from LZPrompt

## Status
Accepted

## Context
LZPrompt is a benchmark/research platform for evaluating prompt compression techniques. It accumulates experimental compressors, benchmark datasets, result files, and evaluation harnesses. Shipping a production proxy from the same repository would conflate research artifacts with operational code.

## Decision
Shiori is a separate repository. LZPrompt remains the research platform and source of truth for benchmark results. Shiori imports no code from LZPrompt — it re-implements the production-grade subset of compressors and the routing logic, and explicitly references LZPrompt in its documentation.

## Consequences
- Shiori has its own versioning, dependency set, and release cycle
- Routing decisions in Shiori must be validated against LZPrompt benchmark results
- Updates to LZPrompt compressors must be manually ported to Shiori (acceptable: the port is straightforward and enforces the "research first, production second" discipline)
- The README states: "LZPrompt discovers what works. Shiori applies it."
