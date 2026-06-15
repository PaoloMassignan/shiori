# ADR-011: JSON-aware compressor for structured tool outputs

## Status
Accepted

## Context
Tool outputs, RAG results, and API responses passed as message content are frequently JSON arrays of homogeneous objects. The existing lossless pipeline (template → caveman → dictionary) treats JSON as plain text and misses the dominant compression opportunities in structured data:

1. Null / empty values (`null`, `""`, `[]`, `{}`) carry zero information but consume tokens.
2. Keys repeated across every object in an array duplicate structure N times.
3. Constant key-value pairs (same value in every object) are pure redundancy.

Whitespace in pretty-printed JSON is also a direct token cost with no semantic value.

Headroom's SmartCrusher benchmarks show 73–92% token reduction on JSON-heavy workloads (GitHub issue triage, code search). The same gains are achievable without an external dependency.

## Decision

Add `JsonCompressor` in `src/shiori/compressors/json_compressor.py`. It implements the existing `Compressor` protocol and applies three passes in order:

1. **Null/empty elimination** — recursively drop keys whose value is `null`, `""`, `[]`, or `{}`.
2. **Compact serialisation** — `json.dumps(separators=(',', ':'))` removes all whitespace.
3. **Constant hoisting** (arrays of objects only, len ≥ 2) — key-value pairs whose value is identical and non-empty across every object are extracted to a `[CONST: k=v, ...]` header and removed from each object.

The output is no longer guaranteed to be valid JSON when the CONST header is present, but it remains unambiguously machine-readable and the LLM can reconstruct the full structure from the header + body.

Safety invariant is unchanged: if the result is not shorter than the original, the original is returned unmodified.

Non-JSON input (parse failure) is returned unmodified without raising.

## Consequences

- `JsonCompressor` is a pure compressor, not a router. The router decides when to invoke it (US-002 wires it to `role=tool` messages).
- `dictionary` is always `{}` — no symbol table is needed. The CONST header is self-describing.
- The compressor has no external dependencies beyond `json` (stdlib) and `tiktoken`.
- New routing rules that invoke `JsonCompressor` require a LZPrompt benchmark result per ADR-010.
