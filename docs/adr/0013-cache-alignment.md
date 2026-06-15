# ADR-013: Cache alignment for Anthropic KV cache

## Status
Accepted

## Context
Anthropic's API supports prompt caching via `cache_control: {"type": "ephemeral"}` markers
on message content blocks. When the same prefix is sent across requests with this marker,
Anthropic caches the KV states and charges reduced input-token prices on cache hits.

Without explicit markers, caching never fires regardless of prefix stability. A single
trailing whitespace difference in the system prompt is enough to invalidate the cache.

OpenAI's caching is automatic for prefixes ≥ 1024 tokens; no client-side change is needed.
CacheAligner is therefore Anthropic-only.

## Decision

Add `apply_cache_align(messages)` as a pure function in
`src/shiori/api/cache_aligner.py`. It converts `role=system` message content from a plain
string to an Anthropic content-block array with `cache_control: {"type": "ephemeral"}` on
the last block. String content is normalised with `rstrip()` before conversion.

The function is called inside `OpenAIProvider.chat_completions`, **after** dictionary
injection (which writes to `content` as a string) and **before** the HTTP call. This
ordering avoids a type conflict: dictionary injection concatenates strings; CacheAligner
converts the final string to blocks.

Activation conditions (all must be true):
- `cfg.cache_align` is `True` (default)
- `"anthropic.com"` appears in `cfg.provider.base_url`

`ShioriConfig` gains a `cache_align: bool = True` field.
`OpenAIProvider.__init__` gains a `cache_align: bool` parameter set by the provider factory.

## Consequences

- `apply_cache_align` is pure and O(1) — no latency cost.
- Non-Anthropic providers are not affected.
- Existing tests that mock `OpenAIProvider.chat_completions` are not affected (they bypass
  the provider internals entirely).
- Provider-level integration tests use `respx` to verify the outgoing HTTP payload contains
  the expected `cache_control` block.
- `from_yaml` in `ShioriConfig` must handle the new `cache_align` key.
