# ADR-014: CCR — reversible lossy compression via in-process retrieval

## Status
Accepted

## Context
Lossy compression (`llmlingua`) permanently discards tokens. In production this is
a risk: if the LLM needs a detail that was removed, there is no way to recover it.
This limits lossy compression to workloads where information loss is acceptable
(e.g. meeting summarisation). For agentic workloads with tool outputs and long
documents, lossy compression is avoided even when it would be safe.

Headroom's CCR pattern solves this: store the original, tell the LLM it can ask
for it, let the LLM decide whether the compressed version is enough.

## Decision

Add a `CcrStore` (in-process dict with lazy TTL expiry) that stores originals and
returns a UUID `content_id`. When `strategy == "llmlingua"` fires AND the incoming
payload already contains a `tools` key:

1. Store the original user text in `CcrStore`. Get `content_id`.
2. Prepend `[SHIORI_CCR id=<content_id>]` to the compressed text.
3. Inject `shiori_retrieve(content_id)` into the payload `tools` list.

The LLM sees the CCR marker, reads the compressed content, and calls
`shiori_retrieve` only when it needs more detail.

A retrieval endpoint `GET /v1/shiori/retrieve/{content_id}` serves the original.
The endpoint returns 404 when the entry is absent or TTL-expired.

**CCR is silently disabled when:**
- The payload has no `tools` key (client does not support function calling)
- `strategy != "llmlingua"` (lossless compression is already reversible by definition)

TTL defaults to 600 seconds, configurable via `ccr_ttl_seconds` in `ShioriConfig`.

## Consequences

- `CcrStore` is created once per app instance in `_build_app` and shared between the
  chat completions handler and the retrieval router.
- `build_router` gains a `ccr_store` parameter.
- No background tasks or external dependencies: TTL is enforced lazily on `get()`.
- Memory usage is bounded by the number of concurrent sessions × average prompt size;
  acceptable for a local proxy. A future US can add a max-entries cap.
- The existing `safe` and `off` modes never trigger CCR: `safe` never routes to
  `llmlingua`, and `off` returns strategy `none`.
