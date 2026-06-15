# ADR-016: Streaming pass-through for `stream: true` requests

## Status
Accepted

## Context
Clients that set `"stream": true` in the chat completions payload expect a
Server-Sent Events (SSE) response, not a JSON blob. The current proxy buffers
the full provider response via `response.json()`, which fails for streaming
requests: httpx raises a decode error when trying to parse an SSE stream as JSON.

## Decision

Detect `payload.get("stream") == True` in the route handler. When streaming:

1. **Provider** — add `stream_chat_completions(payload, dictionary)` as an async
   generator that uses `httpx.AsyncClient.stream()` and yields raw bytes from the
   provider's SSE stream.
2. **Shared prep** — extract `_prepare_messages(payload, dictionary)` from
   `chat_completions` so dictionary injection and cache alignment are shared
   between both code paths. No duplication.
3. **Route handler** — return `fastapi.responses.StreamingResponse` with
   `media_type="text/event-stream"` and `Cache-Control: no-cache`.
4. **Compression** — still applied to user and tool messages before forwarding.
   The stream carries the already-compressed payload.
5. **Telemetry** — recorded before the stream starts with `provider_latency_ms=0.0`
   (timing the stream end would require buffering it, negating the benefit).
6. **Debug mode** — the `shiori` metadata field cannot be injected into an SSE
   stream without buffering. Streaming requests in debug mode receive no
   `shiori` field; routing decisions appear in logs only.

## Consequences

- Streaming requests now work correctly instead of raising a decode error.
- Provider latency is not measured for streaming requests.
- Debug metadata is unavailable in streaming responses.
- The `_prepare_messages` refactor eliminates duplicated logic in the provider.
