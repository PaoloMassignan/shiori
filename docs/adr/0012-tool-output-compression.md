# ADR-012: Tool output compression for role=tool messages

## Status
Accepted

## Context
`openai_routes.py` compresses only the last `role=user` message. Messages with
`role=tool` (function call results) pass through unmodified. These messages are
the most token-dense part of an agentic thread: search results, database rows,
API responses, file contents. They are also the content type where `JsonCompressor`
(ADR-011) achieves the greatest savings.

Tool outputs do not carry a task signal — the user's intent is expressed in the
`role=user` message, not in the tool result. Task-aware routing (ADR-007) is
therefore not applicable to tool messages.

## Decision

Before forwarding a request, compress all `role=tool` messages in the thread:

1. If the message content is valid JSON → apply `JsonCompressor`.
2. If `JsonCompressor` produces no saving (non-JSON or already compact) →
   apply `LosslessCompressor` as fallback.
3. If `LosslessCompressor` also produces no saving → leave unchanged.

Routing signal: `via="tool_output"`. This bypasses the task-aware router
entirely; it is not a routing decision in the ADR-007 sense.

Mode behaviour:
- `off` → tool messages are not compressed (compressor_factory returns NoOp).
- `safe`, `aggressive`, `debug` → tool messages are always compressed.

`role=system` and `role=assistant` messages are not touched.
The existing `role=user` compression path is unchanged.

## Consequences

- `"json"` must be registered in `compressor_factory` in `server.py`.
- Tool output compression is silent in telemetry for now — `RequestRecord`
  continues to track the user-message compression only. A future US can extend
  telemetry to aggregate across all messages.
- No new config knob is needed: tool compression follows the existing mode flag.
