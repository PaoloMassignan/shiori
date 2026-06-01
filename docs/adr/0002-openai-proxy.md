# ADR-002: OpenAI-compatible proxy as primary interface

## Status
Accepted

## Context
Users of LLM APIs are most familiar with the OpenAI chat completions interface. Building an OpenAI-compatible proxy means zero client-side changes — users point their existing code at Shiori's endpoint instead of OpenAI's.

## Decision
The primary interface is POST /v1/chat/completions, accepting and returning the OpenAI chat completions schema. Shiori sits between the client and the provider: it intercepts the request, compresses the prompt, and forwards the modified request to the configured provider.

## Consequences
- Any OpenAI client library works with Shiori out of the box
- Streaming support must be handled transparently (future milestone)
- Anthropic and Ollama support requires translation layers (future milestone)
- Dictionary injection requires modifying the system message, which is a side-effect invisible to the client
