# ADR-009: No external dependency required for routing

## Status
Accepted

## Context
Routing must work in air-gapped environments, CI pipelines, and low-latency production deployments. Any routing path that requires a network call would be fragile and slow.

## Decision
The full routing stack (structural rules + ML classifier + fallback) runs entirely locally. The structural rules require no model. The ML classifier requires a local model file (models/shiori_router/). The fallback requires no model. No routing decision ever makes a network call.

## Consequences
- Shiori can be deployed in environments with no outbound internet access (except to the configured LLM provider)
- The ML model must be distributed separately (training script provided)
- Routing latency is bounded: structural rules <1ms, ML classifier <10ms on CPU
