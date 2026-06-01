# ADR-006: Prompt logging is opt-in

## Status
Accepted

## Context
Prompts contain user data. Logging prompts by default would be a privacy violation in most enterprise contexts. Telemetry (token counts, latency, strategy) is always collected because it contains no content.

## Decision
Prompt content is never logged unless `observability.log_prompts = true` is explicitly set in the config (or SHIORI_LOG_PROMPTS=true in the environment). This setting defaults to false. API keys are loaded from environment variables, never hardcoded.

## Consequences
- Default deployments produce no prompt content in logs
- Debugging compression quality requires opting into prompt logging
- Telemetry (token counts, routing decisions, latency) is always available and does not require opt-in
