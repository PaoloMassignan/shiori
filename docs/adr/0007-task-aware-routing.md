# ADR-007: Task-aware routing, not content-aware

## Status
Accepted

## Context
Most prompt compression systems inspect the content of the prompt to decide how to compress it (e.g., detect long text → apply high compression). Shiori's approach is different: it tries to understand what the user is trying to accomplish, then chooses the compression strategy safe for that task.

## Decision
The primary routing signal is task type, not content volume. Examples:
- Summarization → lossy safe (key facts distributed, redundant tokens can be dropped)
- Retrieval / passphrase → lossless required (exact tokens must survive)
- Code generation → lossless required (identifiers are precise)
- Multi-hop QA → lossless preferred (chain-linking tokens must survive)
- Long prose with no task signal → heuristic fallback (length-based)

The ML classifier is trained on task type labels (lossless/lossy), not on content features.

## Consequences
- Short prompts with retrieval tasks are correctly routed to lossless even if they'd benefit from lossy compression on volume grounds
- The classifier input is the [QUESTION] + [ANSWER_FORMAT] section (task signal), not the [CONTEXT] (content)
- New task types require new training examples and possibly new structural rules
