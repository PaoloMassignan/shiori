# ADR-004: Lossless as the safe default

## Status
Accepted

## Context
When the routing signal is ambiguous, the wrong compression choice has asymmetric consequences: lossless compression on a summarization task costs a few percent saving; lossy compression on a QA or retrieval task can destroy the answer (-0.267 to -0.477 quality delta, per LZPrompt benchmarks).

## Decision
The default strategy when no signal is detected is "lossless". In "safe" mode, lossless is the maximum compression level regardless of ML or fallback signals. "Aggressive" mode allows llmlingua when the signal is clear (summarization, long prose).

## Consequences
- "safe" mode may leave token savings on the table for summarization workloads
- "aggressive" mode may harm quality on edge cases where ML misroutes
- The asymmetry is intentional: a lost 40% saving is recoverable; a destroyed answer is not
