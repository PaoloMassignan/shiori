# ADR-010: Benchmark results drive routing decisions

## Status
Accepted

## Context
Routing heuristics without empirical validation are guesswork. The cost of a wrong routing decision can be severe (-0.477 quality delta on LogBench with llmlingua, per LZPrompt).

## Decision
Every routing rule in Shiori has a corresponding LZPrompt benchmark result that justifies it. The docs/routing.md file documents each rule with its evidence:

| Task type | Route | Evidence |
|---|---|---|
| Log lines (structural) | lossless | LZPrompt LogBench: llmlingua -0.477 delta |
| Code / diff (structural) | lossless | LZPrompt SWEBench: llmlingua -0.267 delta |
| Meeting summarization (ML) | llmlingua | LZPrompt MeetingBank: lossless 5.9% vs llmlingua 51.7% |
| Multi-hop QA (ML) | lossless | LZPrompt HotPotQA: lossless +0.050 vs llmlingua +0.005 |
| MCQ (ML) | lossless | LZPrompt ZeroSCROLLS: llmlingua -0.350 delta |

New routing rules without benchmark evidence are not accepted.

## Consequences
- Routing changes require a benchmark run in LZPrompt first
- The benchmark corpus is documented with train/eval/contaminated/held-out splits
- Out-of-distribution tasks fall back to lossless (safe default, ADR-004)
