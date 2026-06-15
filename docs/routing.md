# Shiori routing rules

Every rule in this table has a corresponding LZPrompt benchmark result (ADR-010).
The eval suite (`pytest tests/evals/` or `python -m shiori.evals`) validates each
rule against a synthetic text that reproduces the benchmark pattern.

## Structural rules (authoritative — fire before ML and fallback)

| Signal | Strategy | Benchmark | Evidence |
|---|---|---|---|
| Git diff / patch lines | lossless | SWEBench | llmlingua: −0.267 quality delta |
| Log lines with timestamps + level | lossless | LogBench | llmlingua: −0.477 quality delta |
| Stack trace | lossless | — | class names and line numbers must survive intact |
| Code fence (``` blocks) | lossless | SWEBench | identifier preservation required |
| Passphrase / retrieval query | lossless | — | exact token match required |

Structural rules are implemented in `router/rules.py` and always win over ML and fallback.

## ML classifier (layer 2 — when no structural signal)

The MiniLM binary classifier (`router/classifier.py`) is trained on task-type labels.
Input: `[QUESTION]` + `[ANSWER_FORMAT]` sections only (not the full context).
Model path: `models/shiori_router` (not committed; see `scripts/`).

When the model is absent, routing falls through to the fallback layer.

## Fallback heuristics (layer 3 — when ML unavailable or inconclusive)

| Condition | Strategy | Benchmark | Evidence |
|---|---|---|---|
| Long prose (> 2000 tokens) + `summarize` instruction | llmlingua | MeetingBank | lossless: 5.9% vs llmlingua: 51.7% |
| Long prose (> 4000 tokens, no other signal) | llmlingua | — | length heuristic |
| Everything else | lossless | HotPotQA, ZeroSCROLLS | lossless preferred for QA |

Fallback is implemented in `router/__init__.py`.

## Benchmark reference

| Benchmark | Task type | Correct strategy | Wrong strategy penalty |
|---|---|---|---|
| LogBench | Log QA | lossless | llmlingua −0.477 |
| SWEBench | Code / diff | lossless | llmlingua −0.267 |
| MeetingBank | Summarization | llmlingua | lossless only 5.9% compression |
| HotPotQA | Multi-hop QA | lossless | llmlingua +0.005 vs lossless +0.050 |
| ZeroSCROLLS | MCQ | lossless | llmlingua −0.350 |

Source: LZPrompt benchmark runs (sibling repository). New routing rules require a
LZPrompt benchmark result before being accepted (ADR-010).
