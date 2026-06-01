# ADR-003: Local MiniLM router — no external call for routing

## Status
Accepted

## Context
Routing must be fast (<10ms) and must not require an external API call. Sending the prompt to an external classifier would add latency, cost, and a privacy risk (the prompt is the user's data).

## Decision
The ML classifier is a fine-tuned all-MiniLM-L6-v2 (22M parameters, ~90MB) that runs locally on CPU. It is optional: if the model directory is absent, Shiori falls back to rule-based routing. No routing decision ever requires a network call.

## Consequences
- First-time users need to download or train the model (~90MB)
- Without the model, MCQ tasks may be misrouted (LZPrompt benchmark shows -0.350 delta on ZeroSCROLLS quality without the ML classifier)
- The model file is not committed to the repository (too large); training instructions are in scripts/train_router.py
