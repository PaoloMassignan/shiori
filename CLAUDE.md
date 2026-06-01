# Shiori — CLAUDE.md

## What this project is

A task-aware prompt compression proxy. Shiori sits between an LLM client and an LLM provider, intercepts `POST /v1/chat/completions`, routes the prompt to the safest compression strategy for the detected task type, and forwards the compressed prompt to the provider.

**"LZPrompt discovers what works. Shiori applies it."**

LZPrompt (sibling repository) is the benchmark/research platform. Every routing rule in Shiori is backed by a LZPrompt benchmark result.

---

## Development methodology

### TDD — Test-Driven Development

Every feature starts with tests. No implementation without a specification or test.

Requirements:
- compressor tests: `tests/compressors/`
- router tests: `tests/router/`
- proxy tests: `tests/api/`
- No test requires real API keys — use mocks everywhere

```bash
pip install -e ".[dev]"
pytest
```

### SDD — Specification-Driven Development

Every module has a docstring that defines:
- Purpose
- Inputs / Outputs
- Invariants
- Failure modes

Implementation follows the specification in the docstring. Read the docstring before modifying a module.

### ADR — Architecture Decision Records

Architectural decisions are documented in `docs/adr/`. Each ADR explains:
- What was decided
- Why (context and constraints)
- What the consequences are

Before adding a new routing rule, compressor, or proxy behavior, create or update the relevant ADR.

Current ADRs: `docs/adr/0001` through `docs/adr/0010`.

### KISS — Keep It Simple, Stupid

Prefer working software over abstraction. Three similar lines are better than a premature abstraction.

- Do not add error handling for scenarios that cannot happen
- Do not design for hypothetical future requirements
- Do not add abstractions beyond what the current feature requires
- A compressor is a class with `name` and `compress()`. That is enough.
- A routing rule is an `if` statement. That is enough.

---

## Project layout

```
src/shiori/
  config.py              — ShioriConfig dataclass (mode, provider, router, compressors)
  api/
    server.py            — FastAPI app + uvicorn entrypoint
    openai_routes.py     — POST /v1/chat/completions
    middleware.py        — logging middleware
  router/
    features.py          — PromptFeatures + extract_features()
    rules.py             — structural routing rules (authoritative)
    classifier.py        — MiniLM binary classifier (lossless/lossy)
    decision.py          — RoutingDecision dataclass
    __init__.py          — route() — top-level routing function
  compressors/
    base.py              — CompressionResult dataclass + Compressor protocol
    caveman.py           — remove articles and weak connectives
    dictionary.py        — repeated phrases → §A §B symbols
    template.py          — LZ-style line structure extraction
    lossless.py          — pipeline: template → caveman → dictionary
    llmlingua.py         — lossy token-level compression (optional dep)
  providers/
    openai.py            — forward to OpenAI (or OpenAI-compatible) API
  metrics/
    telemetry.py         — RequestRecord + Telemetry

tests/
  compressors/           — unit tests for each compressor
  router/                — unit tests for rules, classifier, route()
  api/                   — integration tests for proxy endpoint (mocked provider)

docs/adr/               — Architecture Decision Records (0001–0010)
configs/                 — shiori.yaml, safe.yaml, aggressive.yaml, debug.yaml
models/shiori_router/   — trained MiniLM classifier (not committed, see scripts/)
```

---

## Proxy modes

| Mode | Behavior |
|---|---|
| `safe` | Lossless or none. ML and fallback heuristics cap to lossless. |
| `aggressive` | Full routing: structural rules → ML → fallback → lossless/llmlingua |
| `off` | Pass-through. No compression. |
| `debug` | Same as aggressive + routing decisions included in response JSON |

---

## Routing layers

1. **Structural rules** (`router/rules.py`) — authoritative, always fast
   - git diff, log lines, stack traces, code fences, passphrase/retrieval queries → `lossless`
2. **ML classifier** (`router/classifier.py`) — optional, <10ms on CPU
   - MiniLM binary classifier trained on task type labels
   - Input: [QUESTION] + [ANSWER_FORMAT] only (not full context)
3. **Fallback** (`router/__init__.py`)
   - Long prose + summarize instruction → `llmlingua`
   - Long prose > 4000 tokens → `llmlingua`
   - Otherwise → `lossless`

---

## Compressor invariants

Every compressor must satisfy:
- `result.compressed_tokens <= result.original_tokens` (safety check, hard guarantee)
- `result.dictionary == {}` for lossy compressors (information is dropped permanently)
- Deterministic: same input → same output

---

## Run

```bash
pip install -e ".[dev]"
pytest

# Start proxy (safe mode, port 8000)
cp .env.example .env  # fill in OPENAI_API_KEY
shiori
# or
python -m shiori.api.server

# Use as drop-in for OpenAI
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Hello"}]}'

# Health and metrics
curl http://localhost:8000/health
curl http://localhost:8000/metrics
```
