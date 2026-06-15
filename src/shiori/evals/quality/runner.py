"""
Quality benchmark runner — measures compression fidelity via live model calls.

For each QualityCase:
  1. Build original messages: user = context + question
  2. Compress context with the specified compressor
  3. Build compressed messages: user = compressed_context + question
  4. Call model with both → score responses against expected_facts
  5. Report: compression ratio, original score, compressed score, delta

Score = fraction of expected_facts found (case-insensitive substring) in response.
Score 0.0 = none found, 1.0 = all found.

Both paths run concurrently (asyncio.gather) to halve wall-clock time.

Model: gpt-4o-mini (configurable). temperature=0 for deterministic responses.
API key read from OPENAI_API_KEY env var (.env auto-loaded from cwd).
"""
from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass

import httpx
from dotenv import load_dotenv

from shiori.compressors.ast_compressor import AstCompressor
from shiori.compressors.json_compressor import JsonCompressor
from shiori.compressors.lossless import LosslessCompressor
from shiori.compressors.llmlingua import LLMLinguaCompressor
from shiori.evals.quality.cases import COMPARISON_CASES, QUALITY_CASES, QualityCase
from shiori.router import route

load_dotenv()

_COMPRESSORS = {
    "json": JsonCompressor(),
    "ast": AstCompressor(),
    "lossless": LosslessCompressor(),
    "llmlingua": LLMLinguaCompressor(),
    "none": LosslessCompressor(),  # proxy_mode=off fallback
}

_OPENAI_URL = "https://api.openai.com/v1/chat/completions"


@dataclass
class QualityResult:
    case_name: str
    compressor_name: str
    original_tokens: int
    compressed_tokens: int
    original_score: float
    compressed_score: float
    original_response: str
    compressed_response: str
    error: str | None = None

    @property
    def saving_pct(self) -> float:
        if self.original_tokens == 0:
            return 0.0
        return (1.0 - self.compressed_tokens / self.original_tokens) * 100.0

    @property
    def delta(self) -> float:
        return self.compressed_score - self.original_score


def _score(response: str, expected_facts: list[str]) -> float:
    """Fraction of expected_facts found (case-insensitive) in response."""
    if not expected_facts:
        return 1.0
    lower = response.lower()
    found = sum(1 for fact in expected_facts if fact.lower() in lower)
    return found / len(expected_facts)


def _build_messages(context: str, question: str) -> list[dict]:
    return [{"role": "user", "content": f"{context}\n\n{question}"}]


async def _call_model(messages: list[dict], api_key: str, model: str) -> str:
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            _OPENAI_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            content=json.dumps({
                "model": model,
                "messages": messages,
                "max_tokens": 256,
                "temperature": 0,
            }),
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


async def _run_case(case: QualityCase, api_key: str, model: str) -> QualityResult:
    try:
        compressor = _COMPRESSORS[case.compressor_name]
        cr = compressor.compress(case.context)

        original_msgs = _build_messages(case.context, case.question)
        compressed_msgs = _build_messages(cr.compressed_text, case.question)

        original_response, compressed_response = await asyncio.gather(
            _call_model(original_msgs, api_key, model),
            _call_model(compressed_msgs, api_key, model),
        )

        return QualityResult(
            case_name=case.name,
            compressor_name=case.compressor_name,
            original_tokens=cr.original_tokens,
            compressed_tokens=cr.compressed_tokens,
            original_score=_score(original_response, case.expected_facts),
            compressed_score=_score(compressed_response, case.expected_facts),
            original_response=original_response,
            compressed_response=compressed_response,
        )
    except Exception as exc:
        return QualityResult(
            case_name=case.name,
            compressor_name=case.compressor_name,
            original_tokens=0,
            compressed_tokens=0,
            original_score=0.0,
            compressed_score=0.0,
            original_response="",
            compressed_response="",
            error=str(exc),
        )


async def run_quality_bench(
    model: str = "gpt-4o-mini",
    api_key: str | None = None,
) -> list[QualityResult]:
    key = api_key or os.getenv("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY not set — create a .env file or export the variable"
        )
    return list(await asyncio.gather(*[_run_case(c, key, model) for c in QUALITY_CASES]))


def format_table(results: list[QualityResult]) -> str:
    header = (
        f"{'case':<18} {'compressor':<10} {'saving':>7} "
        f"{'orig':>6} {'compr':>6} {'delta':>6}"
    )
    sep = "-" * len(header)
    rows = [header, sep]
    for r in results:
        if r.error:
            rows.append(f"{r.case_name:<18} ERROR: {r.error}")
            continue
        delta_str = f"{r.delta:+.2f}"
        rows.append(
            f"{r.case_name:<18} {r.compressor_name:<10} {r.saving_pct:>6.1f}% "
            f"{r.original_score:>6.2f} {r.compressed_score:>6.2f} {delta_str:>6}"
        )
    return "\n".join(rows)


def format_verbose(results: list[QualityResult]) -> str:
    lines = []
    for r in results:
        lines.append(f"\n=== {r.case_name} ({r.compressor_name}) ===")
        if r.error:
            lines.append(f"ERROR: {r.error}")
            continue
        lines.append(f"Saving: {r.saving_pct:.1f}%  Delta: {r.delta:+.2f}")
        lines.append(f"[ORIGINAL]   {r.original_response[:300]}")
        lines.append(f"[COMPRESSED] {r.compressed_response[:300]}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Safe vs Aggressive routing comparison
# ---------------------------------------------------------------------------

@dataclass
class ComparisonResult:
    case_name: str
    safe_strategy: str
    aggressive_strategy: str
    safe_saving_pct: float
    aggressive_saving_pct: float
    original_score: float
    safe_score: float
    aggressive_score: float
    safe_response: str
    aggressive_response: str
    error: str | None = None

    @property
    def aggressive_delta(self) -> float:
        return self.aggressive_score - self.original_score

    @property
    def diverges(self) -> bool:
        return self.safe_strategy != self.aggressive_strategy


def _saving_pct(original_tokens: int, compressed_tokens: int) -> float:
    if original_tokens == 0:
        return 0.0
    return (1.0 - compressed_tokens / original_tokens) * 100.0


async def _run_comparison_case(
    case: QualityCase, api_key: str, model: str
) -> ComparisonResult:
    try:
        full_text = f"{case.context}\n\n{case.question}"
        decision_safe = route(full_text, proxy_mode="safe")
        decision_aggressive = route(full_text, proxy_mode="aggressive")

        safe_comp = _COMPRESSORS[decision_safe.strategy]
        aggr_comp = _COMPRESSORS[decision_aggressive.strategy]

        safe_cr = safe_comp.compress(case.context)
        aggr_cr = aggr_comp.compress(case.context)

        original_msgs = _build_messages(case.context, case.question)
        safe_msgs = _build_messages(safe_cr.compressed_text, case.question)
        aggr_msgs = _build_messages(aggr_cr.compressed_text, case.question)

        orig_r, safe_r, aggr_r = await asyncio.gather(
            _call_model(original_msgs, api_key, model),
            _call_model(safe_msgs, api_key, model),
            _call_model(aggr_msgs, api_key, model),
        )

        return ComparisonResult(
            case_name=case.name,
            safe_strategy=decision_safe.strategy,
            aggressive_strategy=decision_aggressive.strategy,
            safe_saving_pct=_saving_pct(safe_cr.original_tokens, safe_cr.compressed_tokens),
            aggressive_saving_pct=_saving_pct(aggr_cr.original_tokens, aggr_cr.compressed_tokens),
            original_score=_score(orig_r, case.expected_facts),
            safe_score=_score(safe_r, case.expected_facts),
            aggressive_score=_score(aggr_r, case.expected_facts),
            safe_response=safe_r,
            aggressive_response=aggr_r,
        )
    except Exception as exc:
        return ComparisonResult(
            case_name=case.name,
            safe_strategy="?",
            aggressive_strategy="?",
            safe_saving_pct=0.0,
            aggressive_saving_pct=0.0,
            original_score=0.0,
            safe_score=0.0,
            aggressive_score=0.0,
            safe_response="",
            aggressive_response="",
            error=str(exc),
        )


async def run_mode_comparison(
    model: str = "gpt-4o-mini",
    api_key: str | None = None,
) -> list[ComparisonResult]:
    key = api_key or os.getenv("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    return list(
        await asyncio.gather(*[_run_comparison_case(c, key, model) for c in COMPARISON_CASES])
    )


def format_comparison_table(results: list[ComparisonResult]) -> str:
    header = (
        f"{'case':<18} {'safe':>9} {'aggr':>9} "
        f"{'safe%':>6} {'aggr%':>6} "
        f"{'orig':>5} {'safe':>5} {'aggr':>5} {'d-aggr':>7}"
    )
    sep = "-" * len(header)
    rows = [header, sep]
    for r in results:
        if r.error:
            rows.append(f"{r.case_name:<18} ERROR: {r.error[:60]}")
            continue
        marker = " <--" if r.diverges else ""
        rows.append(
            f"{r.case_name:<18} {r.safe_strategy:>9} {r.aggressive_strategy:>9} "
            f"{r.safe_saving_pct:>5.1f}% {r.aggressive_saving_pct:>5.1f}% "
            f"{r.original_score:>5.2f} {r.safe_score:>5.2f} {r.aggressive_score:>5.2f} "
            f"{r.aggressive_delta:>+7.2f}{marker}"
        )
    return "\n".join(rows)


def format_comparison_verbose(results: list[ComparisonResult]) -> str:
    lines = []
    for r in results:
        lines.append(f"\n=== {r.case_name}: safe={r.safe_strategy} / aggr={r.aggressive_strategy} ===")
        if r.error:
            lines.append(f"ERROR: {r.error}")
            continue
        lines.append(
            f"safe={r.safe_saving_pct:.1f}%  aggressive={r.aggressive_saving_pct:.1f}%  "
            f"d-aggr={r.aggressive_delta:+.2f}"
        )
        if r.diverges:
            lines.append(f"[SAFE]       {r.safe_response[:300]}")
            lines.append(f"[AGGRESSIVE] {r.aggressive_response[:300]}")
        else:
            lines.append(f"  (same strategy — responses omitted)")
    return "\n".join(lines)
