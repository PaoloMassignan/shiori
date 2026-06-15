"""
Spec: Compression benchmarks

Purpose: Validate that each compressor meets a minimum token-saving threshold
         on its assigned corpus. A failing test means a compressor change has
         introduced a regression of more than the allowed margin.

Thresholds are conservative lower bounds — actual savings should be higher.
If you legitimately lower a threshold, update the ADR and document the reason.

Invariants:
  - Results are deterministic (same corpus → same numbers every run)
  - No API keys, no network calls, no side effects
  - Exit code 1 when any threshold is breached (pytest standard behaviour)
"""
from __future__ import annotations

import pytest

from shiori.bench.runner import BenchResult, run_bench

# Minimum saving % required per content type.
# These are regression guards: lowering them requires justification.
_THRESHOLDS: dict[str, float] = {
    "json_array":   50.0,   # null/empty stripping + constant hoisting
    "log_lines":    20.0,   # repeated structure + function-word removal
    "stack_trace":   4.0,   # path deduplication + whitespace (paths are mostly unique)
    "python_code":  30.0,   # docstrings + comments via AST (AstCompressor)
    "plain_prose":   3.0,   # function-word removal only
}


@pytest.fixture(scope="module")
def bench_results() -> dict[str, BenchResult]:
    return {r.content_type: r for r in run_bench()}


@pytest.mark.parametrize("content_type,min_saving", _THRESHOLDS.items())
def test_saving_meets_threshold(bench_results, content_type, min_saving):
    result = bench_results[content_type]
    assert result.saving_pct >= min_saving, (
        f"{content_type}: saving {result.saving_pct:.1f}% is below "
        f"the {min_saving:.1f}% threshold — compressor regression detected"
    )


def test_results_cover_all_corpus_types(bench_results):
    assert set(bench_results.keys()) == set(_THRESHOLDS.keys())


def test_results_are_deterministic():
    first = {r.content_type: r.compressed_tokens for r in run_bench()}
    second = {r.content_type: r.compressed_tokens for r in run_bench()}
    assert first == second


def test_compressed_never_exceeds_original(bench_results):
    for r in bench_results.values():
        assert r.compressed_tokens <= r.original_tokens, (
            f"{r.content_type}: compressed ({r.compressed_tokens}) > "
            f"original ({r.original_tokens})"
        )
