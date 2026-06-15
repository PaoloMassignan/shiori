"""
Spec: Routing eval suite

Purpose: Validate that the Shiori router makes the correct strategy decision
         for each benchmark case documented in docs/routing.md.

If a routing rule is changed without updating these tests, at least one case
will fail, preventing silent regressions (ADR-010).

Invariants:
  - No API keys, no ML model required (ml_model_path=None)
  - Proxy mode is always "aggressive" (safe caps all to lossless)
  - Deterministic: same text → same decision every run
"""
from __future__ import annotations

import pytest

from shiori.evals.cases import EVAL_CASES, EvalCase
from shiori.evals.runner import EvalResult, run_evals


# ---------------------------------------------------------------------------
# Parametrised correctness tests — one per benchmark case
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def eval_results() -> dict[str, EvalResult]:
    return {r.case.name: r for r in run_evals()}


@pytest.mark.parametrize("case", EVAL_CASES, ids=[c.name for c in EVAL_CASES])
def test_routing_strategy_correct(eval_results, case: EvalCase):
    result = eval_results[case.name]
    assert result.actual_strategy == case.expected_strategy, (
        f"{case.name}: expected strategy '{case.expected_strategy}', "
        f"got '{result.actual_strategy}' (via={result.actual_via}). "
        f"Evidence: {case.evidence}"
    )


@pytest.mark.parametrize("case", EVAL_CASES, ids=[c.name for c in EVAL_CASES])
def test_routing_via_correct(eval_results, case: EvalCase):
    result = eval_results[case.name]
    assert result.actual_via == case.expected_via, (
        f"{case.name}: expected routing via '{case.expected_via}', "
        f"got '{result.actual_via}'"
    )


# ---------------------------------------------------------------------------
# Structural rule coverage — explicit checks
# ---------------------------------------------------------------------------

def test_logbench_fires_structural_rule(eval_results):
    assert eval_results["LogBench"].actual_via == "rules"


def test_swebench_fires_structural_rule(eval_results):
    assert eval_results["SWEBench"].actual_via == "rules"


# ---------------------------------------------------------------------------
# Fallback coverage
# ---------------------------------------------------------------------------

def test_meetingbank_routes_to_llmlingua_via_fallback(eval_results):
    r = eval_results["MeetingBank"]
    assert r.actual_strategy == "llmlingua"
    assert r.actual_via == "fallback"


def test_hotpotqa_defaults_to_lossless(eval_results):
    assert eval_results["HotPotQA"].actual_strategy == "lossless"


def test_zeroscrolls_defaults_to_lossless(eval_results):
    assert eval_results["ZeroSCROLLS"].actual_strategy == "lossless"


# ---------------------------------------------------------------------------
# Suite integrity
# ---------------------------------------------------------------------------

def test_all_benchmarks_covered(eval_results):
    expected_names = {c.name for c in EVAL_CASES}
    assert set(eval_results.keys()) == expected_names


def test_results_are_deterministic():
    first = {r.case.name: r.actual_strategy for r in run_evals()}
    second = {r.case.name: r.actual_strategy for r in run_evals()}
    assert first == second
