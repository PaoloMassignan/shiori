"""
Eval runner — route each eval case and collect pass/fail results.

Purpose:
    Validate that the routing layer makes the correct strategy decision for each
    case documented in docs/routing.md, using only locally generated text.

Inputs / Outputs:
    Input:  EVAL_CASES from cases.py
    Output: list[EvalResult] — one per case

Invariants:
    - No API keys, no ML model required (ml_model_path=None → fallback layer)
    - Proxy mode is always "aggressive" (safe mode caps everything to lossless)
    - Deterministic
"""
from __future__ import annotations

from dataclasses import dataclass

from shiori.evals.cases import EVAL_CASES, EvalCase
from shiori.router import route


@dataclass(frozen=True)
class EvalResult:
    case: EvalCase
    actual_strategy: str
    actual_via: str

    @property
    def passed(self) -> bool:
        return self.actual_strategy == self.case.expected_strategy


def run_evals() -> list[EvalResult]:
    """Route all eval cases and return results."""
    results: list[EvalResult] = []
    for case in EVAL_CASES:
        decision = route(case.text, proxy_mode="aggressive", ml_model_path=None)
        results.append(EvalResult(
            case=case,
            actual_strategy=decision.strategy,
            actual_via=decision.via,
        ))
    return results


def format_table(results: list[EvalResult]) -> str:
    header = (
        f"{'benchmark':<15} {'expected':<12} {'actual':<12} {'via':<10} {'pass':<5}"
    )
    sep = "-" * len(header)
    rows = [header, sep]
    for r in results:
        mark = "PASS" if r.passed else "FAIL"
        rows.append(
            f"{r.case.name:<15} {r.case.expected_strategy:<12} "
            f"{r.actual_strategy:<12} {r.actual_via:<10} {mark}"
        )
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    rows.append(sep)
    rows.append(f"{passed}/{total} passed")
    return "\n".join(rows)
