"""python -m shiori.evals

Validate routing decisions against the benchmark corpus documented in
docs/routing.md. Exits with code 1 if any case fails.
"""
from __future__ import annotations

import sys

from shiori.evals.runner import format_table, run_evals


def main() -> None:
    results = run_evals()
    print(format_table(results))
    if any(not r.passed for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
