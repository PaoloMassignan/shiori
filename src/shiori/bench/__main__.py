"""python -m shiori.bench [--save]

Print a compression benchmark table. With --save, write results to
docs/benchmarks.md relative to the project root.
"""
from __future__ import annotations

import sys
from pathlib import Path

from shiori.bench.runner import format_markdown, format_table, run_bench


def main() -> None:
    results = run_bench()
    print(format_table(results))

    if "--save" in sys.argv:
        docs_dir = Path(__file__).parent.parent.parent.parent / "docs"
        docs_dir.mkdir(exist_ok=True)
        out = docs_dir / "benchmarks.md"
        out.write_text(format_markdown(results), encoding="utf-8")
        print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
