"""python -m shiori.evals.quality

Run the Shiori compression quality benchmark.

Modes:
  (default)  — 5 Claude Code cases, direct compressor assignment
  --compare  — 6 cases routed via safe vs aggressive mode (includes meeting transcript)

Usage:
    python -m shiori.evals.quality
    python -m shiori.evals.quality --compare
    python -m shiori.evals.quality --compare --verbose
    python -m shiori.evals.quality --model gpt-4o
"""
from __future__ import annotations

import argparse
import asyncio
import sys

from shiori.evals.quality.runner import (
    format_comparison_table,
    format_comparison_verbose,
    format_table,
    format_verbose,
    run_mode_comparison,
    run_quality_bench,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Shiori compression quality benchmark")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model to use")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show model responses")
    parser.add_argument(
        "--compare", action="store_true",
        help="Compare safe vs aggressive routing (includes meeting transcript case)"
    )
    args = parser.parse_args()

    if args.compare:
        print(f"Running safe vs aggressive comparison with {args.model}...")
        print("(LLMLingua model loads on first use — may take a moment)\n")
        results = asyncio.run(run_mode_comparison(model=args.model))
        print(format_comparison_table(results))
        if args.verbose:
            print(format_comparison_verbose(results))
        failed = [r for r in results if r.error or r.aggressive_delta < -0.1]
        if failed:
            print(f"\n{len(failed)} case(s) with significant quality degradation or errors.")
            sys.exit(1)
    else:
        print(f"Running quality benchmark with {args.model}...")
        results = asyncio.run(run_quality_bench(model=args.model))
        print(format_table(results))
        if args.verbose:
            print(format_verbose(results))
        failed = [r for r in results if r.error or r.compressed_score < r.original_score]
        if failed:
            print(f"\n{len(failed)} case(s) with quality degradation or errors.")
            sys.exit(1)


if __name__ == "__main__":
    main()
