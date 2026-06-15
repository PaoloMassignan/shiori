# ADR-015: AST-aware Python code compressor

## Status
Accepted

## Context
`LosslessCompressor` treats code as plain text and achieves only ~10% saving on
Python source (bench corpus: 496 → 443 tokens). Python docstrings and comments
account for a large fraction of tokens in typical code passed as context: library
documentation, function descriptions, inline explanations.

Headroom's `CodeCompressor` achieves 30–50% saving on code by parsing AST and
removing semantically redundant constructs. For Python, the stdlib `ast` module
provides parse + unparse without external dependencies.

## Decision

Add `AstCompressor` in `src/shiori/compressors/ast_compressor.py`.

**What it removes:**
- Module, class, function, and async-function docstrings — the first
  `Expr(Constant(str))` node in each body.
- All inline comments — `ast.unparse()` does not emit comments because
  they are not represented in the AST.

**What it preserves:**
- All executable statements, imports, class/function signatures.
- Type annotations — they carry semantic information and affect runtime behaviour
  (Pydantic, dataclasses, FastAPI).

**Failure modes → identity:**
- `SyntaxError` on parse → not Python, return original unchanged.
- `ast.unparse` raises → return original unchanged.
- `compressed_tokens >= original_tokens` → safety check, return original.

**Wiring:**
- Registered as `"ast"` in `compressor_factory`.
- Added to the tool-output fallback chain between JSON and lossless:
  `JsonCompressor → AstCompressor → LosslessCompressor`.
- The bench runner uses `AstCompressor` for the `python_code` corpus entry.

**Scope: Python only.** Other languages fall through to `LosslessCompressor`.
A follow-up US may add regex-based comment stripping for JS/TS/Go/Rust.

## Consequences

- Requires Python 3.9+ for `ast.unparse()`. Shiori already requires 3.10+.
- The `python_code` bench threshold rises from 8% to 30%.
- No new dependencies.
