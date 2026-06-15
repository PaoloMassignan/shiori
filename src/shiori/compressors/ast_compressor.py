"""
AstCompressor — remove docstrings and comments from Python source via AST.

Purpose:
    Reduce token cost of Python code passed as context (tool outputs, code
    fences, file contents) by stripping semantically redundant constructs
    that are not required for the LLM to understand the code's behaviour.

Inputs / Outputs:
    Input:  any string. Non-Python input (SyntaxError on parse) is returned unchanged.
    Output: CompressionResult with docstrings and comments removed.

What is removed:
    - Module, class, function, and async-function docstrings
      (first Expr(Constant(str)) in each body)
    - All inline and block comments (not present in the AST)

What is preserved:
    - All executable statements and expressions
    - Import statements
    - Type annotations (carry semantic information used by dataclasses, Pydantic, etc.)
    - Class and function signatures

Invariants:
    - compressed_tokens <= original_tokens (safety check; falls back to identity)
    - dictionary is always {}
    - SyntaxError on parse → identity, no exception raised
    - Deterministic
"""
from __future__ import annotations

import ast

import tiktoken

from shiori.compressors.base import CompressionResult

_ENCODING = tiktoken.get_encoding("cl100k_base")


def _count(text: str) -> int:
    return len(_ENCODING.encode(text))


def _identity(text: str, original_tokens: int) -> CompressionResult:
    return CompressionResult(
        original_text=text,
        compressed_text=text,
        original_tokens=original_tokens,
        compressed_tokens=original_tokens,
        content_tokens=original_tokens,
    )


class _DocstringRemover(ast.NodeTransformer):
    def _strip_docstring(self, node: ast.AST) -> ast.AST:
        if (
            node.body  # type: ignore[attr-defined]
            and isinstance(node.body[0], ast.Expr)  # type: ignore[attr-defined]
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            node.body = node.body[1:]  # type: ignore[attr-defined]
            if not node.body:  # type: ignore[attr-defined]
                node.body = [ast.Pass()]  # type: ignore[attr-defined]
        return node

    def visit_Module(self, node: ast.Module) -> ast.Module:
        self.generic_visit(node)
        return self._strip_docstring(node)  # type: ignore[return-value]

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        self.generic_visit(node)
        return self._strip_docstring(node)  # type: ignore[return-value]

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
        self.generic_visit(node)
        return self._strip_docstring(node)  # type: ignore[return-value]

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        self.generic_visit(node)
        return self._strip_docstring(node)  # type: ignore[return-value]


class AstCompressor:
    name = "ast"

    def compress(self, text: str) -> CompressionResult:
        original_tokens = _count(text)

        try:
            tree = ast.parse(text)
        except SyntaxError:
            return _identity(text, original_tokens)

        _DocstringRemover().visit(tree)
        ast.fix_missing_locations(tree)

        try:
            compressed_text = ast.unparse(tree)
        except Exception:
            return _identity(text, original_tokens)

        compressed_tokens = _count(compressed_text)
        if compressed_tokens >= original_tokens:
            return _identity(text, original_tokens)

        return CompressionResult(
            original_text=text,
            compressed_text=compressed_text,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            content_tokens=compressed_tokens,
        )
