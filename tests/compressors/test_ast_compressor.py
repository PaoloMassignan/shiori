"""
Spec: AstCompressor

Purpose: Compress Python source code by removing docstrings and comments
         via AST parsing, without touching executable statements or type
         annotations.

Invariants:
  - compressed_tokens <= original_tokens always
  - dictionary is always {}
  - Non-Python input is returned unchanged without raising
  - Deterministic
  - Type annotations are preserved
  - Function/class signatures are preserved
"""
from shiori.compressors.ast_compressor import AstCompressor

_CODE_WITH_DOCSTRINGS = '''\
"""Module docstring."""

def add(x: int, y: int) -> int:
    """Add two numbers and return the result."""
    return x + y


class Calculator:
    """A simple calculator class."""

    def multiply(self, a: float, b: float) -> float:
        """Multiply a by b."""
        return a * b
'''

_CODE_WITH_COMMENTS = '''\
# This is a module-level comment
import os  # inline comment

def greet(name: str) -> str:
    # Build the greeting string
    return f"Hello, {name}"
'''

_CODE_WITH_BOTH = '''\
"""Module-level docstring that explains the purpose of this module."""

# Standard library imports
import json
import time
from dataclasses import dataclass


@dataclass
class Config:
    """Configuration dataclass for the application."""

    host: str = "localhost"  # server host
    port: int = 8080         # server port
    debug: bool = False

    def url(self) -> str:
        """Return the base URL."""
        # Build and return the URL
        return f"http://{self.host}:{self.port}"
'''

_PLAIN_TEXT = "The quick brown fox jumps over the lazy dog."
_PARTIAL_PYTHON = "def broken(:"


# ---------------------------------------------------------------------------
# Docstring removal
# ---------------------------------------------------------------------------

def test_removes_module_docstring():
    result = AstCompressor().compress(_CODE_WITH_DOCSTRINGS)
    assert "Module docstring" not in result.compressed_text


def test_removes_function_docstring():
    result = AstCompressor().compress(_CODE_WITH_DOCSTRINGS)
    assert "Add two numbers" not in result.compressed_text


def test_removes_class_docstring():
    result = AstCompressor().compress(_CODE_WITH_DOCSTRINGS)
    assert "simple calculator" not in result.compressed_text


def test_removes_method_docstring():
    result = AstCompressor().compress(_CODE_WITH_DOCSTRINGS)
    assert "Multiply a by b" not in result.compressed_text


# ---------------------------------------------------------------------------
# Comment removal
# ---------------------------------------------------------------------------

def test_removes_line_comments():
    result = AstCompressor().compress(_CODE_WITH_COMMENTS)
    assert "module-level comment" not in result.compressed_text
    assert "inline comment" not in result.compressed_text
    assert "Build the greeting" not in result.compressed_text


# ---------------------------------------------------------------------------
# Preservation of executable content
# ---------------------------------------------------------------------------

def test_preserves_function_signatures():
    result = AstCompressor().compress(_CODE_WITH_DOCSTRINGS)
    assert "def add" in result.compressed_text
    assert "def multiply" in result.compressed_text


def test_preserves_type_annotations():
    result = AstCompressor().compress(_CODE_WITH_DOCSTRINGS)
    assert "int" in result.compressed_text
    assert "float" in result.compressed_text


def test_preserves_class_definition():
    result = AstCompressor().compress(_CODE_WITH_DOCSTRINGS)
    assert "class Calculator" in result.compressed_text


def test_preserves_return_statements():
    result = AstCompressor().compress(_CODE_WITH_DOCSTRINGS)
    assert "return x + y" in result.compressed_text


def test_preserves_imports():
    result = AstCompressor().compress(_CODE_WITH_COMMENTS)
    assert "import os" in result.compressed_text


# ---------------------------------------------------------------------------
# Non-Python passthrough
# ---------------------------------------------------------------------------

def test_plain_text_returns_unchanged():
    result = AstCompressor().compress(_PLAIN_TEXT)
    assert result.compressed_text == _PLAIN_TEXT
    assert result.changed is False


def test_syntax_error_returns_unchanged():
    result = AstCompressor().compress(_PARTIAL_PYTHON)
    assert result.compressed_text == _PARTIAL_PYTHON
    assert result.changed is False


# ---------------------------------------------------------------------------
# Protocol invariants
# ---------------------------------------------------------------------------

def test_safety_never_expands():
    result = AstCompressor().compress(_CODE_WITH_BOTH)
    assert result.compressed_tokens <= result.original_tokens


def test_significant_saving_on_documented_code():
    result = AstCompressor().compress(_CODE_WITH_BOTH)
    assert result.token_saving_ratio >= 0.25


def test_dictionary_always_empty():
    result = AstCompressor().compress(_CODE_WITH_DOCSTRINGS)
    assert result.dictionary == {}


def test_deterministic():
    r1 = AstCompressor().compress(_CODE_WITH_BOTH)
    r2 = AstCompressor().compress(_CODE_WITH_BOTH)
    assert r1.compressed_text == r2.compressed_text


def test_name():
    assert AstCompressor().name == "ast"


def test_tiny_input_does_not_raise():
    AstCompressor().compress("")
    AstCompressor().compress("x = 1")
