"""
CavemanCompressor — lossless removal of articles and weak discourse connectives.

Operates only on the context section of the prompt (before [QUESTION]).
Never modifies the question, answer format, or instructions.

Invariant: compressed_tokens <= original_tokens (falls back to identity otherwise).
"""
from __future__ import annotations

import re

import tiktoken

from shiori.compressors.base import CompressionResult

_ENCODING = tiktoken.get_encoding("cl100k_base")

_ARTICLE_RE = re.compile(r"\b(a|an|the)\s+", re.IGNORECASE)
_CONNECTIVE_RE = re.compile(
    r"^[ \t]*(additionally|furthermore|moreover|consequently|nevertheless|"
    r"therefore|thus|hence|also)\s*,?\s+",
    re.IGNORECASE | re.MULTILINE,
)


def _count(text: str) -> int:
    return len(_ENCODING.encode(text))


def _context_boundary(text: str) -> int:
    m = re.search(r"\[QUESTION\]", text, re.IGNORECASE)
    return m.start() if m else len(text)


class CavemanCompressor:
    name = "caveman"

    def compress(self, text: str) -> CompressionResult:
        original_tokens = _count(text)
        boundary = _context_boundary(text)
        context, tail = text[:boundary], text[boundary:]

        out = _ARTICLE_RE.sub("", context)
        out = _CONNECTIVE_RE.sub("", out)
        out = re.sub(r"  +", " ", out)
        compressed_text = out + tail
        compressed_tokens = _count(compressed_text)

        if compressed_tokens >= original_tokens:
            return CompressionResult(
                original_text=text,
                compressed_text=text,
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                content_tokens=original_tokens,
            )
        return CompressionResult(
            original_text=text,
            compressed_text=compressed_text,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            content_tokens=compressed_tokens,
        )
