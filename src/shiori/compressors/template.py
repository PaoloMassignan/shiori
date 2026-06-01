"""
TemplateCompressor — LZ-style compression for structured line-based text.

Groups lines by whitespace-token count, identifies fixed vs variable token
positions, and emits:
  [T] fixed_token <*> fixed_token ...
  var1|var2|...

Identical lines are collapsed to [xN] <line>.

Lossless: the LLM reconstructs each original line by slot-filling <*>
with the pipe-separated values in each row.

Falls back to identity when no compressible pattern is found or when the
compressed result is not shorter than the original.
"""
from __future__ import annotations

import re

import tiktoken

from shiori.compressors.base import CompressionResult

_ENCODING = tiktoken.get_encoding("cl100k_base")
_VAR = "<*>"
_SEP = "|"
_TPL_MARKER = "[T]"
_CONTEXT_RE = re.compile(
    r"(\[CONTEXT\])(.*?)(?=\[QUESTION\]|\[ANSWER_FORMAT\]|$)", re.DOTALL
)


def _count(text: str) -> int:
    return len(_ENCODING.encode(text))


def _group_by_count(
    lines: list[str], min_size: int
) -> dict[int, list[tuple[int, list[str]]]]:
    groups: dict[int, list[tuple[int, list[str]]]] = {}
    for i, line in enumerate(lines):
        toks = line.split()
        if not toks:
            continue
        groups.setdefault(len(toks), []).append((i, toks))
    return {n: g for n, g in groups.items() if len(g) >= min_size}


def _build_block(group: list[tuple[int, list[str]]]) -> tuple[str, list[int]]:
    indices = [i for i, _ in group]
    all_toks = [t for _, t in group]
    n = len(all_toks[0])
    var_positions = [p for p in range(n) if len({t[p] for t in all_toks}) > 1]

    if not var_positions:
        line = " ".join(all_toks[0])
        block = f"[x{len(group)}] {line}" if len(group) > 1 else line
        return block, indices

    template = " ".join(
        _VAR if p in var_positions else all_toks[0][p] for p in range(n)
    )
    rows = [_SEP.join(toks[p] for p in var_positions) for toks in all_toks]
    return f"{_TPL_MARKER} {template}\n" + "\n".join(rows), indices


class TemplateCompressor:
    name = "template"

    def __init__(self, min_group_size: int = 3) -> None:
        self.min_group_size = min_group_size

    def compress(self, text: str) -> CompressionResult:
        original_tokens = _count(text)

        ctx_match = _CONTEXT_RE.search(text)
        if ctx_match:
            prefix = text[: ctx_match.start(2)]
            body = ctx_match.group(2)
            suffix = text[ctx_match.end(2) :]
        else:
            prefix = suffix = ""
            body = text

        lines = [l for l in body.split("\n") if l.strip()]
        if len(lines) < self.min_group_size:
            return self._identity(text, original_tokens)

        groups = _group_by_count(lines, self.min_group_size)
        if not groups:
            return self._identity(text, original_tokens)

        replacement: dict[int, str] = {}
        suppressed: set[int] = set()
        for _n, group in sorted(groups.items(), key=lambda x: -len(x[1])):
            block, indices = _build_block(group)
            first = indices[0]
            if first not in suppressed:
                replacement[first] = block
                suppressed.update(indices[1:])

        parts = [replacement.get(i, line) for i, line in enumerate(lines) if i not in suppressed]
        compressed_ctx = "\n".join(parts)
        compressed_text = prefix + "\n" + compressed_ctx + "\n" + suffix
        compressed_tokens = _count(compressed_text)

        if compressed_tokens >= original_tokens:
            return self._identity(text, original_tokens)

        return CompressionResult(
            original_text=text,
            compressed_text=compressed_text,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            content_tokens=compressed_tokens,
        )

    def _identity(self, text: str, original_tokens: int) -> CompressionResult:
        return CompressionResult(
            original_text=text,
            compressed_text=text,
            original_tokens=original_tokens,
            compressed_tokens=original_tokens,
            content_tokens=original_tokens,
        )
