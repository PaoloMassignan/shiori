"""
DictionaryCompressor — replace repeated long phrases with §A §B symbols.

Symbol table is passed to the LLM in the system prompt for reconstruction.
Only phrases that appear >= min_occurrences times AND that produce a net token
saving (phrase_tokens * count > symbol_tokens * count + dict_line_tokens) are
substituted.

Invariant: compressed_tokens <= original_tokens (falls back to identity otherwise).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import tiktoken

from shiori.compressors.base import CompressionResult

_ENCODING = tiktoken.get_encoding("cl100k_base")
_SYMBOL_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

_NUMBER_RE = re.compile(r"\b\d[\d,. ]*\d\b|\b\d\b")
_DATE_RE = re.compile(
    r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{2}[/-]\d{2}|"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4})\b",
    re.IGNORECASE,
)
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_QUOTED_RE = re.compile(r'"[^"]{2,}"')
_QUESTION_STOPWORDS = {
    "the", "a", "an", "of", "in", "is", "was", "are", "were", "has", "have",
    "had", "for", "to", "and", "or", "but", "what", "which", "who", "how",
    "when", "where", "does", "did", "do", "not", "its", "it", "this", "that",
}


def _count(text: str) -> int:
    return len(_ENCODING.encode(text))


def _make_symbol(index: int) -> str:
    result = ""
    index += 1
    while index > 0:
        index, r = divmod(index - 1, 26)
        result = _SYMBOL_CHARS[r] + result
    return f"§{result}"


def _protected_spans(text: str, min_chars: int, protect_question: bool) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for pattern in (_NUMBER_RE, _DATE_RE, _URL_RE, _QUOTED_RE):
        spans += [m.span() for m in pattern.finditer(text)]
    if protect_question:
        q = re.search(r"\[QUESTION\](.*?)(?:\[|$)", text, re.DOTALL | re.IGNORECASE)
        if q:
            for word in re.findall(r"\b[A-Za-z]\w*\b", q.group(1)):
                if len(word) >= min_chars and word.lower() not in _QUESTION_STOPWORDS:
                    for m in re.finditer(r"\b" + re.escape(word) + r"\b", text, re.IGNORECASE):
                        spans.append(m.span())
    return spans


def _overlaps(start: int, end: int, protected: list[tuple[int, int]]) -> bool:
    return any(start < pe and end > ps for ps, pe in protected)


class DictionaryCompressor:
    name = "dictionary"

    def __init__(
        self,
        min_occurrences: int = 2,
        min_chars: int = 12,
        protect_question_terms: bool = True,
    ) -> None:
        self.min_occurrences = min_occurrences
        self.min_chars = min_chars
        self.protect_question_terms = protect_question_terms

    def compress(self, text: str) -> CompressionResult:
        original_tokens = _count(text)
        protected = _protected_spans(text, self.min_chars, self.protect_question_terms)

        words = re.split(r"(\s+)", text)
        positions: list[tuple[str, int]] = []
        pos = 0
        for token in words:
            positions.append((token, pos))
            pos += len(token)
        word_tokens = [(w, p) for w, p in positions if w.strip()]

        counts: dict[str, int] = {}
        for si in range(len(word_tokens)):
            phrase = ""
            for ei in range(si, min(si + 20, len(word_tokens))):
                w, wpos = word_tokens[ei]
                phrase = (phrase + " " + w) if phrase else w
                if len(phrase) < self.min_chars:
                    continue
                phrase_end = wpos + len(w)
                if _overlaps(word_tokens[si][1], phrase_end, protected):
                    break
                counts[phrase] = counts.get(phrase, 0) + 1

        candidates = [(p, c) for p, c in counts.items() if c >= self.min_occurrences]

        sample_sym = _make_symbol(0)
        sample_sym_tokens = _count(sample_sym)

        scored = []
        for phrase, count in candidates:
            pt = _count(phrase)
            dict_cost = _count(f"{sample_sym} = {phrase}")
            net = count * (pt - sample_sym_tokens) - dict_cost
            if net > 0:
                scored.append((net, phrase, count))
        scored.sort(key=lambda x: (x[0], len(x[1])), reverse=True)

        dictionary: dict[str, str] = {}
        idx = 0
        working = text
        for _, phrase, _ in scored:
            if working.count(phrase) < self.min_occurrences:
                continue
            sym = _make_symbol(idx)
            idx += 1
            dictionary[sym] = phrase
            working = working.replace(phrase, sym)

        if not dictionary:
            return CompressionResult(
                original_text=text,
                compressed_text=text,
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                content_tokens=original_tokens,
            )

        header = ["[LZP_DICTIONARY]"]
        header += [f"{s} = {p}" for s, p in dictionary.items()]
        header += ["", "[LZP_CONTENT]"]
        compressed_text = "\n".join(header) + "\n" + working
        compressed_tokens = _count(compressed_text)
        content_tokens = _count(working)
        dict_tokens = compressed_tokens - content_tokens

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
            dictionary=dictionary,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            dictionary_tokens=dict_tokens,
            content_tokens=content_tokens,
        )
