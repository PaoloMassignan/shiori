"""
LosslessCompressor — pipeline: template → caveman → dictionary.

Covers three orthogonal compression opportunities:
  1. template: structured line-based text (logs, records, tables)
  2. caveman: function-word removal (articles, discourse connectives)
  3. dictionary: exact phrase substitution for repeated strings

On narrative text, template is a no-op and the pipeline reduces to caveman+dictionary.
On structured log data, template extraction adds significant saving before caveman runs.

Invariant: if the final result is not shorter than the original, returns the original unchanged.
"""
from __future__ import annotations

import tiktoken

from shiori.compressors.base import CompressionResult
from shiori.compressors.caveman import CavemanCompressor
from shiori.compressors.dictionary import DictionaryCompressor
from shiori.compressors.template import TemplateCompressor

_ENCODING = tiktoken.get_encoding("cl100k_base")


def _count(text: str) -> int:
    return len(_ENCODING.encode(text))


class LosslessCompressor:
    name = "lossless"

    def __init__(
        self,
        min_occurrences: int = 2,
        min_chars: int = 12,
        llmlingua_rate: float = 0.5,
        protect_question_terms: bool = True,
        template_min_group_size: int = 3,
    ) -> None:
        self._template = TemplateCompressor(min_group_size=template_min_group_size)
        self._caveman = CavemanCompressor()
        self._dict = DictionaryCompressor(
            min_occurrences=min_occurrences,
            min_chars=min_chars,
            protect_question_terms=protect_question_terms,
        )

    def compress(self, text: str) -> CompressionResult:
        original_tokens = _count(text)

        step1 = self._template.compress(text)
        step2 = self._caveman.compress(step1.compressed_text)
        step3 = self._dict.compress(step2.compressed_text)

        final_text = step3.compressed_text
        final_tokens = step3.compressed_tokens

        if final_tokens >= original_tokens:
            return CompressionResult(
                original_text=text,
                compressed_text=text,
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                content_tokens=original_tokens,
            )

        return CompressionResult(
            original_text=text,
            compressed_text=final_text,
            dictionary=step3.dictionary,
            original_tokens=original_tokens,
            compressed_tokens=final_tokens,
            dictionary_tokens=step3.dictionary_tokens,
            content_tokens=step3.content_tokens,
        )
