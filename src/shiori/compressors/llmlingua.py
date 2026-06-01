"""
LLMLinguaCompressor — lossy token-level compression via LLMLingua-2.

Requires: pip install llmlingua
Model: microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank (~700 MB)

This compressor drops tokens permanently — no dictionary is produced.
The system prompt is NOT modified. Only the user prompt is compressed.

Failure modes:
  - llmlingua not installed: raises ImportError
  - rate too aggressive: key tokens may be dropped (caller's responsibility to use
    safe routing — llmlingua should never be used for retrieval or QA tasks)
"""
from __future__ import annotations

import tiktoken

from shiori.compressors.base import CompressionResult

_ENCODING = tiktoken.get_encoding("cl100k_base")

_COMPRESSOR = None


def _get_compressor():
    global _COMPRESSOR
    if _COMPRESSOR is None:
        from llmlingua import PromptCompressor
        _COMPRESSOR = PromptCompressor(
            model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
            use_llmlingua2=True,
            device_map="cpu",
        )
    return _COMPRESSOR


def _count(text: str) -> int:
    return len(_ENCODING.encode(text))


class LLMLinguaCompressor:
    name = "llmlingua"

    def __init__(self, rate: float = 0.5, question_hint: str = "") -> None:
        self.rate = rate
        self.question_hint = question_hint

    def compress(self, text: str) -> CompressionResult:
        original_tokens = _count(text)
        compressor = _get_compressor()

        question = self.question_hint or "Answer the question."
        result = compressor.compress_prompt(
            text,
            rate=self.rate,
            question=question,
            use_sentence_level_filter=False,
            condition_in_question="none",
            reorder_context="sort",
            dynamic_context_compression_ratio=0.3,
            condition_compare=True,
            target_token=2000,
        )
        compressed_text = result["compressed_prompt"]
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
