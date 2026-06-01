"""
Base types for compressors.

CompressionResult is the return type of every compressor.
Compressor is the protocol every compressor must satisfy.

Invariants:
  - compressed_tokens <= original_tokens (safety check enforced by each compressor)
  - if dictionary is non-empty, the LLM must receive it alongside the compressed text
  - dictionary is empty for lossy compressors (llmlingua) — information is dropped permanently
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class CompressionResult:
    original_text: str
    compressed_text: str
    dictionary: dict[str, str] = field(default_factory=dict)
    original_tokens: int = 0
    compressed_tokens: int = 0
    dictionary_tokens: int = 0
    content_tokens: int = 0

    @property
    def token_saving_ratio(self) -> float:
        if self.original_tokens == 0:
            return 0.0
        effective = self.content_tokens + self.dictionary_tokens
        return max(0.0, 1.0 - effective / self.original_tokens)

    @property
    def changed(self) -> bool:
        return self.compressed_text != self.original_text


class Compressor(Protocol):
    """Contract every compression strategy must satisfy."""

    @property
    def name(self) -> str: ...

    def compress(self, text: str) -> CompressionResult: ...
