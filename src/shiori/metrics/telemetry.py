"""
Telemetry — structured record of a single Shiori proxy request.

Collected for every request regardless of proxy mode.
Exposed via GET /metrics (aggregate counters) and structured JSON logs.

Prompt content is NEVER included unless observability.log_prompts = true (opt-in).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class RequestRecord:
    request_id: str
    model: str
    provider: str
    strategy: str
    routing_via: str
    routing_reason: str
    original_tokens: int
    compressed_tokens: int
    dictionary_tokens: int
    token_saving_ratio: float
    compression_latency_ms: float
    provider_latency_ms: float
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "model": self.model,
            "provider": self.provider,
            "strategy": self.strategy,
            "routing_via": self.routing_via,
            "routing_reason": self.routing_reason,
            "original_tokens": self.original_tokens,
            "compressed_tokens": self.compressed_tokens,
            "dictionary_tokens": self.dictionary_tokens,
            "token_saving_ratio": round(self.token_saving_ratio, 4),
            "compression_latency_ms": round(self.compression_latency_ms, 2),
            "provider_latency_ms": round(self.provider_latency_ms, 2),
            "timestamp": self.timestamp,
        }


class Telemetry:
    def __init__(self) -> None:
        self._records: list[RequestRecord] = []

    def record(self, r: RequestRecord) -> None:
        self._records.append(r)

    def summary(self) -> dict:
        if not self._records:
            return {"total_requests": 0}
        total = len(self._records)
        avg_saving = sum(r.token_saving_ratio for r in self._records) / total
        total_original = sum(r.original_tokens for r in self._records)
        total_compressed = sum(r.compressed_tokens for r in self._records)
        by_strategy: dict[str, int] = {}
        for r in self._records:
            by_strategy[r.strategy] = by_strategy.get(r.strategy, 0) + 1
        return {
            "total_requests": total,
            "avg_token_saving_ratio": round(avg_saving, 4),
            "total_original_tokens": total_original,
            "total_compressed_tokens": total_compressed,
            "by_strategy": by_strategy,
        }
