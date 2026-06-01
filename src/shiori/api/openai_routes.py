"""
OpenAI-compatible chat completions endpoint.

POST /v1/chat/completions

Workflow:
  1. Extract the last user message as the text to route/compress
  2. Router → RoutingDecision (strategy, reason, via)
  3. Apply compressor to get CompressionResult
  4. Forward modified messages + dictionary to provider
  5. Return provider response (OpenAI-compatible)
  6. Record telemetry
"""
from __future__ import annotations

import time
import uuid
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from shiori.router import route
from shiori.metrics.telemetry import RequestRecord


def _extract_user_text(messages: list[dict]) -> str:
    """Return the last user message content for routing and compression."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return " ".join(
                    part.get("text", "") for part in content if part.get("type") == "text"
                )
    return ""


def _replace_last_user_message(messages: list[dict], new_content: str) -> list[dict]:
    result = list(messages)
    for i in reversed(range(len(result))):
        if result[i].get("role") == "user":
            result[i] = {**result[i], "content": new_content}
            return result
    return result


def build_router(cfg, compressor_factory, provider_factory, telemetry) -> APIRouter:
    router = APIRouter()

    @router.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> JSONResponse:
        payload: dict[str, Any] = await request.json()
        messages = payload.get("messages", [])
        model = payload.get("model", "unknown")

        user_text = _extract_user_text(messages)

        t0 = time.perf_counter()
        decision = route(
            user_text,
            proxy_mode=cfg.mode if cfg.mode != "debug" else "aggressive",
            ml_model_path=cfg.router.ml_model_path,
        )
        compressor = compressor_factory(decision.strategy)
        result = compressor.compress(user_text)
        compression_ms = (time.perf_counter() - t0) * 1000

        if result.changed:
            modified_messages = _replace_last_user_message(messages, result.compressed_text)
        else:
            modified_messages = messages

        forward_payload = {**payload, "messages": modified_messages}

        t1 = time.perf_counter()
        provider = provider_factory()
        response_data = await provider.chat_completions(forward_payload, result.dictionary)
        provider_ms = (time.perf_counter() - t1) * 1000

        if cfg.mode == "debug":
            response_data["shiori"] = {
                "strategy": decision.strategy,
                "via": decision.via,
                "reason": decision.reason,
                "original_tokens": result.original_tokens,
                "compressed_tokens": result.compressed_tokens,
                "token_saving_ratio": result.token_saving_ratio,
            }

        telemetry.record(RequestRecord(
            request_id=str(uuid.uuid4()),
            model=model,
            provider=cfg.provider.name,
            strategy=decision.strategy,
            routing_via=decision.via,
            routing_reason=decision.reason,
            original_tokens=result.original_tokens,
            compressed_tokens=result.compressed_tokens,
            dictionary_tokens=result.dictionary_tokens,
            token_saving_ratio=result.token_saving_ratio,
            compression_latency_ms=compression_ms,
            provider_latency_ms=provider_ms,
        ))

        return JSONResponse(content=response_data)

    return router
