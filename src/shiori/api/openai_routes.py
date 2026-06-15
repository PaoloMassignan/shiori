"""
OpenAI-compatible chat completions endpoint.

POST /v1/chat/completions

Workflow:
  1. Compress role=tool messages (JSON-first, lossless fallback)
  2. Extract the last user message as the text to route/compress
  3. Router → RoutingDecision (strategy, reason, via)
  4. Apply compressor to get CompressionResult
  5. If strategy=llmlingua and payload has tools: apply CCR (store original, inject tool)
  6. Forward modified messages + dictionary to provider
  7. Return provider response (OpenAI-compatible)
  8. Record telemetry
"""
from __future__ import annotations

import time
import uuid
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from shiori.router import route
from shiori.metrics.telemetry import RequestRecord
from shiori.ccr.store import CcrStore

_CCR_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "shiori_retrieve",
        "description": (
            "Retrieve the full original content that was lossily compressed by Shiori. "
            "The [SHIORI_CCR id=...] marker in the context shows what was compressed. "
            "Call this when you need complete details that may have been omitted."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "content_id": {
                    "type": "string",
                    "description": "The content ID from the [SHIORI_CCR id=...] marker.",
                }
            },
            "required": ["content_id"],
        },
    },
}


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


def _compress_tool_messages(
    messages: list[dict], json_comp, ast_comp, lossless_comp
) -> tuple[list[dict], int, int]:
    """Compress role=tool messages: JSON → AST → lossless fallback chain.

    Returns (modified_messages, tool_original_tokens, tool_compressed_tokens).
    """
    result = list(messages)
    tool_original = 0
    tool_compressed = 0
    for i, msg in enumerate(result):
        if msg.get("role") != "tool":
            continue
        content = msg.get("content", "")
        if not isinstance(content, str) or not content:
            continue
        compression = json_comp.compress(content)
        if not compression.changed:
            compression = ast_comp.compress(content)
        if not compression.changed:
            compression = lossless_comp.compress(content)
        tool_original += compression.original_tokens
        tool_compressed += compression.compressed_tokens
        if compression.changed:
            result[i] = {**msg, "content": compression.compressed_text}
    return result, tool_original, tool_compressed


def _replace_last_user_message(messages: list[dict], new_content: str) -> list[dict]:
    result = list(messages)
    for i in reversed(range(len(result))):
        if result[i].get("role") == "user":
            result[i] = {**result[i], "content": new_content}
            return result
    return result


def build_router(cfg, compressor_factory, provider_factory, telemetry, ccr_store: CcrStore) -> APIRouter:
    router = APIRouter()

    @router.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        payload: dict[str, Any] = await request.json()
        messages = payload.get("messages", [])
        model = payload.get("model", "unknown")

        messages, tool_original_tokens, tool_compressed_tokens = _compress_tool_messages(
            messages,
            json_comp=compressor_factory("json"),
            ast_comp=compressor_factory("ast"),
            lossless_comp=compressor_factory("lossless"),
        )

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

        compressed_text = result.compressed_text
        forward_tools = payload.get("tools")

        if decision.strategy == "llmlingua" and result.changed and forward_tools is not None:
            content_id = ccr_store.put(result.original_text)
            compressed_text = f"[SHIORI_CCR id={content_id}]\n{result.compressed_text}"
            forward_tools = list(forward_tools) + [_CCR_TOOL]

        if result.changed:
            modified_messages = _replace_last_user_message(messages, compressed_text)
        else:
            modified_messages = messages

        forward_payload = {**payload, "messages": modified_messages}
        if forward_tools is not None:
            forward_payload = {**forward_payload, "tools": forward_tools}

        provider = provider_factory()

        if payload.get("stream") is True:
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
                tool_original_tokens=tool_original_tokens,
                tool_compressed_tokens=tool_compressed_tokens,
                compression_latency_ms=compression_ms,
                provider_latency_ms=0.0,
            ))
            return StreamingResponse(
                provider.stream_chat_completions(forward_payload, result.dictionary),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache"},
            )

        t1 = time.perf_counter()
        response_data = await provider.chat_completions(forward_payload, result.dictionary)
        provider_ms = (time.perf_counter() - t1) * 1000

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
            tool_original_tokens=tool_original_tokens,
            tool_compressed_tokens=tool_compressed_tokens,
            compression_latency_ms=compression_ms,
            provider_latency_ms=provider_ms,
        ))

        if cfg.mode == "debug":
            response_data["shiori"] = {
                "strategy": decision.strategy,
                "via": decision.via,
                "reason": decision.reason,
                "original_tokens": result.original_tokens,
                "compressed_tokens": result.compressed_tokens,
                "token_saving_ratio": result.token_saving_ratio,
            }

        return JSONResponse(content=response_data)

    return router
