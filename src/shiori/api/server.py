"""
Shiori proxy server.

Entrypoint: shiori (via pyproject.toml script) or python -m shiori.api.server

Starts a FastAPI/uvicorn server exposing:
  POST /v1/chat/completions  — OpenAI-compatible proxy
  GET  /health               — liveness probe
  GET  /metrics              — aggregate telemetry
"""
from __future__ import annotations

import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from shiori.config import ShioriConfig
from shiori.compressors.caveman import CavemanCompressor
from shiori.compressors.dictionary import DictionaryCompressor
from shiori.compressors.lossless import LosslessCompressor
from shiori.compressors.template import TemplateCompressor
from shiori.compressors.base import CompressionResult
from shiori.providers.openai import OpenAIProvider
from shiori.metrics.telemetry import Telemetry
from shiori.api.openai_routes import build_router


def _build_app(cfg: ShioriConfig) -> FastAPI:
    app = FastAPI(title="Shiori", version="0.1.0", docs_url="/docs")
    telemetry = Telemetry()

    def compressor_factory(strategy: str):
        if strategy == "none" or cfg.mode == "off":
            class _NoOp:
                name = "none"
                def compress(self, text):
                    import tiktoken
                    enc = tiktoken.get_encoding("cl100k_base")
                    t = len(enc.encode(text))
                    return CompressionResult(
                        original_text=text, compressed_text=text,
                        original_tokens=t, compressed_tokens=t, content_tokens=t,
                    )
            return _NoOp()
        if strategy == "lossless":
            return LosslessCompressor(
                min_occurrences=cfg.compressors.min_occurrences,
                min_chars=cfg.compressors.min_chars,
                protect_question_terms=cfg.compressors.protect_question_terms,
            )
        if strategy == "llmlingua":
            from shiori.compressors.llmlingua import LLMLinguaCompressor
            return LLMLinguaCompressor(rate=cfg.compressors.llmlingua_rate)
        if strategy == "caveman":
            return CavemanCompressor()
        if strategy == "template":
            return TemplateCompressor()
        if strategy == "dictionary":
            return DictionaryCompressor(
                min_occurrences=cfg.compressors.min_occurrences,
                min_chars=cfg.compressors.min_chars,
            )
        return LosslessCompressor()

    def provider_factory():
        api_key = cfg.provider.api_key or os.getenv("OPENAI_API_KEY", "")
        return OpenAIProvider(base_url=cfg.provider.base_url, api_key=api_key)

    api_router = build_router(cfg, compressor_factory, provider_factory, telemetry)
    app.include_router(api_router)

    @app.get("/health")
    async def health():
        return {"status": "ok", "mode": cfg.mode, "provider": cfg.provider.name}

    @app.get("/metrics")
    async def metrics():
        return JSONResponse(content=telemetry.summary())

    return app


def main():
    from dotenv import load_dotenv
    load_dotenv()

    config_path = os.getenv("SHIORI_CONFIG", "configs/shiori.yaml")
    if Path(config_path).exists():
        cfg = ShioriConfig.from_yaml(config_path)
    else:
        cfg = ShioriConfig.defaults()

    app = _build_app(cfg)
    uvicorn.run(
        app,
        host=cfg.server.host,
        port=int(os.getenv("SHIORI_PORT", cfg.server.port)),
    )


if __name__ == "__main__":
    main()
