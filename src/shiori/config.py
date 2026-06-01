from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ProviderConfig:
    name: str = "openai"
    base_url: str = "https://api.openai.com/v1"
    api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))


@dataclass
class RouterConfig:
    ml_model_path: str = "models/shiori_router"


@dataclass
class CompressorConfig:
    llmlingua_rate: float = 0.5
    min_occurrences: int = 2
    min_chars: int = 12
    preserve_numbers: bool = True
    preserve_dates: bool = True
    preserve_quoted_strings: bool = True
    preserve_urls: bool = True
    protect_question_terms: bool = True


@dataclass
class ObservabilityConfig:
    log_prompts: bool = False
    telemetry: bool = True


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000


@dataclass
class ShioriConfig:
    """
    Top-level configuration for the Shiori proxy.

    mode:
      safe       - lossless or none only (never lossy)
      aggressive - full routing: lossless or llmlingua based on task type
      off        - pass-through, no compression
      debug      - aggressive + extended logging of routing decisions
    """
    mode: str = "safe"
    provider: ProviderConfig = field(default_factory=ProviderConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    compressors: CompressorConfig = field(default_factory=CompressorConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    server: ServerConfig = field(default_factory=ServerConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ShioriConfig":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(
            mode=os.getenv("SHIORI_MODE", data.get("mode", "safe")),
            provider=ProviderConfig(**data.get("provider", {})),
            router=RouterConfig(**data.get("router", {})),
            compressors=CompressorConfig(**data.get("compressors", {})),
            observability=ObservabilityConfig(**data.get("observability", {})),
            server=ServerConfig(**data.get("server", {})),
        )

    @classmethod
    def defaults(cls) -> "ShioriConfig":
        return cls()
