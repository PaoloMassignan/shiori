# ADR-008: Compression strategies are plug-ins

## Status
Accepted

## Context
The set of useful compression strategies is not fixed. LZPrompt research discovers new approaches (template compressor, adaptive compressor, future semantic compressors). Shiori must be able to add new strategies without restructuring the proxy.

## Decision
Each compression strategy implements the `Compressor` protocol: `name: str` and `compress(text: str) -> CompressionResult`. The router returns a strategy name string; the server's `compressor_factory` maps names to instances. Adding a new strategy requires adding a file in `src/shiori/compressors/` and a mapping in `compressor_factory`.

## Consequences
- New compressors can be added without changing the router or proxy logic
- The Compressor protocol is minimal — any class with `name` and `compress()` qualifies
- llmlingua is an optional dependency (ImportError is raised only when the strategy is selected, not at import time)
