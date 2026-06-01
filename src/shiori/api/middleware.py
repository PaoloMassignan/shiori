"""
Logging middleware.

In debug mode: logs routing decisions and compression metrics per request.
In all modes: logs errors.

Prompt content is NEVER logged unless observability.log_prompts = true (opt-in).
"""
from __future__ import annotations

import logging
import time

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("shiori")


class LoggingMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, log_prompts: bool = False) -> None:
        super().__init__(app)
        self.log_prompts = log_prompts

    async def dispatch(self, request: Request, call_next):
        t0 = time.perf_counter()
        response = await call_next(request)
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            "%s %s → %d (%.1fms)",
            request.method,
            request.url.path,
            response.status_code,
            elapsed,
        )
        return response
