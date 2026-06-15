"""
Fixed benchmark corpus for Shiori compression benchmarks.

Each entry is a realistic sample of a content type seen in agentic workloads.
Samples are embedded as constants so benchmarks are reproducible with no
external files or API calls.
"""
from __future__ import annotations

import json

# ---------------------------------------------------------------------------
# JSON array — simulated RAG / search tool output
# ---------------------------------------------------------------------------

_SEARCH_ROWS = [
    {
        "doc_id": f"doc_{i:04d}",
        "title": f"Understanding transformer architectures — part {i}",
        "score": round(0.97 - i * 0.02, 4),
        "source": "knowledge_base",
        "language": "en",
        "content_type": "article",
        "author": f"Author {i % 5}",
        "published_at": f"2024-{(i % 12) + 1:02d}-15",
        "content": (
            f"Transformers were introduced in 'Attention Is All You Need' (Vaswani et al., 2017). "
            f"This article (part {i}) covers attention mechanisms, positional encodings, and "
            f"the encoder-decoder structure used in modern LLMs."
        ),
        "tags": ["ml", "transformers", "llm"],
        "metadata": None,
        "deleted_at": None,
        "flagged": False,
        "internal_score": None,
        "review_notes": "",
        "extra": {},
    }
    for i in range(18)
]

JSON_ARRAY = json.dumps(_SEARCH_ROWS, indent=2)

# ---------------------------------------------------------------------------
# Log lines — typical application log block
# ---------------------------------------------------------------------------

LOG_LINES = """\
2024-03-15 08:00:01.234 INFO  [main] dfs.DataNode$PacketResponder: Received block blk_1073741825 of size 67108864 from /10.0.0.12
2024-03-15 08:00:01.891 INFO  [main] dfs.DataNode$PacketResponder: Received block blk_1073741826 of size 67108864 from /10.0.0.12
2024-03-15 08:00:02.103 INFO  [main] dfs.DataNode$PacketResponder: Received block blk_1073741827 of size 67108864 from /10.0.0.13
2024-03-15 08:00:02.441 INFO  [main] dfs.DataNode$PacketResponder: Received block blk_1073741828 of size 67108864 from /10.0.0.13
2024-03-15 08:00:03.012 WARN  [main] dfs.DataNode$PacketResponder: Slow write for block blk_1073741829 — 350ms
2024-03-15 08:00:03.219 INFO  [main] dfs.DataNode$PacketResponder: Received block blk_1073741829 of size 67108864 from /10.0.0.14
2024-03-15 08:00:03.887 INFO  [main] dfs.DataNode$PacketResponder: Received block blk_1073741830 of size 67108864 from /10.0.0.14
2024-03-15 08:00:04.334 INFO  [main] dfs.DataNode$PacketResponder: Received block blk_1073741831 of size 67108864 from /10.0.0.12
2024-03-15 08:00:04.901 ERROR [main] dfs.DataNode$PacketResponder: Exception for block blk_1073741832: java.io.IOException: Connection reset by peer
2024-03-15 08:00:05.112 INFO  [main] dfs.DataNode$PacketResponder: Received block blk_1073741833 of size 67108864 from /10.0.0.15
2024-03-15 08:00:05.778 INFO  [main] dfs.DataNode$PacketResponder: Received block blk_1073741834 of size 67108864 from /10.0.0.15
2024-03-15 08:00:06.023 INFO  [main] dfs.DataNode$PacketResponder: Received block blk_1073741835 of size 67108864 from /10.0.0.12
2024-03-15 08:00:06.445 WARN  [main] dfs.DataNode$PacketResponder: Slow write for block blk_1073741836 — 412ms
2024-03-15 08:00:06.891 INFO  [main] dfs.DataNode$PacketResponder: Received block blk_1073741836 of size 67108864 from /10.0.0.13
2024-03-15 08:00:07.234 INFO  [main] dfs.DataNode$PacketResponder: Received block blk_1073741837 of size 67108864 from /10.0.0.16
2024-03-15 08:00:07.779 INFO  [main] dfs.DataNode$PacketResponder: Received block blk_1073741838 of size 67108864 from /10.0.0.16
2024-03-15 08:00:08.102 INFO  [main] dfs.DataNode$PacketResponder: Received block blk_1073741839 of size 67108864 from /10.0.0.14
2024-03-15 08:00:08.567 INFO  [main] dfs.DataNode$PacketResponder: Received block blk_1073741840 of size 67108864 from /10.0.0.14
2024-03-15 08:00:09.003 ERROR [main] dfs.DataNode$PacketResponder: Exception for block blk_1073741841: java.io.IOException: Broken pipe
2024-03-15 08:00:09.441 INFO  [main] dfs.DataNode$PacketResponder: Received block blk_1073741841 of size 67108864 from /10.0.0.12
"""

# ---------------------------------------------------------------------------
# Stack trace — Python traceback
# ---------------------------------------------------------------------------

STACK_TRACE = """\
Traceback (most recent call last):
  File "/app/shiori/api/openai_routes.py", line 72, in chat_completions
    response_data = await provider.chat_completions(forward_payload, result.dictionary)
  File "/app/shiori/providers/openai.py", line 41, in chat_completions
    response = await self._client.post(self._url, json=payload, headers=headers)
  File "/usr/local/lib/python3.12/site-packages/httpx/_client.py", line 1574, in post
    return await self.request("POST", url, content=content, data=data, json=json, headers=headers)
  File "/usr/local/lib/python3.12/site-packages/httpx/_client.py", line 1521, in request
    return await self.send(request, auth=auth, follow_redirects=follow_redirects)
  File "/usr/local/lib/python3.12/site-packages/httpx/_client.py", line 1618, in send
    response = await self._send_with_response(request, auth, history)
  File "/usr/local/lib/python3.12/site-packages/httpx/_client.py", line 1654, in _send_with_response
    response = await transport.handle_async_request(request)
  File "/usr/local/lib/python3.12/site-packages/httpx/_transports/default.py", line 372, in handle_async_request
    with map_httpcore_exceptions():
  File "/usr/local/lib/python3.12/contextlib.py", line 158, in __exit__
    self.gen.throw(value)
  File "/usr/local/lib/python3.12/site-packages/httpx/_transports/default.py", line 86, in map_httpcore_exceptions
    raise mapped_exc(message) from exc
httpx.ConnectTimeout: timed out while connecting to host
"""

# ---------------------------------------------------------------------------
# Python code — a realistic module with type hints and docstrings
# ---------------------------------------------------------------------------

PYTHON_CODE = '''\
"""
Cache-aware request batcher for the Shiori metrics pipeline.

Collects RequestRecord objects and flushes them in configurable batches
to avoid per-request I/O overhead on high-throughput deployments.
"""
from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

from shiori.metrics.telemetry import RequestRecord


@dataclass
class BatcherConfig:
    """Configuration for the RequestBatcher."""
    max_batch_size: int = 128
    flush_interval_seconds: float = 5.0
    max_queue_size: int = 10_000


class RequestBatcher:
    """Accumulate RequestRecords and flush them in batches.

    Parameters
    ----------
    config:
        Batcher configuration.
    flush_fn:
        Callable that receives a list of RequestRecord and persists them.
        Called on the event loop that owns this batcher.
    """

    def __init__(
        self,
        config: BatcherConfig,
        flush_fn: Callable[[list[RequestRecord]], None],
    ) -> None:
        self._config = config
        self._flush_fn = flush_fn
        self._queue: deque[RequestRecord] = deque()
        self._last_flush: float = time.monotonic()
        self._lock = asyncio.Lock()

    async def add(self, record: RequestRecord) -> None:
        """Enqueue a record; flush if batch size or interval threshold is reached."""
        async with self._lock:
            if len(self._queue) >= self._config.max_queue_size:
                self._queue.popleft()
            self._queue.append(record)
            should_flush = (
                len(self._queue) >= self._config.max_batch_size
                or (time.monotonic() - self._last_flush) >= self._config.flush_interval_seconds
            )
        if should_flush:
            await self.flush()

    async def flush(self) -> None:
        """Drain the queue and call flush_fn with all accumulated records."""
        async with self._lock:
            if not self._queue:
                return
            batch = list(self._queue)
            self._queue.clear()
            self._last_flush = time.monotonic()
        self._flush_fn(batch)

    async def close(self) -> None:
        """Flush remaining records on shutdown."""
        await self.flush()
'''

# ---------------------------------------------------------------------------
# Plain prose — narrative document chunk (RAG-style)
# ---------------------------------------------------------------------------

PLAIN_PROSE = """\
The attention mechanism is the cornerstone of the transformer architecture introduced by
Vaswani et al. in 2017. Unlike recurrent neural networks, which process tokens sequentially
and therefore struggle to capture long-range dependencies, the transformer computes
relationships between all token pairs simultaneously via scaled dot-product attention.

In the self-attention layer, each token generates three vectors — a query, a key, and a
value — through learned linear projections. The attention score between any two tokens is
computed as the dot product of their query and key vectors, scaled by the square root of
the key dimension to prevent gradient instability in deep networks. A softmax function
normalises these scores into a probability distribution, which is then used to compute a
weighted sum of the value vectors.

Multi-head attention extends this idea by running several attention operations in parallel,
each with independently learned projections. The outputs are concatenated and projected
back to the model dimension. This allows the model to simultaneously attend to information
from different representation subspaces — for example, one head might track syntactic
dependencies while another captures semantic similarity.

Positional encodings are added to the input embeddings to inject information about token
order, since the attention mechanism itself is permutation-invariant. The original paper
used fixed sinusoidal encodings, but modern architectures increasingly favour learned
absolute positions or relative position encodings such as RoPE (Rotary Position Embedding),
which has shown strong performance across a wide range of sequence lengths.

The feed-forward sublayer that follows each attention block applies two linear
transformations with a non-linear activation (typically GELU or SwiGLU) between them.
This sublayer operates independently on each position, functioning as a key-value memory
store for factual knowledge, as demonstrated by Geva et al. in their 2021 analysis of
transformer feed-forward layers.
"""

# ---------------------------------------------------------------------------
# Public mapping
# ---------------------------------------------------------------------------

CORPUS: dict[str, str] = {
    "json_array": JSON_ARRAY,
    "log_lines": LOG_LINES,
    "stack_trace": STACK_TRACE,
    "python_code": PYTHON_CODE,
    "plain_prose": PLAIN_PROSE,
}
