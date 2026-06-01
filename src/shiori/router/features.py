"""
Prompt feature extraction for the Shiori router.

PromptFeatures is a frozen dataclass — immutable, hashable, cheap to compute.
extract_features() is the single entry point; it never raises.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import tiktoken

_ENCODING = tiktoken.get_encoding("cl100k_base")

_DIFF_RE = re.compile(r"^(?:---|\+\+\+|@@|diff --git)\s", re.MULTILINE)
_CODE_FENCE_RE = re.compile(r"```")
_LOG_LINE_RE = re.compile(
    r"(?:\d{4}-\d{2}-\d{2}|\d{2}/\w{3}/\d{4}|\d{2}:\d{2}:\d{2})"
    r".*?\b(?:DEBUG|INFO|WARN(?:ING)?|ERROR|FATAL|CRITICAL)\b",
    re.IGNORECASE,
)
_STACK_TRACE_RE = re.compile(
    r"Traceback \(most recent call last\)"
    r"|^\s+at (?:com|org|java|net|io)\."
    r"|File \"[^\"]+\", line \d+",
    re.MULTILINE,
)
_PASSPHRASE_RE = re.compile(
    r"\b(?:passphrase|pass.?key|secret phrase|needle|hidden phrase|the phrase is"
    r"|secret code|codes? for|follow the chain|value of variable|chain of"
    r"|variable.*equal|assigned to|value associated with|the value.*key)\b",
    re.IGNORECASE,
)
_SUMMARIZE_RE = re.compile(
    r"\b(?:summarize|summary|summarise|summarisation|summarization|key points?|main points?)\b",
    re.IGNORECASE,
)

_LONG_PROSE_MIN_TOKENS = 2000
_LONG_PROSE_MAX_CODE_FENCES = 1


@dataclass(frozen=True)
class PromptFeatures:
    prompt_length_tokens: int
    has_diff: bool
    has_code_fence: bool
    has_log_lines: bool
    has_stack_trace: bool
    has_passphrase_query: bool
    is_long_prose: bool
    has_summarize_instruction: bool

    @property
    def has_structured_content(self) -> bool:
        return self.has_diff or self.has_log_lines or self.has_stack_trace or self.has_code_fence


def extract_features(text: str) -> PromptFeatures:
    tokens = len(_ENCODING.encode(text))
    has_diff = bool(_DIFF_RE.search(text))
    fence_count = len(_CODE_FENCE_RE.findall(text))
    has_code_fence = fence_count >= 2
    has_log_lines = bool(_LOG_LINE_RE.search(text))
    has_stack_trace = bool(_STACK_TRACE_RE.search(text))

    question_match = re.search(r"\[QUESTION\](.*?)(?:\[|$)", text, re.DOTALL | re.IGNORECASE)
    question_text = question_match.group(1).strip() if question_match else text[-500:]
    has_passphrase_query = bool(_PASSPHRASE_RE.search(question_text))
    has_summarize_instruction = bool(_SUMMARIZE_RE.search(question_text))

    structured = has_diff or has_log_lines or has_stack_trace or (fence_count > _LONG_PROSE_MAX_CODE_FENCES)
    is_long_prose = tokens >= _LONG_PROSE_MIN_TOKENS and not structured

    return PromptFeatures(
        prompt_length_tokens=tokens,
        has_diff=has_diff,
        has_code_fence=has_code_fence,
        has_log_lines=has_log_lines,
        has_stack_trace=has_stack_trace,
        has_passphrase_query=has_passphrase_query,
        is_long_prose=is_long_prose,
        has_summarize_instruction=has_summarize_instruction,
    )
