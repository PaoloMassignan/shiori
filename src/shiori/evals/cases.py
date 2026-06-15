"""
Synthetic eval corpus for Shiori routing validation.

Each case reproduces the pattern of a LZPrompt benchmark (documented in
docs/routing.md and ADR-010) using only locally generated text — no API
keys, no external files.

Cases are designed to activate a specific routing layer:
  - Structural rules fire first (LogBench, SWEBench)
  - Fallback heuristics fire when no structural signal + no ML model
    (MeetingBank, HotPotQA, ZeroSCROLLS)
"""
from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Synthetic text helpers
# ---------------------------------------------------------------------------

_LOG_BLOCK = "\n".join([
    f"2024-03-{i+1:02d} 08:{i:02d}:00.{i*100:03d} INFO  [worker-{i%4}] "
    f"dfs.DataNode$PacketResponder: Received block blk_{1073741825+i} "
    f"of size 67108864 from /10.0.0.{10+i%6}"
    for i in range(20)
])

_LOG_TEXT = (
    f"[CONTEXT]\n{_LOG_BLOCK}\n\n"
    "[QUESTION]\nWhat errors occurred in the log?"
)

_DIFF_TEXT = """\
[CONTEXT]
diff --git a/src/router/rules.py b/src/router/rules.py
index a1b2c3d..e4f5g6h 100644
--- a/src/router/rules.py
+++ b/src/router/rules.py
@@ -12,6 +12,10 @@ def route_structural(features):
     if features.has_log_lines:
         return "lossless", "log lines detected"
+    if features.has_json_array:
+        return "json", "json array detected"
+
     if features.has_stack_trace:
         return "lossless", "stack trace detected"

[QUESTION]
What was changed in this diff?
"""

# MeetingBank: long prose (>2000 tokens) with summarize instruction.
# A single meeting round is ~120 words; 20 rounds exceed the 2000-token threshold.
_MEETING_ROUND = """\
Alice: Q{i} revenue came in at {rev} million, up {pct}% versus the same quarter last year. \
The enterprise segment led with strong bookings from financial services and healthcare. \
SMB continued to face headwinds from budget cycles but showed improvement late in the quarter.

Bob: On the product side, we shipped {features} new features in Q{i}. \
The dashboard redesign saw 34% adoption after three weeks, ahead of our 30% target. \
The mobile application is on track for general availability next quarter.

Carol: Engineering velocity improved after the team restructuring. \
We reduced mean time to deploy from 45 minutes to 12 minutes. \
Technical debt reduction accounted for 20% of engineering capacity this quarter.

David: For next quarter I recommend three priorities: expanding the enterprise sales team \
by four headcount, launching the partner programme targeting system integrators, \
and improving onboarding to reduce SMB churn which currently stands at 8% monthly.
"""

_MEETING_TRANSCRIPT = "\n".join(
    _MEETING_ROUND.format(i=q + 1, rev=round(3.8 + q * 0.4, 1), pct=8 + q * 2, features=12 + q * 3)
    for q in range(22)
)

_MEETING_TEXT = (
    f"[CONTEXT]\n{_MEETING_TRANSCRIPT}\n\n"
    "[QUESTION]\nPlease summarize the key points from this meeting transcript."
)

_HOTPOTQA_TEXT = (
    "[CONTEXT]\n"
    "Paris is the capital and largest city of France. "
    "France is a country located in Western Europe, bordered by Belgium, "
    "Luxembourg, Germany, Switzerland, Italy, Monaco, Spain, and Andorra. "
    "The Eiffel Tower, built in 1889, is located in Paris on the Champ de Mars.\n\n"
    "[QUESTION]\nIn which continent is the city where the Eiffel Tower stands located?"
)

_MCQ_TEXT = (
    "[CONTEXT]\n"
    "The speed of light in a vacuum is approximately 299,792 kilometres per second. "
    "Albert Einstein's theory of special relativity establishes that no object with "
    "mass can reach or exceed the speed of light. Light takes about 8 minutes and "
    "20 seconds to travel from the Sun to the Earth.\n\n"
    "[QUESTION]\nWhich of the following best describes the speed of light?\n"
    "A) Approximately 100,000 km/s\n"
    "B) Approximately 299,792 km/s\n"
    "C) Approximately 500,000 km/s\n"
    "D) Variable depending on the observer\n"
)

# ---------------------------------------------------------------------------
# EvalCase dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EvalCase:
    name: str
    text: str
    expected_strategy: str
    expected_via: str
    evidence: str


EVAL_CASES: list[EvalCase] = [
    EvalCase(
        name="LogBench",
        text=_LOG_TEXT,
        expected_strategy="lossless",
        expected_via="rules",
        evidence="llmlingua −0.477 quality delta on log QA (ADR-010)",
    ),
    EvalCase(
        name="SWEBench",
        text=_DIFF_TEXT,
        expected_strategy="lossless",
        expected_via="rules",
        evidence="llmlingua −0.267 quality delta on code/diff tasks (ADR-010)",
    ),
    EvalCase(
        name="MeetingBank",
        text=_MEETING_TEXT,
        expected_strategy="llmlingua",
        expected_via="fallback",
        evidence="lossless 5.9% vs llmlingua 51.7% compression on meetings (ADR-010)",
    ),
    EvalCase(
        name="HotPotQA",
        text=_HOTPOTQA_TEXT,
        expected_strategy="lossless",
        expected_via="fallback",
        evidence="lossless +0.050 vs llmlingua +0.005 on multi-hop QA (ADR-010)",
    ),
    EvalCase(
        name="ZeroSCROLLS",
        text=_MCQ_TEXT,
        expected_strategy="lossless",
        expected_via="fallback",
        evidence="llmlingua −0.350 quality delta on MCQ (ADR-010)",
    ),
]
