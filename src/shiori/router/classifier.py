"""
ML-based router: loads a fine-tuned MiniLM binary classifier from models/shiori_router/
and classifies prompts as lossless (0) or lossy (1).

Input: [QUESTION] + [ANSWER_FORMAT] sections (≤256 tokens).
Output: "lossless" | "lossy" | None (model unavailable)

The model runs entirely locally. No network call is made.
Falls back silently to None when the model directory is not present.
"""
from __future__ import annotations

import re
from pathlib import Path

_QUESTION_RE = re.compile(r"\[QUESTION\](.*?)(?:\[ANSWER_FORMAT\]|$)", re.DOTALL | re.IGNORECASE)
_FORMAT_RE = re.compile(r"\[ANSWER_FORMAT\](.*?)$", re.DOTALL | re.IGNORECASE)

_tokenizer = None
_model = None
_loaded = False


def _model_dir(override: str | None = None) -> Path:
    if override:
        return Path(override)
    candidates = [
        Path(__file__).parent.parent.parent.parent / "models" / "shiori_router",
        Path("models/shiori_router"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def _load(model_path: str | None = None) -> bool:
    global _tokenizer, _model, _loaded
    if _loaded:
        return _model is not None
    _loaded = True

    d = _model_dir(model_path)
    if not d.exists():
        return False

    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        _tokenizer = AutoTokenizer.from_pretrained(str(d))
        _model = AutoModelForSequenceClassification.from_pretrained(str(d))
        _model.eval()
        return True
    except Exception:
        return False


def _extract_task_text(prompt: str) -> str:
    parts = []
    q = _QUESTION_RE.search(prompt)
    f = _FORMAT_RE.search(prompt)
    if q:
        parts.append(q.group(1).strip())
    if f:
        parts.append(f.group(1).strip())
    return " | ".join(parts) if parts else prompt[-400:].strip()


def classify(prompt: str, model_path: str | None = None) -> str | None:
    """Return 'lossless', 'lossy', or None if model unavailable."""
    if not _load(model_path):
        return None

    import torch
    text = _extract_task_text(prompt)
    enc = _tokenizer(
        text,
        max_length=256,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    with torch.no_grad():
        logits = _model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
        ).logits
    label = int(logits.argmax(dim=-1).item())
    return "lossy" if label == 1 else "lossless"
