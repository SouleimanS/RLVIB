"""Metrics for AVHBench-style yes/no tasks (baseline eval).

The full grounding + abstention suite (ΔAcc across modality ablations, AURC/E-AURC,
Abstain-ECE) lands once multiple datasets are wired (see the design doc §6). For
the AVHBench baseline, per-task yes/no accuracy + a parse rate is the headline.
"""
from __future__ import annotations

import re

_YES = {"yes", "true", "correct", "yeah", "yep"}
_NO = {"no", "false", "incorrect", "nope"}


def parse_yes_no(text: str | None) -> str | None:
    """Map a free-text model answer to 'yes' / 'no' / None (unparseable)."""
    if not text:
        return None
    t = text.strip().lower()
    first = re.split(r"[\s,.!?:;]+", t)[0] if t else ""
    if first in _YES:
        return "yes"
    if first in _NO:
        return "no"
    m = re.search(r"\b(yes|no)\b", t)  # fall back to first yes/no anywhere
    return m.group(1) if m else None


def accuracy(preds: list[str | None], golds: list[str]) -> dict:
    """preds/golds are 'yes'/'no' (preds may contain None = unparsed -> counted wrong)."""
    n = len(golds)
    if n == 0:
        return {"accuracy": 0.0, "n": 0, "correct": 0, "parse_rate": 0.0}
    correct = sum(1 for p, g in zip(preds, golds) if p == g)
    parsed = sum(1 for p in preds if p is not None)
    return {
        "accuracy": correct / n,
        "n": n,
        "correct": correct,
        "parse_rate": parsed / n,
    }


def parse_choice(text: str | None) -> str | None:
    """Extract a single MC letter (A-E) from an answer like '(B)' or 'B) ...' (DAVE)."""
    if not text:
        return None
    t = text.strip().upper()
    m = re.search(r"\(([A-E])\)", t)  # prefer parenthesized "(B)"
    if m:
        return m.group(1)
    m = re.match(r"([A-E])\b", t)  # else a leading "B" / "B)"
    return m.group(1) if m else None

