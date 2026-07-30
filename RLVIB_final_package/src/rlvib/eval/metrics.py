"""Metrics for AVHBench-style yes/no tasks (baseline eval).

The full grounding + abstention suite (ΔAcc across modality ablations, AURC/E-AURC,
Abstain-ECE) lands once multiple datasets are wired (see the design doc §6). For
the AVHBench baseline, per-task yes/no accuracy + a parse rate is the headline.
"""
from __future__ import annotations

import re

_YES = {"yes", "true", "correct", "yeah", "yep", "yup"}
_NO = {"no", "false", "incorrect", "nope", "nah"}

# Negation / absence cues -> "no" (the model is rejecting presence). The earlier parser
# only matched a standalone \bno\b, so verbose absence answers like "I did not see a tree"
# or "I didn't hear a bell" parsed as None (-> scored wrong), flooring CMM-HR / AVHBench for
# backbones that answer in full sentences (VideoLLaMA2). See scripts/audit_cmm_hr.py and
# docs/research/videollama2-yesbias-and-metric-audit.md.
_NEG = re.compile(
    r"\b(?:no|not|never|none|nothing|cannot|can't|cant|don't|dont|doesn't|doesnt|"
    r"didn't|didnt|isn't|isnt|aren't|arent|wasn't|wasnt|weren't|werent|won't|wont|"
    r"without|neither|nor|absent|unable|unheard|inaudible|invisible)\b"
)
# Affirmation / presence cues -> "yes".
_AFF = re.compile(
    r"\b(?:yes|yeah|yep|yup|correct|true|present|audible|visible|"
    r"there (?:is|are|'s)|i can (?:see|hear)|can be (?:seen|heard))\b"
)


def parse_yes_no(text: str | None) -> str | None:
    """Map a free-text model answer to 'yes' / 'no' / None (unparseable).

    Negation-aware: a leading yes/no wins; otherwise decide by negation vs affirmation
    cues (a negation scopes any affirmation, e.g. "there is no tree" -> no), then fall
    back to a bare yes/no anywhere. This reads sentence answers like "I did not see X"
    (-> no) that the old \\bno\\b-only parser dropped.
    """
    if not text:
        return None
    t = text.strip().lower()
    first = re.split(r"[\s,.!?:;]+", t)[0] if t else ""
    if first in _YES:
        return "yes"
    if first in _NO:
        return "no"
    neg, aff = bool(_NEG.search(t)), bool(_AFF.search(t))
    if neg and not aff:
        return "no"
    if aff and not neg:
        return "yes"
    if neg and aff:
        return "no"  # a negation usually scopes the affirmation ("there is no tree")
    m = re.search(r"\b(yes|no)\b", t)  # last resort: a bare yes/no anywhere
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

