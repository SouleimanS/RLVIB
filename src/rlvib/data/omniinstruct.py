"""OmniInstruct-v1 loader (`m-a-p/OmniInstruct_v1`) — audio + single-image MCQ.

Primary labeled-QA *training* corpus (see docs/research/training-data-plan.md).
HF dataset with inline `Audio` + `Image` features. Per-row fields:
  question      : str
  options       : list[str]
  answer        : str   (the correct option's TEXT, not a letter)
  audio         : decoded audio (array + sr / AudioDecoder)
  image         : PIL image (single frame)
  audio_label   : str   (JSON of AudioSet-style tag probabilities -> audio semantics,
                          handy for semantic audio-swap counterfactual pairs)
  source        : str   (AVQA / MUSIC-AVQA2.0 / MSRVTT-QA — audio-dependence is mixed)

Requires `torchcodec` (datasets>=5 audio decoding). Use the full local download
(not streaming=True) for training — streaming + torchcodec has a teardown crash.
"""
from __future__ import annotations

import string

REPO = "m-a-p/OmniInstruct_v1"


def load(split: str = "train", streaming: bool = False):
    """Return the raw HF dataset (apply `normalize` per row when iterating)."""
    from datasets import load_dataset

    return load_dataset(REPO, split=split, streaming=streaming)


def gold_index(answer, options) -> int | None:
    """Index of the correct option (answer is the option text); None if unmatched."""
    a = (answer or "").strip()
    for i, o in enumerate(options):
        if str(o).strip() == a:
            return i
    al = a.lower()
    for i, o in enumerate(options):
        if str(o).strip().lower() == al:
            return i
    return None


def normalize(row: dict) -> dict:
    """Parse a raw row into a training-ready example (keeps decoded audio/image)."""
    options = row.get("options") or []
    gi = gold_index(row.get("answer"), options)
    return {
        "question": row.get("question"),
        "options": options,
        "answer": row.get("answer"),
        "gold_index": gi,
        "gold_letter": string.ascii_uppercase[gi] if gi is not None else None,
        "audio": row.get("audio"),
        "image": row.get("image"),
        "audio_label": row.get("audio_label"),
        "source": row.get("source"),
    }


def format_mcq(question: str, options: list[str]) -> str:
    """Lettered multiple-choice prompt (parseable by rlvib.eval.metrics.parse_choice)."""
    opts = "\n".join(f"({string.ascii_uppercase[i]}) {o}" for i, o in enumerate(options))
    return f"{question}\n{opts}\nAnswer with only the letter."
