"""MMAU loader (Massive Multi-Task Audio Understanding, arXiv:2410.19168).

Audio-only multiple-choice QA over three categories -- Sound, Music, Speech. The common
`mmau-test-mini.json` (1000 items, answers public) is a list of:
  {audio_id, question, choices: [...], answer: <choice text>, category|task|dataset, ...}
Field names vary a little across releases, so this reads them defensively.

  data/MMAU/mmau-test-mini.json                         # the split file
  data/MMAU/test-mini-audios/<...>.wav                  # the audio (audio_id is relative)

Download (login node):  https://github.com/Sakshi113/MMAU  (or the HF mirror); put the json at
data/MMAU/mmau-test-mini.json and the audios under data/MMAU/ so audio_id resolves.
"""
from __future__ import annotations

import json
import os

DEFAULT_JSON = "data/MMAU/mmau-test-mini.json"
DEFAULT_ROOT = "data/MMAU"
CATEGORIES = ("sound", "music", "speech")


def _category(s: dict) -> str:
    """Normalize an item's category to one of sound/music/speech (best-effort)."""
    raw = " ".join(str(s.get(k, "")) for k in ("category", "task", "dataset", "difficulty")).lower()
    return next((c for c in CATEGORIES if c in raw), "other")


def _resolve(audio: str | None, root: str) -> str | None:
    if not audio:
        return None
    if os.path.isabs(audio) or os.path.exists(audio):
        return audio
    for cand in (os.path.join(root, audio), os.path.join(root, os.path.basename(audio))):
        if os.path.exists(cand):
            return cand
    return os.path.join(root, audio)            # let the eval surface a clean "missing file"


def load_mmau(json_path: str = DEFAULT_JSON, audio_root: str = DEFAULT_ROOT) -> list[dict]:
    """Return [{audio_path, question, choices, answer, category, id}] for the split."""
    with open(json_path) as f:
        data = json.load(f)
    if isinstance(data, dict):                  # some dumps wrap the list under a key
        data = data.get("data") or data.get("questions") or next(iter(data.values()))
    items = []
    for s in data:
        audio = s.get("audio_id") or s.get("audio") or s.get("audio_path")
        items.append({
            "audio_path": _resolve(audio, audio_root),
            "question": s["question"],
            "choices": list(s.get("choices") or s.get("options") or []),
            "answer": str(s.get("answer", "")).strip(),
            "category": _category(s),
            "id": s.get("id") or s.get("audio_id"),
        })
    return items


def format_mmau(question: str, choices: list[str]) -> str:
    """Lettered MCQ prompt (parse the reply with rlvib.eval.metrics.parse_choice)."""
    import string
    opts = "\n".join(f"({string.ascii_uppercase[i]}) {c}" for i, c in enumerate(choices))
    return f"{question}\n{opts}\nAnswer with the letter of the correct option."
