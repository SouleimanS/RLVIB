"""Temporal-alignment test beds (the deck's 'Scoring when' slide).

Two loaders, both returning balanced 2-way sync-judgment items scored like every other MCQ:

  AVE-Shift (ours)   -- build_avshift(): held-out AVE clips, half untouched (gold 'in sync'),
                        half shifted by +/-{0.5,1,2}s (gold 'out of sync', with a lags/leads
                        direction). Materialized on demand via pairs.make_shift_examples.
  VGGSound-Sync      -- load_vggsound_sync(): an out-of-distribution sync benchmark. Reads a
                        json/parquet manifest of {video, offset_seconds}; offset != 0 => shifted.

Each item: {video_path, question, options, sync_letter (gold), shifted, delta[, dir_options,
dir_letter]}. Balanced => a constant answer scores 0.50.
"""
from __future__ import annotations

import glob
import json
import os
import random

DEFAULT_AVE_SHIFT_DIR = "data/AVE/shifted"
DEFAULT_VGGSYNC_DIR = "data/VGGSoundSync"


def build_avshift(n: int = 200, out_dir: str = DEFAULT_AVE_SHIFT_DIR,
                  offsets=(0.5, 1.0, 2.0), split: str = "val",
                  seed: int = 0) -> list[dict]:
    """Materialize (or reuse) `n` AVE-Shift items via pairs.make_shift_examples on a held-out
    split. Deterministic given `seed`; disjoint from training by construction (uses `split`)."""
    from rlvib.data import ave
    from rlvib.data.pairs import make_shift_examples

    items = ave.load_ave(split)
    rng = random.Random(seed)
    rng.shuffle(items)
    return make_shift_examples(items, n, out_dir, offsets=offsets, rng=rng)


def _sync_item(video_path: str, delta: float) -> dict:
    from rlvib.data.pairs import _sync_mcq
    rec = _sync_mcq(abs(delta) > 1e-6, float(delta))
    return dict(rec, video_path=video_path)


def load_vggsound_sync(data_dir: str = DEFAULT_VGGSYNC_DIR,
                       manifest: str | None = None) -> list[dict]:
    """Load VGGSound-Sync as balanced sync-judgment items.

    Manifest (json list or parquet) entries need a clip path and a signed audio offset:
    keys tried are {video|path|filename} and {offset|offset_seconds|delta}. offset == 0 is an
    in-sync positive control; nonzero is shifted (sign = lags/leads).
    """
    manifest = manifest or _find_manifest(data_dir)
    rows = _read_rows(manifest)
    root = os.path.join(data_dir, "videos")
    items = []
    for r in rows:
        v = r.get("video") or r.get("path") or r.get("filename")
        if not v:
            continue
        vp = v if os.path.isabs(v) or os.path.exists(v) else os.path.join(root, v)
        delta = float(r.get("offset", r.get("offset_seconds", r.get("delta", 0.0))))
        items.append(_sync_item(vp, delta))
    return items


def _find_manifest(data_dir: str) -> str:
    for pat in ("*.json", "*.parquet", "meta/*.json"):
        hits = sorted(glob.glob(os.path.join(data_dir, pat)))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"no VGGSound-Sync manifest under {data_dir} (see module docstring)")


def _read_rows(manifest: str) -> list[dict]:
    if manifest.endswith(".parquet"):
        import pandas as pd
        return pd.read_parquet(manifest).to_dict("records")
    with open(manifest) as f:
        data = json.load(f)
    return data if isinstance(data, list) else data.get("data") or list(data.values())


def format_sync(question: str, options: list[str]) -> str:
    """Lettered MCQ prompt (parse the reply with rlvib.eval.metrics.parse_choice)."""
    import string
    opts = "\n".join(f"({string.ascii_uppercase[i]}) {c}" for i, c in enumerate(options))
    return f"{question}\n{opts}\nAnswer with the letter of the correct option."
