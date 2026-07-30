#!/usr/bin/env python
"""Materialize MMAU test-mini audio from the HF parquet (the audio bytes are embedded).

data/MMAU/ ships mmau-test-mini.json + test_mini.parquet but no audio files -> every eval
item errors on a missing file. The parquet's `context` column holds {bytes, path} per row and
`other_attributes` carries the item id, so we can write each clip to the path the json expects
and skip any download.

  PYTHONPATH=src python scripts/extract_mmau_audio.py [--root data/MMAU]

Writes to <root>/<audio_id> for relative ids, <root>/<basename> for absolute ones (the loader's
fallback), verifies with rlvib.data.mmau.load_mmau, and prints the missing count (want 0).
"""
from __future__ import annotations

import argparse
import json
import os


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/MMAU")
    ap.add_argument("--parquet", default=None, help="default: <root>/test_mini.parquet")
    ap.add_argument("--json", dest="json_path", default=None,
                    help="default: <root>/mmau-test-mini.json")
    args = ap.parse_args()
    pq = args.parquet or os.path.join(args.root, "test_mini.parquet")
    jp = args.json_path or os.path.join(args.root, "mmau-test-mini.json")

    import pandas as pd
    df = pd.read_parquet(pq)
    by_id = {}
    for _, r in df.iterrows():
        try:
            by_id[json.loads(r["other_attributes"])["id"]] = r
        except Exception:  # noqa: BLE001
            continue
    print(f"parquet rows: {len(df)} (with ids: {len(by_id)})")

    data = json.load(open(jp))
    if isinstance(data, dict):
        data = data.get("data") or next(iter(data.values()))
    written = existed = unmatched = 0
    for s in data:
        audio = s.get("audio_id") or s.get("audio") or s.get("audio_path")
        r = by_id.get(s.get("id"))
        if r is None or not audio:
            unmatched += 1
            continue
        rel = audio if not os.path.isabs(audio) else os.path.basename(audio)
        dest = os.path.join(args.root, rel)
        if os.path.exists(dest):
            existed += 1
            continue
        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
        with open(dest, "wb") as f:
            f.write(r["context"]["bytes"])
        written += 1
    print(f"written: {written}  already-there: {existed}  unmatched: {unmatched}")

    from rlvib.data.mmau import load_mmau
    ds = load_mmau(jp, args.root)
    missing = [x for x in ds if not (x["audio_path"] and os.path.exists(x["audio_path"]))]
    print(f"loader check: {len(ds)} items, missing audio after extract: {len(missing)}")
    for m in missing[:3]:
        print("  e.g. missing:", m["audio_path"])
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
