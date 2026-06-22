#!/usr/bin/env python
"""Dump the global-comparison numbers from the _full eval JSONs, one row per model/checkpoint:
AVHBench per task (A->V, V->A, AV-match) + overall, CMM PA/HR + overall, and DAVE. Full-set
accuracy (held-out McNemar significance comes from paired_stats). Missing/partial cells show
'-'. Paste the output and the paper's global table is filled directly from it.

  python scripts/make_table.py
"""
from __future__ import annotations

import glob
import json
import os
import re

MODELS = ["qwen3-omni", "qwen2.5-omni", "videollama2", "gemini", "gpt4o"]
A2V, V2A, AVM = "Audio-driven Video Hallucination", "Video-driven Audio Hallucination", "AV Matching"


def _variant(b: str) -> str:
    m = re.search(r"_(broad\w*)_full_step(\d+)", b)
    return f"{m.group(1)}@{m.group(2)}" if m else "base"


def _load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _f(x):
    return f"{x:.3f}" if isinstance(x, (int, float)) else "  -  "


def _avh(blob):
    r = (blob or {}).get("results", {})
    g = lambda k: r.get(k, {}).get("accuracy")  # noqa: E731
    return g(A2V), g(V2A), g(AVM), r.get("overall", {}).get("accuracy")


def _cmm(blob):
    o = (blob or {}).get("results", {}).get("overall", {})
    return o.get("PA"), o.get("HR"), o.get("acc")


def main() -> int:
    data: dict = {}
    for path in glob.glob("runs/*_full*.json"):
        b = os.path.basename(path)
        bench = b.split("_", 1)[0]
        if bench not in ("avhbench", "cmm", "dave"):
            continue
        model = next((m for m in MODELS if m in b), None)
        if model:
            data.setdefault(model, {}).setdefault(_variant(b), {})[bench] = _load(path)

    hdr = f"  {'variant':14s} | {'A->V':5s} {'V->A':5s} {'AVm':5s} {'AVH':5s} | " \
          f"{'PA':5s} {'HR':5s} {'CMM':5s} | {'DAVE':5s}"
    for model in MODELS:
        if model not in data:
            continue
        print(f"\n=== {model} ===")
        print(hdr)
        for var in sorted(data[model], key=lambda v: (v != "base", v)):
            d = data[model][var]
            a2v, v2a, avm, avh = _avh(d.get("avhbench"))
            pa, hr, cmm = _cmm(d.get("cmm"))
            dave = (d.get("dave") or {}).get("accuracy")
            print(f"  {var:14s} | {_f(a2v)} {_f(v2a)} {_f(avm)} {_f(avh)} | "
                  f"{_f(pa)} {_f(hr)} {_f(cmm)} | {_f(dave)}")
    print("\nA->V audio-driven video hall.; V->A video-driven AUDIO hall. (the grounding probe); "
          "AVm AV-matching; AVH overall.")
    print("CMM: PA perception acc / HR hallucination-resistance / CMM overall. Full-set accuracy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
