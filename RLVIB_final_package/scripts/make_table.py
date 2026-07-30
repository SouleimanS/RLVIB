#!/usr/bin/env python
"""Dump the global-comparison numbers from the eval JSONs, one row per model/checkpoint:
AVHBench per task (A->V, V->A, AV-match) + overall, CMM PA/HR + overall, DAVE.

Reads every eval JSON under runs/ across the naming schemes in use:
  avhbench_<model>[_<exp>_sysfull_step<N>].json   cmm_<model>...        (corrected protocol)
  avhbench_<model>_full[_<exp>_step<N>].json       cmm_<model>...        (API/older full runs)
  smoke_avh_<model>[_<exp>_step<N>].json           smoke_cmm_<model>...  (PARTIAL smoke runs)
When both a full and a smoke exist for the same cell, the full one wins; the `src` column marks
the source and `n` is the AVHBench item count, so partial (smoke) rows are obvious. Missing: '-'.

  python scripts/make_table.py
"""
from __future__ import annotations

import glob
import json
import os
import re

MODELS = ["qwen3-omni", "qwen2.5-omni", "videollama2", "gemini", "gpt4o"]
A2V, V2A, AVM = "Audio-driven Video Hallucination", "Video-driven Audio Hallucination", "AV Matching"
_PRIO = {"sysfull": 3, "full": 2, "smoke": 1}   # which source wins when a cell has several files


def _bench(b: str):
    if b.startswith(("avhbench_", "smoke_avh_")):
        return "avhbench"
    if b.startswith(("cmm_", "smoke_cmm_")):
        return "cmm"
    if b.startswith(("dave_", "smoke_dave_")):
        return "dave"
    return None


def _model(b: str):
    return next((m for m in MODELS if m in b), None)


def _variant(b: str, model: str) -> str:
    tail = b.split(model, 1)[-1]                       # strip the prefix+model, parse the rest
    step = re.search(r"step(\d+)", tail)
    if not step:
        return "base"
    em = re.search(r"_([A-Za-z]\w*?)(?:_sysfull|_full)?_step\d+", tail)
    return f"{em.group(1) if em else 'broad'}@{step.group(1)}"


def _source(b: str) -> str:
    if b.startswith("smoke_"):
        return "smoke"
    return "sysfull" if "_sysfull" in b else "full"


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
    data: dict = {}                                   # data[model][variant][bench] = (prio, src, blob)
    for path in glob.glob("runs/*.json"):
        b = os.path.basename(path)
        bench, model = _bench(b), _model(b)
        if not bench or not model:
            continue
        src = _source(b)
        cell = data.setdefault(model, {}).setdefault(_variant(b, model), {})
        if bench not in cell or _PRIO[src] > cell[bench][0]:
            cell[bench] = (_PRIO[src], src, _load(path))

    hdr = (f"  {'variant':14s} {'src':7s} {'n':>5s} | {'A->V':5s} {'V->A':5s} {'AVm':5s} {'AVH':5s}"
           f" | {'PA':5s} {'HR':5s} {'CMM':5s} | {'DAVE':5s}")
    for model in MODELS:
        if model not in data:
            continue
        print(f"\n=== {model} ===")
        print(hdr)
        for var in sorted(data[model], key=lambda v: (v != "base", v)):
            cell = data[model][var]
            avh, cmm, dave = cell.get("avhbench"), cell.get("cmm"), cell.get("dave")
            avh_blob = (avh[2] if avh else None) or {}
            a2v, v2a, avm, avho = _avh(avh_blob)
            pa, hr, cmmo = _cmm((cmm[2] if cmm else None) or {})
            daveo = ((dave[2] if dave else None) or {}).get("accuracy")
            src = (avh or cmm or (0, "-", None))[1]
            n = avh_blob.get("results", {}).get("overall", {}).get("n")
            print(f"  {var:14s} {src:7s} {str(n) if n else '-':>5s} | "
                  f"{_f(a2v)} {_f(v2a)} {_f(avm)} {_f(avho)} | {_f(pa)} {_f(hr)} {_f(cmmo)} | {_f(daveo)}")
    print("\nA->V audio-driven video hall.; V->A video-driven AUDIO hall. (grounding probe); "
          "AVm AV-matching; AVH overall; n = AVHBench items.")
    print("CMM: PA perception / HR hallucination-resistance / CMM overall.")
    print("src: sysfull/full = complete run; smoke = PARTIAL (--limit) -- not final numbers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
