#!/usr/bin/env python
"""Assemble the results table from the _full eval JSONs.

Per model/checkpoint: headline accuracy on AVHBench / CMM / DAVE, plus the thesis
sub-metrics -- AVHBench audio probe (Video-driven Audio Hallucination) and CMM audio-subset
HR + the three overrely_* hallucination-resistance scores (the over-reliance tasks). These
are FULL-SET numbers (incl. the dev-300 the checkpoint was selected on); the held-out split
and significance come from paired_stats. Reads whatever is on disk; missing/partial files
show their count or '-'. A trailing '*' flags a benchmark below its full size.

  python scripts/make_table.py
"""
from __future__ import annotations

import glob
import json
import os
import re

MODELS = ["qwen3-omni", "qwen2.5-omni", "videollama2", "gemini", "gpt4o"]
FULL = {"avhbench": 5302, "cmm": 2400, "dave": 1572}
AVH_AUDIO = "Video-driven Audio Hallucination"
OVR = ["overrely_visual_ignore_audio", "overrely_audio_ignore_visual",
       "overrely_language_ignore_visual"]


def _variant(b: str) -> str:
    m = re.search(r"_(broad\w*)_full_step(\d+)", b)
    return f"{m.group(1)}@{m.group(2)}" if m else "base"


def _load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _fmt(x, partial=False):
    if not isinstance(x, (int, float)):
        return "  -  "
    return f"{x:.3f}{'*' if partial else ' '}"


def _partial(blob, bench):
    """True if this benchmark is below its full size (still running / clips skipped)."""
    if not blob:
        return False
    n = blob.get("n") if bench == "dave" else blob.get("results", {}).get("overall", {}).get("n")
    return isinstance(n, int) and n < FULL[bench]


def main() -> int:
    data: dict = {}
    for path in glob.glob("runs/*_full*.json"):
        b = os.path.basename(path)
        bench = b.split("_", 1)[0]
        if bench not in FULL:
            continue
        model = next((m for m in MODELS if m in b), None)
        if not model:
            continue
        data.setdefault(model, {}).setdefault(_variant(b), {})[bench] = _load(path)

    for model in MODELS:
        if model not in data:
            continue
        print(f"\n=== {model} ===")
        print(f"  {'checkpoint':14s} |  AVHBench  audio | "
              f" CMM   aHR  | ovr V/A A/V L/V |  DAVE")
        for var in sorted(data[model], key=lambda v: (v != "base", v)):
            d = data[model][var]
            av, cm, dv = d.get("avhbench"), d.get("cmm"), d.get("dave")
            ar = (av or {}).get("results", {})
            cr = (cm or {}).get("results", {})
            pa, pc, pd = _partial(av, "avhbench"), _partial(cm, "cmm"), _partial(dv, "dave")
            avh = _fmt(ar.get("overall", {}).get("accuracy"), pa)
            aud = _fmt(ar.get(AVH_AUDIO, {}).get("accuracy"), pa)
            cmm = _fmt(cr.get("overall", {}).get("acc"), pc)
            ahr = _fmt(cr.get("audio_subsets", {}).get("HR"), pc)
            ovr = " ".join(_fmt(cr.get(o, {}).get("HR")).strip() for o in OVR)
            dav = _fmt((dv or {}).get("accuracy"), pd)
            print(f"  {var:14s} | {avh} {aud} | {cmm} {ahr} | {ovr} | {dav}")

    print("\n  audio = Video-driven Audio Hallucination (the audio-grounding probe)")
    print("  aHR   = CMM audio-subset Hallucination-Resistance; ovr = overrely_* HR "
          "(Visual/Audio/Language ignore)")
    print("  '*' = benchmark below full size (still running / clips skipped). "
          "Full-set incl. dev-300; use paired_stats for held-out + significance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
