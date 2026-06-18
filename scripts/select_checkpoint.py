#!/usr/bin/env python
"""Pick the best anchored swap-DPO checkpoint from the per-step eval JSONs.

Reads the tagged JSONs written by select_checkpoint.sh
(runs/avhbench_<model>_step*.json, runs/cmm_<model>_step*.json, runs/dave_<split>_<model>_
audio_visual_alignment_step*.json), prints a tradeoff table, and applies the rule:

    maximize AVHBench overall  subject to  CMM PA >= --min-pa  and  DAVE >= --min-dave

The base (no-bottleneck) row is shown as the reference when its JSONs are present.

  python scripts/select_checkpoint.py
  python scripts/select_checkpoint.py --min-pa 0.92 --min-dave 0.36
"""
from __future__ import annotations

import argparse
import glob
import json
import re


def _load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _avh(model, tag):
    d = _load(f"runs/avhbench_{model}{tag}.json")
    if not d:
        return None
    return (d["results"].get("overall") or {}).get("accuracy")


def _cmm(model, tag):
    d = _load(f"runs/cmm_{model}{tag}.json")
    if not d:
        return None, None
    r = next(iter(d["results"].values()))
    return r.get("PA"), r.get("HR")


def _dave(model, tag, split):
    d = _load(f"runs/dave_{split}_{model}_audio_visual_alignment{tag}.json")
    return d.get("accuracy") if d else None


def _fmt(x):
    return f"{x:.3f}" if isinstance(x, float) else "  -  "


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-omni")
    ap.add_argument("--split", default="ego4d")
    ap.add_argument("--min-pa", type=float, default=0.90, help="CMM perception-accuracy guard")
    ap.add_argument("--min-hr", type=float, default=0.70, help="CMM hallucination-resistance guard")
    ap.add_argument("--min-dave", type=float, default=0.36, help="DAVE accuracy guard (skipped if unevaluated)")
    args = ap.parse_args()

    steps = sorted({int(m.group(1))
                    for p in glob.glob(f"runs/avhbench_{args.model}_step*.json")
                    for m in [re.search(r"_step(\d+)\.json$", p)] if m})
    if not steps:
        print("No per-step AVHBench JSONs found in runs/. Run scripts/select_checkpoint.sh first.")
        return 1
    rows = [("base", "")] + [(f"step{s}", f"_step{s}") for s in steps]

    print(f"{'ckpt':>8}  {'AVHBench':>8}  {'CMM_PA':>7}  {'CMM_HR':>7}  {'DAVE':>6}  guard")
    cand = []
    for name, tag in rows:
        avh = _avh(args.model, tag)
        pa, hr = _cmm(args.model, tag)
        dave = _dave(args.model, tag, args.split)
        ok = ((pa is None or pa >= args.min_pa) and (hr is None or hr >= args.min_hr)
              and (dave is None or dave >= args.min_dave))
        guard = "" if name == "base" else ("ok" if ok else "FAIL")
        print(f"{name:>8}  {_fmt(avh):>8}  {_fmt(pa):>7}  {_fmt(hr):>7}  {_fmt(dave):>6}  {guard}")
        if name != "base" and isinstance(avh, float) and ok:
            cand.append((avh, name))

    if cand:
        avh, name = max(cand)
        print(f"\nSELECTED: {name}  (AVHBench={avh:.3f}; passes CMM_PA>={args.min_pa}, DAVE>={args.min_dave})")
    else:
        print("\nNo checkpoint passed the guards -- loosen --min-pa/--min-dave or inspect the runs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
