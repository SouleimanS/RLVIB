#!/usr/bin/env python
"""Compact table of frozen-model baseline results in runs/.

  python scripts/summarize_baselines.py

Reads the per-benchmark JSONs the eval runners write (DAVE: one per split/mode;
AVHBench/CMM: a {results: {...}} dict) and prints one line per result, so progress
can be shared without pasting whole files.
"""
import glob
import json
import os


def _load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def main() -> None:
    dave = sorted(glob.glob("runs/dave_*.json"))
    if dave:
        print("=== DAVE (multiple choice) ===")
        for p in dave:
            d = _load(p)
            if d and "accuracy" in d:
                tag = os.path.basename(p)[5:-5]  # strip "dave_" + ".json"
                print(f"  {tag:36s} acc={d['accuracy']:.3f}  parse={d.get('parse_rate', 0):.2f}  n={d['n']}")

    for name, path in [("AVHBench", "runs/avhbench_baseline.json"), ("CMM", "runs/cmm_baseline.json")]:
        d = _load(path)
        if not d:
            continue
        print(f"=== {name} ===")
        for k, v in d.get("results", {}).items():
            if "PA" in v:  # CMM: Perception Acc / Hallucination Resistance
                print(f"  {k:34s} PA={v['PA']:.3f} HR={v['HR']:.3f} acc={v['acc']:.3f}  n={v['n']}")
            else:  # AVHBench: plain accuracy
                print(f"  {k:34s} acc={v['accuracy']:.3f}  parse={v.get('parse_rate', 0):.2f}  n={v['n']}")


if __name__ == "__main__":
    main()
