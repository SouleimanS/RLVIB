#!/usr/bin/env python
"""Compact table of frozen-model baseline results in runs/.

  python scripts/summarize_baselines.py

Globs the per-benchmark JSONs the eval runners write (now per model), so multiple
models' baselines can be compared without pasting whole files.
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
                print(f"  {os.path.basename(p)[:-5]:46s} acc={d['accuracy']:.3f}  "
                      f"parse={d.get('parse_rate', 0):.2f}  n={d['n']}")

    for label, pattern in [("AVHBench", "runs/avhbench_*.json"), ("CMM", "runs/cmm_*.json")]:
        files = sorted(glob.glob(pattern))
        if not files:
            continue
        print(f"=== {label} ===")
        for p in files:
            d = _load(p)
            if not d:
                continue
            print(f"  [{os.path.basename(p)[:-5]}]")
            for k, v in d.get("results", {}).items():
                if "PA" in v:  # CMM: Perception Acc / Hallucination Resistance
                    print(f"    {k:32s} PA={v['PA']:.3f} HR={v['HR']:.3f} acc={v['acc']:.3f}  n={v['n']}")
                else:  # AVHBench: plain accuracy
                    print(f"    {k:32s} acc={v['accuracy']:.3f}  parse={v.get('parse_rate', 0):.2f}  n={v['n']}")


if __name__ == "__main__":
    main()
