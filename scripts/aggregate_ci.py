#!/usr/bin/env python
"""Bootstrap CIs + across-seed aggregation for selected-checkpoint eval JSONs.

Per file: re-reads the per-example `records` and bootstraps a 95% CI on the headline
metric (AVHBench overall acc; CMM PA & HR; DAVE acc). Given several files of the SAME
recipe (e.g. seeds), also reports mean +/- std of the point estimates.

  python scripts/aggregate_ci.py runs/avhbench_qwen3-omni_broad_step60.json \
                                 runs/cmm_qwen3-omni_broad_step60.json
  python scripts/aggregate_ci.py runs/avhbench_qwen3-omni_broad_s*_step60.json   # across seeds
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def _correct(path, d):
    """Per-example correctness {metric: [0/1, ...]} for the file's benchmark."""
    recs = d.get("records", [])
    base = os.path.basename(path)
    if base.startswith("avhbench"):
        return {"AVHBench": [int((r["pred"] or "") == str(r["label"]).strip().lower()) for r in recs]}
    if base.startswith("cmm"):
        return {"CMM_PA": [int(r["pred"] == r["answer"]) for r in recs if r["answer"] == "yes"],
                "CMM_HR": [int(r["pred"] == r["answer"]) for r in recs if r["answer"] == "no"]}
    if base.startswith("dave"):
        return {"DAVE": [int(r["pred"] == r["gt"]) for r in recs]}
    return {}


def _ci(correct, n_boot=10000, seed=0):
    a = np.asarray(correct, float)
    if a.size == 0:
        return float("nan"), float("nan"), float("nan"), 0
    rng = np.random.default_rng(seed)
    boot = a[rng.integers(0, a.size, size=(n_boot, a.size))].mean(axis=1)
    return a.mean(), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)), a.size


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="selected-checkpoint eval JSONs")
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()

    points: dict[str, list[float]] = {}
    for path in args.files:
        if not os.path.exists(path):
            print(f"!! missing {path}")
            continue
        with open(path) as f:
            d = json.load(f)
        for metric, correct in _correct(path, d).items():
            mean, lo, hi, n = _ci(correct, args.n_boot)
            print(f"{os.path.basename(path):52s} {metric:8s} "
                  f"{mean:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  (n={n})")
            points.setdefault(metric, []).append(mean)

    multi = {k: v for k, v in points.items() if len(v) > 1}
    if multi:
        print("\nAcross runs/seeds (mean +/- std):")
        for metric, vals in multi.items():
            a = np.asarray(vals)
            print(f"  {metric:8s} {a.mean():.3f} +/- {a.std(ddof=1):.3f}  (k={a.size})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
