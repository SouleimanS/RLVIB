#!/usr/bin/env python
"""Leakage-free checkpoint selection: split each benchmark's per-example records into a
fixed val/test half, SELECT the checkpoint on VAL, REPORT on TEST. Uses a per-model
RELATIVE guard (vs the base row) so it is calibrated per backbone. Reads the same eval
JSONs as select_checkpoint.py (their `records`) -- NO retraining, NO re-evaluation.

Also prints the OLD leaky number (argmax over the full set) so the selection bias is visible.

  python scripts/select_holdout.py --model qwen3-omni --exp broad
  python scripts/select_holdout.py --model qwen2.5-omni --exp broad --tol 0.05
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re


def _records(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f).get("records", [])


def _valset(n, frac=0.5, seed=12345):
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    return set(idx[: int(n * frac)])


def _avh(recs):
    """-> (val_acc, test_acc, full_acc) for AVHBench (pred == label)."""
    val = _valset(len(recs))
    vc = vn = tc = tn = 0
    for i, r in enumerate(recs):
        ok = int((r["pred"] or "") == str(r["label"]).strip().lower())
        if i in val:
            vc += ok; vn += 1
        else:
            tc += ok; tn += 1
    return (vc / max(vn, 1), tc / max(tn, 1), (vc + tc) / max(vn + tn, 1))


def _cmm(recs):
    """-> dict of val/test/full PA (answer=='yes') and HR (answer=='no')."""
    val = _valset(len(recs))
    agg = {k: [0, 0] for k in ("vy", "vn", "ty", "tn")}  # [correct, total]
    for i, r in enumerate(recs):
        a = r.get("answer")
        if a not in ("yes", "no"):
            continue
        side = "v" if i in val else "t"
        key = side + ("y" if a == "yes" else "n")
        agg[key][1] += 1
        agg[key][0] += int(r["pred"] == a)

    def acc(*keys):
        c = sum(agg[k][0] for k in keys); n = sum(agg[k][1] for k in keys)
        return c / n if n else float("nan")
    return {"val_pa": acc("vy"), "val_hr": acc("vn"),
            "test_pa": acc("ty"), "test_hr": acc("tn"),
            "full_pa": acc("vy", "ty"), "full_hr": acc("vn", "tn")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-omni")
    ap.add_argument("--exp", default="")
    ap.add_argument("--tol", type=float, default=0.05, help="relative guard slack vs base PA/HR")
    args = ap.parse_args()
    xt = f"_{args.exp}" if args.exp else ""

    steps = sorted({int(m.group(1))
                    for p in glob.glob(f"runs/avhbench_{args.model}{xt}_step*.json")
                    for m in [re.search(r"_step(\d+)\.json$", p)] if m})
    if not steps:
        print("no per-step AVHBench JSONs; run select_checkpoint.sh first")
        return 1

    base_cmm = _cmm(_records(f"runs/cmm_{args.model}.json") or [])
    base_avh = _avh(_records(f"runs/avhbench_{args.model}.json") or [{"pred": None, "label": "x"}])
    g_pa, g_hr = base_cmm["val_pa"] - args.tol, base_cmm["val_hr"] - args.tol
    print(f"relative guard (on VAL): PA>={g_pa:.3f}  HR>={g_hr:.3f}   "
          f"(base val PA={base_cmm['val_pa']:.3f} HR={base_cmm['val_hr']:.3f})")
    print(f"base TEST: AVH={base_avh[1]:.3f} PA={base_cmm['test_pa']:.3f} HR={base_cmm['test_hr']:.3f}\n")
    print(f"{'step':>6}{'val_AVH':>9}{'test_AVH':>9}{'val_PA':>8}{'val_HR':>8}"
          f"{'test_PA':>8}{'test_HR':>8}  guard")

    rows, val_cand, full_cand = {}, [], []
    for s in steps:
        a = _records(f"runs/avhbench_{args.model}{xt}_step{s}.json")
        c = _records(f"runs/cmm_{args.model}{xt}_step{s}.json")
        if a is None or c is None:
            continue
        av = _avh(a); cm = _cmm(c)
        rows[s] = (av, cm)
        ok = cm["val_pa"] >= g_pa and cm["val_hr"] >= g_hr
        print(f"{s:>6}{av[0]:>9.3f}{av[1]:>9.3f}{cm['val_pa']:>8.3f}{cm['val_hr']:>8.3f}"
              f"{cm['test_pa']:>8.3f}{cm['test_hr']:>8.3f}  {'ok' if ok else 'FAIL'}")
        if ok:
            val_cand.append((av[0], s))                                  # select on VAL acc
        if cm["full_pa"] >= base_cmm["full_pa"] - args.tol and cm["full_hr"] >= base_cmm["full_hr"] - args.tol:
            full_cand.append((av[2], s))                                # OLD leaky: select on FULL acc

    if val_cand:
        s = max(val_cand)[1]
        av, cm = rows[s]
        print(f"\nHONEST  (select on val -> report on test): step{s}  "
              f"AVHBench={av[1]:.3f}  CMM_PA={cm['test_pa']:.3f}  CMM_HR={cm['test_hr']:.3f}")
    else:
        print("\nHONEST: no checkpoint passes the relative val guard")
    if full_cand:
        s = max(full_cand)[1]
        print(f"LEAKY   (old: argmax on the full reported set): step{s}  "
              f"AVHBench={rows[s][0][2]:.3f}   <- the selection-biased number")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
