#!/usr/bin/env python
"""Paired base-vs-adapted statistics on the SAME test examples (review #1 + #7).

The headline 0.647->0.694 compares two independently-noisy point estimates; this does the
statistically correct thing instead. It reads the eval JSONs' per-example records (NO GPU,
NO re-eval), restricts to the same val/test split as select_holdout (test half, seed 12345),
pairs base vs a selected checkpoint item-by-item, and reports per benchmark axis:

  * McNemar exact test on the discordant pairs:
        b = base-right / adapted-wrong,  c = base-wrong / adapted-right
    (the correct paired comparison -- both models see identical items). Prints b, c, p.
  * bootstrap 95% CI over examples for base acc, adapted acc, and Delta = adapted - base.
  * the adapted model's yes-rate on the test half (constant-answer / collapse detector).

--pool aggregates discordant pairs across seeds (combined McNemar -> larger effective n).

Stdlib only (json/math/random) -- runs anywhere, including the login node.
  python scripts/paired_stats.py --model qwen3-omni --exp broad --step 60
  python scripts/paired_stats.py --model qwen3-omni --pool broad:60,broad_s1:90,broad_s2:60
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random


def _records(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f).get("records", [])


def _testset(n, frac=0.5, seed=12345):
    """Test half = complement of select_holdout's validation half (same seed/order)."""
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    return set(idx[int(n * frac):])


def _mcnemar_p(b, c):
    """Exact two-sided McNemar (binomial on the b+c discordant pairs, p0=0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * tail)


def _ci(xs, lo=0.025, hi=0.975):
    xs = sorted(xs)
    n = len(xs)
    return xs[int(lo * n)], xs[min(n - 1, int(hi * n))]


def _bootstrap(bc, ac, B=5000, seed=0):
    """95% CIs for base acc, adapted acc, Delta over example resamples (paired)."""
    rng = random.Random(seed)
    n = len(bc)
    base_s, adapt_s, delta_s = [], [], []
    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]
        b = sum(bc[i] for i in idx) / n
        a = sum(ac[i] for i in idx) / n
        base_s.append(b)
        adapt_s.append(a)
        delta_s.append(a - b)
    return _ci(base_s), _ci(adapt_s), _ci(delta_s)


def _paired(base, adapt, keep, correct):
    """Aligned (base_correct, adapt_correct) over indices in `keep` that pass `select`."""
    n = min(len(base), len(adapt))
    bc, ac = [], []
    for i in range(n):
        if i in keep and correct(base[i]) is not None:
            bc.append(correct(base[i]))
            ac.append(correct(adapt[i]))
    return bc, ac


def _report(name, bc, ac, adapt_yes=None):
    if not bc:
        print(f"  {name}: no paired examples")
        return
    n = len(bc)
    b = sum(1 for x, y in zip(bc, ac) if x and not y)   # base right, adapted wrong
    c = sum(1 for x, y in zip(bc, ac) if not x and y)   # base wrong, adapted right
    p = _mcnemar_p(b, c)
    (bl, bh), (al, ah), (dl, dh) = _bootstrap(bc, ac)
    base_acc, adapt_acc = sum(bc) / n, sum(ac) / n
    star = "  *" if p < 0.05 else ("  (marginal)" if p < 0.10 else "")
    print(f"  {name:10s} n={n:3d}  base={base_acc:.3f}[{bl:.3f},{bh:.3f}]  "
          f"adapted={adapt_acc:.3f}[{al:.3f},{ah:.3f}]")
    print(f"             Delta={adapt_acc - base_acc:+.3f} [{dl:+.3f},{dh:+.3f}]   "
          f"McNemar b={b} c={c} p={p:.4f}{star}"
          + (f"   adapted yes-rate={adapt_yes:.2f}" if adapt_yes is not None else ""))
    return b, c


def _avh_correct(r):
    pred = r.get("pred")
    return int((pred or "") == str(r.get("label", "")).strip().lower())


def _cmm_correct_for(answer):
    def f(r):
        if r.get("answer") != answer:
            return None
        return int(r.get("pred") == answer)
    return f


def _yes_rate(recs, keep):
    sub = [r for i, r in enumerate(recs) if i in keep]
    return (sum(1 for r in sub if r.get("pred") == "yes") / len(sub)) if sub else float("nan")


def _one(model, exp, step):
    xt = f"_{exp}" if exp else ""
    base_a = _records(f"runs/avhbench_{model}.json")
    base_c = _records(f"runs/cmm_{model}.json")
    adt_a = _records(f"runs/avhbench_{model}{xt}_step{step}.json")
    adt_c = _records(f"runs/cmm_{model}{xt}_step{step}.json")
    if not all((base_a, base_c, adt_a, adt_c)):
        print(f"  missing JSONs for {model} exp={exp} step={step}")
        return None, None
    keep_a = _testset(min(len(base_a), len(adt_a)))
    keep_c = _testset(min(len(base_c), len(adt_c)))
    print(f"\n=== {model}  exp={exp or '-'}  step={step}  (test half) ===")
    bc, ac = _paired(base_a, adt_a, keep_a, _avh_correct)
    bcd = _report("AVHBench", bc, ac, adapt_yes=_yes_rate(adt_a, keep_a))
    for axis, ans in (("CMM-PA", "yes"), ("CMM-HR", "no")):
        bc, ac = _paired(base_c, adt_c, keep_c, _cmm_correct_for(ans))
        _report(axis, bc, ac)
    return bcd  # AVHBench (b, c) for pooling


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-omni")
    ap.add_argument("--exp", default="broad")
    ap.add_argument("--step", type=int, default=60)
    ap.add_argument("--pool", default="", help="exp:step,exp:step,... -> combined AVHBench McNemar")
    args = ap.parse_args()

    if args.pool:
        B = C = 0
        for tok in args.pool.split(","):
            exp, step = tok.split(":")
            bc = _one(args.model, exp, int(step))
            if bc:
                B += bc[0]
                C += bc[1]
        p = _mcnemar_p(B, C)
        print(f"\n=== POOLED AVHBench across seeds: b={B} c={C} "
              f"McNemar p={p:.4f}{'  *' if p < 0.05 else ''} ===")
    else:
        _one(args.model, args.exp, args.step)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
