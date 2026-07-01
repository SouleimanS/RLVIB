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
    """Exact two-sided McNemar (binomial on the b+c discordant pairs, p0=0.5).

    Computed in log-space (lgamma) so math.comb doesn't overflow float for large b+c
    (AVHBench has thousands of pairs -> comb(n, n/2) exceeds ~1.8e308 around n>1024)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    ln_half = n * math.log(0.5)
    tail = math.fsum(
        math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + ln_half)
        for i in range(k + 1)
    )
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


def _keep(n, dev):
    """Held-out test indices: records[dev:] if dev>0 (full-eval; dev = the dev-subset size);
    else select_holdout's 50/50 split (subset eval)."""
    return set(range(dev, n)) if dev > 0 else _testset(n)


def _side(model, exp, step, sx):
    """(avhbench, cmm) records for one side: exp=None -> the no-adapter base run; else the
    exp/step checkpoint. Lets --vs pair two ADAPTED checkpoints (e.g. DPO vs FiLM)."""
    if exp is None:
        return _records(f"runs/avhbench_{model}{sx}.json"), _records(f"runs/cmm_{model}{sx}.json")
    return (_records(f"runs/avhbench_{model}_{exp}{sx}_step{step}.json"),
            _records(f"runs/cmm_{model}_{exp}{sx}_step{step}.json"))


def _one(model, exp, step, dev=0, suffix="", base=None):
    sx = f"_{suffix}" if suffix else ""
    base_a, base_c = _side(model, base[0] if base else None, base[1] if base else None, sx)
    adt_a, adt_c = _side(model, exp, step, sx)
    if not all((base_a, base_c, adt_a, adt_c)):
        print(f"  missing JSONs ({model} {base or 'base'} vs {exp}@{step} suffix={suffix or '-'})")
        return None
    keep_a = _keep(min(len(base_a), len(adt_a)), dev)
    keep_c = _keep(min(len(base_c), len(adt_c)), dev)
    scope = f"held-out records[{dev}:]" if dev else "test half"
    lhs = f"{base[0]}@{base[1]}" if base else "base"
    print(f"\n=== {model}  {lhs} -> {exp}@{step}  ({scope})  [left=base col, right=adapted col] ===")
    bc, ac = _paired(base_a, adt_a, keep_a, _avh_correct)
    bcd = _report("AVHBench", bc, ac, adapt_yes=_yes_rate(adt_a, keep_a))
    for short, full in (("A->V", "Audio-driven Video Hallucination"),
                        ("V->A", "Video-driven Audio Hallucination"),
                        ("AV-match", "AV Matching")):           # per hallucination type
        keep_t = {i for i in keep_a if i < len(base_a) and base_a[i].get("task") == full}
        bt, at = _paired(base_a, adt_a, keep_t, _avh_correct)
        _report("  " + short, bt, at)
    for axis, ans in (("CMM-PA", "yes"), ("CMM-HR", "no")):
        bc2, ac2 = _paired(base_c, adt_c, keep_c, _cmm_correct_for(ans))
        _report(axis, bc2, ac2)
    return bcd  # AVHBench (b, c) for pooling


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-omni")
    ap.add_argument("--exp", default="broad")
    ap.add_argument("--step", type=int, default=60)
    ap.add_argument("--pool", default="", help="exp:step,exp:step,... -> combined AVHBench McNemar")
    ap.add_argument("--dev", type=int, default=0,
                    help="full-eval: report on records[DEV:] (full minus the DEV-size dev subset)")
    ap.add_argument("--suffix", default="", help="read *_{suffix}.json eval files (e.g. 'full')")
    ap.add_argument("--vs", default="", help="exp:step -> use THIS checkpoint (e.g. broad:60) as the "
                    "baseline instead of the no-adapter base, for a paired DPO-vs-FiLM test")
    args = ap.parse_args()

    base = None
    if args.vs:
        _e, _s = args.vs.split(":")
        base = (_e, int(_s))
    if args.pool:
        B = C = 0
        for tok in args.pool.split(","):
            exp, step = tok.split(":")
            bc = _one(args.model, exp, int(step), args.dev, args.suffix)
            if bc:
                B += bc[0]
                C += bc[1]
        p = _mcnemar_p(B, C)
        print(f"\n=== POOLED AVHBench across seeds: b={B} c={C} "
              f"McNemar p={p:.4f}{'  *' if p < 0.05 else ''} ===")
    else:
        _one(args.model, args.exp, args.step, args.dev, args.suffix, base=base)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
