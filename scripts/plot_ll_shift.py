#!/usr/bin/env python
"""Plot the likelihood-shift figure (KDE of gold-answer log-likelihood, true vs swapped audio).

One panel per probe JSON (base, +DPO, +FiLM, ...), styled after the MoD-DPO++ figure: teal =
log p(y | a, v, x) with the TRUE audio, magenta = log p(y | a', v, x) with SWAPPED audio, dashed
mean lines, and a "Shift = x" box. A bigger (more negative) shift = the answer depends more on
what the model hears.

  python scripts/plot_ll_shift.py runs/llshift_qwen3-omni.json runs/llshift_qwen3-omni_film_step160.json \
      --out paper/figures/ll_shift.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

TEAL, MAGENTA = "#5f9ea0", "#c74f9e"


def _kde(xs, grid):
    """Gaussian KDE (scipy if present, else a small hand-rolled one -- Silverman bandwidth)."""
    xs = np.asarray(xs, dtype=float)
    try:
        from scipy.stats import gaussian_kde
        return gaussian_kde(xs)(grid)
    except ImportError:
        n = len(xs)
        bw = 1.06 * xs.std(ddof=1) * n ** (-1 / 5) + 1e-9
        d = (grid[:, None] - xs[None, :]) / bw
        return np.exp(-0.5 * d ** 2).sum(axis=1) / (n * bw * np.sqrt(2 * np.pi))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsons", nargs="+", help="llshift_*.json files, one panel each")
    ap.add_argument("--labels", default=None, help="comma-separated panel titles (default: model/tag)")
    ap.add_argument("--out", default="paper/figures/ll_shift.png")
    args = ap.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed -- `pip install matplotlib` in the rlvib env.", file=sys.stderr)
        return 1

    runs = [json.load(open(p)) for p in args.jsons]
    titles = (args.labels.split(",") if args.labels
              else [f"{r.get('model', '?')}" + (f" + {r['tag']}" if r.get("tag") else " (base)")
                    for r in runs])

    fig, axes = plt.subplots(1, len(runs), figsize=(4.6 * len(runs), 3.9), squeeze=False)
    for ax, r, title in zip(axes[0], runs, titles):
        full, pert = np.asarray(r["ll_full"]), np.asarray(r["ll_pert"])
        lo = min(full.min(), pert.min()) - 0.5
        hi = max(full.max(), pert.max()) + 0.5
        grid = np.linspace(lo, hi, 400)
        for xs, c, lab in ((full, TEAL, r"$\log p(y\,|\,a,v,x)$"),
                           (pert, MAGENTA, r"$\log p(y\,|\,a',v,x)$")):
            d = _kde(xs, grid)
            ax.plot(grid, d, color=c, lw=1.8)
            ax.fill_between(grid, d, alpha=0.35, color=c, label=lab)
            ax.axvline(xs.mean(), color=c, ls="--", lw=1.4)
        shift = pert.mean() - full.mean()
        ax.annotate(f"Shift = {shift:+.2f}", xy=(0.97, 0.90), xycoords="axes fraction",
                    ha="right", fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.8))
        ax.set_title(f"{title}   (n={r['n']})", fontsize=10)
        ax.set_xlabel("Log-Likelihood")
        ax.set_ylabel("Density")
        ax.grid(alpha=0.3, ls=":")
        ax.legend(fontsize=8, loc="upper left", frameon=True)
        ax.spines[["top", "right"]].set_visible(False)
        print(f"  {title:32s} n={r['n']:3d}  shift={shift:+.3f}")
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
