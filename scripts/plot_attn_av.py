#!/usr/bin/env python
"""Box-plot the AV-attention fractions (MoD-DPO++ Figure 6) from attn_av_analysis.py JSONs.

Pass the per-run JSONs in the order you want them on the x-axis; base rows (no bottleneck)
are drawn blue, trained rows (a bottleneck) orange -- matching the paper's figure.

  python scripts/plot_attn_av.py \
      runs/attnav_qwen2.5-omni.json runs/attnav_qwen2.5-omni_broad_step60.json \
      runs/attnav_qwen3-omni.json   runs/attnav_qwen3-omni_film_step90.json \
      --out paper/figures/attn_av.png

Each box's label is "<model>\n<base|tag>". Writes a PNG (300 dpi) ready for the slides.
"""
from __future__ import annotations

import argparse
import json
import sys


def _load(path):
    with open(path) as f:
        d = json.load(f)
    fr = [100.0 * x for x in d.get("fractions", [])]          # -> percent
    base = not d.get("bottleneck")
    label = f"{d.get('model', '?')}\n{'base' if base else (d.get('tag') or 'ours')}"
    return fr, base, label, d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsons", nargs="+", help="attnav_*.json files, in x-axis order")
    ap.add_argument("--out", default="paper/figures/attn_av.png")
    ap.add_argument("--title", default="Total attention to audio-visual tokens (CMM)")
    args = ap.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed -- `pip install matplotlib` (or conda) in the rlvib env.",
              file=sys.stderr)
        return 1

    data, colors, labels = [], [], []
    for p in args.jsons:
        fr, base, label, d = _load(p)
        if not fr:
            print(f"skip {p}: no fractions (n=0)", file=sys.stderr)
            continue
        data.append(fr)
        colors.append("#4878a8" if base else "#e8862a")       # blue base / orange trained
        labels.append(label)
        print(f"  {label.replace(chr(10), ' / '):32s} n={len(fr):4d}  mean={sum(fr) / len(fr):.2f}%")
    if not data:
        print("nothing to plot.", file=sys.stderr)
        return 1

    import os
    fig, ax = plt.subplots(figsize=(1.6 * len(data) + 1.5, 4.2))
    bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=False,
                    medianprops={"color": "black", "linewidth": 1.3})
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.85)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Attention to AV tokens (% of total input attention)", fontsize=9)
    ax.set_title(args.title, fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
