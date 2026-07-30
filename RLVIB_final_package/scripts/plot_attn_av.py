#!/usr/bin/env python
"""Box-plot the AV-attention fractions (MoD-DPO++ Figure 6 style) from attn_av_analysis.py JSONs.

Pass the per-run JSONs in x-axis order. Labels are auto-derived: no bottleneck -> "base",
tag starting with broad/dpo -> "DPO", tag starting with film -> "FiLM" (override with --labels).
Boxes are colored by training type and visually grouped by model. This is a WITHIN-model
base-vs-trained figure -- attention-mass is not comparable across architectures, so read each
model's group on its own (don't rank Qwen2.5 vs Qwen3 by bar height; use the benchmarks for that).

  python scripts/plot_attn_av.py \
      runs/attnav_qwen2.5-omni.json runs/attnav_qwen2.5-omni_broad_step60.json \
      runs/attnav_qwen3-omni.json   runs/attnav_qwen3-omni_broad_step60.json \
      runs/attnav_qwen3-omni_film_step220.json \
      --out paper/figures/attn_av.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_COLOR = {"base": "#4878a8", "DPO": "#e8862a", "FiLM": "#4a9a4a"}     # blue / orange / green


def _kind(d) -> str:
    if not d.get("bottleneck"):
        return "base"
    tag = (d.get("tag") or "").lower()
    if tag.startswith("broad") or "dpo" in tag:
        return "DPO"
    if tag.startswith("film"):
        return "FiLM"
    return d.get("tag") or "ours"


def _load(path):
    with open(path) as f:
        d = json.load(f)
    fr = [100.0 * x for x in d.get("fractions", [])]                 # -> percent
    return fr, d.get("model", "?"), _kind(d)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsons", nargs="+", help="attnav_*.json files, in x-axis order")
    ap.add_argument("--labels", default=None, help="comma-separated x labels (override auto)")
    ap.add_argument("--out", default="paper/figures/attn_av.png")
    ap.add_argument("--title", default="Attention to audio-visual tokens (CMM)")
    args = ap.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed -- `pip install matplotlib` in the rlvib env.", file=sys.stderr)
        return 1

    data, colors, labels, models, override = [], [], [], [], (args.labels.split(",") if args.labels else None)
    for j, p in enumerate(args.jsons):
        fr, model, kind = _load(p)
        if not fr:
            print(f"skip {p}: no fractions (n=0)", file=sys.stderr)
            continue
        data.append(fr)
        colors.append(_COLOR.get(kind, "#888888"))
        models.append(model)
        labels.append(override[j] if override and j < len(override) else f"{model}\n{kind}")
        print(f"  {model:14s} {kind:5s}  n={len(fr):4d}  mean={sum(fr) / len(fr):.2f}%")
    if not data:
        print("nothing to plot.", file=sys.stderr)
        return 1

    # positions: add a gap whenever the model changes -> visual grouping per backbone
    pos, x = [], 1.0
    for i, mdl in enumerate(models):
        if i and mdl != models[i - 1]:
            x += 1.0
        pos.append(x)
        x += 1.0

    fig, ax = plt.subplots(figsize=(0.95 * (pos[-1] + 1) + 1.0, 4.4))
    bp = ax.boxplot(data, positions=pos, patch_artist=True, widths=0.7, showfliers=False,
                    medianprops={"color": "black", "linewidth": 1.3})
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.85)
    ax.set_xticks(pos)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Attention to AV tokens (% of total input attention)", fontsize=9)
    ax.set_title(args.title, fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=_COLOR[k], alpha=0.85) for k in _COLOR]
    ax.legend(handles, list(_COLOR), loc="upper right", fontsize=8, frameon=False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
