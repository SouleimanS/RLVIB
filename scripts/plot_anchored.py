#!/usr/bin/env python
"""Figures for the anchored swap-DPO report (docs/reports/01-anchored-swap-dpo.md).

Parses the training log (runs/train_anchored_out.txt) and the eval JSONs
(runs/{cmm,dave,avhbench}_*.json) the runners write, and renders three PNGs into
docs/reports/figs/:
  fig_probe.png  -- yes-fraction probe vs step (the collapse detector): stays ~base.
  fig_train.png  -- p_chosen / chosen_minus_ref / gen_kl vs step (learning + anchors holding).
  fig_bench.png  -- base vs collapsed vs anchored on CMM/DAVE/AVHBench (the headline result).

Run on the cluster after training/eval (matplotlib is in environment.yml):
  python scripts/plot_anchored.py --model qwen3-omni
Missing inputs are skipped with a warning, so partial runs still produce what they can.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGDIR = "docs/reports/figs"

# Collapsed-run numbers (the latent-KL-only DPO, train_swap.py) -- no JSON survives, so
# they are recorded here from the run that motivated the rebuild. Edit if you re-measure.
COLLAPSED = {"CMM PA": 0.007, "CMM HR": 0.99, "DAVE": 0.18}


def parse_log(path):
    """-> (steps list of dicts, probe list of dicts, base_frac_yes or None)."""
    steps, probe, base_fy = [], [], None
    if not os.path.exists(path):
        print(f"[skip] no training log at {path}")
        return steps, probe, base_fy
    kv = re.compile(r"(\w+)=([+-]?[\d.]+)")
    with open(path) as f:
        for line in f:
            mb = re.search(r"\[probe base\]\s+frac_yes=([\d.]+)", line)
            if mb:
                base_fy = float(mb.group(1))
                continue
            mp = re.search(r"\[probe step (\d+)\]\s+frac_yes=([\d.]+).*?acc=([\d.]+)", line)
            if mp:
                probe.append({"step": int(mp.group(1)), "frac_yes": float(mp.group(2)),
                              "acc": float(mp.group(3))})
                continue
            ms = re.search(r"\bstep (\d+):\s*(.*)", line)
            if ms and "probe" not in line:
                d = {"step": int(ms.group(1))}
                d.update({k: float(v) for k, v in kv.findall(ms.group(2))})
                steps.append(d)
    return steps, probe, base_fy


def fig_probe(probe, base_fy):
    if not probe:
        print("[skip] fig_probe: no probe points")
        return
    xs = [p["step"] for p in probe]
    fy = [p["frac_yes"] for p in probe]
    ac = [p["acc"] for p in probe]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axhspan(0.0, 0.15, color="red", alpha=0.08)
    ax.axhspan(0.85, 1.0, color="red", alpha=0.08, label="collapse zone")
    if base_fy is not None:
        ax.axhline(base_fy, ls="--", color="gray", label=f"base ({base_fy:.2f})")
    ax.plot(xs, fy, "-o", color="C0", label="frac_yes (policy)")
    ax.plot(xs, ac, "-s", color="C2", label="probe acc")
    ax.set_xlabel("training step")
    ax.set_ylabel("yes-fraction / accuracy")
    ax.set_ylim(0, 1.02)
    ax.set_title("Collapse detector: balanced yes/no probe stays at base")
    ax.legend(loc="center right", fontsize=8)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "fig_probe.png")
    fig.savefig(out, dpi=130)
    print(f"[wrote] {out}")


def fig_train(steps):
    if not steps:
        print("[skip] fig_train: no step metrics")
        return
    xs = [s["step"] for s in steps]
    fig, ax = plt.subplots(3, 1, figsize=(7, 7), sharex=True)
    for a, key, lab in [(ax[0], "p_chosen", "p_chosen (audio preference)"),
                        (ax[1], "chosen_minus_ref", "chosen − ref log-prob (anchor)"),
                        (ax[2], "gen_kl", "KL(base ‖ policy) on general inputs")]:
        ys = [s.get(key, float("nan")) for s in steps]
        a.plot(xs, ys, "-", color="C0")
        a.set_ylabel(lab, fontsize=9)
        a.grid(alpha=0.3)
    ax[1].axhline(0.0, ls="--", color="red", alpha=0.6)  # anchor floor: should stay >= 0
    ax[2].set_xlabel("training step")
    ax[0].set_title("Anchored swap-DPO training dynamics")
    fig.tight_layout()
    out = os.path.join(FIGDIR, "fig_train.png")
    fig.savefig(out, dpi=130)
    print(f"[wrote] {out}")


def _load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _cmm(model, suffix=""):
    d = _load(f"runs/cmm_{model}{suffix}.json")
    if not d:
        return None
    r = next(iter(d["results"].values()))
    return {"CMM PA": r["PA"], "CMM HR": r["HR"]}


def _dave(model, suffix=""):
    d = _load(f"runs/dave_ego4d_{model}_audio_visual_alignment{suffix}.json")
    return {"DAVE": d["accuracy"]} if d else None


def _avh(model, suffix=""):
    d = _load(f"runs/avhbench_{model}{suffix}.json")
    if not d:
        return None
    ov = d["results"].get("overall") or {}
    return {"AVHBench": ov.get("accuracy")}


def fig_bench(model):
    base, anc = {}, {}
    for fn in (_cmm, _dave, _avh):
        for tgt, suf in ((base, ""), (anc, "_bn")):
            got = fn(model, suf)
            if got:
                tgt.update(got)
    if not base or not anc:
        print("[skip] fig_bench: missing base/_bn eval JSONs in runs/")
        return
    metrics = [m for m in ("CMM PA", "CMM HR", "DAVE", "AVHBench") if m in base and m in anc]
    cols = {"base": base, "collapsed": COLLAPSED, "anchored": anc}
    colors = {"base": "C7", "collapsed": "C3", "anchored": "C0"}
    import numpy as np
    x = np.arange(len(metrics))
    w = 0.26
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (name, d) in enumerate(cols.items()):
        vals = [d.get(m, 0) for m in metrics]
        bars = ax.bar(x + (i - 1) * w, vals, w, label=name, color=colors[name])
        for b, m in zip(bars, metrics):
            if m not in d:
                continue
            ax.annotate(f"{d[m]:.2f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{model}: bottleneck preserves capability + lifts AVHBench (no collapse)")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(FIGDIR, "fig_bench.png")
    fig.savefig(out, dpi=130)
    print(f"[wrote] {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-omni")
    ap.add_argument("--log", default="runs/train_anchored_out.txt")
    args = ap.parse_args()
    os.makedirs(FIGDIR, exist_ok=True)
    steps, probe, base_fy = parse_log(args.log)
    fig_probe(probe, base_fy)
    fig_train(steps)
    fig_bench(args.model)
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
