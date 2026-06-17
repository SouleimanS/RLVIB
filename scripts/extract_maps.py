#!/usr/bin/env python
"""Extract the IB rate maps (per-token KL) from a trained bottleneck checkpoint.

The VIB's per-token KL = "bits the bottleneck spends on this token" = an
information-theoretic saliency, and it falls out of the method (no attention needed):
  audio  -> (T_a,)            a 1-D temporal saliency over the audio timeline
  vision -> (T_v,) -> (t,h,w) a coarse spatiotemporal heatmap (via video_grid_thw)

This is also a sanity check on the +0.30 heard_rate: if the bits land on the actual
sounding segments (not uniformly on all audio), the gain is real grounding, not an
audio-bias. Prints token counts + grid so we can verify the reshape before trusting it.

  python scripts/extract_maps.py --ckpt runs/bottleneck_swap.pt --n 3
"""
from __future__ import annotations

import argparse
import math
import os
import random
import shutil

import numpy as np
import torch

from rlvib.data import ave
from rlvib.data.pairs import make_swap_examples
from rlvib.models import get_model
from rlvib.models.bottleneck import VariationalBottleneck, attach_bottlenecks, set_bypass


def _san(s: str) -> str:
    """Filesystem-safe slug for an event name (categories have spaces/commas)."""
    return "".join(c if c.isalnum() else "-" for c in s).strip("-")[:28]


def _load_frames(video_path, t):
    """Sample t frames evenly across the clip (approx. temporal alignment to the tokens)."""
    import decord

    vr = decord.VideoReader(video_path)
    n = len(vr)
    idx = np.linspace(0, n - 1, t).round().astype(int).tolist()
    return vr.get_batch(idx).asnumpy()  # (t, H, W, 3) uint8 RGB


def _render_combined(plt, path, audio, vmap, frames, k, heard, seen, prompt):
    """One figure per clip: header + prompt, audio rate (full width), and the real video
    frames with the rate heatmap overlaid (percentile-normalized; bare heatmap if no frames)."""
    header = f"clip{k}:   heard = {heard}    |    seen = {seen}\n\n{prompt}"
    nlines = header.count("\n") + 1
    if vmap is not None:
        t = int(vmap.shape[0])
        ncol = min(t, 4)
        nrow = math.ceil(t / ncol)
        lo, hi = float(np.percentile(vmap, 5)), float(np.percentile(vmap, 95))
    else:
        t, ncol, nrow, lo, hi = 0, 1, 0, 0.0, 1.0
    head_in = 0.20 * nlines + 0.3
    fig_h = head_in + 2.4 + 2.6 * nrow
    fig = plt.figure(figsize=(3.6 * max(ncol, 2), fig_h))
    fig.text(0.01, 0.99, header, va="top", ha="left", fontsize=8, family="monospace")
    top = 1.0 - head_in / fig_h
    gs = fig.add_gridspec(nrow + 1, max(ncol, 1), height_ratios=[1.1] + [1] * nrow,
                          top=top, bottom=0.03, left=0.06, right=0.98, hspace=0.28, wspace=0.05)
    ax = fig.add_subplot(gs[0, :])
    ax.plot(audio)
    ax.set_title("audio rate -- KL bits per token (time ->)", fontsize=9)
    ax.set_xlabel("audio token (time)")
    ax.set_ylabel("KL bits")
    for fi in range(t):
        r, c = divmod(fi, ncol)
        axf = fig.add_subplot(gs[1 + r, c])
        heat = np.clip((vmap[fi] - lo) / (hi - lo + 1e-6), 0.0, 1.0)
        if frames is not None:
            H, W = frames[fi].shape[:2]
            axf.imshow(frames[fi], extent=(0, W, H, 0))
            axf.imshow(heat, cmap="turbo", alpha=0.5, extent=(0, W, H, 0),
                       interpolation="bilinear", vmin=0.0, vmax=1.0)
            axf.set_xlim(0, W)
            axf.set_ylim(H, 0)
        else:
            axf.imshow(heat, cmap="turbo", vmin=0.0, vmax=1.0)
        axf.set_title(f"frame {fi}", fontsize=8)
        axf.axis("off")
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-omni")
    ap.add_argument("--ckpt", default="runs/bottleneck_swap.pt")
    ap.add_argument("--out", default="runs/maps")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--swap-dir", default="data/AVE/swapped")
    args = ap.parse_args()

    m = get_model(args.model)
    bns, handles = attach_bottlenecks(m, cls=VariationalBottleneck)
    ckpt = torch.load(args.ckpt, map_location=m.device)
    bns.load_state_dict(ckpt["state_dict"])
    bns.eval()                 # z = mu (deterministic rate)
    set_bypass(bns, False)     # bottleneck active -> KL is computed
    os.makedirs(args.out, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:           # noqa: BLE001
        plt = None
        print("matplotlib not available -> saving .npy only")

    vis = m.model.thinker.visual
    merge = (getattr(getattr(vis, "config", vis), "spatial_merge_size", None)
             or getattr(vis, "spatial_merge_size", 2))

    cats = ave.categories()
    items = ave.load_ave("train")
    rng = random.Random(123)   # different draw from train/val
    pairs = make_swap_examples(items, args.n, args.swap_dir, cats, rng=rng)

    lm = getattr(m.model, "thinker", m.model)
    for k, rec in enumerate(pairs):
        msg = m.message(video=rec["video_path"],
                        prompt=ave.format_mcq(rec["question"], rec["options"]))
        inputs = m.build_inputs(msg, use_audio_in_video=True)
        with torch.no_grad():
            lm(**inputs)        # triggers the adapter hooks -> populates last_kl_per_token

        a = bns["audio"].last_kl_per_token.float().cpu().numpy().reshape(-1)
        v = bns["vision"].last_kl_per_token.float().cpu().numpy().reshape(-1)
        grid = inputs.get("video_grid_thw", None)
        grid_l = None if grid is None else grid[0].tolist()
        print(f"[{k}] heard={rec['audio_event']}  seen={rec['visual_event']}  "
              f"audio_tokens={a.shape[0]}  vision_tokens={v.shape[0]}  "
              f"video_grid_thw={grid_l}  merge={merge}", flush=True)

        np.save(os.path.join(args.out, f"clip{k}_audio_kl.npy"), a)
        np.save(os.path.join(args.out, f"clip{k}_vision_kl.npy"), v)

        vmap = None
        if grid_l is not None:
            t, h, w = grid_l
            hm, wm = h // merge, w // merge
            if t * hm * wm == v.shape[0]:
                vmap = v.reshape(t, hm, wm)
                np.save(os.path.join(args.out, f"clip{k}_vision_kl_grid.npy"), vmap)
                print(f"     vision reshaped -> (t={t}, h={hm}, w={wm})  OK", flush=True)
            else:
                print(f"     vision reshape MISMATCH: t*h/m*w/m={t * hm * wm} != "
                      f"vision_tokens={v.shape[0]} (inspect layout/merge before trusting)",
                      flush=True)

        seen, heard = rec["visual_event"], rec["audio_event"]
        prompt = ave.format_mcq(rec["question"], rec["options"])
        shutil.copy(rec["video_path"],  # the swapped clip the model saw/heard, for reference
                    os.path.join(args.out, f"clip{k}_video_seen-{_san(seen)}_heard-{_san(heard)}.mp4"))
        frames = None
        if vmap is not None:
            try:
                frames = _load_frames(rec["video_path"], int(vmap.shape[0]))
            except Exception as e:  # noqa: BLE001
                print(f"     frame load failed ({e}); heatmap-only", flush=True)
        if plt is not None:  # combined PNG: header+prompt, audio rate, frames + heatmap overlay
            _render_combined(plt, os.path.join(args.out, f"clip{k}_combined.png"),
                             a, vmap, frames, k, heard, seen, prompt)

    for hd in handles:
        hd.remove()
    print(f"=== maps written to {args.out} ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
