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
import os
import random

import numpy as np
import torch

from rlvib.data import ave
from rlvib.data.pairs import make_swap_examples
from rlvib.models import get_model
from rlvib.models.bottleneck import VariationalBottleneck, attach_bottlenecks, set_bypass


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

        if grid_l is not None:
            t, h, w = grid_l
            hm, wm = h // merge, w // merge
            if t * hm * wm == v.shape[0]:
                vmap = v.reshape(t, hm, wm)
                np.save(os.path.join(args.out, f"clip{k}_vision_kl_grid.npy"), vmap)
                print(f"     vision reshaped -> (t={t}, h={hm}, w={wm})  OK", flush=True)
                if plt is not None:
                    for fi in range(t):
                        plt.imsave(os.path.join(args.out, f"clip{k}_frame{fi}_visionkl.png"),
                                   vmap[fi], cmap="hot")
            else:
                print(f"     vision reshape MISMATCH: t*h/m*w/m={t * hm * wm} != "
                      f"vision_tokens={v.shape[0]} (inspect layout/merge before trusting)",
                      flush=True)

        if plt is not None:
            plt.figure(figsize=(6, 2))
            plt.plot(a)
            plt.title(f"clip{k} audio rate (heard={rec['audio_event']})")
            plt.xlabel("audio token (time)")
            plt.ylabel("KL bits")
            plt.tight_layout()
            plt.savefig(os.path.join(args.out, f"clip{k}_audio_kl.png"))
            plt.close()

    for hd in handles:
        hd.remove()
    print(f"=== maps written to {args.out} ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
