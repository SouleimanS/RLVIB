#!/usr/bin/env python
"""Visual saliency maps -- where the vision tokens drive the yes/no answer -- BASE vs OURS,
on a frozen Qwen3-Omni (GPU). We hook the merger output (the vision tokens that feed the
Thinker = the bottleneck attach point), backprop the (pos-neg) answer logit to it, and take
|grad . token| per token -> reshape to the patch grid -> overlay on a frame. This is
gradient x input saliency, which the repo memo (docs/research/audio-visual-localization-maps.md)
prefers over RAW attention (decoder-only models dump attention on sink tokens). Run with the
bottleneck bypassed (base) and active (ours) on the same clip to see how the adapter
redistributes visual grounding. UNTESTED on GPU -- prints shapes/grid so the reshape can be
fixed; iterate.

  PYTHONPATH=src python scripts/visual_attention.py \
      --ckpt runs/anchored_qwen3-omni_broad/bottleneck_step60.pt \
      --video data/AVHBench/videos/02301.mp4 \
      --prompt "Is the power tool visible in the video? Answer yes or no." --pos Yes --neg No
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from rlvib.models import get_model
from rlvib.models.bottleneck import load_attached, set_bypass


def _frames(video_path, t):
    import decord
    vr = decord.VideoReader(video_path)
    idx = np.linspace(0, len(vr) - 1, max(t, 1)).round().astype(int).tolist()
    return vr.get_batch(idx).asnumpy()                       # (t, H, W, 3)


def _merge_size(model):
    cfg = getattr(model.model, "config", None)
    for path in ("thinker_config.vision_config.spatial_merge_size",
                 "vision_config.spatial_merge_size"):
        o = cfg
        ok = True
        for p in path.split("."):
            o = getattr(o, p, None)
            if o is None:
                ok = False
                break
        if ok:
            return int(o)
    return 2


def _upsample(m, hw):
    from PIL import Image
    h, w = hw
    return np.asarray(Image.fromarray((m * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-omni")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--prompt", default="Answer yes or no.")
    ap.add_argument("--pos", default="Yes", help="positive answer token (saliency = pos - neg)")
    ap.add_argument("--neg", default="No")
    ap.add_argument("--out", default="runs/attn")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tag = os.path.splitext(os.path.basename(a.video))[0]

    m = get_model(a.model)
    bns, handles = load_attached(m, a.ckpt)
    tok = getattr(m, "tokenizer", None) or m.processor.tokenizer
    pos_id = tok(a.pos, add_special_tokens=False).input_ids[0]
    neg_id = tok(a.neg, add_special_tokens=False).input_ids[0]
    lm = getattr(m.model, "thinker", m.model)               # Qwen3 -> .thinker (proven path)
    merger = m.adapter_modules()["vision"]
    ms = _merge_size(m)

    saved = {}

    def cap(_mod, _inp, out):                                # runs after the VIB hook -> post-VIB tokens
        y = out.detach().requires_grad_(True)
        saved["v"] = y
        return y
    cap_h = merger.register_forward_hook(cap)

    msg = m.message(video=a.video, prompt=a.prompt)
    inputs = m.build_inputs(msg)
    grid = inputs.get("video_grid_thw")
    if grid is None:
        grid = inputs.get("image_grid_thw")
    gt, gh, gw = (int(x) for x in grid[0].tolist())
    hm, wm = gh // ms, gw // ms

    smaps = {}
    for name, byp in (("base", True), ("ours", False)):
        set_bypass(bns, byp)
        saved.clear()
        with torch.enable_grad():
            logits = lm(**inputs).logits[:, -1, :]
            target = (logits[0, pos_id] - logits[0, neg_id]).float()
            lm.zero_grad(set_to_none=True)
            target.backward()
        v, g = saved["v"][0], saved["v"].grad[0]            # (T_v, d)
        sal = (g * v).sum(-1).abs().float().cpu().numpy()
        tv = int(sal.shape[0])
        print(f"{name}: T_v={tv}  grid(t,h,w)=({gt},{gh},{gw})  merge={ms} -> ({gt},{hm},{wm}) "
              f"= {gt * hm * wm}  pred={a.pos if float(target) > 0 else a.neg}", flush=True)
        if tv == gt * hm * wm:
            sal = sal.reshape(gt, hm, wm).mean(0)           # mean over frames -> (hm, wm)
        else:
            s = int(round(tv ** 0.5))
            sal = sal[:s * s].reshape(s, s)                 # fallback (grid mismatch -> debug)
        sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
        smaps[name] = sal
    cap_h.remove()
    for h in handles:
        h.remove()

    np.savez(os.path.join(a.out, f"{tag}.npz"), base=smaps["base"], ours=smaps["ours"])
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        frames = _frames(a.video, gt)
        frame = frames[len(frames) // 2]
        fig, ax = plt.subplots(1, 3, figsize=(9, 3))
        ax[0].imshow(frame); ax[0].set_title("frame"); ax[0].axis("off")
        for k, name in enumerate(("base", "ours"), start=1):
            ax[k].imshow(frame)
            ax[k].imshow(_upsample(smaps[name], frame.shape[:2]), cmap="jet", alpha=0.45)
            ax[k].set_title(name); ax[k].axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, f"{tag}.png"), dpi=150, bbox_inches="tight")
        print(f"wrote {a.out}/{tag}.png  and  {a.out}/{tag}.npz", flush=True)
    except Exception as e:  # noqa: BLE001 -- overlay is best-effort; raw maps are saved
        print(f"(overlay skipped: {type(e).__name__}: {e}); raw maps in {a.out}/{tag}.npz", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
