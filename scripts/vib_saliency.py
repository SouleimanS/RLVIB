#!/usr/bin/env python
"""Per-token information map of a trained VIB -- interpretability readout (option 1).

For each clip we run ONE forward pass with the held bottleneck attached and read each
modality VIB's `last_kl_per_token`: the KL "bits" it spends per token (the code's built-in
saliency map, models/bottleneck.py). This shows WHERE the VIB invests its information
budget -- audio vs vision, and concentrated on a few tokens vs spread thin -- i.e. what the
trained bottleneck is actually doing to the media tokens. Forward-pass only, no training.

  python scripts/vib_saliency.py \
      --bottleneck runs/anchored_qwen3-omni_broad/bottleneck_step60.pt --n 8

`tot` = total bits the VIB adds over all tokens (near 0 => near-identity / barely editing);
`mean` = bits/token; `top10%` = share of the bits in the top 10% of tokens (high => the VIB
focuses on a few tokens; ~10% => diffuse).
"""
from __future__ import annotations

import argparse
import math
import random

import torch

from rlvib.data import ave
from rlvib.models import get_model
from rlvib.models.bottleneck import load_attached
from rlvib.train.dpo import answer_logp_vec

NAT2BIT = 1.0 / math.log(2.0)


def _stats(bn):
    kpt = getattr(bn, "last_kl_per_token", None)
    if kpt is None:
        return None
    v = kpt.detach().float().flatten() * NAT2BIT          # nats -> bits, per token
    n = v.numel()
    tot = float(v.sum())
    k = max(1, n // 10)
    top = float(v.topk(k).values.sum())
    return {"T": n, "total": tot, "mean": tot / n, "max": float(v.max()),
            "top10": top / tot if tot > 0 else 0.0}


@torch.no_grad()
def probe(model, bns, msg):
    answer_logp_vec(model, msg)                            # forward; hooks fill last_kl_per_token
    return {name: _stats(bns[name]) for name in ("audio", "vision")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-omni")
    ap.add_argument("--bottleneck", default="runs/anchored_qwen3-omni_broad/bottleneck_step60.pt")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--prompt", default="What do you see and hear in this clip?")
    args = ap.parse_args()

    m = get_model(args.model)
    bns, handles = load_attached(m, args.bottleneck)
    print(f"held bottleneck <- {args.bottleneck}\n", flush=True)

    items = ave.load_ave("train")
    random.Random(0).shuffle(items)
    items = items[: args.n]

    sums = {"audio": [0.0, 0.0], "vision": [0.0, 0.0]}    # [sum total, sum mean] over clips
    for i, it in enumerate(items):
        st = probe(m, bns, m.message(video=it["video_path"], prompt=args.prompt))
        v, a = st["vision"], st["audio"]
        print(f"clip {i:2d} {it['category']:>18}  "
              f"VISION T={v['T']:4d} tot={v['total']:7.1f} mean={v['mean']:.3f} "
              f"max={v['max']:5.2f} top10%={v['top10']:.0%}   "
              f"AUDIO T={a['T']:4d} tot={a['total']:7.1f} mean={a['mean']:.3f} "
              f"max={a['max']:5.2f} top10%={a['top10']:.0%}", flush=True)
        for nm, s in (("vision", v), ("audio", a)):
            sums[nm][0] += s["total"]
            sums[nm][1] += s["mean"]

    k = max(1, len(items))
    vt, vm = sums["vision"][0] / k, sums["vision"][1] / k
    at, am = sums["audio"][0] / k, sums["audio"][1] / k
    print(f"\nSUMMARY (n={k})  mean per clip:")
    print(f"  VISION  total={vt:7.1f} bits   per-token={vm:.3f}")
    print(f"  AUDIO   total={at:7.1f} bits   per-token={am:.3f}")
    print(f"  -> the VIB invests more bits/token in "
          f"{'VISION' if vm > am else 'AUDIO'} (mean {max(vm, am):.3f} vs {min(vm, am):.3f})", flush=True)

    for h in handles:
        h.remove()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
