#!/usr/bin/env python
"""Per-token information + edit map of a trained VIB -- interpretability readout.

Forward-pass only. For each clip we read, per modality VIB:
  * the KL "rate" map  (last_kl_per_token, bits/token) -- where it spends information;
  * the EDIT map       (||out(z)|| per token, and relative to ||x||) -- how much it
    actually changes the tokens. KL is the rate, not the edit: a big rate can still be a
    small perturbation, so we report both.
Note each VIB is per-modality (vision sees only the video tokens, audio only the audio
tokens), so its transform is necessarily unconditional w.r.t. the other modality.

  python scripts/vib_saliency.py --bottleneck runs/anchored_qwen3-omni_broad/bottleneck_step60.pt --n 8
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
    kb = kpt.detach().float().flatten() * NAT2BIT                 # bits/token
    rpt = bn.last_residual_per_token.detach().float().flatten()   # ||out(z)|| per token
    ipt = bn.last_input_norm_per_token.detach().float().flatten().clamp(min=1e-6)
    n = kb.numel()
    tot = float(kb.sum())
    k = max(1, n // 10)
    return {"T": n, "bits_mean": tot / n,
            "top10": float(kb.topk(k).values.sum()) / tot if tot > 0 else 0.0,
            "edit": float(rpt.mean()), "rel": float((rpt / ipt).mean())}


@torch.no_grad()
def probe(model, bns, msg):
    answer_logp_vec(model, msg)                                   # forward; hooks fill the maps
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
    sel = items[: args.n]

    agg = {"audio": [0.0, 0.0], "vision": [0.0, 0.0]}            # [bits_mean, rel]
    for i, it in enumerate(sel):
        st = probe(m, bns, m.message(video=it["video_path"], prompt=args.prompt))
        v, a = st["vision"], st["audio"]
        print(f"clip {i:2d} {it['category']:>16}  "
              f"VIS T={v['T']:4d} bits/tok={v['bits_mean']:7.2f} edit={v['edit']:6.2f} "
              f"rel={v['rel']:6.1%} top10%={v['top10']:.0%}   "
              f"AUD bits/tok={a['bits_mean']:.3f} rel={a['rel']:.1%}", flush=True)
        for nm, s in (("vision", v), ("audio", a)):
            agg[nm][0] += s["bits_mean"]
            agg[nm][1] += s["rel"]

    k = max(1, len(sel))
    print(f"\nSUMMARY (n={k})")
    print(f"  VISION  bits/tok={agg['vision'][0]/k:7.2f}   rel-edit={agg['vision'][1]/k:.1%}")
    print(f"  AUDIO   bits/tok={agg['audio'][0]/k:7.3f}   rel-edit={agg['audio'][1]/k:.1%}")
    print("  rel-edit = mean ||out(z)|| / ||x|| : the fraction of each token's magnitude the VIB "
          "actually changes.\n  (big bits + big rel = heavy rewrite; big bits + tiny rel = encodes "
          "a lot but perturbs little.)", flush=True)

    for h in handles:
        h.remove()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
