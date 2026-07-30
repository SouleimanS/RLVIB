#!/usr/bin/env python
"""Step-5 swap-DPO smoke: contrastive DPO on audio-swapped AVE clips.

Builds audio-swap pairs (video = seen event, audio = heard event) and runs dpo_step
preferring the HEARD letter over the SEEN letter -> trains the bottleneck to read
audio against the visual shortcut. Expect `margin` and `p_chosen` to climb off the
floor (the base shortcuts to the seen event, so p_chosen starts near 0).

  python scripts/train_swap_smoke.py --model qwen3-omni --steps 8 --batch 2
"""
from __future__ import annotations

import argparse
import random

import torch

from rlvib.data import ave
from rlvib.data.pairs import make_swap_examples
from rlvib.models import get_model
from rlvib.models.bottleneck import VariationalBottleneck, attach_bottlenecks
from rlvib.train.dpo import dpo_step


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-omni")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--beta-kl", type=float, default=0.01)
    ap.add_argument("--swap-dir", default="data/AVE/swapped")
    args = ap.parse_args()

    m = get_model(args.model)
    bottlenecks, handles = attach_bottlenecks(m, cls=VariationalBottleneck)
    bottlenecks.train()
    opt = torch.optim.AdamW(bottlenecks.parameters(), lr=args.lr)

    cats = ave.categories()
    items = ave.load_ave("train")
    rng = random.Random(0)
    pairs = make_swap_examples(items, args.steps * args.batch, args.swap_dir, cats, rng=rng)
    print(f"swap pairs: {len(pairs)} | steps={args.steps} batch={args.batch}", flush=True)

    idx = 0
    for step in range(args.steps):
        batch = []
        while len(batch) < args.batch and idx < len(pairs):
            rec = pairs[idx]
            idx += 1
            msg = m.message(video=rec["video_path"],
                            prompt=ave.format_mcq(rec["question"], rec["options"]))
            batch.append({"messages": msg, "chosen_letter": rec["audio_letter"],
                          "rejected_letter": rec["visual_letter"]})
        if not batch:
            break
        metrics = dpo_step(m, bottlenecks, opt, batch, beta=args.beta, beta_kl=args.beta_kl)
        print(f"step {step}: " + "  ".join(f"{k}={v:+.4f}" for k, v in metrics.items()), flush=True)

    for h in handles:
        h.remove()
    print("=== swap-DPO smoke done ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
