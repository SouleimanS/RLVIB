#!/usr/bin/env python
"""Step-5 training smoke: a few mDPO steps on AVE (audio-drop counterfactual).

Validates the training loop end-to-end before a real run:
  - attach the VIB bottleneck, optimizer over its params only
  - per step: build MCQ from AVE clips, run mDPO (full-AV vs audio-dropped), step
  - expect a finite, generally decreasing loss and a non-trivial margin/KL.

  python scripts/train_smoke.py --model qwen3-omni --steps 5 --batch 2
"""
from __future__ import annotations

import argparse
import random

import torch

from rlvib.data import ave
from rlvib.models import get_model
from rlvib.models.bottleneck import VariationalBottleneck, attach_bottlenecks
from rlvib.train.dpo import mdpo_step


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-omni")
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--beta-kl", type=float, default=0.01)
    args = ap.parse_args()

    m = get_model(args.model)
    bottlenecks, handles = attach_bottlenecks(m, cls=VariationalBottleneck)
    bottlenecks.train()
    opt = torch.optim.AdamW(bottlenecks.parameters(), lr=args.lr)

    cats = ave.categories()
    items = ave.load_ave("train")
    rng = random.Random(0)
    rng.shuffle(items)
    print(f"AVE train items: {len(items)} | steps={args.steps} batch={args.batch}", flush=True)

    idx = 0
    for step in range(args.steps):
        batch = []
        while len(batch) < args.batch and idx < len(items):
            it = items[idx]
            idx += 1
            mcq = ave.make_mcq(it["category"], cats, rng=rng)
            msg = m.message(video=it["video_path"],
                            prompt=ave.format_mcq(mcq["question"], mcq["options"]))
            batch.append({"messages": msg, "gold_letter": mcq["gold_letter"]})
        metrics = mdpo_step(m, bottlenecks, opt, batch, beta=args.beta, beta_kl=args.beta_kl)
        print(f"step {step}: " + "  ".join(f"{k}={v:+.4f}" for k, v in metrics.items()), flush=True)

    for h in handles:
        h.remove()
    print("=== training smoke done ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
