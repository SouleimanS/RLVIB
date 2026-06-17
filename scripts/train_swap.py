#!/usr/bin/env python
"""Step-5 swap-DPO training run: contrastive DPO on audio-swapped AVE clips.

Trains the VIB bottleneck to prefer the HEARD event over the SEEN one on audio-swapped
clips (anti-shortcut faithful fusion). Improvements over train_swap_smoke:
  - held-out val split of swap pairs;
  - gradient accumulation (effective batch = --accum) to denoise the DPO step;
  - the headline metric: heard_rate on held-out pairs, trained-vs-base
    (= how often the model answers what it HEARS instead of what it SEES);
  - saves the bottleneck checkpoint for downstream benchmark eval.

  python scripts/train_swap.py --model qwen3-omni --pairs 200 --epochs 2 --accum 8
"""
from __future__ import annotations

import argparse
import os
import random

import torch

from rlvib.data import ave
from rlvib.data.pairs import make_swap_examples
from rlvib.models import get_model
from rlvib.models.bottleneck import VariationalBottleneck, attach_bottlenecks, set_bypass
from rlvib.train.dpo import answer_logp_vec, dpo_step, letter_id


def _batch(model, recs):
    return [{"messages": model.message(video=r["video_path"],
                                       prompt=ave.format_mcq(r["question"], r["options"])),
             "chosen_letter": r["audio_letter"], "rejected_letter": r["visual_letter"]}
            for r in recs]


@torch.no_grad()
def _heard_rate(model, recs) -> float:
    """Fraction of swap pairs the model answers with the HEARD event (its current state)."""
    hit = 0
    for rec in recs:
        msg = model.message(video=rec["video_path"],
                            prompt=ave.format_mcq(rec["question"], rec["options"]))
        lp = answer_logp_vec(model, msg, use_audio_in_video=True)
        letters = [chr(65 + i) for i in range(len(rec["options"]))]
        pred = max(letters, key=lambda L: float(lp[letter_id(model, L)]))
        hit += pred == rec["audio_letter"]
    return hit / max(1, len(recs))


def eval_heard(model, bottlenecks, recs, bypass: bool) -> float:
    """heard_rate with the bottleneck bypassed (base) or active (trained); eval mode = mu."""
    set_bypass(bottlenecks, bypass)
    bottlenecks.eval()
    r = _heard_rate(model, recs)
    bottlenecks.train()
    set_bypass(bottlenecks, False)
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-omni")
    ap.add_argument("--pairs", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8, help="examples per optimizer step")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--beta-kl", type=float, default=0.01)
    ap.add_argument("--eval-every", type=int, default=5, help="optimizer steps between val evals")
    ap.add_argument("--log-every", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--swap-dir", default="data/AVE/swapped")
    ap.add_argument("--save", default="runs/bottleneck_swap.pt")
    args = ap.parse_args()

    m = get_model(args.model)
    bns, handles = attach_bottlenecks(m, cls=VariationalBottleneck)
    opt = torch.optim.AdamW(bns.parameters(), lr=args.lr)

    cats = ave.categories()
    items = ave.load_ave("train")
    rng = random.Random(args.seed)
    pairs = make_swap_examples(items, args.pairs, args.swap_dir, cats, rng=rng)
    rng.shuffle(pairs)
    n_val = max(8, len(pairs) // 10)
    val, train = pairs[:n_val], pairs[n_val:]
    print(f"swap pairs: {len(pairs)} (train {len(train)} / val {len(val)})  "
          f"epochs={args.epochs} accum={args.accum}", flush=True)

    base_hr = eval_heard(m, bns, val, bypass=True)
    print(f"[eval] base heard_rate={base_hr:.3f}", flush=True)

    step = 0
    for epoch in range(args.epochs):
        rng.shuffle(train)
        for i in range(0, len(train) - args.accum + 1, args.accum):
            metrics = dpo_step(m, bns, opt, _batch(m, train[i:i + args.accum]),
                               beta=args.beta, beta_kl=args.beta_kl)
            step += 1
            if step % args.log_every == 0:
                print(f"epoch {epoch} step {step}: "
                      + "  ".join(f"{k}={v:+.4f}" for k, v in metrics.items()), flush=True)
            if args.eval_every and step % args.eval_every == 0:
                hr = eval_heard(m, bns, val, bypass=False)
                print(f"[eval] step {step}: heard_rate={hr:.3f}  (base {base_hr:.3f})", flush=True)

    final_hr = eval_heard(m, bns, val, bypass=False)
    print(f"[eval] FINAL heard_rate={final_hr:.3f}  (base {base_hr:.3f})  "
          f"delta={final_hr - base_hr:+.3f}", flush=True)

    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        torch.save({"state_dict": bns.state_dict(), "dim": m.hidden_dim,
                    "cls": "VariationalBottleneck", "model": args.model}, args.save)
        print(f"saved bottleneck -> {args.save}", flush=True)

    for h in handles:
        h.remove()
    print("=== swap-DPO training done ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
