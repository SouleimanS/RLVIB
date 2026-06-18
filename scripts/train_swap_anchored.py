#!/usr/bin/env python
"""Collapse-resistant anchored swap-DPO (the rebuild). See docs/research/dpo-collapse-and-fixes.md.

Fixes the degenerate-"no"/garbage collapse of train_swap.py by adding:
  1. the mDPO chosen anchor (pins the chosen log-prob >= the frozen base), and
  2. an explicit KL-to-base on GENERAL (non-swap) prompts -- a mix of the matched MCQ and
     yes/no probes -- so the always-on adapter stays identity where it should.
Plus the monitoring we lacked: a per-step yes-fraction probe (the cheap collapse detector),
and periodic checkpoints so model selection happens on the held-out BENCHMARKS (run
run_bottleneck_eval.sh on the saved checkpoints), never on the in-distribution proxy.

  python scripts/train_swap_anchored.py --pairs 300 --epochs 2 --accum 4
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
from rlvib.train.dpo import anchored_dpo_step, answer_logp_vec, letter_id


def _anchor_msg(model, it, cats, rng):
    """A general, non-swap prompt (matched MCQ or yes/no) where the adapter should match base."""
    if rng.random() < 0.5:
        mcq = ave.make_mcq(it["category"], cats, rng=rng)
        return model.message(video=it["video_path"],
                             prompt=ave.format_mcq(mcq["question"], mcq["options"]))
    cat = it["category"] if rng.random() < 0.5 else rng.choice([c for c in cats if c != it["category"]])
    return model.message(video=it["video_path"],
                         prompt=f"Do you HEAR the sound of {cat} in this clip? Answer yes or no.")


def _yesno_probe(model, items, cats, rng):
    probe = []
    for it in items:
        if rng.random() < 0.5:
            cat, gold = it["category"], "yes"
        else:
            cat, gold = rng.choice([c for c in cats if c != it["category"]]), "no"
        msg = model.message(video=it["video_path"],
                            prompt=f"Do you HEAR the sound of {cat} in this clip? Answer yes or no.")
        probe.append({"messages": msg, "gold": gold})
    return probe


@torch.no_grad()
def _frac_yes(model, probe):
    """Fraction the model answers 'yes' (the cheap collapse detector) + balanced accuracy."""
    yes_ids = [letter_id(model, t) for t in ("yes", "Yes", " yes")]
    no_ids = [letter_id(model, t) for t in ("no", "No", " no")]
    nyes = nacc = 0
    for ex in probe:
        lp = answer_logp_vec(model, ex["messages"])
        pred = "yes" if max(float(lp[i]) for i in yes_ids) > max(float(lp[i]) for i in no_ids) else "no"
        nyes += pred == "yes"
        nacc += pred == ex["gold"]
    k = max(1, len(probe))
    return nyes / k, nacc / k


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-omni")
    ap.add_argument("--pairs", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--accum", type=int, default=4, help="swap pairs per optimizer step")
    ap.add_argument("--anchor-batch", type=int, default=4, help="general anchor prompts per step")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--beta-kl", type=float, default=0.01)
    ap.add_argument("--lam-anchor", type=float, default=1.0)
    ap.add_argument("--delta", type=float, default=0.0)
    ap.add_argument("--lam-kl", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--swap-dir", default="data/AVE/swapped")
    ap.add_argument("--save-dir", default=None, help="default: runs/anchored_<model>")
    args = ap.parse_args()
    args.save_dir = args.save_dir or f"runs/anchored_{args.model}"

    m = get_model(args.model)
    bns, handles = attach_bottlenecks(m, cls=VariationalBottleneck)
    opt = torch.optim.AdamW(bns.parameters(), lr=args.lr)
    os.makedirs(args.save_dir, exist_ok=True)

    cats = ave.categories()
    items = ave.load_ave("train")
    rng = random.Random(0)
    rng.shuffle(items)
    swap = make_swap_examples(items, args.pairs, args.swap_dir, cats, rng=rng)
    anchor_items = items[:400]
    probe = _yesno_probe(m, items[400:440], cats, random.Random(1))
    print(f"swap pairs: {len(swap)} | anchor pool: {len(anchor_items)} | probe: {len(probe)}", flush=True)

    set_bypass(bns, True)  # true frozen base
    bns.eval()
    base_yes, base_acc = _frac_yes(m, probe)
    bns.train()
    set_bypass(bns, False)
    print(f"[probe base] frac_yes={base_yes:.2f} acc={base_acc:.2f}  (collapse = frac_yes -> 0 or 1)",
          flush=True)

    step = 0
    for epoch in range(args.epochs):
        rng.shuffle(swap)
        for i in range(0, len(swap) - args.accum + 1, args.accum):
            sb = [{"messages": m.message(video=r["video_path"],
                                         prompt=ave.format_mcq(r["question"], r["options"])),
                   "chosen_letter": r["audio_letter"], "rejected_letter": r["visual_letter"]}
                  for r in swap[i:i + args.accum]]
            ab = [{"messages": _anchor_msg(m, rng.choice(anchor_items), cats, rng)}
                  for _ in range(args.anchor_batch)]
            mt = anchored_dpo_step(m, bns, opt, sb, ab, beta=args.beta, beta_kl=args.beta_kl,
                                   lam_anchor=args.lam_anchor, delta=args.delta, lam_kl=args.lam_kl)
            step += 1
            print(f"epoch {epoch} step {step}: "
                  + "  ".join(f"{k}={v:+.4f}" for k, v in mt.items()), flush=True)
            if step % args.eval_every == 0:
                bns.eval()
                fy, ac = _frac_yes(m, probe)
                bns.train()
                ckpt = os.path.join(args.save_dir, f"bottleneck_step{step}.pt")
                torch.save({"state_dict": bns.state_dict(), "dim": m.hidden_dim,
                            "cls": "VariationalBottleneck", "model": args.model}, ckpt)
                flag = "  <-- COLLAPSE ALARM" if (fy < 0.15 or fy > 0.85) else ""
                print(f"[probe step {step}] frac_yes={fy:.2f} (base {base_yes:.2f}) acc={ac:.2f}  "
                      f"saved {os.path.basename(ckpt)}{flag}", flush=True)
                if fy < 0.05 or fy > 0.95:
                    print("frac_yes collapsed past 0.05/0.95 -> stopping early.", flush=True)
                    for h in handles:
                        h.remove()
                    return 0

    for h in handles:
        h.remove()
    print("=== anchored training done ===  select a checkpoint with run_bottleneck_eval.sh", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
