#!/usr/bin/env python
"""Train the RLVIB adapter with the MoD-DPO / MoD-DPO++ objective (baseline comparison).

Same data substrate as our anchored recipe (AVE audio-swapped clips, HEAR + SEE prompts --
the paper alternates audio-related and vision-related prompts, which HEAR/SEE gives us
exactly), same adapter capacity (the unconditional VIB), same selection tooling: checkpoints
land in runs/anchored_<model>_moddpo[pp]/ so EXP=moddpo / EXP=moddpopp works with
select_checkpoint.sh, select_holdout.py, eval_one.sh and paired_stats.py unchanged.

  PYTHONPATH=src python baselines/mod_dpo/train_mod_dpo.py --pairs 300 --epochs 2          # MoD-DPO
  PYTHONPATH=src python baselines/mod_dpo/train_mod_dpo.py --plus --pairs 300 --epochs 2   # MoD-DPO++
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import warnings

warnings.filterwarnings("ignore")  # librosa/audioread deprecation spam floods the training log

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mod_dpo import DEFAULTS, build_corruptions, mod_dpo_step  # noqa: E402

from rlvib.data import ave  # noqa: E402
from rlvib.data.pairs import make_swap_examples  # noqa: E402
from rlvib.models import get_model  # noqa: E402
from rlvib.models.bottleneck import VariationalBottleneck, attach_bottlenecks  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-omni")
    ap.add_argument("--pairs", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--beta", type=float, default=DEFAULTS["beta"])
    ap.add_argument("--beta-inv", type=float, default=DEFAULTS["beta_inv"])
    ap.add_argument("--beta-sens", type=float, default=DEFAULTS["beta_sens"])
    ap.add_argument("--gamma-lpd", type=float, default=DEFAULTS["gamma_lpd"])
    ap.add_argument("--plus", action="store_true", help="MoD-DPO++ (adds the LPD term)")
    ap.add_argument("--beta-kl", type=float, default=0.01, help="our adapter's usual VIB rate term")
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--swap-dir", default="data/AVE/swapped")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-dir", default=None)
    args = ap.parse_args()
    tag = "moddpopp" if args.plus else "moddpo"
    gamma_lpd = args.gamma_lpd if args.plus else 0.0
    args.save_dir = args.save_dir or f"runs/anchored_{args.model}_{tag}"
    torch.manual_seed(args.seed)

    m = get_model(args.model)
    bns, handles = attach_bottlenecks(m, cls=VariationalBottleneck,
                                      normalize_input=args.model == "videollama2")
    opt = torch.optim.AdamW(bns.parameters(), lr=args.lr)
    os.makedirs(args.save_dir, exist_ok=True)

    cats = ave.categories()
    items = ave.load_ave("train")
    rng = random.Random(args.seed)
    rng.shuffle(items)
    swaps = make_swap_examples(items, args.pairs, args.swap_dir, cats, rng=rng)
    corr_dir = os.path.join(args.swap_dir, "mod_corrupt")
    print(f"swap records: {len(swaps)} | objective: MoD-DPO{'++' if args.plus else ''} "
          f"(beta={args.beta}, inv={args.beta_inv}, sens={args.beta_sens}, lpd={gamma_lpd})", flush=True)

    # Build the alternating audio-/vision-related example stream. For an audio-related (HEAR)
    # prompt: relevant = audio -> sensitivity target corrupts the AUDIO (silenced), invariance
    # target corrupts the VIDEO (mismatched clip). Vision-related (SEE): roles swap.
    examples = []
    for rec in swaps:
        corr = build_corruptions(rec, items, corr_dir, rng)
        hear_q = ave.format_mcq(rec["question"], rec["options"])          # "which do you HEAR?"
        see = ave.make_see_mcq(rec["audio_event"], rec["visual_event"], cats, rng=rng)
        see_q = ave.format_mcq(see["question"], see["options"])
        examples.append({  # audio-related
            "messages": m.message(video=rec["video_path"], prompt=hear_q),
            "messages_rel": m.message(video=corr["audio_corrupt"], prompt=hear_q),
            "messages_irr": m.message(video=corr["video_corrupt"], prompt=hear_q),
            "messages_text": m.message(prompt=hear_q),
            "chosen_letter": rec["audio_letter"], "rejected_letter": rec["visual_letter"]})
        examples.append({  # vision-related
            "messages": m.message(video=rec["video_path"], prompt=see_q),
            "messages_rel": m.message(video=corr["video_corrupt"], prompt=see_q),
            "messages_irr": m.message(video=corr["audio_corrupt"], prompt=see_q),
            "messages_text": m.message(prompt=see_q),
            "chosen_letter": see["visual_letter"], "rejected_letter": see["audio_letter"]})

    step = 0
    for epoch in range(args.epochs):
        rng.shuffle(examples)
        for i in range(0, len(examples) - args.accum + 1, args.accum):
            mt = mod_dpo_step(m, bns, opt, examples[i:i + args.accum], beta=args.beta,
                              beta_inv=args.beta_inv, beta_sens=args.beta_sens,
                              gamma_lpd=gamma_lpd, beta_kl=args.beta_kl)
            step += 1
            print(f"epoch {epoch} step {step}: " + "  ".join(f"{k}={v:+.4f}" for k, v in mt.items()),
                  flush=True)
            if step % args.eval_every == 0:
                ckpt = os.path.join(args.save_dir, f"bottleneck_step{step}.pt")
                torch.save({"state_dict": bns.state_dict(), "dim": m.hidden_dim,
                            "cls": "VariationalBottleneck", "model": args.model,
                            "normalize_input": args.model == "videollama2"}, ckpt)
                print(f"saved {os.path.basename(ckpt)}", flush=True)

    for h in handles:
        h.remove()
    print(f"=== MoD-DPO{'++' if args.plus else ''} training done -> {args.save_dir} ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
