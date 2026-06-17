#!/usr/bin/env python
"""Confirm the audio-SWAP counterfactual carries the signal that audio-drop lacked.

Builds n audio-swapped AVE clips (video = SEEN event A, audio = HEARD event B != A),
asks "which event do you HEAR?", and measures — on the BASE model (no bottleneck) —
whether it shortcuts to the SEEN event instead of the HEARD one. Uses the exact
first-token letter scoring the trainer uses. How to read the summary:
  - shortcut_rate high (pred == seen)        => the visual shortcut is real here
  - mean(lp_seen - lp_heard) >> 0            => chosen starts low -> big DPO gradient
  => audio-swap DPO has signal; safe to run train_swap_smoke.

  python scripts/diag_swap_sensitivity.py --model qwen3-omni --n 8
"""
from __future__ import annotations

import argparse
import random

import torch

from rlvib.data import ave
from rlvib.data.pairs import make_swap_examples
from rlvib.models import get_model
from rlvib.train.dpo import answer_logp_vec, letter_id


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-omni")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--swap-dir", default="data/AVE/swapped")
    args = ap.parse_args()

    m = get_model(args.model)
    cats = ave.categories()
    items = ave.load_ave("train")
    rng = random.Random(0)
    pairs = make_swap_examples(items, args.n, args.swap_dir, cats, rng=rng)
    print(f"built {len(pairs)} swap pairs in {args.swap_dir}", flush=True)

    shortcut = heard = n = 0
    gaps = []
    for rec in pairs:
        msg = m.message(video=rec["video_path"],
                        prompt=ave.format_mcq(rec["question"], rec["options"]))
        with torch.no_grad():
            lp = answer_logp_vec(m, msg, use_audio_in_video=True)
        letters = [chr(65 + i) for i in range(len(rec["options"]))]
        pred = max(letters, key=lambda L: float(lp[letter_id(m, L)]))
        a = float(lp[letter_id(m, rec["audio_letter"])])    # heard (gold)
        v = float(lp[letter_id(m, rec["visual_letter"])])   # seen (shortcut)
        shortcut += pred == rec["visual_letter"]
        heard += pred == rec["audio_letter"]
        gaps.append(v - a)
        n += 1
        print(f"heard={rec['audio_event']}({rec['audio_letter']})  "
              f"seen={rec['visual_event']}({rec['visual_letter']})  pred={pred}  "
              f"lp_heard={a:+.2f}  lp_seen={v:+.2f}  seen-heard={v - a:+.2f}", flush=True)

    print(f"\nN={n}  shortcut_rate(pred==seen)={shortcut}/{n}  "
          f"heard_rate(pred==heard)={heard}/{n}  "
          f"mean(lp_seen-lp_heard)={sum(gaps) / max(1, n):+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
