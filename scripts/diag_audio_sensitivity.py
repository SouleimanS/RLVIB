#!/usr/bin/env python
"""Diagnose the mDPO signal: does the AVE-MCQ answer depend on the audio?

The training smoke showed margin ~0 (loss pinned at ln 2). Two candidate causes:
  (M) mechanical — we score the wrong first token ("A" vs " A"), so logprobs sit
      near the floor and full-vs-drop differences vanish;
  (S) scientific — the frozen base answers from video alone (the grounding gap),
      so dropping the audio doesn't move the answer.

For N AVE clips (BASE model, no bottleneck) this prints the model's actual top-1
first token, its predicted letter WITH vs WITHOUT audio, and the gold-letter
logprob delta (full - drop). Encoding-robust: each letter is scored as the better
of "L" / " L". How to read the summary line:
  - top1_is_letter ~ N/N        => letter scoring is sane (rules out M)
  - acc_full > acc_drop, pred_changed high, mean|d_gold| large
        => audio matters; audio-drop is a usable DPO counterfactual -> scale up
  - all ~equal                  => base ignores audio here; the audio-drop signal
        is null -> need audio-SWAP pairs (Tier B) or a harder, audio-necessary MCQ.

  python scripts/diag_audio_sensitivity.py --model qwen3-omni --n 8
"""
from __future__ import annotations

import argparse
import random

import torch

from rlvib.data import ave
from rlvib.models import get_model


def first_token_logp(model, messages, use_audio_in_video: bool = True):
    """log-softmax over the vocab at the first generated position (no grad)."""
    inputs = model.build_inputs(messages, use_audio_in_video=use_audio_in_video)
    lm = getattr(model.model, "thinker", model.model)
    with torch.no_grad():
        logits = lm(**inputs).logits[:, -1, :]
    return torch.log_softmax(logits.float(), dim=-1)[0]


def _tid(model, s: str) -> int:
    return model.processor.tokenizer(s, add_special_tokens=False).input_ids[0]


def letter_lp(model, lp, letter: str) -> float:
    """Logprob the model puts on `letter`, robust to leading-space encoding."""
    return max(float(lp[_tid(model, letter)]), float(lp[_tid(model, " " + letter)]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-omni")
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()

    m = get_model(args.model)
    tok = m.processor.tokenizer
    cats = ave.categories()
    items = ave.load_ave("train")
    rng = random.Random(0)
    rng.shuffle(items)

    acc_full = acc_drop = changed = top1_letter = n = 0
    dsum = 0.0
    for it in items[: args.n]:
        mcq = ave.make_mcq(it["category"], cats, rng=rng)
        letters = [chr(65 + i) for i in range(len(mcq["options"]))]
        msg = m.message(video=it["video_path"],
                        prompt=ave.format_mcq(mcq["question"], mcq["options"]))
        gold = mcq["gold_letter"]

        lpf = first_token_logp(m, msg, True)
        lpd = first_token_logp(m, msg, False)

        pf = max(letters, key=lambda L: letter_lp(m, lpf, L))
        pd = max(letters, key=lambda L: letter_lp(m, lpd, L))
        d = letter_lp(m, lpf, gold) - letter_lp(m, lpd, gold)

        top = int(lpf.argmax())
        top_str = tok.decode([top])
        top1_letter += top_str.strip() in letters
        acc_full += pf == gold
        acc_drop += pd == gold
        changed += pf != pd
        dsum += abs(d)
        n += 1
        print(f"gold={gold}  full:pred={pf}({letter_lp(m, lpf, gold):+.2f})  "
              f"drop:pred={pd}({letter_lp(m, lpd, gold):+.2f})  "
              f"top1={top_str!r}({float(lpf[top]):+.2f})  d_gold={d:+.3f}  "
              f"changed={'Y' if pf != pd else 'n'}", flush=True)

    print(f"\nN={n}  acc_full={acc_full}/{n}  acc_drop={acc_drop}/{n}  "
          f"pred_changed={changed}/{n}  mean|d_gold|={dsum / max(1, n):.4f}  "
          f"top1_is_letter={top1_letter}/{n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
