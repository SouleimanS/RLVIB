#!/usr/bin/env python
"""Likelihood-shift probe: does the model's answer DEPEND on the audio? (causal-ish meter)

For AVE clips with a cached audio-swapped counterpart, ask the audio-presence question about
the clip's TRUE event A ("Do you HEAR the sound of A? ... yes") and measure the gold answer's
log-likelihood twice:

    ll_full = log p("yes" | original clip)          (audio matches: sound of A present)
    ll_pert = log p("yes" | audio-SWAPPED clip)     (sound of A absent -> should drop)

shift = mean(ll_pert - ll_full). A model that ignores audio shifts ~0; the more it relies on
what it hears, the more negative the shift. Run for base and for a trained bottleneck (FiLM
condition set automatically), then plot with scripts/plot_ll_shift.py (the KDE "Shift =" figure).

  python scripts/probe_ll_shift.py --model qwen3-omni --pairs 40                     # base
  python scripts/probe_ll_shift.py --model qwen3-omni --pairs 40 \
      --bottleneck runs/anchored_qwen3-omni_film/bottleneck_step160.pt --tag film_step160
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import statistics
import warnings

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")
for _n in ("transformers", "qwen_vl_utils", "qwen_omni_utils"):
    logging.getLogger(_n).setLevel(logging.ERROR)

import torch  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

from rlvib.data import ave  # noqa: E402
from rlvib.data.pairs import make_swap_examples  # noqa: E402
from rlvib.models import get_model  # noqa: E402
from rlvib.models.bottleneck import load_attached, question_embedding, set_condition  # noqa: E402
from rlvib.train.dpo import answer_logp_vec, letter_id  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-omni")
    ap.add_argument("--bottleneck", default=None)
    ap.add_argument("--pairs", type=int, default=40, help="original/swapped clip pairs to score")
    ap.add_argument("--swap-dir", default="data/AVE/swapped")
    ap.add_argument("--ave-root", default=ave.DEFAULT_ROOT)
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="", help="output tag (base if empty)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    tag = f"_{args.tag}" if args.tag else ""
    args.out = args.out or f"runs/llshift_{args.model}{tag}.json"

    model = get_model(args.model)
    bns, cond = None, False
    if args.bottleneck:
        bns, _h = load_attached(model, args.bottleneck)
        cond = "q_proj" in bns
        print(f"bottleneck <- {args.bottleneck}" + ("  (FiLM)" if cond else ""), flush=True)
    yes_ids = [letter_id(model, t) for t in ("yes", "Yes", " yes")]

    cats = ave.categories(args.ave_root)
    items = ave.load_ave("train", args.ave_root)
    rng = random.Random(args.seed)
    rng.shuffle(items)
    swaps = make_swap_examples(items, args.pairs, args.swap_dir, cats, rng=rng)
    vdir = os.path.join(args.ave_root, "AVE")
    print(f"scoring {len(swaps)} original/swapped pairs", flush=True)

    ll_full, ll_pert = [], []
    for r in tqdm(swaps, desc="ll-shift", unit="pair", dynamic_ncols=True):
        # original clip path: swapped file is <orig_id>__aud_<donor_id>.mp4
        orig_id = os.path.basename(r["video_path"]).split("__aud_")[0]
        orig = os.path.join(vdir, f"{orig_id}.mp4")
        cat = r["visual_event"]                       # the original clip's (seen+heard) event
        qtext = f"Do you HEAR the sound of {cat} in this clip? Answer yes or no."
        if cond:
            set_condition(bns, question_embedding(model, qtext))
        try:
            with torch.no_grad():
                lp_f = answer_logp_vec(model, model.message(video=orig, prompt=qtext, fps=args.fps))
                lp_p = answer_logp_vec(model, model.message(video=r["video_path"], prompt=qtext,
                                                            fps=args.fps))
        except Exception as e:  # noqa: BLE001 -- skip unreadable clips
            print(f"\n[skip {orig_id}] {type(e).__name__}: {e}", flush=True)
            continue
        ll_full.append(max(float(lp_f[i]) for i in yes_ids))
        ll_pert.append(max(float(lp_p[i]) for i in yes_ids))

    if not ll_full:
        print("no pairs scored")
        return 1
    shift = statistics.fmean(ll_pert) - statistics.fmean(ll_full)
    res = {"model": args.model, "tag": args.tag, "bottleneck": args.bottleneck,
           "n": len(ll_full), "shift": shift,
           "mean_full": statistics.fmean(ll_full), "mean_pert": statistics.fmean(ll_pert),
           "ll_full": ll_full, "ll_pert": ll_pert}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n=== ll-shift {args.model}{tag} ===")
    print(f"  mean ll (true audio)    = {res['mean_full']:+.3f}")
    print(f"  mean ll (swapped audio) = {res['mean_pert']:+.3f}")
    print(f"  SHIFT = {shift:+.3f}   (more negative = answers depend more on the audio)")
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
