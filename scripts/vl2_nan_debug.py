#!/usr/bin/env python
"""Pinpoint VideoLLaMA2 NaN: a clean clip forwards fine, so test the ACTUAL training
inputs. Probes (all active VIB + grad) a clean+open-ended, a clean+MCQ, and a
SWAPPED+MCQ input -> tells us whether it's the swapped media or the MCQ prompt.

  CONDA_ENV=rlvib_vl2 ... :  PYTHONPATH=src python scripts/vl2_nan_debug.py
"""
import random

import torch

from rlvib.data import ave
from rlvib.data.pairs import make_swap_examples
from rlvib.models import get_model
from rlvib.models.bottleneck import VariationalBottleneck, attach_bottlenecks, set_bypass
from rlvib.train.dpo import answer_logp_vec


def fin(t):
    return bool(torch.isfinite(t).all()) if t is not None else None


def probe(m, bns, video, prompt, tag):
    set_bypass(bns, False)
    bns.train()
    lp = answer_logp_vec(m, m.message(video=video, prompt=prompt))
    print(f"[{tag:11s}] logits_finite={fin(lp)}  "
          f"vis_x={fin(bns['vision'].last_input_norm_per_token)} "
          f"aud_x={fin(bns['audio'].last_input_norm_per_token)}", flush=True)


def main() -> int:
    m = get_model("videollama2")
    print("model dtype:", m.dtype, flush=True)
    bns, handles = attach_bottlenecks(m, cls=VariationalBottleneck)

    cats = ave.categories()
    items = ave.load_ave("train")
    rng = random.Random(0)
    rng.shuffle(items)
    clean = items[0]

    probe(m, bns, clean["video_path"], "What do you see and hear in this clip?", "clean+open")
    mcq = ave.make_mcq(clean["category"], cats, rng=rng)
    probe(m, bns, clean["video_path"], ave.format_mcq(mcq["question"], mcq["options"]), "clean+mcq")

    swap = make_swap_examples(items, 4, "data/AVE/swapped", cats, rng=rng)
    r = swap[0]
    print("swap clip:", r["video_path"], flush=True)
    probe(m, bns, r["video_path"], ave.format_mcq(r["question"], r["options"]), "swap+mcq")

    for h in handles:
        h.remove()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
