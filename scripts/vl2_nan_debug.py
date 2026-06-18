#!/usr/bin/env python
"""VideoLLaMA2 finiteness diagnostic (matches the VL2 training config: normalize_input=True).

Reports, per real training input (clean+open / clean+MCQ / swapped+MCQ), with the VIB
active + grad:
  model dtype  -- did the bf16 cast take? (should be torch.bfloat16)
  logits_finite -- is the LLM forward finite? (bf16 should fix attention overflow)
  vis_featmax  -- magnitude of the mm_projector features into the VIB (~1e9 = massive acts)
  vis_kl       -- the VIB KL rate + finiteness (normalize_input should keep it O(1))

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
    if t is None:
        return None
    return bool(torch.isfinite(t if torch.is_tensor(t) else torch.tensor(float(t))).all())


def _max(t):
    return float(t.max()) if t is not None else float("nan")


def probe(m, bns, video, prompt, tag):
    set_bypass(bns, False)
    bns.train()
    lp = answer_logp_vec(m, m.message(video=video, prompt=prompt))
    v, a = bns["vision"], bns["audio"]
    print(f"[{tag:11s}] logits_fin={fin(lp)}  "
          f"vis_featmax={_max(v.last_input_norm_per_token):.2e} "
          f"vis_kl={float(v.last_kl):.2e}({fin(v.last_kl)})  "
          f"aud_featmax={_max(a.last_input_norm_per_token):.2e} "
          f"aud_kl={float(a.last_kl):.2e}({fin(a.last_kl)})", flush=True)


def main() -> int:
    m = get_model("videollama2")
    print("model dtype:", m.dtype, flush=True)
    bns, handles = attach_bottlenecks(m, cls=VariationalBottleneck, normalize_input=True)
    print("vib param dtype:", next(bns["vision"].parameters()).dtype,
          "| normalize_input:", bns["vision"].normalize_input, flush=True)

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
