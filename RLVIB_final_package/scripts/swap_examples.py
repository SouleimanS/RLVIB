#!/usr/bin/env python
"""Generate real audio-swap examples for the hallucination figure (GPU; small).

For a few held-out AVE clips, overlay a DIFFERENT-category clip's audio (so the seen event
!= the heard event), then run the FROZEN BASE and the ADAPTED model (same checkpoint, with
the bottleneck bypass toggled) on the "which event do you HEAR?" MCQ. Prints, per clip, the
seen event, the overlaid (heard) event, and each model's choice -- flagging the ones where
the base picks the SEEN event (ignored the audio = hallucination) and ours picks the HEARD
event. These are the figure examples that SHOW the underlying sound and EXPLAIN the
audio-grounding failure our method targets (unlike the visual-language probes, where audio
is not the driver).

  PYTHONPATH=src python scripts/swap_examples.py \
      --ckpt runs/anchored_qwen3-omni_broad/bottleneck_step60.pt --n 10
"""
from __future__ import annotations

import argparse
import os
import random
import subprocess

from rlvib.data import ave
from rlvib.data.pairs import swap_audio
from rlvib.eval.metrics import parse_choice
from rlvib.models import get_model
from rlvib.models.bottleneck import load_attached, set_bypass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-omni")
    ap.add_argument("--ckpt", required=True, help="trained bottleneck checkpoint (.pt)")
    ap.add_argument("--n", type=int, default=10, help="examples to show")
    ap.add_argument("--split", default="test", help="AVE split to draw clips from (held-out)")
    ap.add_argument("--out-dir", default="data/AVE/swapped_fig")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    os.makedirs(a.out_dir, exist_ok=True)

    cats = ave.categories()
    items = ave.load_ave(a.split)
    rng.shuffle(items)
    by_cat: dict[str, list] = {}
    for it in items:
        by_cat.setdefault(it["category"], []).append(it)

    m = get_model(a.model)
    bns, _h = load_attached(m, a.ckpt)        # adapted; toggle bypass for the frozen base
    print(f"model={a.model}  ckpt={a.ckpt}  split={a.split}\n", flush=True)

    shown = 0
    for it in items:
        if shown >= a.n:
            break
        other = [c for c in by_cat if c != it["category"]]
        if not other:
            continue
        jt = rng.choice(by_cat[rng.choice(other)])
        seen, heard = it["category"], jt["category"]
        out_path = os.path.join(a.out_dir, f"{it['video_id']}__aud_{jt['video_id']}.mp4")
        if not os.path.exists(out_path):
            try:
                swap_audio(it["video_path"], jt["video_path"], out_path)
            except subprocess.CalledProcessError:
                continue
        mcq = ave.make_hear_mcq(heard, seen, cats, k=4, rng=rng)
        msg = m.message(video=out_path, prompt=ave.format_mcq(mcq["question"], mcq["options"]))
        set_bypass(bns, True)
        base = m.generate(msg, use_audio_in_video=True, max_new_tokens=8)
        set_bypass(bns, False)
        ours = m.generate(msg, use_audio_in_video=True, max_new_tokens=8)
        pb, po = parse_choice(base), parse_choice(ours)
        shown += 1
        tag = "BASE=SEEN/OURS=HEARD" if (pb == mcq["visual_letter"] and po == mcq["audio_letter"]) else ""
        print(f"[{tag}] video: {out_path}")
        print(f"   SEE: {seen}   |   overlaid SOUND (heard): {heard}")
        print(f"   Q: {mcq['question']}  options={mcq['options']}")
        print(f"      (heard letter={mcq['audio_letter']}  seen letter={mcq['visual_letter']})")
        print(f"   base -> {pb!r}  {base!r}")
        print(f"   ours -> {po!r}  {ours!r}\n", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
