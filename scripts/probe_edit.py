#!/usr/bin/env python
"""Realized per-modality edit probe -- reconciles the weight analysis with the "59%" claim.

`analyze_bottleneck.py` showed the audio and vision output MAPS are equal magnitude. This probe
measures the REALIZED edit during a forward: per modality, mean  ||W_out z|| / ||x||  (relative
edit), and its raw parts  ||edit||  and  ||x|| (token norm). The point:

  * if audio ||edit|| ~ vision ||edit|| but audio ||x|| >> vision ||x||, then audio's smaller
    RELATIVE edit is a TOKEN-NORM effect, not weight inertness (weights were equal) -- and the old
    "audio ~ pass-through" was measuring the ratio, not the module.
  * if audio ||edit|| << vision ||edit|| here too, the asymmetry is real for this checkpoint.

Interpretability, ~15 clips, no benchmark. Needs a GPU (one forward per clip).

  python scripts/probe_edit.py --model qwen3-omni \
      --bottleneck runs/anchored_qwen3-omni_broad/bottleneck_step60.pt --limit 15
"""
from __future__ import annotations

import argparse
import collections
import logging
import os
import statistics
import warnings

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")
for _n in ("transformers", "qwen_vl_utils", "qwen_omni_utils"):
    logging.getLogger(_n).setLevel(logging.ERROR)

import torch  # noqa: E402

from rlvib.data.avhbench import AVHBenchDataset  # noqa: E402
from rlvib.models import get_model  # noqa: E402
from rlvib.models.bottleneck import load_attached, question_embedding, set_condition  # noqa: E402
from rlvib.train.dpo import answer_logp_vec  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-omni")
    ap.add_argument("--bottleneck", required=True)
    ap.add_argument("--qa-json", default="data/AVHBench/qa.json")
    ap.add_argument("--video-root", default="data/AVHBench/videos")
    ap.add_argument("--limit", type=int, default=15)
    ap.add_argument("--fps", type=float, default=2.0)
    args = ap.parse_args()

    model = get_model(args.model)
    bns, _h = load_attached(model, args.bottleneck)      # eval mode -> z = mu (deterministic)
    bns.eval()
    cond = "q_proj" in bns
    print(f"bottleneck <- {args.bottleneck}" + ("  (FiLM)" if cond else ""), flush=True)

    ds = AVHBenchDataset(args.qa_json, args.video_root)
    n = min(args.limit, len(ds))
    agg = collections.defaultdict(lambda: {"rel": [], "edit": [], "tok": []})
    done = 0
    for i in range(n):
        item = ds[i]
        q = item["text"].rstrip()
        if cond:
            set_condition(bns, question_embedding(model, q))
        try:
            msg = model.message(video=item["video_path"], prompt=q + " Answer yes or no.", fps=args.fps)
            with torch.no_grad():
                answer_logp_vec(model, msg, use_audio_in_video=True)   # triggers the adapter hooks
        except Exception as e:  # noqa: BLE001
            print(f"  skip clip {i}: {type(e).__name__}: {e}", flush=True)
            continue
        for mod in ("audio", "vision"):
            b = bns[mod]
            if getattr(b, "last_residual_per_token", None) is None:
                continue
            edit = b.last_residual_per_token.float()             # ||W_out z|| per token
            tok = b.last_input_norm_per_token.float()            # ||x|| per token
            agg[mod]["rel"].append(float((edit / (tok + 1e-6)).mean()))
            agg[mod]["edit"].append(float(edit.mean()))
            agg[mod]["tok"].append(float(tok.mean()))
        done += 1

    print(f"\n=== realized edit over {done} clips ===")
    m = lambda xs: statistics.fmean(xs) if xs else float("nan")   # noqa: E731
    for mod in ("audio", "vision"):
        a = agg[mod]
        print(f"  [{mod:6s}] relative edit ||W_out z||/||x|| = {100 * m(a['rel']):6.2f}%   "
              f"||edit|| = {m(a['edit']):8.3f}   ||token|| = {m(a['tok']):8.3f}")
    ea, ev = m(agg["audio"]["edit"]), m(agg["vision"]["edit"])
    ta, tv = m(agg["audio"]["tok"]), m(agg["vision"]["tok"])
    print("\n=== verdict ===")
    if ea == ea and ev == ev:  # not nan
        print(f"  raw edit  audio/vision = {ea / max(ev, 1e-9):.2f}x   "
              f"token-norm audio/vision = {ta / max(tv, 1e-9):.2f}x")
        print("  edit ratio ~1 but token ratio >>1  => audio's smaller RELATIVE edit is a token-norm "
              "effect (weights are equal), not an inert audio module.")
        print("  edit ratio <<1  => the audio edit really is smaller for this checkpoint.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
