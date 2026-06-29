#!/usr/bin/env python
"""Per-clip AV-attention fraction over CMM -- the MoD-DPO++ "Figure 6" probe, base vs trained.

For each CMM clip, measures the share of the answer position's attention that lands on the
audio+visual tokens (vs all input tokens incl. text). Run once for the base model and once with
a trained bottleneck; plot_attn_av.py box-plots the two distributions. Qwen-Omni only (loads
with eager attention so attentions are exposed). Memory ~ O(L*H*S^2) -> keep --fps low.

  # base:
  python scripts/attn_av_analysis.py --model qwen2.5-omni --limit 150 --fps 1
  # trained (broad VIB or FiLM -- FiLM condition is set automatically):
  python scripts/attn_av_analysis.py --model qwen2.5-omni --limit 150 --fps 1 \
      --bottleneck runs/anchored_qwen2.5-omni_broad/bottleneck_step60.pt --tag broad_step60

Writes runs/attnav_<model>[_<tag>].json = {fractions:[...], mean, median, n, av_tokens_mean}.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import warnings

from tqdm.auto import tqdm

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")
for _n in ("transformers", "qwen_vl_utils", "qwen_omni_utils"):
    logging.getLogger(_n).setLevel(logging.ERROR)

from rlvib.data.cmm import CMMDataset  # noqa: E402
from rlvib.eval.attention_av import av_attention_fraction, av_token_ids  # noqa: E402
from rlvib.eval.timeout import time_limit  # noqa: E402
from rlvib.models import get_model  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5-omni", help="qwen2.5-omni | qwen3-omni (Qwen-Omni only)")
    ap.add_argument("--bottleneck", default=None, help="trained checkpoint; omit for the base row")
    ap.add_argument("--json-path", default="data/CMM/all_data_final_reorg.json")
    ap.add_argument("--data-root", default="data/CMM")
    ap.add_argument("--subsets", nargs="*", default=None, help="CMM sub_category filter; default all")
    ap.add_argument("--limit", type=int, default=150, help="clips to analyze (0 = all)")
    ap.add_argument("--fps", type=float, default=1.0, help="LOW keeps S (hence attn memory) small")
    ap.add_argument("--query", default="last", choices=["last", "mean"])
    ap.add_argument("--gen-timeout", type=int, default=180, help="per-clip wall cap (s); 0 disables")
    ap.add_argument("--tag", default="", help="output tag, e.g. broad_step60 (base if empty)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    tag = f"_{args.tag}" if args.tag else ""
    args.out = args.out or f"runs/attnav_{args.model}{tag}.json"

    model = get_model(args.model, attn="eager")              # eager -> attentions exposed
    cond = False
    if args.bottleneck:
        from rlvib.models.bottleneck import load_attached, question_embedding, set_condition
        bns, _h = load_attached(model, args.bottleneck)
        cond = "q_proj" in bns                               # FiLM -> set the question per clip
        print(f"attached bottleneck <- {args.bottleneck}" + ("  (prompt-aware/FiLM)" if cond else ""),
              flush=True)
    av_ids = av_token_ids(model)
    print(f"AV token ids: {sorted(av_ids) or 'NONE (mask would be empty!)'}", flush=True)
    if not av_ids:
        print("WARNING: no AV placeholder tokens found -- the fraction will be 0. Check the "
              "tokenizer/config token names in rlvib.eval.attention_av.", flush=True)

    ds = CMMDataset(args.json_path, args.data_root, sub_categories=args.subsets)
    n = len(ds) if args.limit in (0, None) else min(args.limit, len(ds))
    print(f"AV-attention over CMM: {n}/{len(ds)} clips | fps={args.fps} | query={args.query}", flush=True)

    fractions, av_counts = [], []
    bar = tqdm(range(n), desc="attn-av", unit="clip", dynamic_ncols=True)
    for i in bar:
        item = ds[i]
        v, a = item["video_path"], item["audio_path"]
        uaiv = bool(v) and not a and item.get("modality") == "audio"
        if cond:
            set_condition(bns, question_embedding(model, item["question"]))
        try:
            msg = model.message(video=v, audio=a, prompt=item["question"], fps=args.fps)
            with time_limit(args.gen_timeout):
                frac, n_av, _ = av_attention_fraction(model, msg, av_ids=av_ids,
                                                      use_audio_in_video=uaiv, query=args.query)
            fractions.append(frac)
            av_counts.append(n_av)
            bar.set_postfix(mean=f"{100 * statistics.fmean(fractions):.2f}%", refresh=False)
        except Exception as e:  # noqa: BLE001 -- skip a bad/oom/hanging clip, keep going
            print(f"\n[skip clip {i}] {type(e).__name__}: {e}", flush=True)

    res = {
        "model": args.model, "tag": args.tag, "bottleneck": args.bottleneck,
        "fps": args.fps, "query": args.query, "n": len(fractions),
        "mean": statistics.fmean(fractions) if fractions else None,
        "median": statistics.median(fractions) if fractions else None,
        "stdev": statistics.pstdev(fractions) if len(fractions) > 1 else 0.0,
        "av_tokens_mean": statistics.fmean(av_counts) if av_counts else None,
        "fractions": fractions,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    m = res["mean"]
    print(f"\n=== AV-attention {args.model}{tag} ===")
    print(f"  mean={100 * m:.2f}%  median={100 * res['median']:.2f}%  (n={res['n']}, "
          f"avg AV tokens={res['av_tokens_mean']:.0f})" if m is not None else "  (no clips scored)")
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
