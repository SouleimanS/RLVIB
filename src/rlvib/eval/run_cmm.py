"""Frozen Qwen3-Omni baseline on CMM (The Curse of Multi-Modalities, arXiv:2410.12787).

  python -m rlvib.eval.run_cmm --json-path cmm.json --data-root data/CMM [--limit N]

Per sub_category reports CMM's two metrics + overall:
  PA (Perception Accuracy)      = acc on answer=="yes" (detect present)
  HR (Hallucination Resistance) = acc on answer=="no"  (reject absent)
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import time

from rlvib.data.cmm import AUDIO_SUBSETS, CMMDataset
from rlvib.eval.metrics import parse_yes_no
from rlvib.models import get_model


def _scores(pairs: list) -> dict:
    """pairs: list of (gold, pred); gold in {'yes','no'}, pred in {'yes','no',None}."""
    yes = [(g, p) for g, p in pairs if g == "yes"]
    no = [(g, p) for g, p in pairs if g == "no"]
    acc = lambda ps: (sum(1 for g, p in ps if p == g) / len(ps)) if ps else 0.0  # noqa: E731
    parsed = sum(1 for _, p in pairs if p is not None)
    return {
        "PA": acc(yes), "HR": acc(no), "acc": acc(pairs),
        "n": len(pairs), "n_yes": len(yes), "n_no": len(no),
        "parse_rate": parsed / len(pairs) if pairs else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-omni")
    ap.add_argument("--json-path", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--subsets", nargs="*", default=None, help="sub_category filter; default all")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=8)
    ap.add_argument("--out", default="runs/cmm_baseline.json")
    args = ap.parse_args()

    model = get_model(args.model)
    ds = CMMDataset(args.json_path, args.data_root, sub_categories=args.subsets)
    n = len(ds) if args.limit in (0, None) else min(args.limit, len(ds))
    print(f"CMM: {n}/{len(ds)} questions | subsets={args.subsets or 'all'}", flush=True)

    by_sub = collections.defaultdict(list)
    records = []
    t0 = time.time()
    for i in range(n):
        item = ds[i]
        gold = item["answer"]
        v, a = item["video_path"], item["audio_path"]
        # separate audio file wins; else only extract audio from video for an audio probe
        uaiv = bool(v) and not a and item.get("modality") == "audio"
        msg = model.message(video=v, audio=a, prompt=item["question"])
        try:
            ans = model.generate(msg, use_audio_in_video=uaiv, max_new_tokens=args.max_new_tokens)
            pred = parse_yes_no(ans)
        except Exception as e:  # noqa: BLE001 — skip bad/missing media, keep going
            ans, pred = f"ERROR: {e}", None
        by_sub[item["sub_category"]].append((gold, pred))
        records.append({
            "sub_category": item["sub_category"], "modality": item.get("modality"),
            "question": item["question"], "answer": gold, "pred": pred, "raw": ans,
        })
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{n} ({(time.time() - t0) / (i + 1):.1f}s/it)", flush=True)

    results = {sub: _scores(p) for sub, p in by_sub.items()}
    results["overall"] = _scores([pr for p in by_sub.values() for pr in p])
    audio_pairs = [pr for sub, p in by_sub.items() if sub in AUDIO_SUBSETS for pr in p]
    if audio_pairs:
        results["audio_subsets"] = _scores(audio_pairs)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"results": results, "records": records}, f, indent=2)

    print("\n=== CMM baseline (frozen Qwen3-Omni) ===")
    for sub, m in results.items():
        print(f"  {sub:34s} PA={m['PA']:.3f} HR={m['HR']:.3f} acc={m['acc']:.3f} "
              f"(n={m['n']}, parse={m['parse_rate']:.2f})")
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
