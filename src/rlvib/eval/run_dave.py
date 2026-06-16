"""Frozen Qwen3-Omni baseline on DAVE (multiple choice; every Q needs both modalities).

  python -m rlvib.eval.run_dave --json data/DAVE/ego4d.json --media-root data/DAVE/ego4d \
      --mode audio_visual_alignment

Run all four modes to get the modality-ablation ΔAcc:
  audio_visual_alignment vs visual_only / audio_only / text_only.

The prompt references the overlaid sound's moment (DAVE's design) and is kept
identical across modes, so ablating a modality fairly drops accuracy. (Approximate
vs the official DAVE prompts.py; the cross-mode ΔAcc is the signal we want.)
"""
from __future__ import annotations

import argparse
import json
import os
import string
import time

from rlvib.data.dave import MODE_SPEC, DaveDataset
from rlvib.eval.metrics import parse_choice
from rlvib.models import get_model


def build_prompt(choices: list[str]) -> str:
    opts = "\n".join(f"({string.ascii_uppercase[i]}) {c}" for i, c in enumerate(choices))
    return (
        "An extra sound was overlaid onto this clip at one moment. "
        "What is the person doing at the moment that sound is heard?\n"
        f"{opts}\nAnswer with only the letter."
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-omni")
    ap.add_argument("--json", required=True)
    ap.add_argument("--media-root", required=True)
    ap.add_argument("--mode", default="audio_visual_alignment", choices=list(MODE_SPEC))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model = get_model(args.model)
    ds = DaveDataset(args.json, args.media_root, mode=args.mode)
    n = len(ds) if args.limit in (0, None) else min(args.limit, len(ds))
    out = args.out or f"runs/dave_{args.mode}.json"
    print(f"DAVE[{args.mode}]: {n}/{len(ds)}", flush=True)

    correct = parsed = 0
    records = []
    t0 = time.time()
    for i in range(n):
        item = ds[i]
        gt = string.ascii_uppercase[item["gt_index"]] if item["gt_index"] is not None else None
        v = item["media_path"] if item["kind"] == "video" else None
        a = item["media_path"] if item["kind"] == "audio" else None
        msg = model.message(video=v, audio=a, prompt=build_prompt(item["choices"]))
        try:
            ans = model.generate(msg, use_audio_in_video=item["use_audio"],
                                  max_new_tokens=args.max_new_tokens)
            pred = parse_choice(ans)
        except Exception as e:  # noqa: BLE001 — skip bad/missing media, keep going
            ans, pred = f"ERROR: {e}", None
        parsed += pred is not None
        correct += int(pred == gt)
        records.append({"gt": gt, "pred": pred, "raw": ans, "type": item["type"]})
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{n} ({(time.time() - t0) / (i + 1):.1f}s/it)", flush=True)

    acc = correct / n if n else 0.0
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump({"mode": args.mode, "accuracy": acc, "n": n,
                   "parse_rate": parsed / n if n else 0.0, "records": records}, f, indent=2)
    print(f"\n=== DAVE [{args.mode}] acc={acc:.3f} (n={n}, parse={parsed / n if n else 0:.2f}) ===")
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
