"""Frozen Qwen3-Omni baseline on AVHBench (the 3 binary yes/no tasks).

  python -m rlvib.eval.run_avhbench --qa-json qa.json --video-root videos/ [--limit N]

Captioning (METEOR/CIDEr/GAVIE) is out of scope for the baseline — default tasks
are the three binary ones. Establishes the number to beat before any training.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import time

from rlvib.data.avhbench import BINARY_TASKS, AVHBenchDataset
from rlvib.eval.metrics import accuracy, parse_yes_no
from rlvib.models import get_model

YN_SUFFIX = " Answer with a single word: Yes or No."


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-omni")
    ap.add_argument("--bottleneck", default=None, help="attach a trained bottleneck checkpoint")
    ap.add_argument("--qa-json", required=True)
    ap.add_argument("--video-root", required=True)
    ap.add_argument("--tasks", nargs="*", default=list(BINARY_TASKS))
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--max-new-tokens", type=int, default=8)
    ap.add_argument("--out", default="runs/avhbench_baseline.json")
    args = ap.parse_args()

    model = get_model(args.model)
    if args.bottleneck:
        from rlvib.models.bottleneck import load_attached
        _bn, _h = load_attached(model, args.bottleneck)
        print(f"attached bottleneck <- {args.bottleneck}", flush=True)
    ds = AVHBenchDataset(args.qa_json, args.video_root, tasks=args.tasks)
    n = len(ds) if args.limit in (0, None) else min(args.limit, len(ds))
    print(f"AVHBench: {n}/{len(ds)} samples | tasks={args.tasks}", flush=True)

    per_task = collections.defaultdict(lambda: {"preds": [], "golds": []})
    records = []
    t0 = time.time()
    for i in range(n):
        item = ds[i]
        gold = str(item["label"]).strip().lower()  # "yes" / "no"
        msg = model.message(video=item["video_path"], prompt=item["text"] + YN_SUFFIX)
        try:
            ans = model.generate(msg, use_audio_in_video=True, max_new_tokens=args.max_new_tokens)
            pred = parse_yes_no(ans)
        except Exception as e:  # noqa: BLE001 — skip bad/missing clips, keep going
            ans, pred = f"ERROR: {e}", None
        per_task[item["task"]]["preds"].append(pred)
        per_task[item["task"]]["golds"].append(gold)
        records.append({
            "video_path": item["video_path"], "task": item["task"],
            "text": item["text"], "label": item["label"], "answer": ans, "pred": pred,
        })
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{n}  ({(time.time() - t0) / (i + 1):.1f}s/it)", flush=True)

    results, all_preds, all_golds = {}, [], []
    for task, d in per_task.items():
        results[task] = accuracy(d["preds"], d["golds"])
        all_preds += d["preds"]
        all_golds += d["golds"]
    results["overall"] = accuracy(all_preds, all_golds)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"results": results, "records": records}, f, indent=2)

    print("\n=== AVHBench baseline (frozen Qwen3-Omni) ===")
    for task, m in results.items():
        print(f"  {task:28s} acc={m['accuracy']:.3f}  (n={m['n']}, parse={m['parse_rate']:.2f})")
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
