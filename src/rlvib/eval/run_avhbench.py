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
from rlvib.eval.contrastive import contrastive_answer
from rlvib.eval.metrics import accuracy, parse_yes_no
from rlvib.eval.timeout import time_limit
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
    ap.add_argument("--save-every", type=int, default=25, help="checkpoint the out JSON every N items")
    ap.add_argument("--no-resume", action="store_true", help="start fresh, ignoring any existing --out")
    ap.add_argument("--gen-timeout", type=int, default=120,
                    help="per-item wall-clock cap (s); a clip that hangs generate() is skipped "
                         "(pred=None) instead of stalling the whole run. 0 disables.")
    ap.add_argument("--audio-cd", type=float, default=0.0,
                    help="audio-aware contrastive decoding strength alpha (0=off); composes with "
                         "the attached bottleneck. Qwen3-/Qwen2.5-Omni only.")
    ap.add_argument("--cd-plausibility", type=float, default=0.1,
                    help="VCD plausibility constraint for --audio-cd (keep tokens within "
                         "log(plausibility) of the full pass's max).")
    args = ap.parse_args()

    model = get_model(args.model)
    if args.bottleneck:
        from rlvib.models.bottleneck import load_attached
        _bn, _h = load_attached(model, args.bottleneck)
        print(f"attached bottleneck <- {args.bottleneck}", flush=True)
    cd_alpha = args.audio_cd
    if cd_alpha > 0 and args.model not in ("qwen3-omni", "qwen2.5-omni"):
        print(f"[audio-cd] unsupported for {args.model} (sentence answers); plain decoding", flush=True)
        cd_alpha = 0.0
    elif cd_alpha > 0:
        print(f"[audio-cd] audio-aware contrastive decoding ON (alpha={cd_alpha})", flush=True)
    ds = AVHBenchDataset(args.qa_json, args.video_root, tasks=args.tasks)
    n = len(ds) if args.limit in (0, None) else min(args.limit, len(ds))
    print(f"AVHBench: {n}/{len(ds)} samples | tasks={args.tasks}", flush=True)

    # Resume a partial run (API evals are long/flaky): reload saved records and continue.
    per_task = collections.defaultdict(lambda: {"preds": [], "golds": []})
    records = []
    if not args.no_resume and os.path.exists(args.out):
        with open(args.out) as f:
            records = json.load(f).get("records", [])[:n]
        for r in records:
            per_task[r["task"]]["preds"].append(r["pred"])
            per_task[r["task"]]["golds"].append(str(r["label"]).strip().lower())
        if records:
            print(f"resuming from {len(records)} saved records in {args.out}", flush=True)

    def _write():
        res, ap_, ag_ = {}, [], []
        for task, d in per_task.items():
            res[task] = accuracy(d["preds"], d["golds"])
            ap_ += d["preds"]
            ag_ += d["golds"]
        res["overall"] = accuracy(ap_, ag_)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"results": res, "records": records}, f, indent=2)
        return res

    start, t0 = len(records), time.time()
    for i in range(start, n):
        item = ds[i]
        gold = str(item["label"]).strip().lower()  # "yes" / "no"
        prompt = item["text"] + YN_SUFFIX
        try:
            with time_limit(args.gen_timeout):
                if cd_alpha > 0:  # audio-aware contrastive decoding, composed with the bottleneck
                    ans = contrastive_answer(model, video=item["video_path"], audio=None,
                                             prompt=prompt, alpha=cd_alpha, use_audio_in_video=True,
                                             plausibility=args.cd_plausibility)
                else:
                    ans = model.generate(model.message(video=item["video_path"], prompt=prompt),
                                         use_audio_in_video=True, max_new_tokens=args.max_new_tokens)
            pred = parse_yes_no(ans)
        except Exception as e:  # noqa: BLE001 — skip bad/missing/hanging clips, keep going
            ans, pred = f"ERROR: {e}", None
        per_task[item["task"]]["preds"].append(pred)
        per_task[item["task"]]["golds"].append(gold)
        records.append({
            "video_path": item["video_path"], "task": item["task"],
            "text": item["text"], "label": item["label"], "answer": ans, "pred": pred,
        })
        done = i + 1
        if done - start <= 3 or done % 10 == 0 or done == n:  # early feedback, then every 10
            print(f"  {done}/{n}  ({(time.time() - t0) / max(done - start, 1):.1f}s/it)", flush=True)
        if args.save_every and done % args.save_every == 0:
            _write()                                          # checkpoint so a crash loses <= N items

    results = _write()
    print("\n=== AVHBench baseline ===")
    for task, m in results.items():
        print(f"  {task:28s} acc={m['accuracy']:.3f}  (n={m['n']}, parse={m['parse_rate']:.2f})")
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
