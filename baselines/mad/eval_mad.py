#!/usr/bin/env python
"""Evaluate MAD (training-free modality-adaptive decoding) on AVHBench or CMM.

Reuses the repo's datasets, model wrappers, metrics and answer suffix; only the decoding
changes (baselines/mad/mad.py). ~5 forwards per decoded token -> use --limit generously.

  PYTHONPATH=src python baselines/mad/eval_mad.py --bench avhbench --model qwen2.5-omni --limit 300
  PYTHONPATH=src python baselines/mad/eval_mad.py --bench cmm      --model qwen2.5-omni --limit 300
Writes runs/mad_<bench>_<model>.json (records + per-task metrics; resumable).
"""
from __future__ import annotations

import argparse
import collections
import json
import logging
import os
import sys
import warnings

from tqdm.auto import tqdm

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")
for _n in ("transformers", "qwen_vl_utils", "qwen_omni_utils"):
    logging.getLogger(_n).setLevel(logging.ERROR)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mad import GAMMA, mad_answer  # noqa: E402

from rlvib.eval.metrics import accuracy, parse_yes_no  # noqa: E402
from rlvib.eval.run_avhbench import DEFAULT_YN_SUFFIX  # noqa: E402
from rlvib.eval.timeout import time_limit  # noqa: E402
from rlvib.models import get_model  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", choices=["avhbench", "cmm"], required=True)
    ap.add_argument("--model", default="qwen2.5-omni")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--gamma", type=float, default=GAMMA)
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=8)
    ap.add_argument("--gen-timeout", type=int, default=300)
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.fps is None:
        args.fps = {"qwen3-omni": 2.0, "qwen2.5-omni": 1.0}.get(args.model)
    args.out = args.out or f"runs/mad_{args.bench}_{args.model}.json"

    model = get_model(args.model)
    if args.bench == "avhbench":
        from rlvib.data.avhbench import AVHBenchDataset
        ds = AVHBenchDataset("data/AVHBench/qa.json", "data/AVHBench/videos")
        get = lambda it: (it["video_path"], None, it["text"].rstrip() + " " + DEFAULT_YN_SUFFIX,  # noqa: E731
                          str(it["label"]).strip().lower(), it["task"])
    else:
        from rlvib.data.cmm import CMMDataset
        ds = CMMDataset("data/CMM/all_data_final_reorg.json", "data/CMM")
        get = lambda it: (it["video_path"], it["audio_path"], it["question"],  # noqa: E731
                          it["answer"], it["sub_category"])

    n = len(ds) if args.limit in (0, None) else min(args.limit, len(ds))
    per = collections.defaultdict(lambda: {"preds": [], "golds": []})
    records = []
    if os.path.exists(args.out):                          # resume
        with open(args.out) as f:
            records = json.load(f).get("records", [])[:n]
        for r in records:
            per[r["task"]]["preds"].append(r["pred"])
            per[r["task"]]["golds"].append(r["gold"])
        print(f"resuming from {len(records)}", flush=True)

    def _write():
        res = {t: accuracy(d["preds"], d["golds"]) for t, d in per.items()}
        allp = [p for d in per.values() for p in d["preds"]]
        allg = [g for d in per.values() for g in d["golds"]]
        res["overall"] = accuracy(allp, allg)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"gamma": args.gamma, "results": res, "records": records}, f, indent=2)
        return res

    bar = tqdm(range(len(records), n), total=n, initial=len(records),
               desc=f"MAD/{args.bench}", unit="q", dynamic_ncols=True)
    live = [0, 0]
    for i in bar:
        v, a, prompt, gold, task = get(ds[i])
        try:
            with time_limit(args.gen_timeout):
                ans = mad_answer(model, video=v, audio=a, prompt=prompt, fps=args.fps,
                                 gamma=args.gamma, max_new_tokens=args.max_new_tokens)
            pred = parse_yes_no(ans)
        except Exception as e:  # noqa: BLE001
            ans, pred = f"ERROR: {e}", None
        per[task]["preds"].append(pred)
        per[task]["golds"].append(gold)
        records.append({"task": task, "gold": gold, "pred": pred, "raw": ans})
        live[1] += 1
        live[0] += int(pred == gold)
        bar.set_postfix(acc=f"{100 * live[0] / live[1]:.1f}", refresh=False)
        if args.save_every and (i + 1) % args.save_every == 0:
            _write()

    res = _write()
    print(f"\n=== MAD ({args.bench}, gamma={args.gamma}) ===")
    for t, m_ in res.items():
        print(f"  {t:34s} acc={m_['accuracy']:.3f} (n={m_['n']})")
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
