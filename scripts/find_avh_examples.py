#!/usr/bin/env python
"""Mine AVHBench examples where the BASE answers wrong and OURS answers right, per
hallucination task. NO GPU. AVHBench eval records store the clip path, so this directly
lists the videos -- pick one per task (A->V, V->A, AV-matching).

  python scripts/find_avh_examples.py --model qwen3-omni --exp broad --step 60 --n 6
  python scripts/find_avh_examples.py ... --suffix full      # after the full-set eval
"""
from __future__ import annotations

import argparse
import collections
import json
import os

# (task string in the data, short label, what the question probes)
TASKS = [
    ("Audio-driven Video Hallucination", "A->V", "is the queried object VISIBLE? (audio-driven visual hallucination)"),
    ("Video-driven Audio Hallucination", "V->A", "is the queried sound AUDIBLE?  (video-driven audio hallucination)"),
    ("AV Matching", "AV-match", "do the audio and video correspond?"),
]


def _recs(p):
    return json.load(open(p)).get("records", []) if os.path.exists(p) else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-omni")
    ap.add_argument("--exp", default="broad")
    ap.add_argument("--step", type=int, default=60)
    ap.add_argument("--n", type=int, default=6, help="examples shown per task")
    ap.add_argument("--suffix", default="", help="read *_{suffix}.json (e.g. 'full')")
    a = ap.parse_args()
    sx = f"_{a.suffix}" if a.suffix else ""
    xt = f"_{a.exp}" if a.exp else ""
    base = _recs(f"runs/avhbench_{a.model}{sx}.json")
    adt = _recs(f"runs/avhbench_{a.model}{xt}{sx}_step{a.step}.json")
    if base is None or adt is None:
        print("missing base/adapted AVHBench JSONs (run the evals first)")
        return 1

    n = min(len(base), len(adt))
    by_task = collections.defaultdict(list)
    for i in range(n):
        b, d = base[i], adt[i]
        gold = str(b.get("label", "")).strip().lower()
        if b.get("pred") != gold and d.get("pred") == gold:        # base wrong, ours right
            by_task[b.get("task")].append((i, b, d, gold))

    print(f"AVHBench  base-WRONG / ours-RIGHT  ({a.model} base vs {a.exp} step{a.step}"
          f"{', ' + a.suffix if a.suffix else ''})\n")
    for task, short, probes in TASKS:
        items = by_task.get(task, [])
        print(f"### {short}  [{task}] -- {probes}\n    ({len(items)} base-wrong/ours-right)")
        for i, b, d, gold in items[: a.n]:
            print(f"  idx={i}")
            print(f"     video: {b.get('video_path')}")
            print(f"     Q: {b.get('text')}")
            print(f"     gold={gold!r} | base={b.get('pred')!r} <= {(b.get('answer') or '')[:80]!r}")
            print(f"               | ours={d.get('pred')!r} <= {(d.get('answer') or '')[:80]!r}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
