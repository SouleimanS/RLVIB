#!/usr/bin/env python
"""Mine REAL dominance/hallucination examples for an original "over-reliance on unimodal
priors" figure (our own version, not a copy). NO GPU: reads the CMM eval records (base +
selected checkpoint), and per sub_category lists 'no'-gold probes where the BASE model
hallucinated (predicted 'yes') and, ideally, the ADAPTED model corrected it (predicted
'no') -- with the clip path, question, audio modality, and both raw outputs, so you can
pick one clip per dominance type (language / visual / audio) and extract frames.

  python scripts/find_dominance_examples.py --model qwen3-omni --exp broad --step 60 --n 4
"""
from __future__ import annotations

import argparse
import collections
import json
import os


def _recs(p):
    return json.load(open(p)).get("records", []) if os.path.exists(p) else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-omni")
    ap.add_argument("--exp", default="broad")
    ap.add_argument("--step", type=int, default=60)
    ap.add_argument("--n", type=int, default=4, help="examples shown per sub_category")
    a = ap.parse_args()
    xt = f"_{a.exp}" if a.exp else ""
    base = _recs(f"runs/cmm_{a.model}.json")
    adt = _recs(f"runs/cmm_{a.model}{xt}_step{a.step}.json")
    if base is None or adt is None:
        print("missing base/adapted CMM JSONs (run the evals + rederive_preds first)")
        return 1

    n = min(len(base), len(adt))
    bycat = collections.defaultdict(list)
    for i in range(n):
        b = base[i]
        if b.get("answer") == "no":           # hallucination probes are the 'no'-gold ones
            bycat[b.get("sub_category")].append((i, b, adt[i]))

    print(f"CMM 'no'-gold probes: {a.model} base vs {a.exp} step{a.step}  "
          f"(pick one clip per dominance type)\n")
    for cat, items in sorted(bycat.items()):
        fixed = [(i, b, d) for (i, b, d) in items if b.get("pred") == "yes" and d.get("pred") == "no"]
        halluc = [(i, b, d) for (i, b, d) in items if b.get("pred") == "yes"]
        print(f"### {cat}   ({len(items)} probes | base hallucinated {len(halluc)} | "
              f"OURS FIXED {len(fixed)})")
        for i, b, d in (fixed or halluc)[: a.n]:
            tag = "BASE-FAILS/OURS-FIXES" if (b.get("pred") == "yes" and d.get("pred") == "no") \
                else "base-hallucinates"
            print(f"  [{tag}] idx={i}  modality={b.get('modality')}")
            print(f"     video: {b.get('video_path')}")
            if b.get("audio_path"):
                print(f"     audio: {b.get('audio_path')}")
            print(f"     Q: {b.get('question')}")
            print(f"     gold=no | base={b.get('pred')!r} <= {(b.get('raw') or '')[:90]!r}")
            print(f"            | ours={d.get('pred')!r} <= {(d.get('raw') or '')[:90]!r}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
