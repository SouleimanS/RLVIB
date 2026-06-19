#!/usr/bin/env python
"""Mine REAL hallucination examples for an original "over-reliance on unimodal priors"
figure (our own version, not a copy). NO GPU.

Reads the CMM eval records (base + selected checkpoint) and, per sub_category, lists
'no'-gold probes where the BASE model hallucinated (pred='yes') and ideally the ADAPTED
model corrected it (pred='no'). The eval records don't store the clip path, so we resolve
it from the SOURCE CMM json by index (eval record i <-> source sample i, since the eval
runs the dataset in order with no subset filter). Also prints which sub_categories the eval
actually covered -- a --limit run may only touch the first (the json is grouped).

  python scripts/find_dominance_examples.py --model qwen3-omni --exp broad --step 60 --n 4
"""
from __future__ import annotations

import argparse
import collections
import json
import os


def _recs(p):
    return json.load(open(p)).get("records", []) if os.path.exists(p) else None


def _resolve(p, root):
    return os.path.join(root, p.lstrip("./")) if p else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-omni")
    ap.add_argument("--exp", default="broad")
    ap.add_argument("--step", type=int, default=60)
    ap.add_argument("--n", type=int, default=4, help="examples shown per sub_category")
    ap.add_argument("--json-path", default="data/CMM/all_data_final_reorg.json")
    ap.add_argument("--data-root", default="data/CMM")
    a = ap.parse_args()
    xt = f"_{a.exp}" if a.exp else ""

    base = _recs(f"runs/cmm_{a.model}.json")
    adt = _recs(f"runs/cmm_{a.model}{xt}_step{a.step}.json")
    if base is None or adt is None:
        print("missing base/adapted CMM JSONs (run the evals + rederive_preds first)")
        return 1
    src = json.load(open(a.json_path)) if os.path.exists(a.json_path) else None
    if src is None:
        print(f"(source CMM json {a.json_path} not found -- clip paths/captions unavailable)\n")

    n = min(len(base), len(adt))
    if src:
        allcat = collections.Counter(s.get("sub_category") for s in src)
        covered = collections.Counter(s.get("sub_category") for s in src[:n])
        print(f"source CMM sub_categories (total | covered by your first-{n} eval):")
        for c, k in allcat.most_common():
            mark = "" if covered.get(c, 0) else "   <- NOT in your eval (raise --limit or use --subsets)"
            print(f"   {str(c):30s} {k:5d} | covered {covered.get(c, 0):4d}{mark}")
        print()

    bycat = collections.defaultdict(list)
    for i in range(n):
        b = base[i]
        if b.get("answer") == "no":            # hallucination probes are the 'no'-gold ones
            bycat[b.get("sub_category")].append((i, b, adt[i]))

    print(f"'no'-gold probes mined (base vs {a.exp} step{a.step}):\n")
    for cat, items in sorted(bycat.items()):
        fixed = [(i, b, d) for (i, b, d) in items if b.get("pred") == "yes" and d.get("pred") == "no"]
        halluc = [(i, b, d) for (i, b, d) in items if b.get("pred") == "yes"]
        print(f"### {cat}   ({len(items)} probes | base hallucinated {len(halluc)} | "
              f"OURS FIXED {len(fixed)})")
        for i, b, d in (fixed or halluc)[: a.n]:
            tag = "BASE-FAILS/OURS-FIXES" if (b.get("pred") == "yes" and d.get("pred") == "no") \
                else "base-hallucinates"
            s = src[i] if (src and i < len(src)) else {}
            vp = _resolve(s.get("video_path"), a.data_root)
            ap_ = _resolve(s.get("audio_path"), a.data_root)
            extra = {k: v for k, v in s.items()
                     if k not in ("video_path", "audio_path", "question", "answer", "sub_category")}
            print(f"  [{tag}] idx={i}  modality={b.get('modality')}")
            print(f"     video: {vp}")
            if ap_:
                print(f"     audio: {ap_}")
            if extra:
                print(f"     meta:  {extra}")
            print(f"     Q: {b.get('question')}")
            print(f"     gold=no | base={b.get('pred')!r} <= {(b.get('raw') or '')[:90]!r}")
            print(f"            | ours={d.get('pred')!r} <= {(d.get('raw') or '')[:90]!r}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
