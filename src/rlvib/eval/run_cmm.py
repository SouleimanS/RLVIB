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
from rlvib.eval.timeout import time_limit
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
    ap.add_argument("--bottleneck", default=None, help="attach a trained bottleneck checkpoint")
    ap.add_argument("--json-path", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--subsets", nargs="*", default=None, help="sub_category filter; default all")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=8)
    ap.add_argument("--out", default="runs/cmm_baseline.json")
    ap.add_argument("--save-every", type=int, default=25, help="checkpoint the out JSON every N items")
    ap.add_argument("--no-resume", action="store_true", help="start fresh, ignoring any existing --out")
    ap.add_argument("--gen-timeout", type=int, default=120,
                    help="per-item wall-clock cap (s); a clip that hangs generate() is skipped "
                         "(pred=None) instead of stalling the whole run. 0 disables.")
    ap.add_argument("--skip-clips", default="rEFeVc",
                    help="comma-separated clip-name substrings to skip WITHOUT calling generate() "
                         "-- a hard backstop for clips that wedge the decoder below the Python level "
                         "(where --gen-timeout's signal can't reach). '' to disable.")
    ap.add_argument("--skip-file", default="runs/cmm_skip_clips.txt",
                    help="optional file of extra clip substrings (comma/newline separated) to skip, "
                         "e.g. the output of scripts/scan_bad_clips.py. Merged with --skip-clips.")
    args = ap.parse_args()

    model = get_model(args.model)
    if args.bottleneck:
        from rlvib.models.bottleneck import load_attached
        _bn, _h = load_attached(model, args.bottleneck)
        print(f"attached bottleneck <- {args.bottleneck}", flush=True)
    ds = CMMDataset(args.json_path, args.data_root, sub_categories=args.subsets)
    n = len(ds) if args.limit in (0, None) else min(args.limit, len(ds))
    print(f"CMM: {n}/{len(ds)} questions | subsets={args.subsets or 'all'}", flush=True)

    # Resume a partial run (API evals are long/flaky): reload saved records and continue.
    records, by_sub = [], collections.defaultdict(list)
    if not args.no_resume and os.path.exists(args.out):
        with open(args.out) as f:
            records = json.load(f).get("records", [])[:n]
        for r in records:
            by_sub[r["sub_category"]].append((r["answer"], r["pred"]))
        if records:
            print(f"resuming from {len(records)} saved records in {args.out}", flush=True)

    def _write():
        res = {sub: _scores(p) for sub, p in by_sub.items()}
        res["overall"] = _scores([pr for p in by_sub.values() for pr in p])
        audio = [pr for sub, p in by_sub.items() if sub in AUDIO_SUBSETS for pr in p]
        if audio:
            res["audio_subsets"] = _scores(audio)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"results": res, "records": records}, f, indent=2)
        return res

    skip = [s for s in args.skip_clips.split(",") if s]
    if args.skip_file and os.path.exists(args.skip_file):
        with open(args.skip_file) as f:
            skip += [s.strip() for s in f.read().replace("\n", ",").split(",") if s.strip()]
    skip = sorted(set(skip))
    if skip:
        print(f"skip-clips (hard backstop): {skip}", flush=True)
    start, t0 = len(records), time.time()
    for i in range(start, n):
        item = ds[i]
        gold = item["answer"]
        v, a = item["video_path"], item["audio_path"]
        # separate audio file wins; else only extract audio from video for an audio probe
        uaiv = bool(v) and not a and item.get("modality") == "audio"
        if any(s in (v or "") or s in (a or "") for s in skip):
            ans, pred = "SKIPPED (skip-clips)", None        # decoder-hang clip; keep indices aligned
        else:
            msg = model.message(video=v, audio=a, prompt=item["question"])
            try:
                with time_limit(args.gen_timeout):
                    ans = model.generate(msg, use_audio_in_video=uaiv, max_new_tokens=args.max_new_tokens)
                pred = parse_yes_no(ans)
            except Exception as e:  # noqa: BLE001 — skip bad/missing/hanging media, keep going
                ans, pred = f"ERROR: {e}", None
        by_sub[item["sub_category"]].append((gold, pred))
        records.append({
            "sub_category": item["sub_category"], "modality": item.get("modality"),
            "question": item["question"], "answer": gold, "pred": pred, "raw": ans,
        })
        done = i + 1
        if done - start <= 3 or done % 10 == 0 or done == n:  # early feedback, then every 10
            print(f"  {done}/{n} ({(time.time() - t0) / max(done - start, 1):.1f}s/it)", flush=True)
        if args.save_every and done % args.save_every == 0:
            _write()                                          # checkpoint so a crash loses <= N items

    results = _write()
    print("\n=== CMM baseline ===")
    for sub, m in results.items():
        print(f"  {sub:34s} PA={m['PA']:.3f} HR={m['HR']:.3f} acc={m['acc']:.3f} "
              f"(n={m['n']}, parse={m['parse_rate']:.2f})")
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
