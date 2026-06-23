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
import logging
import os
import warnings

from tqdm.auto import tqdm

from rlvib.data.cmm import AUDIO_SUBSETS, CMMDataset
from rlvib.eval.contrastive import contrastive_answer
from rlvib.eval.metrics import parse_yes_no
from rlvib.eval.timeout import time_limit
from rlvib.models import get_model

# Quiet the eval logs (see run_avhbench): transformers config/LOAD-REPORT + librosa/decord noise.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")
for _n in ("transformers", "qwen_vl_utils", "qwen_omni_utils"):
    logging.getLogger(_n).setLevel(logging.ERROR)


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
    ap.add_argument("--fps", type=float, default=None,
                    help="video fps for the Qwen frame sampler (None = model default). Ignored by "
                         "non-Qwen models.")
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
    # live tqdm: running overall acc + PA (gold=yes) + HR (gold=no), CMM's two headline metrics
    live = {"c": 0, "n": 0, "pa_c": 0, "pa_n": 0, "hr_c": 0, "hr_n": 0}
    for pairs in by_sub.values():                            # seed from any resumed records
        for g, p in pairs:
            live["n"] += 1
            live["c"] += int(p == g)
            if g == "yes":
                live["pa_n"] += 1
                live["pa_c"] += int(p == g)
            elif g == "no":
                live["hr_n"] += 1
                live["hr_c"] += int(p == g)

    def _postfix():
        d = {"acc": f"{100 * live['c'] / live['n']:.1f}" if live["n"] else "—"}
        if live["pa_n"]:
            d["PA"] = f"{100 * live['pa_c'] / live['pa_n']:.1f}"
        if live["hr_n"]:
            d["HR"] = f"{100 * live['hr_c'] / live['hr_n']:.1f}"
        return d

    msg_kwargs = {} if args.fps is None else {"fps": args.fps}
    start = len(records)
    bar = tqdm(range(start, n), total=n, initial=start, desc="CMM", unit="q", dynamic_ncols=True)
    for i in bar:
        item = ds[i]
        gold = item["answer"]
        v, a = item["video_path"], item["audio_path"]
        # separate audio file wins; else only extract audio from video for an audio probe
        uaiv = bool(v) and not a and item.get("modality") == "audio"
        if any(s in (v or "") or s in (a or "") for s in skip):
            ans, pred = "SKIPPED (skip-clips)", None        # decoder-hang clip; keep indices aligned
        else:
            try:
                with time_limit(args.gen_timeout):
                    if cd_alpha > 0:  # audio-aware contrastive decoding, composed with the bottleneck
                        ans = contrastive_answer(model, video=v, audio=a, prompt=item["question"],
                                                 alpha=cd_alpha, use_audio_in_video=uaiv,
                                                 plausibility=args.cd_plausibility)
                    else:
                        msg = model.message(video=v, audio=a, prompt=item["question"], **msg_kwargs)
                        ans = model.generate(msg, use_audio_in_video=uaiv,
                                             max_new_tokens=args.max_new_tokens)
                pred = parse_yes_no(ans)
            except Exception as e:  # noqa: BLE001 — skip bad/missing/hanging media, keep going
                ans, pred = f"ERROR: {e}", None
        by_sub[item["sub_category"]].append((gold, pred))
        live["n"] += 1
        live["c"] += int(pred == gold)
        if gold == "yes":
            live["pa_n"] += 1
            live["pa_c"] += int(pred == gold)
        elif gold == "no":
            live["hr_n"] += 1
            live["hr_c"] += int(pred == gold)
        records.append({
            "sub_category": item["sub_category"], "modality": item.get("modality"),
            "question": item["question"], "answer": gold, "pred": pred, "raw": ans,
        })
        bar.set_postfix(_postfix(), refresh=False)
        if args.save_every and (i + 1) % args.save_every == 0:
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
