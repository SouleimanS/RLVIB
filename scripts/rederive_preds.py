#!/usr/bin/env python
"""Re-derive eval `pred` from saved raw outputs with the fixed (negation-aware) parser.

NO GPU, NO re-eval. The old parser matched only a standalone "\\bno\\b", so verbose absence
answers ("I did not see a tree") parsed as None -> scored wrong, flooring CMM-HR/AVHBench for
backbones that answer in full sentences (VideoLLaMA2). This rewrites `pred` for every record
from its saved raw text using rlvib.eval.metrics.parse_yes_no (now fixed), recomputes the
`results` block, backs up the originals under runs/_preparse_backup/, and prints before/after.

Schema-aware: CMM records are {answer=gold, raw=output}; AVHBench are {label=gold, answer=output}.

  PYTHONPATH=src python scripts/rederive_preds.py --model videollama2 --exp broad
  PYTHONPATH=src python scripts/rederive_preds.py --model qwen3-omni              # control (~no-op)
  PYTHONPATH=src python scripts/rederive_preds.py --model videollama2 --exp broad --dry-run
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil

from rlvib.eval.metrics import parse_yes_no  # the FIXED canonical parser


def _gold_raw(rec):
    """(gold 'yes'/'no', raw_output_text) for either benchmark schema."""
    if "raw" in rec:                                   # CMM: answer=gold, raw=output
        return rec.get("answer"), rec.get("raw")
    return str(rec.get("label", "")).strip().lower(), rec.get("answer")  # AVHBench


def _scores(pairs):
    """pairs: (gold, pred). Mirrors run_cmm._scores / run_avhbench accuracy."""
    yes = [(g, p) for g, p in pairs if g == "yes"]
    no = [(g, p) for g, p in pairs if g == "no"]
    acc = lambda ps: (sum(1 for g, p in ps if p == g) / len(ps)) if ps else 0.0  # noqa: E731
    return {"PA": acc(yes), "HR": acc(no), "acc": acc(pairs), "n": len(pairs),
            "n_yes": len(yes), "n_no": len(no),
            "parse_rate": (sum(1 for _, p in pairs if p is not None) / len(pairs)) if pairs else 0.0}


def _group_key(rec):
    return rec.get("sub_category") or rec.get("task") or "all"


def rederive(path, dry_run=False, backup_dir="runs/_preparse_backup"):
    with open(path) as f:
        blob = json.load(f)
    recs = blob.get("records", [])
    if not recs:
        return None

    old = [(g, rec.get("pred")) for rec in recs for g, _ in [_gold_raw(rec)]]
    new_pairs, groups = [], {}
    for rec in recs:
        gold, raw = _gold_raw(rec)
        pred = parse_yes_no(raw)
        rec["pred"] = pred
        new_pairs.append((gold, pred))
        groups.setdefault(_group_key(rec), []).append((gold, pred))

    old_overall, new_overall = _scores(old), _scores(new_pairs)
    results = {k: _scores(v) for k, v in groups.items()}
    results["overall"] = new_overall
    audio = [(g, p) for rec, (g, p) in zip(recs, new_pairs) if rec.get("modality") == "audio"]
    if audio:
        results["audio_subsets"] = _scores(audio)   # recomputed by modality (proxy for AUDIO_SUBSETS)
    blob["results"] = results

    if not dry_run:
        os.makedirs(backup_dir, exist_ok=True)
        shutil.copy2(path, os.path.join(backup_dir, os.path.basename(path)))
        with open(path, "w") as f:
            json.dump(blob, f, indent=2)
    return old_overall, new_overall


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="videollama2")
    ap.add_argument("--exp", default="")
    ap.add_argument("--dry-run", action="store_true", help="report before/after; don't write")
    args = ap.parse_args()
    xt = f"_{args.exp}" if args.exp else ""

    paths = []
    for bench in ("cmm", "avhbench"):
        base = f"runs/{bench}_{args.model}.json"
        if os.path.exists(base):
            paths.append(base)
        paths += sorted(glob.glob(f"runs/{bench}_{args.model}{xt}_step*.json"))
    if not paths:
        print(f"no eval JSONs for model={args.model} exp={args.exp or '-'}")
        return 1

    tag = "DRY-RUN (no write)" if args.dry_run else "rewriting in place (originals -> runs/_preparse_backup/)"
    print(f"=== re-derive preds: {args.model} exp={args.exp or '-'}  [{tag}] ===")
    print(f"{'file':46s}{'HR':>16}{'PA':>14}{'acc':>14}{'parse':>14}")
    print(f"{'':46s}{'old->new':>16}{'old->new':>14}{'old->new':>14}{'old->new':>14}")
    for p in paths:
        r = rederive(p, dry_run=args.dry_run)
        if r is None:
            print(f"{os.path.basename(p):46s}   (no records)")
            continue
        o, n = r
        print(f"{os.path.basename(p):46s}"
              f"{o['HR']:.3f}->{n['HR']:.3f}   {o['PA']:.3f}->{n['PA']:.3f}   "
              f"{o['acc']:.3f}->{n['acc']:.3f}   {o['parse_rate']:.2f}->{n['parse_rate']:.2f}")
    print("\nNext: PYTHONPATH=src python scripts/select_holdout.py "
          f"--model {args.model}{' --exp ' + args.exp if args.exp else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
