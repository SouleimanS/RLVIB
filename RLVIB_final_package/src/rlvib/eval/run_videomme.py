"""Video-MME MCQ eval -- general video understanding, our capability guard at scale.

  python -m rlvib.eval.run_videomme --model qwen3-omni --durations short --limit 6      # smoke
  python -m rlvib.eval.run_videomme --model qwen3-omni --durations short \
      --bottleneck runs/anchored_qwen3-omni_film/bottleneck_step160.pt                  # + FiLM

Without-subtitles track. The video's own audio is used (use_audio_in_video=True); pass
--no-audio for the vision-only ablation. Answers are single letters (A-D), parsed with
parse_choice and scored against the gold letter. Per-duration + per-task accuracy,
resumable JSON, same shape as run_mmau. Qwen-Omni base or +bottleneck (FiLM condition
auto-set from the question).
"""
from __future__ import annotations

import argparse
import collections
import json
import logging
import os
import warnings

from tqdm.auto import tqdm

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")
for _n in ("transformers", "qwen_vl_utils", "qwen_omni_utils"):
    logging.getLogger(_n).setLevel(logging.ERROR)

from rlvib.data.videomme import DURATIONS, format_videomme, load_videomme  # noqa: E402
from rlvib.eval.metrics import parse_choice  # noqa: E402
from rlvib.eval.timeout import time_limit  # noqa: E402
from rlvib.models import get_model  # noqa: E402


def _acc(pairs):
    """pairs: list of (correct_bool, parsed_bool)."""
    n = len(pairs)
    return {"acc": sum(c for c, _ in pairs) / n if n else 0.0, "n": n,
            "parse_rate": sum(p for _, p in pairs) / n if n else 0.0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-omni")
    ap.add_argument("--bottleneck", default=None, help="attach a trained bottleneck checkpoint")
    ap.add_argument("--data-dir", default="data/VideoMME")
    ap.add_argument("--video-root", default=None, help="default: <data-dir>/videos")
    ap.add_argument("--durations", default="short",
                    help='comma-separated subset of short,medium,long -- or "all". '
                         "Default short: medium/long blow up audio+frame token budgets.")
    ap.add_argument("--fps", type=float, default=None,
                    help="frame-sampling fps override (e.g. 0.5 for medium/long)")
    ap.add_argument("--no-audio", action="store_true", help="vision-only ablation")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--max-new-tokens", type=int, default=8)
    ap.add_argument("--out", default=None, help="default: runs/videomme_<model>.json")
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--gen-timeout", type=int, default=600)
    args = ap.parse_args()
    args.out = args.out or f"runs/videomme_{args.model}.json"

    model = get_model(args.model)
    cond = False
    if args.bottleneck:
        from rlvib.models.bottleneck import load_attached, question_embedding, set_condition
        bns, _h = load_attached(model, args.bottleneck)
        cond = "q_proj" in bns
        print(f"attached bottleneck <- {args.bottleneck}" + ("  (prompt-aware/FiLM)" if cond else ""),
              flush=True)

    durations = None if args.durations.strip().lower() == "all" else args.durations.split(",")
    ds = load_videomme(args.data_dir, args.video_root, durations)
    n = len(ds) if args.limit in (0, None) else min(args.limit, len(ds))
    missing = sum(not os.path.exists(x["video_path"]) for x in ds[:n])
    print(f"Video-MME: {n}/{len(ds)} questions (durations={args.durations}, "
          f"audio={'off' if args.no_audio else 'on'}, missing videos among them: {missing})",
          flush=True)

    records = []
    by_dur = collections.defaultdict(list)
    by_task = collections.defaultdict(list)
    if not args.no_resume and os.path.exists(args.out):
        with open(args.out) as f:
            records = json.load(f).get("records", [])[:n]
        for r in records:
            by_dur[r["duration"]].append((r["correct"], r["pred"] is not None))
            by_task[r["task_type"]].append((r["correct"], r["pred"] is not None))
        if records:
            print(f"resuming from {len(records)} saved records in {args.out}", flush=True)

    def _write():
        res = {d: _acc(by_dur[d]) for d in DURATIONS if by_dur[d]}
        allp = [pr for d in by_dur for pr in by_dur[d]]
        res["overall"] = _acc(allp)
        res["by_task"] = {t: _acc(by_task[t]) for t in sorted(by_task)}
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"results": res, "records": records}, f, indent=2)
        return res

    live = {"c": sum(int(r["correct"]) for r in records), "n": len(records)}
    start = len(records)
    bar = tqdm(range(start, n), total=n, initial=start, desc="Video-MME", unit="q",
               dynamic_ncols=True)
    for i in bar:
        item = ds[i]
        if cond:
            set_condition(bns, question_embedding(model, item["question"]))
        try:
            with time_limit(args.gen_timeout):
                msg = model.message(video=item["video_path"],
                                    prompt=format_videomme(item["question"], item["options"]),
                                    fps=args.fps)
                ans = model.generate(msg, use_audio_in_video=not args.no_audio,
                                     max_new_tokens=args.max_new_tokens)
            pred = parse_choice(ans)                                     # "A".."Z" or None
        except Exception as e:  # noqa: BLE001
            ans, pred = f"ERROR: {e}", None
        correct = pred is not None and pred == item["answer"]
        by_dur[item["duration"]].append((correct, pred is not None))
        by_task[item["task_type"]].append((correct, pred is not None))
        live["n"] += 1
        live["c"] += int(correct)
        records.append({"id": item["id"], "duration": item["duration"],
                        "task_type": item["task_type"], "domain": item["domain"],
                        "answer": item["answer"], "pred": pred, "correct": correct, "raw": ans})
        bar.set_postfix(acc=f"{100 * live['c'] / live['n']:.1f}" if live["n"] else "—",
                        refresh=False)
        if args.save_every and (i + 1) % args.save_every == 0:
            _write()

    results = _write()
    print("\n=== Video-MME (w/o subs) ===")
    for d in DURATIONS:
        if d in results:
            print(f"  {d.capitalize():8s} acc={results[d]['acc']:.4f}  (n={results[d]['n']})")
    print(f"  {'Overall':8s} acc={results['overall']['acc']:.4f}  (n={results['overall']['n']})")
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
