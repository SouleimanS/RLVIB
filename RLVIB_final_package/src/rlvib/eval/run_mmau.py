"""MMAU (audio understanding) MCQ eval: per-category Sound/Music/Speech accuracy + overall.

  python -m rlvib.eval.run_mmau --json-path data/MMAU/mmau-test-mini.json --audio-root data/MMAU [--limit N]

Audio-only multiple choice. We prompt for the option LETTER, map it back to the chosen text,
and compare to the gold answer text. Reports accuracy per category and overall (the paper's
"Avg." column is the overall accuracy). Qwen-Omni base or +bottleneck (FiLM condition auto-set).
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

from rlvib.data.mmau import CATEGORIES, format_mmau, load_mmau  # noqa: E402
from rlvib.eval.metrics import parse_choice  # noqa: E402
from rlvib.eval.timeout import time_limit  # noqa: E402
from rlvib.models import get_model  # noqa: E402


def _norm(s: str) -> str:
    return " ".join(str(s).strip().lower().split())


def _acc(pairs):
    """pairs: list of (correct_bool, parsed_bool)."""
    n = len(pairs)
    return {"acc": sum(c for c, _ in pairs) / n if n else 0.0, "n": n,
            "parse_rate": sum(p for _, p in pairs) / n if n else 0.0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5-omni")
    ap.add_argument("--bottleneck", default=None, help="attach a trained bottleneck checkpoint")
    ap.add_argument("--json-path", default="data/MMAU/mmau-test-mini.json")
    ap.add_argument("--audio-root", default="data/MMAU")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--max-new-tokens", type=int, default=8)
    ap.add_argument("--out", default="runs/mmau_baseline.json")
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--gen-timeout", type=int, default=120)
    args = ap.parse_args()

    model = get_model(args.model)
    cond = False
    if args.bottleneck:
        from rlvib.models.bottleneck import load_attached, question_embedding, set_condition
        bns, _h = load_attached(model, args.bottleneck)
        cond = "q_proj" in bns
        print(f"attached bottleneck <- {args.bottleneck}" + ("  (prompt-aware/FiLM)" if cond else ""),
              flush=True)

    ds = load_mmau(args.json_path, args.audio_root)
    n = len(ds) if args.limit in (0, None) else min(args.limit, len(ds))
    print(f"MMAU: {n}/{len(ds)} questions", flush=True)

    records, by_cat = [], collections.defaultdict(list)
    if not args.no_resume and os.path.exists(args.out):
        with open(args.out) as f:
            records = json.load(f).get("records", [])[:n]
        for r in records:
            by_cat[r["category"]].append((r["correct"], r["pred"] is not None))
        if records:
            print(f"resuming from {len(records)} saved records in {args.out}", flush=True)

    def _write():
        res = {c: _acc(by_cat[c]) for c in CATEGORIES if by_cat[c]}
        allp = [pr for c in CATEGORIES for pr in by_cat[c]]
        res["overall"] = _acc(allp)
        res["avg_of_categories"] = {
            "acc": (sum(res[c]["acc"] for c in CATEGORIES if c in res)
                    / max(1, sum(c in res for c in CATEGORIES)))}
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"results": res, "records": records}, f, indent=2)
        return res

    live = {"c": 0, "n": 0}
    for c in CATEGORIES:
        for ok, _ in by_cat[c]:
            live["n"] += 1
            live["c"] += int(ok)

    start = len(records)
    bar = tqdm(range(start, n), total=n, initial=start, desc="MMAU", unit="q", dynamic_ncols=True)
    for i in bar:
        item = ds[i]
        choices = item["choices"]
        gold = _norm(item["answer"])
        if cond:
            set_condition(bns, question_embedding(model, item["question"]))
        try:
            with time_limit(args.gen_timeout):
                msg = model.message(audio=item["audio_path"], prompt=format_mmau(item["question"], choices))
                ans = model.generate(msg, use_audio_in_video=False, max_new_tokens=args.max_new_tokens)
            letter = parse_choice(ans)                                   # "A".."Z" or None
            idx = (ord(letter) - 65) if letter else -1
            pred = _norm(choices[idx]) if 0 <= idx < len(choices) else None
        except Exception as e:  # noqa: BLE001
            ans, pred = f"ERROR: {e}", None
        correct = pred is not None and pred == gold
        by_cat[item["category"]].append((correct, pred is not None))
        live["n"] += 1
        live["c"] += int(correct)
        records.append({"id": item["id"], "category": item["category"], "question": item["question"],
                        "answer": item["answer"], "pred": pred, "correct": correct, "raw": ans})
        bar.set_postfix(acc=f"{100 * live['c'] / live['n']:.1f}" if live["n"] else "—", refresh=False)
        if args.save_every and (i + 1) % args.save_every == 0:
            _write()

    results = _write()
    print("\n=== MMAU ===")
    for c in CATEGORIES:
        if c in results:
            print(f"  {c.capitalize():8s} acc={results[c]['acc']:.4f}  (n={results[c]['n']})")
    print(f"  {'Overall':8s} acc={results['overall']['acc']:.4f}  (n={results['overall']['n']})")
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
