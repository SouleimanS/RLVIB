"""Temporal-alignment (sync) eval: synced-vs-shifted detection + lags-vs-leads direction.

  # AVE-Shift (ours), base:
  python -m rlvib.eval.run_sync --bench avshift --model qwen3-omni --limit 200
  # + shift-DPO / clock checkpoint:
  python -m rlvib.eval.run_sync --bench avshift --model qwen3-omni \
      --bottleneck runs/anchored_qwen3-omni_shift/bottleneck_step60.pt
  # VGGSound-Sync (out-of-distribution):
  python -m rlvib.eval.run_sync --bench vggsync --model qwen3-omni

Balanced 2-way tasks => chance 0.50. Reports detection accuracy and, over the shifted items,
direction accuracy. Same harness discipline as run_mmau (letter answers, parse_choice, resumable
JSON, FiLM/temporal condition auto-set from the question). -> runs/sync_<bench>_<model>.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import warnings

from tqdm.auto import tqdm

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")
for _n in ("transformers", "qwen_vl_utils", "qwen_omni_utils"):
    logging.getLogger(_n).setLevel(logging.ERROR)

from rlvib.data.avshift import build_avshift, format_sync, load_vggsound_sync  # noqa: E402
from rlvib.eval.metrics import parse_choice  # noqa: E402
from rlvib.eval.timeout import time_limit  # noqa: E402
from rlvib.models import get_model  # noqa: E402


def _acc(pairs):
    n = len(pairs)
    return {"acc": sum(c for c, _ in pairs) / n if n else 0.0, "n": n}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", choices=("avshift", "vggsync"), default="avshift")
    ap.add_argument("--model", default="qwen3-omni")
    ap.add_argument("--bottleneck", default=None)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--offsets", default="0.5,1,2",
                    help="AVE-Shift displacement magnitudes in seconds. Use e.g. '2' to test\n                         only clearly-resolvable shifts (at 2 fps, 0.5 s is a single frame).")
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=8)
    ap.add_argument("--gen-timeout", type=int, default=180)
    ap.add_argument("--out", default=None)
    ap.add_argument("--save-every", type=int, default=20)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()
    args.out = args.out or f"runs/sync_{args.bench}_{args.model}.json"

    model = get_model(args.model)
    cond = False
    if args.bottleneck:
        from rlvib.models.bottleneck import load_attached
        bns, _h = load_attached(model, args.bottleneck)
        cond = "q_proj" in bns
        print(f"attached bottleneck <- {args.bottleneck}" + ("  (prompt-aware)" if cond else ""),
              flush=True)

    offs = tuple(float(x) for x in args.offsets.split(",") if x.strip())
    ds = (build_avshift(n=args.limit or 200, offsets=offs) if args.bench == "avshift"
          else load_vggsound_sync())
    n = len(ds) if args.limit in (0, None) else min(args.limit, len(ds))
    print(f"sync[{args.bench}]: {n}/{len(ds)} items", flush=True)

    records = []
    if not args.no_resume and os.path.exists(args.out):
        records = json.load(open(args.out)).get("records", [])[:n]
        if records:
            print(f"resuming from {len(records)} saved records", flush=True)

    def _ask(item, question, options):
        if cond:
            from rlvib.models.bottleneck import question_embedding, set_condition
            set_condition(bns, question_embedding(model, question))
        with time_limit(args.gen_timeout):
            msg = model.message(video=item["video_path"], prompt=format_sync(question, options),
                                fps=args.fps)
            ans = model.generate(msg, use_audio_in_video=True, max_new_tokens=args.max_new_tokens)
        return parse_choice(ans), ans

    def _write():
        det = [(r["det_correct"], True) for r in records]
        dirp = [(r["dir_correct"], True) for r in records if r.get("dir_correct") is not None]
        res = {"detection": _acc(det), "direction": _acc(dirp)}
        # per-|offset| detection: does accuracy scale with displacement? (chance if no signal)
        by_off = {}
        for r in records:
            d = r.get("delta")
            if d:
                by_off.setdefault(f"{abs(float(d)):.1f}s", []).append((r["det_correct"], True))
        res["detection_by_offset"] = {k: _acc(v) for k, v in sorted(by_off.items())}
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump({"results": res, "records": records}, open(args.out, "w"), indent=2)
        return res

    bar = tqdm(range(len(records), n), total=n, initial=len(records), desc=f"sync:{args.bench}")
    for i in bar:
        item = ds[i]
        try:
            det_pred, det_raw = _ask(item, item["question"], item["options"])
            det_correct = det_pred == item["sync_letter"]
            dir_correct = None
            if item.get("shifted") and item.get("dir_options"):
                dir_pred, _ = _ask(item, "Does the audio lag or lead the video?",
                                   item["dir_options"])
                dir_correct = dir_pred == item["dir_letter"]
        except Exception as e:  # noqa: BLE001
            det_correct, dir_correct, det_raw = False, None, f"ERROR: {e}"
        records.append({"video_path": item["video_path"], "delta": item.get("delta"),
                        "det_correct": bool(det_correct), "dir_correct": dir_correct,
                        "raw": det_raw})
        if args.save_every and (i + 1) % args.save_every == 0:
            _write()

    res = _write()
    print(f"\n=== sync[{args.bench}] ===")
    print(f"  detection acc={res['detection']['acc']:.4f} (n={res['detection']['n']})")
    print(f"  direction acc={res['direction']['acc']:.4f} (n={res['direction']['n']})")
    for k, v in res.get("detection_by_offset", {}).items():
        print(f"    |offset|={k}: detection={v['acc']:.4f} (n={v['n']})")
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
