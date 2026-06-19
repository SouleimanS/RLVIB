#!/usr/bin/env python
"""Run CMM to completion, auto-skipping ANY clip that stalls the eval -- wherever the stall
is (decode, processor, or the GPU forward), which is why a CPU pre-scan can't find them.

Runs `python -m rlvib.eval.run_cmm` as a child with --save-every 1 (so the record count is
exact), watches the --out JSON's count, and if it doesn't grow for --stall seconds, kills the
child, appends the stuck clip to the skip-file (which run_cmm re-reads on resume), and
restarts from where it left off. Repeats until the count reaches the dataset size.

  PYTHONPATH=src python scripts/run_cmm_autoskip.py \
      --out runs/cmm_qwen3-omni_full.json --json data/CMM/all_data_final_reorg.json \
      --root data/CMM -- --model qwen3-omni --limit 0
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time


def _count(path: str) -> int:
    try:
        with open(path) as f:
            return len(json.load(f).get("records", []))
    except (OSError, ValueError):
        return 0


def _append_skip(skip_file: str, clip: str) -> bool:
    """Add clip to the skip-file; return False if it was already there (=> not a clip hang)."""
    cur = open(skip_file).read().strip() if os.path.exists(skip_file) else ""
    have = {s.strip() for s in cur.replace("\n", ",").split(",") if s.strip()}
    if clip in have:
        return False
    os.makedirs(os.path.dirname(skip_file) or ".", exist_ok=True)
    with open(skip_file, "w") as f:
        f.write(",".join(sorted(have | {clip})))
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--skip-file", default="runs/cmm_skip_clips.txt")
    ap.add_argument("--stall", type=int, default=180, help="seconds of no progress => hung")
    ap.add_argument("--poll", type=int, default=20)
    ap.add_argument("--max-skips", type=int, default=60, help="give up after this many skips")
    ap.add_argument("rest", nargs=argparse.REMAINDER, help="-- then extra run_cmm args")
    a = ap.parse_args()

    from rlvib.data.cmm import CMMDataset
    ds = CMMDataset(a.json, a.root)
    n = len(ds)
    extra = a.rest[1:] if a.rest and a.rest[0] == "--" else a.rest
    skips = 0

    while _count(a.out) < n and skips < a.max_skips:
        cmd = [sys.executable, "-u", "-m", "rlvib.eval.run_cmm",
               "--json-path", a.json, "--data-root", a.root, "--out", a.out,
               "--skip-file", a.skip_file, "--save-every", "1", *extra]
        print(f"[autoskip] launch: {' '.join(cmd)}", flush=True)
        p = subprocess.Popen(cmd)
        last, stamp = _count(a.out), time.time()
        while p.poll() is None:
            time.sleep(a.poll)
            c = _count(a.out)
            if c > last:
                last, stamp = c, time.time()
            elif time.time() - stamp > a.stall:
                idx = _count(a.out)                       # next item to process == stuck one
                it = ds[idx] if idx < n else {}
                media = it.get("video_path") or it.get("audio_path") or f"idx{idx}"
                clip = os.path.splitext(os.path.basename(media))[0]
                print(f"[autoskip] STALL at item {idx} -> skip '{clip}'", flush=True)
                p.kill()
                p.wait()
                if not _append_skip(a.skip_file, clip):
                    print(f"[autoskip] '{clip}' already skipped yet still stalls -- aborting "
                          f"(not a per-clip issue). At {idx}/{n}.", flush=True)
                    return 2
                skips += 1
                break
        else:
            p.wait()

    done = _count(a.out)
    print(f"[autoskip] finished: {done}/{n} ({skips} clip(s) skipped)", flush=True)
    return 0 if done >= n else 1


if __name__ == "__main__":
    raise SystemExit(main())
