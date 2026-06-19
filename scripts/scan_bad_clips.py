#!/usr/bin/env python
"""Find CMM video clips that HANG decord (the reader the Qwen wrappers force) so they can be
skipped wholesale instead of one clip at a time. Each clip's video is opened + decoded in a
CHILD process with a hard timeout; any that time out (a true hang, which SIGALRM can't catch
in-process) -- or error -- are written to a skip-file that run_cmm reads automatically
(default runs/cmm_skip_clips.txt). Run once, then relaunch CMM.

  # scan only the unfinished part (items already done never hang), ~fast:
  PYTHONPATH=src python scripts/scan_bad_clips.py --start 925
  # or the whole set:
  PYTHONPATH=src python scripts/scan_bad_clips.py
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os


def _probe(path: str, q: mp.Queue) -> None:
    try:
        import decord
        vr = decord.VideoReader(path)
        _ = vr[0]                       # force an actual frame decode (where it wedges)
        q.put("ok")
    except Exception as e:              # noqa: BLE001 -- any failure is informative
        q.put(f"err:{type(e).__name__}")


def check(path: str, timeout: int) -> str:
    q: mp.Queue = mp.Queue()
    p = mp.Process(target=_probe, args=(path, q), daemon=True)
    p.start()
    p.join(timeout)
    if p.is_alive():                    # never returned -> the hang we care about
        p.terminate()
        p.join()
        return "HANG"
    return q.get() if not q.empty() else "err:noresult"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="data/CMM/all_data_final_reorg.json")
    ap.add_argument("--root", default="data/CMM")
    ap.add_argument("--start", type=int, default=0, help="first dataset index to scan")
    ap.add_argument("--timeout", type=int, default=20, help="seconds before a clip is called a hang")
    ap.add_argument("--errors-too", action="store_true",
                    help="also skip clips that ERROR (default: skip only true hangs; errors recover "
                         "via the torchvision fallback)")
    ap.add_argument("--out", default="runs/cmm_skip_clips.txt")
    a = ap.parse_args()

    from rlvib.data.cmm import CMMDataset
    ds = CMMDataset(a.json, a.root)
    bad, seen = [], set()
    for i in range(a.start, len(ds)):
        v = ds[i].get("video_path")
        if not v or v in seen:
            continue
        seen.add(v)
        r = check(v, a.timeout)
        if r == "HANG" or (a.errors_too and r.startswith("err")):
            name = os.path.splitext(os.path.basename(v))[0]
            bad.append(name)
            print(f"  BAD [{i}] {r:12s} {name}", flush=True)
        elif i % 100 == 0:
            print(f"  .. scanned to {i}/{len(ds)} ({len(bad)} bad so far)", flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        f.write(",".join(sorted(set(bad))))
    print(f"\n{len(bad)} hanging clip(s) -> {a.out}")
    print("SKIP_LIST:", ",".join(sorted(set(bad))) or "(none)")
    print("run_cmm reads this file automatically; just relaunch CMM.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
