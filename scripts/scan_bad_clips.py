#!/usr/bin/env python
"""Find CMM clips that HANG the Qwen media pipeline so they can be skipped wholesale instead
of one stall at a time. Each clip is run through the EXACT call the eval uses --
qwen_omni_utils.process_mm_info (video decode + audio load + the use_audio_in_video combine) --
in a CHILD process with a hard timeout. Anything that times out (a true hang, which SIGALRM
can't catch in-process) is written to a skip-file that run_cmm reads automatically
(default runs/cmm_skip_clips.txt). Run once, then relaunch CMM.

  # scan the unfinished tail (items already done never hang):
  PYTHONPATH=src python scripts/scan_bad_clips.py --start 925
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os


def _probe(v, a, uaiv, q: mp.Queue) -> None:
    os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "decord")   # match the model wrapper
    try:
        from qwen_omni_utils import process_mm_info
        content = []
        if v:
            content.append({"type": "video", "video": v})
        if a:
            content.append({"type": "audio", "audio": a})
        content.append({"type": "text", "text": "x"})
        process_mm_info([{"role": "user", "content": content}], use_audio_in_video=uaiv)
        q.put("ok")
    except Exception as e:              # noqa: BLE001 -- any failure is informative
        q.put(f"err:{type(e).__name__}")


def check(v, a, uaiv, timeout: int) -> str:
    q: mp.Queue = mp.Queue()
    p = mp.Process(target=_probe, args=(v, a, uaiv, q), daemon=True)
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
    ap.add_argument("--timeout", type=int, default=30, help="seconds before a clip is called a hang")
    ap.add_argument("--errors-too", action="store_true",
                    help="also skip clips that ERROR (default: only true hangs; errors recover)")
    ap.add_argument("--out", default="runs/cmm_skip_clips.txt")
    a = ap.parse_args()

    from rlvib.data.cmm import CMMDataset
    ds = CMMDataset(a.json, a.root)
    bad, seen = [], set()
    for i in range(a.start, len(ds)):
        it = ds[i]
        v, aud = it.get("video_path"), it.get("audio_path")
        key = (v, aud)
        if key in seen:
            continue
        seen.add(key)
        uaiv = bool(v) and not aud and it.get("modality") == "audio"   # same rule as run_cmm
        r = check(v, aud, uaiv, a.timeout)
        if r == "HANG" or (a.errors_too and r.startswith("err")):
            name = os.path.splitext(os.path.basename(v or aud))[0]      # basename -> matches .mp4 & .wav
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
