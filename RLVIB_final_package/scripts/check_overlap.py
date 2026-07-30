#!/usr/bin/env python
"""Clip-overlap audit (data-leakage check): does any AVHBench/CMM/DAVE clip share a video id
with the AVE training pool? Lists video-file stems under each benchmark's media dir and
compares to AVE video_ids + AVE path stems. Run where data/ lives (cluster login node, no GPU).

  python scripts/check_overlap.py
  AVHBENCH_VIDEOS=... CMM_ROOT=... DAVE_ROOT=... python scripts/check_overlap.py
"""
from __future__ import annotations

import glob
import os

from rlvib.data import ave

VID_EXT = (".mp4", ".mkv", ".avi", ".webm", ".flv", ".mov")


def stems(root):
    s = set()
    if root and os.path.isdir(root):
        for e in VID_EXT:
            for p in glob.glob(os.path.join(root, "**", "*" + e), recursive=True):
                s.add(os.path.splitext(os.path.basename(p))[0])
    return s


def main() -> int:
    ave_ids, ave_stems = set(), set()
    for split in ("train", "val", "test"):
        try:
            for it in ave.load_ave(split):
                ave_ids.add(str(it.get("video_id")))
                ave_stems.add(os.path.splitext(os.path.basename(it["video_path"]))[0])
        except Exception as e:  # noqa: BLE001
            print(f"(ave {split}: {e})")
    print(f"AVE pool: {len(ave_ids)} video_ids, {len(ave_stems)} path stems\n")

    roots = {
        "AVHBench": os.environ.get("AVHBENCH_VIDEOS", "data/AVHBench/videos"),
        "CMM": os.environ.get("CMM_ROOT", "data/CMM"),
        "DAVE": os.environ.get("DAVE_ROOT", "data/DAVE"),
    }
    leak = False
    for name, root in roots.items():
        bs = stems(root)
        ov = (ave_ids | ave_stems) & bs
        flag = "  <-- OVERLAP (potential leakage)" if ov else ""
        print(f"{name:10s} root={root}  {len(bs)} clips | shared with AVE: {len(ov)}{flag}")
        for x in list(ov)[:8]:
            print(f"    {x}")
        leak = leak or bool(ov)
    print("\n" + ("!! overlap found -- exclude shared clips from eval or training."
                  if leak else "clean: no AVE clip ids appear in the benchmark media dirs."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
