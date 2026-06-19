#!/usr/bin/env python
"""Health-check the full-set eval JSONs (the '_full' tagged outputs).

Why: re-running scripts/launch_full_evals.sh while a batch is still running can put
two jobs on the SAME runs/*_full*.json. The real question is only whether that left a
file damaged. It cannot leave EXTRA records: every resume path truncates to [:n]
(run_cmm.py:62, run_avhbench.py:51, run_dave.py:63), so a file can never grow past the
dataset size however many jobs wrote it. The only damage mode is a torn concurrent
write -> invalid JSON, which a still-running job overwrites on its next save.

So the two signals that actually matter are: (1) the file PARSES, and (2) len(records)
equals the reported n (internal consistency). This reports both, plus the headline
score + parse_rate, reading each benchmark's own schema:
  - AVHBench / CMM : metrics under results.overall  (accuracy|acc, n, parse_rate)
  - DAVE          : metrics at the TOP level         (accuracy, n, parse_rate)
It deliberately does NOT try to count "duplicate" records: CMM/DAVE records carry no
per-clip id and CMM reuses the same question across many clips, so any content key
collides on legitimately-distinct items (that was a false alarm in the first cut).

Exit code is nonzero if any file fails to parse or is count-inconsistent.

  python scripts/check_full_jsons.py                 # all runs/*_full*.json
  python scripts/check_full_jsons.py runs/cmm_qwen3-omni_full.json
"""
from __future__ import annotations

import collections
import glob
import json
import os
import sys


def _metrics(blob: dict):
    """(n, score, parse_rate) tolerant of AVHBench/CMM (results.overall) vs DAVE (top level)."""
    res = blob.get("results")
    if isinstance(res, dict) and isinstance(res.get("overall"), dict):
        o = res["overall"]
        return o.get("n"), o.get("accuracy", o.get("acc")), o.get("parse_rate")
    return blob.get("n"), blob.get("accuracy"), blob.get("parse_rate")  # DAVE


def _benchmark(path: str) -> str:
    b = os.path.basename(path)
    for p in ("avhbench", "cmm", "dave"):
        if b.startswith(p):
            return p
    return "other"


def check(path: str, counts: dict) -> bool:
    try:
        with open(path) as f:
            blob = json.load(f)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"  CORRUPT  {path}\n           !! does not parse: {e}")
        return False
    recs = blob.get("records", [])
    n, score, pr = _metrics(blob)
    counts[_benchmark(path)].add(len(recs))

    flags = []
    if n is not None and len(recs) != n:
        flags.append(f"records={len(recs)} != reported n={n} (partial/torn write -- re-check after the job's next save)")
    ok = not flags
    score_s = f"{score:.3f}" if isinstance(score, (int, float)) else "  -  "
    pr_s = f"{pr:.2f}" if isinstance(pr, (int, float)) else " - "
    print(f"  {'OK    ' if ok else 'CHECK '} {path}\n           n={len(recs):<5} score={score_s} parse={pr_s}"
          + "".join(f"\n           !! {m}" for m in flags))
    return ok


def main() -> int:
    paths = sys.argv[1:] or sorted(glob.glob("runs/*_full*.json"))
    if not paths:
        print("no runs/*_full*.json found (cwd must be the repo root).")
        return 0
    print(f"checking {len(paths)} file(s):")
    counts: dict = collections.defaultdict(set)
    allok = all([check(p, counts) for p in paths])  # list, not generator -> every file checked

    print("\nrecord counts per benchmark (all files of one benchmark should match WHEN FINISHED;")
    print("while jobs run, lower counts are just in-progress -- not an error):")
    for b in sorted(counts):
        cs = sorted(counts[b])
        note = "  <- converged" if len(cs) == 1 else "  <- still spread (jobs running / resuming)"
        print(f"  {b:9s} {cs}{note}")

    print("\nall files parse and are count-consistent." if allok else
          "\n^ a file failed to parse or is count-inconsistent -- a running job self-heals it on its next save.")
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())
