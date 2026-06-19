#!/usr/bin/env python
"""Health-check the full-set eval JSONs (the '_full' tagged outputs).

Why: re-running scripts/launch_full_evals.sh while a batch is still running can put
two jobs on the SAME runs/*_full*.json. Each job's _write() does a full overwrite
(open(out,"w") + json.dump of its whole in-memory records), so the survivor converges
the file to a clean, self-consistent state -- the only way to be left bad is a torn
concurrent write (caught here as a JSON parse error) or duplicate/short records.

This checks every runs/*_full*.json for: (1) it parses; (2) len(records) == results
.overall.n; (3) no duplicate record keys (a concurrent-resume artifact); and prints
the headline score + parse_rate so you can eyeball progress. Exit code is nonzero if
ANY file is not OK, so it is safe to gate on.

  python scripts/check_full_jsons.py                 # all runs/*_full*.json
  python scripts/check_full_jsons.py runs/cmm_qwen3-omni_full.json   # specific files
"""
from __future__ import annotations

import glob
import json
import sys


def _key(rec: dict):
    """A stable identity for a record, schema-tolerant across CMM / AVHBench / DAVE."""
    for ks in (("video_path", "task", "text"),     # AVHBench
               ("sub_category", "question"),        # CMM
               ("question",), ("text",), ("video_path",)):
        if all(k in rec for k in ks):
            return tuple(str(rec.get(k)) for k in ks)
    return json.dumps({k: rec.get(k) for k in sorted(rec)}, sort_keys=True)


def check(path: str) -> bool:
    try:
        with open(path) as f:
            blob = json.load(f)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"  CORRUPT  {path}\n           !! does not parse: {e}")
        return False
    recs = blob.get("records", [])
    overall = (blob.get("results") or {}).get("overall", {})
    n = overall.get("n")
    score = overall.get("accuracy", overall.get("acc"))
    pr = overall.get("parse_rate")
    keys = [_key(r) for r in recs]
    dupes = len(keys) - len(set(keys))

    flags = []
    if n is not None and len(recs) != n:
        flags.append(f"records={len(recs)} != overall.n={n}")
    if dupes:
        flags.append(f"{dupes} DUPLICATE record(s)")
    ok = not flags
    status = "OK     " if ok else "CHECK  "
    score_s = f"{score:.3f}" if isinstance(score, (int, float)) else "  -  "
    pr_s = f"{pr:.2f}" if isinstance(pr, (int, float)) else " - "
    print(f"  {status}{path}\n           n={len(recs):<5} score={score_s} parse={pr_s}"
          + ("".join(f"\n           !! {m}" for m in flags) if flags else ""))
    return ok


def main() -> int:
    paths = sys.argv[1:] or sorted(glob.glob("runs/*_full*.json"))
    if not paths:
        print("no runs/*_full*.json found (cwd must be the repo root).")
        return 0
    print(f"checking {len(paths)} file(s):")
    allok = all([check(p) for p in paths])  # list (not generator) so every file is checked
    print("\nall clean." if allok else "\n^ files flagged CHECK/CORRUPT need a look "
          "(a still-running job will self-heal them on its next 25-item save).")
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())
