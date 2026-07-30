#!/usr/bin/env python
"""Audit CMM HR/PA parsing: is a model's near-floor HR real, or a harness/parse artifact?

Our VideoLLaMA2 base HR (~0.01-0.09) sits 5-30x below CMM's published open-model HR
(~34-59%) and is inconsistent with VL2's 62.4% AVHBench yes-rate -- a red flag that the
HR harness manufactures extra "yes". This reads the eval JSONs' `records` (pred, answer,
raw) -- NO GPU, NO model -- and for the base + each per-step bottleneck checkpoint reports,
on the HR ("no") and PA ("yes") subsets:

  * HR_strict / PA_strict : the score eval ACTUALLY recorded (stored `pred`).
  * none% (parse-fail rate): unparsed answers are counted wrong -> floor HR if high.
  * yes%  (overall yes-rate): compare vs the model's AVHBench yes-rate (VL2 = 62.4%).
  * HR_lenient            : a negation-aware re-parse that recovers absence answers
                            phrased WITHOUT a literal "no" (e.g. "does not contain"),
                            which the eval parser (matches \\bno\\b, not "not"/"cannot")
                            drops. A large strict->lenient HR jump == parse artifact.
  * Wilson 95% CI on HR_strict (so a near-floor point estimate is reported as an interval).

It then DUMPS the raw outputs of "no"-gold records the strict parser missed, so the model's
actual words are visible. Self-contained (no PYTHONPATH needed).

  python scripts/audit_cmm_hr.py --model videollama2 --exp broad --dump 15
  python scripts/audit_cmm_hr.py --model qwen3-omni              # sanity: a clean backbone
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

# --- strict parser: mirrors rlvib.eval.metrics.parse_yes_no EXACTLY (the parser eval used) ---
_YES = {"yes", "true", "correct", "yeah", "yep"}
_NO = {"no", "false", "incorrect", "nope"}


def parse_yes_no(text):
    if not text:
        return None
    t = text.strip().lower()
    first = re.split(r"[\s,.!?:;]+", t)[0] if t else ""
    if first in _YES:
        return "yes"
    if first in _NO:
        return "no"
    m = re.search(r"\b(yes|no)\b", t)
    return m.group(1) if m else None


# --- negation/affirmation-aware fallback for answers the strict parser can't read ---
_NEG = re.compile(r"(\bnot\b|n't\b|\bcannot\b|\bno\b|\bnone\b|\bnothing\b|\bnever\b|"
                  r"\bwithout\b|\bneither\b|\babsent\b|\bsilent\b|\bunable\b|\bunheard\b)")
_AFF = re.compile(r"(\byes\b|\byeah\b|\byep\b|\bcorrect\b|\btrue\b|\bpresent\b|\baudible\b|"
                  r"\bvisible\b|\bthere is\b|\bthere's\b|\bi can (?:hear|see)\b)")


def lenient(raw):
    """Strict first; if unparsed, decide by negation/affirmation cues (negation wins ties)."""
    s = parse_yes_no(raw)
    if s is not None:
        return s
    t = (raw or "").lower()
    neg, aff = bool(_NEG.search(t)), bool(_AFF.search(t))
    if neg and not aff:
        return "no"
    if aff and not neg:
        return "yes"
    if neg and aff:
        return "no"           # "there is no X" -> absence dominates
    return None


def wilson(k, n, z=1.96):
    if not n:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5
    return ((c - h) / d, (c + h) / d)


def _records(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f).get("records", [])


def audit(records):
    yes = [r for r in records if r.get("answer") == "yes"]
    no = [r for r in records if r.get("answer") == "no"]
    n = len(records)

    def acc(rs, fn):
        return (sum(1 for r in rs if fn(r) == r["answer"]) / len(rs)) if rs else float("nan")

    strict = lambda r: r.get("pred")                       # noqa: E731  what eval stored
    lenf = lambda r: lenient(r.get("raw"))                 # noqa: E731
    preds = [strict(r) for r in records]
    k_no = sum(1 for r in no if strict(r) == "no")         # HR_strict numerator
    return {
        "n": n, "n_yes": len(yes), "n_no": len(no),
        "PA": acc(yes, strict), "HR": acc(no, strict),
        "PA_len": acc(yes, lenf), "HR_len": acc(no, lenf),
        "yes_rate": (sum(1 for p in preds if p == "yes") / n) if n else float("nan"),
        "none_rate": (sum(1 for p in preds if p is None) / n) if n else float("nan"),
        "hr_ci": wilson(k_no, len(no)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="videollama2")
    ap.add_argument("--exp", default="")
    ap.add_argument("--dump", type=int, default=12, help="raw 'no'-miss outputs to print (base)")
    args = ap.parse_args()
    xt = f"_{args.exp}" if args.exp else ""

    paths = [("base", f"runs/cmm_{args.model}.json")]
    steps = sorted({int(m.group(1))
                    for p in glob.glob(f"runs/cmm_{args.model}{xt}_step*.json")
                    for m in [re.search(r"_step(\d+)\.json$", p)] if m})
    paths += [(f"step{s}", f"runs/cmm_{args.model}{xt}_step{s}.json") for s in steps]

    print(f"=== CMM HR/PA parse audit: model={args.model} exp={args.exp or '-'} ===")
    print(f"{'ckpt':>8}{'n_no':>6}{'PA':>7}{'HR':>7}{'HR_len':>8}{'yes%':>7}{'none%':>7}"
          f"   HR_strict 95% CI")
    base_recs = None
    for tag, path in paths:
        recs = _records(path)
        if recs is None:
            print(f"{tag:>8}   (missing {path})")
            continue
        if tag == "base":
            base_recs = recs
        a = audit(recs)
        lo, hi = a["hr_ci"]
        flag = ""
        if a["HR_len"] - a["HR"] >= 0.05:
            flag += "  <- parse-recovered HR"
        if a["none_rate"] >= 0.10:
            flag += f"  <- {a['none_rate']:.0%} UNPARSED"
        print(f"{tag:>8}{a['n_no']:>6}{a['PA']:>7.3f}{a['HR']:>7.3f}{a['HR_len']:>8.3f}"
              f"{a['yes_rate']:>7.2f}{a['none_rate']:>7.2f}   [{lo:.3f},{hi:.3f}]{flag}")

    # Dump the base's "no"-gold misses so the model's actual words are visible.
    if base_recs and args.dump:
        miss = [r for r in base_recs if r.get("answer") == "no" and r.get("pred") != "no"]
        print(f"\n--- base: {len(miss)} of {sum(1 for r in base_recs if r.get('answer') == 'no')}"
              f" 'no'-gold answers scored NOT-no (the HR misses); showing {min(args.dump, len(miss))} ---")
        for r in miss[: args.dump]:
            print(f"  stored={str(r.get('pred')):>4} lenient={str(lenient(r.get('raw'))):>4}"
                  f" | {repr((r.get('raw') or '')[:110])}")
        print("\nRead: if these raw outputs clearly say 'no'/negate, the eval parser floored HR "
              "(harness bug) -> fix the parser and re-derive HR. If they truly say 'yes', the "
              "yes-bias is real.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
