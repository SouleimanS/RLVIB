#!/usr/bin/env python
"""Validate the closed-API baselines (keys + SDK + ffmpeg) with ONE cheap call each,
before running a full 300-example benchmark. Run on a node with internet (ABCI login
node), keys exported, in the rlvib env with `pip install google-genai openai`.

  export GEMINI_API_KEY=...  OPENAI_API_KEY=...
  PYTHONPATH=src python scripts/api_smoketest.py --video data/CMM/<some>.mp4
  PYTHONPATH=src python scripts/api_smoketest.py --providers gemini   # one provider
"""
from __future__ import annotations

import argparse
import sys

from rlvib.models import get_model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="path to a sample clip (mp4 with audio)")
    ap.add_argument("--providers", nargs="*", default=["gemini", "gpt4o"])
    ap.add_argument("--prompt", default="Briefly: what do you see and hear in this clip? "
                                         "Then answer yes or no: is a person speaking?")
    args = ap.parse_args()

    ok = True
    for name in args.providers:
        print(f"\n=== {name} ===", flush=True)
        try:
            model = get_model(name)
            msg = model.message(video=args.video, prompt=args.prompt)
            out = model.generate(msg, use_audio_in_video=True, max_new_tokens=64)
            print(f"  response: {out!r}")
            print(f"  OK ({getattr(model, 'model_id', '?')})" if out.strip()
                  else "  WARN: empty response")
            ok = ok and bool(out.strip())
        except Exception as e:  # noqa: BLE001 -- smoke test: report and continue
            print(f"  FAILED: {type(e).__name__}: {e}")
            ok = False
    print("\nall providers responded ✓" if ok else "\nsome providers failed -- see above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
