#!/usr/bin/env python
"""Diagnose why audio tokens aren't counted in the AV-attention probe (Qwen-Omni).

Loads the model, builds a real CMM input BOTH ways (use_audio_in_video off/on), and shows the
actual placeholder tokens so we can tell:
  (a) audio simply isn't in the input (no audio file + use_audio_in_video off), vs
  (b) audio IS in the input but attention_av's id detection misses it (different token name).
A high-count token flagged [?] under use_audio_in_video=True is the undetected audio token.

Run on a GPU node:  python scripts/diag_av_tokens.py --model qwen3-omni
"""
from __future__ import annotations

import argparse
import collections
import logging
import os
import warnings

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")
for _n in ("transformers", "qwen_vl_utils", "qwen_omni_utils"):
    logging.getLogger(_n).setLevel(logging.ERROR)

from rlvib.data.cmm import CMMDataset  # noqa: E402
from rlvib.eval.attention_av import audio_token_ids, visual_token_ids  # noqa: E402
from rlvib.models import get_model  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-omni")
    ap.add_argument("--json-path", default="data/CMM/all_data_final_reorg.json")
    ap.add_argument("--data-root", default="data/CMM")
    ap.add_argument("--fps", type=float, default=1.0)
    args = ap.parse_args()

    m = get_model(args.model)
    tok = getattr(m, "tokenizer", None) or m.processor.tokenizer
    aids, vids = audio_token_ids(m), visual_token_ids(m)
    dec = lambda S: [tok.convert_ids_to_tokens(i) for i in sorted(S)]  # noqa: E731
    print(f"detected audio ids:  {sorted(aids)}  {dec(aids)}")
    print(f"detected vision ids: {sorted(vids)}  {dec(vids)}")

    print("\n=== special/added tokens mentioning audio/vision/image/video ===")
    for t, i in sorted(((t, i) for t, i in tok.get_added_vocab().items()
                        if any(k in t.lower() for k in ("audio", "vision", "image", "video"))),
                       key=lambda x: x[1]):
        print(f"  {i:>8}  {t!r}")

    ds = CMMDataset(args.json_path, args.data_root)
    it = next((ds[i] for i in range(len(ds)) if ds[i].get("audio_path")), None) or ds[0]
    print(f"\nclip: audio={it.get('audio_path')}  video={it.get('video_path')}  "
          f"modality={it.get('modality')}")

    for uaiv in (False, True):
        try:
            msg = m.message(video=it["video_path"], audio=it["audio_path"], prompt=it["question"],
                            fps=args.fps)
            inp = m.build_inputs(msg, use_audio_in_video=uaiv)
        except Exception as e:  # noqa: BLE001
            print(f"\n--- use_audio_in_video={uaiv}: build_inputs failed: {type(e).__name__}: {e}")
            continue
        ids = inp["input_ids"][0].tolist()
        na = sum(i in aids for i in ids)
        nv = sum(i in vids for i in ids)
        print(f"\n--- use_audio_in_video={uaiv}: S={len(ids)}  audio-matched={na}  vision-matched={nv} ---")
        for i, c in collections.Counter(ids).most_common(8):
            flag = "AUDIO" if i in aids else ("VISION" if i in vids else "?")
            print(f"  id={i:>8}  count={c:>6}  tok={tok.convert_ids_to_tokens(i)!r}  [{flag}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
