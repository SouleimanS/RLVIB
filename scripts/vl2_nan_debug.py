#!/usr/bin/env python
"""Pinpoint the VideoLLaMA2 NaN: is it the VIB value, the backbone feats, or the GRAD path?

  [1] bypass, no_grad   -- the working base path (sanity)
  [2] active, no_grad   -- isolates the VIB's *value* effect (delta ~ 0 at init)
  [3] active, with grad -- isolates the grad-tracked forward (fp16 attention instability)

If [2] is NaN  -> the VIB/backbone feats are the problem (look at *_x_finite below).
If [2] is finite but [3] is NaN -> it's the grad-enabled fp16 forward, not the VIB.

  CONDA_ENV=rlvib_vl2 ... :  PYTHONPATH=src python scripts/vl2_nan_debug.py
"""
import torch

from rlvib.data import ave
from rlvib.models import get_model
from rlvib.models.bottleneck import VariationalBottleneck, attach_bottlenecks, set_bypass
from rlvib.train.dpo import answer_logp_vec


def fin(t):
    return bool(torch.isfinite(t).all()) if t is not None else None


def main() -> int:
    m = get_model("videollama2")
    print("model dtype:", m.dtype, flush=True)
    bns, handles = attach_bottlenecks(m, cls=VariationalBottleneck)
    print("vib param dtype:", next(bns["vision"].parameters()).dtype, flush=True)

    it = ave.load_ave("train")[0]
    msg = m.message(video=it["video_path"], prompt="What do you see and hear in this clip?")

    set_bypass(bns, True)
    bns.eval()
    with torch.no_grad():
        lp = answer_logp_vec(m, msg)
    print(f"[1] bypass no_grad : logits_finite={fin(lp)}", flush=True)

    set_bypass(bns, False)
    with torch.no_grad():
        lp = answer_logp_vec(m, msg)
    print(f"[2] active no_grad : logits_finite={fin(lp)}  "
          f"vis_x_finite={fin(bns['vision'].last_input_norm_per_token)}  "
          f"vis_resid_finite={fin(bns['vision'].last_residual_per_token)}  "
          f"aud_x_finite={fin(bns['audio'].last_input_norm_per_token)}  "
          f"aud_resid_finite={fin(bns['audio'].last_residual_per_token)}", flush=True)

    bns.train()
    lp = answer_logp_vec(m, msg)
    print(f"[3] active GRAD    : logits_finite={fin(lp)}", flush=True)

    for h in handles:
        h.remove()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
