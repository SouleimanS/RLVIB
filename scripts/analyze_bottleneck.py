#!/usr/bin/env python
"""Explainability probe: what did the trained bottleneck LEARN? -- WEIGHTS ONLY (no eval, no GPU).

Reads a bottleneck checkpoint and reports, per modality, the structure of the learned output
re-projection  y = x + out(z)  (z = mu(x) at inference). No benchmark, no forward pass -- this
interrogates the mechanism directly from the parameters:

  * ||out.W||_F, spectral norm, singular-value spectrum, effective rank (energy) -> is the vision
    edit a big LOW-RANK re-projection (few directions) or full-rank?
  * ||out.bias|| -> the INPUT-INDEPENDENT (constant) part of the edit; if it dominates the map,
    the adapter is a near-fixed shift, not an input-driven computation.
  * vision/audio output-map ratio -> quantifies "the audio module is ~inert; the mechanism is in
    vision" (the central mechanistic claim), turning an observation into a number.
  * (FiLM) film / gate / q_proj norms -> how much the question actually modulates the bottleneck.

  python scripts/analyze_bottleneck.py runs/anchored_qwen3-omni_broad/bottleneck_step60.pt
  python scripts/analyze_bottleneck.py runs/anchored_qwen3-omni_film/bottleneck_step160.pt
Runs on CPU in seconds.
"""
from __future__ import annotations

import argparse

import torch


def _norm(t):
    return t.float().norm().item() if t is not None else 0.0


def _spectrum(w):
    """Singular values (desc), and the rank capturing 90% of the spectral energy."""
    s = torch.linalg.svdvals(w.float())
    energy = torch.cumsum(s ** 2, dim=0) / (s ** 2).sum().clamp_min(1e-12)
    rank90 = int((energy < 0.90).sum()) + 1
    return s, rank90


def _module(sd, name):
    ow, ob = sd.get(f"{name}.out.weight"), sd.get(f"{name}.out.bias")
    if ow is None:
        return None
    s, rank90 = _spectrum(ow)
    print(f"\n[{name}]  out.weight {tuple(ow.shape)}")
    print(f"  ||out.W||_F = {_norm(ow):8.3f}   spectral = {s[0].item():.3f}   "
          f"top-5 sv = " + " ".join(f"{v:.2f}" for v in s[:5].tolist()))
    print(f"  effective rank (90% energy) = {rank90} / {min(ow.shape)}"
          f"   ({100 * rank90 / min(ow.shape):.0f}% of dims)")
    print(f"  ||out.bias|| (constant edit) = {_norm(ob):.3f}")
    print(f"  ||enc.W||_F = {_norm(sd.get(f'{name}.enc.weight')):.3f}   "
          f"||to_mu.W||_F = {_norm(sd.get(f'{name}.to_mu.weight')):.3f}   "
          f"||to_logvar.W||_F = {_norm(sd.get(f'{name}.to_logvar.weight')):.3f}")
    fw = sd.get(f"{name}.film.weight")
    if fw is not None:                                   # FiLM: conditioning generators
        gb = sd.get(f"{name}.gate.bias")
        print(f"  [FiLM] ||film.W||_F = {_norm(fw):.3f}   ||gate.W||_F = "
              f"{_norm(sd.get(f'{name}.gate.weight')):.3f}   gate.bias(mean) = "
              f"{gb.float().mean().item():+.2f}  (>0 => gate open)")
    return _norm(ow)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck)
    print(f"checkpoint: {args.ckpt}")
    print(f"  cls={ck.get('cls')}  dim={ck.get('dim')}  cond_dim={ck.get('cond_dim')}  model={ck.get('model')}")

    norms = {m: _module(sd, m) for m in ("audio", "vision")}
    if "q_proj.weight" in sd:
        print(f"\n[q_proj]  {tuple(sd['q_proj.weight'].shape)}   ||W||_F = {_norm(sd['q_proj.weight']):.3f}")

    a, v = norms.get("audio") or 0.0, norms.get("vision") or 0.0
    print("\n=== verdict (mechanism localization) ===")
    if a > 0:
        print(f"  vision/audio output-map norm ratio = {v / max(a, 1e-9):5.2f}x")
        print("  >> 1  => the audio module is ~inert; the learned edit lives in the VISION stream.")
    print("  Low vision effective-rank => the vision re-projection is a few directions (simple,"
          " ~query-blind).  Large ||out.bias|| vs ||out.W|| => the edit is closer to a fixed shift.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
