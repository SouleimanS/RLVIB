#!/usr/bin/env python
"""Step-4 smoke test: frozen Qwen3-Omni + trainable ResidualBottlenecks.

Validates the architecture before any training:
  1. zero-init bottleneck is identity (answer unchanged after attaching)
  2. only the bottleneck params are trainable (LLM/encoders frozen)
  3. gradients flow into the bottleneck from a dummy LM loss (through the frozen model)

  python scripts/smoketest_bottleneck.py --video ~/sample_av.mp4
"""
from __future__ import annotations

import argparse
import traceback

from rlvib.models import QwenOmni
from rlvib.models.bottleneck import attach_bottlenecks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    args = ap.parse_args()

    m = QwenOmni()
    msg = m.message(video=args.video, prompt="What is happening in this clip?")

    print("=== baseline (no bottleneck) ===", flush=True)
    a0 = m.generate(msg, max_new_tokens=32)
    print(a0)

    bottlenecks, handles = attach_bottlenecks(m)

    print("\n=== identity check (zero-init bottleneck) ===", flush=True)
    a1 = m.generate(msg, max_new_tokens=32)
    print(a1)
    print("identical to baseline:", a0 == a1)

    n_bn = sum(p.numel() for p in bottlenecks.parameters() if p.requires_grad)
    n_model = sum(p.numel() for p in m.model.parameters() if p.requires_grad)
    print(f"\ntrainable params -> bottleneck: {n_bn:,} | rest of model: {n_model:,}")

    print("\n=== gradient flow through the frozen model (dummy LM loss) ===", flush=True)
    try:
        inputs = m.build_inputs(msg)
        out = m.model(**inputs, labels=inputs["input_ids"])
        loss = out.loss if hasattr(out, "loss") else out["loss"]
        loss.backward()
        for name, bn in bottlenecks.items():
            g = bn.fc2.weight.grad  # fc2 is zero-init; it receives gradient first
            print(f"  {name}: fc2.grad_norm = {None if g is None else float(g.norm()):.4e}")
        print(f"  LM loss = {float(loss):.4f}")
    except Exception:  # noqa: BLE001 — surface the real forward signature
        print("forward/backward failed -- traceback:")
        traceback.print_exc()

    for h in handles:
        h.remove()
    print("\n=== bottleneck smoke test done ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
