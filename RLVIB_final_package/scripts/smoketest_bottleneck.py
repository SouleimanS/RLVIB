#!/usr/bin/env python
"""Step-5 smoke test: frozen model + trainable VariationalBottleneck (VIB).

Validates the v1 IB bottleneck before training:
  1. zero-init output => identity at init (answer unchanged after attaching)
  2. only the bottleneck params train (LLM/encoders frozen)
  3. gradients flow from (LM loss + beta*KL) into the bottleneck; the KL rate is finite

  python scripts/smoketest_bottleneck.py --model qwen3-omni --video clip.mp4
"""
from __future__ import annotations

import argparse
import traceback

from rlvib.models import get_model
from rlvib.models.bottleneck import VariationalBottleneck, attach_bottlenecks, total_kl


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-omni")
    ap.add_argument("--video", required=True)
    ap.add_argument("--beta", type=float, default=0.01)
    args = ap.parse_args()

    m = get_model(args.model)
    msg = m.message(video=args.video, prompt="What is happening in this clip?")

    print("=== baseline (no bottleneck) ===", flush=True)
    a0 = m.generate(msg, max_new_tokens=32)
    print(a0)

    bottlenecks, handles = attach_bottlenecks(m, cls=VariationalBottleneck)

    print("\n=== identity check (zero-init VIB, eval) ===", flush=True)
    bottlenecks.eval()
    a1 = m.generate(msg, max_new_tokens=32)
    print(a1)
    print("identical to baseline:", a0 == a1)

    n_bn = sum(p.numel() for p in bottlenecks.parameters() if p.requires_grad)
    n_model = sum(p.numel() for p in m.model.parameters() if p.requires_grad)
    print(f"\ntrainable params -> bottleneck: {n_bn:,} | rest of model: {n_model:,}")

    print("\n=== gradient flow + KL (LM loss + beta*KL) ===", flush=True)
    bottlenecks.train()
    try:
        inputs = m.build_inputs(msg)
        lm = getattr(m.model, "thinker", m.model)  # Qwen3 full Omni -> .thinker; Qwen2.5 -> itself
        out = lm(**inputs, labels=inputs["input_ids"])
        lm_loss = out.loss if hasattr(out, "loss") else out["loss"]
        kl = total_kl(bottlenecks)
        (lm_loss + args.beta * kl).backward()
        for name, bn in bottlenecks.items():
            g = bn.out.weight.grad
            gn = "None" if g is None else f"{float(g.norm()):.4e}"
            print(f"  {name}: out.grad_norm = {gn} | KL = {float(bn.last_kl.detach()):.4f}")
        kl_v = float(kl.detach()) if hasattr(kl, "detach") else float(kl)
        print(f"  LM loss = {float(lm_loss.detach()):.4f} | total KL = {kl_v:.4f}")
    except Exception:  # noqa: BLE001 — surface the real forward signature
        print("forward/backward failed -- traceback:")
        traceback.print_exc()

    for h in handles:
        h.remove()
    print("\n=== VIB bottleneck smoke test done ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
