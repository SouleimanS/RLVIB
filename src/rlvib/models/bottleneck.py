"""v0 trainable bottleneck on Qwen3-Omni's per-modality adapter outputs.

Inserts a small *zero-initialized residual* transform after each adapter
  thinker.audio_tower.proj2 -> (T_a, 2048)
  thinker.visual.merger     -> (T_v, 2048)
via forward hooks, with the LLM + encoders frozen. Zero-init => identity at attach,
so it doesn't change the model until trained. This is the v0 placeholder for the
conditional / synergy VIB in the design doc (§2); it exists to validate the
"frozen LLM + trainable bottleneck" path end-to-end before we add the IB objective
and counterfactual-RL training.
"""
from __future__ import annotations

import torch.nn as nn


class ResidualBottleneck(nn.Module):
    """y = x + W2(GELU(W1 x)), with W2 zero-initialized -> identity at init."""

    def __init__(self, dim: int = 2048, hidden: int | None = None):
        super().__init__()
        hidden = hidden or dim
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        return x + self.fc2(self.act(self.fc1(x)))


def attach_bottlenecks(model, dim: int | None = None):
    """Freeze the whole model; attach trainable ResidualBottlenecks on the audio +
    vision adapters via forward hooks. `dim` defaults to the model's adapter output
    dim (`model.hidden_dim`: 2048 for Qwen3-Omni, 3584 for Qwen2.5-Omni / VideoLLaMA2).

    Returns (ModuleDict of bottlenecks, list of hook handles).
    Detach with:  for h in handles: h.remove()
    """
    dim = dim or getattr(model, "hidden_dim", 2048)
    for p in model.model.parameters():
        p.requires_grad_(False)

    bottlenecks = nn.ModuleDict({
        "audio": ResidualBottleneck(dim),
        "vision": ResidualBottleneck(dim),
    }).to(model.device, model.dtype)

    handles = []
    for name, adapter in model.adapter_modules().items():
        bn = bottlenecks[name]

        def hook(_module, _inputs, output, bn=bn):
            if isinstance(output, tuple):
                return (bn(output[0]),) + tuple(output[1:])
            return bn(output)

        handles.append(adapter.register_forward_hook(hook))
    return bottlenecks, handles
