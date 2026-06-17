"""Trainable bottlenecks on the per-modality adapter outputs of a frozen AV-LLM.

  ResidualBottleneck     (v0) — deterministic zero-init residual (identity at init).
  VariationalBottleneck  (v1) — per-modality VIB: z = mu + sigma*eps, zero-init output
                                (identity at init), exposes a KL rate term for the IB loss.

Both attach via forward hooks on `model.adapter_modules()`, LLM + encoders frozen.
Zero-init output => attaching doesn't change the model until trained. A `bypass` flag
makes a bottleneck a pass-through (identity) — used to get the DPO reference logprobs
(the frozen base) without holding a second model. See ib-rl-method-and-framing.md (§2).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBottleneck(nn.Module):
    """y = x + W2(GELU(W1 x)), W2 zero-initialized -> identity at init."""

    def __init__(self, dim: int = 2048, hidden: int | None = None):
        super().__init__()
        hidden = hidden or dim
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        self.bypass = False
        self.last_kl = None

    def forward(self, x):
        if self.bypass:
            return x
        return x + self.fc2(self.act(self.fc1(x)))


class VariationalBottleneck(nn.Module):
    """Per-modality variational information bottleneck on an adapter output (T, dim).

    z = mu(x) + sigma(x)*eps ;  y = x + W_out(z)  with W_out zero-init (identity at init).
    `last_kl` = KL(N(mu, sigma^2) || N(0, I)) (mean over tokens+dims) for the IB loss term.
    """

    def __init__(self, dim: int = 2048, hidden: int | None = None):
        super().__init__()
        hidden = hidden or dim
        self.enc = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.to_mu = nn.Linear(hidden, hidden)
        self.to_logvar = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.bypass = False
        self.last_kl = None
        self.last_kl_per_token = None  # (..., T) bits the bottleneck allocates per token

    def forward(self, x):
        if self.bypass:
            return x
        h = self.act(self.enc(x))
        mu = self.to_mu(h)
        logvar = self.to_logvar(h).clamp(-8.0, 8.0)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if self.training else mu
        kl_elem = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
        self.last_kl = kl_elem.mean()                       # scalar rate for the IB loss
        self.last_kl_per_token = kl_elem.sum(dim=-1).detach()  # per-token rate -> saliency map
        return x + self.out(z)


def attach_bottlenecks(model, dim: int | None = None, cls=ResidualBottleneck):
    """Freeze the model; attach trainable bottlenecks (`cls`) on the audio + vision
    adapters via forward hooks. `dim` defaults to `model.hidden_dim`.

    Returns (ModuleDict, handles). Detach with: for h in handles: h.remove().
    """
    dim = dim or getattr(model, "hidden_dim", 2048)
    for p in model.model.parameters():
        p.requires_grad_(False)

    bottlenecks = nn.ModuleDict(
        {"audio": cls(dim), "vision": cls(dim)}
    ).to(model.device, model.dtype)

    handles = []
    for name, adapter in model.adapter_modules().items():
        bn = bottlenecks[name]

        def hook(_module, _inputs, output, bn=bn):
            if isinstance(output, tuple):
                return (bn(output[0]),) + tuple(output[1:])
            return bn(output)

        handles.append(adapter.register_forward_hook(hook))
    return bottlenecks, handles


def total_kl(bottlenecks):
    """Sum of the bottlenecks' last KL rate terms (0.0 if none recorded)."""
    kls = [b.last_kl for b in bottlenecks.values() if getattr(b, "last_kl", None) is not None]
    return sum(kls) if kls else 0.0


def set_bypass(bottlenecks, on: bool) -> None:
    """Toggle pass-through (identity) on all bottlenecks — used for the DPO reference."""
    for b in bottlenecks.values():
        b.bypass = on
