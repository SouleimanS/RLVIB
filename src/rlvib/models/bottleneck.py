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

import os

import torch
import torch.nn as nn
import torch.nn.functional as F


def _finmax(t):
    """'<finite?>|<max|abs|>' for the VIB debug trace."""
    ok = bool(torch.isfinite(t).all())
    return f"{ok}|{(float(t.abs().max()) if ok else float('nan')):.1e}"


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

    def __init__(self, dim: int = 2048, hidden: int | None = None, normalize_input: bool = False):
        super().__init__()
        hidden = hidden or dim
        self.enc = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.to_mu = nn.Linear(hidden, hidden)
        self.to_logvar = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        # VideoLLaMA2's disable_torch_init() no-ops nn.Linear.reset_parameters process-wide,
        # so enc/to_mu/to_logvar can come out UNINITIALIZED (garbage ~1e34 -> inf KL, nan z,
        # NaN logits). Re-init any Linear whose weight/bias is non-finite or absurdly large
        # (proper Kaiming is ~1/sqrt(fan) << 10); a backbone with a working default init
        # (Qwen-Omni) leaves these well below threshold, so it is untouched.
        for lin in (self.enc, self.to_mu, self.to_logvar):
            w, b = lin.weight, lin.bias
            if (not torch.isfinite(w).all() or float(w.abs().max()) > 10.0
                    or not torch.isfinite(b).all() or float(b.abs().max()) > 10.0):
                nn.init.kaiming_uniform_(lin.weight, a=5 ** 0.5)
                nn.init.zeros_(lin.bias)
        # Parameter-free LayerNorm on the ENCODER input only (the residual still carries the
        # raw x), so mu/logvar -> KL rate are scale-invariant. Needed for backbones with
        # "massive activations" (VideoLLaMA2, ~1e9 features); off by default so normal-scale
        # backbones (Qwen-Omni) are byte-for-byte unchanged. (See docs/research.)
        self.normalize_input = normalize_input
        self.bypass = False
        self.last_kl = None
        self.last_kl_per_token = None  # (..., T) bits the bottleneck allocates per token
        self.last_residual_per_token = None    # (..., T) ||out(z)|| -> actual edit magnitude
        self.last_input_norm_per_token = None  # (..., T) ||x||      -> for the relative edit

    def forward(self, x):
        if self.bypass:
            return x
        pd = self.enc.weight.dtype          # fp32 on fp16 backbones -> no overflow / 0*inf NaN
        # Force the VIB to compute in its param dtype even under an outer bf16 autocast
        # (fp16 backbones): otherwise enc()/mu overflow bf16 and the KL rate blows up to inf.
        with torch.autocast(x.device.type, enabled=False):
            xc = torch.nan_to_num(x.to(pd))     # guard inf/nan backbone feats (enc() spreads NaN)
            enc_in = F.layer_norm(xc, (xc.shape[-1],)) if self.normalize_input else xc
            h = self.act(self.enc(enc_in))
            mu = self.to_mu(h)
            logvar = self.to_logvar(h).clamp(-8.0, 8.0)
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if self.training else mu
            kl_elem = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
            self.last_kl = kl_elem.mean()                       # scalar rate for the IB loss
            if os.environ.get("RLVIB_VIB_DEBUG"):
                print(f"  [vibdbg] x_in={_finmax(x)} xc={_finmax(xc)} enc_in={_finmax(enc_in)} "
                      f"h={_finmax(h)} mu={_finmax(mu)} logvar={_finmax(logvar)} "
                      f"kl_elem={_finmax(kl_elem)} encW={_finmax(self.enc.weight)}", flush=True)
            self.last_kl_per_token = kl_elem.sum(dim=-1).detach()  # per-token rate -> saliency map
            delta = self.out(z)
            self.last_residual_per_token = delta.detach().norm(dim=-1)    # ||edit|| per token
            self.last_input_norm_per_token = xc.detach().norm(dim=-1)     # ||token|| (relative edit)
        return x + delta.to(x.dtype)


def attach_bottlenecks(model, dim: int | None = None, cls=ResidualBottleneck,
                       normalize_input: bool = False):
    """Freeze the model; attach trainable bottlenecks (`cls`) on the audio + vision
    adapters via forward hooks. `dim` defaults to `model.hidden_dim`. `normalize_input`
    LayerNorms the VIB encoder input (for massive-activation backbones, e.g. VideoLLaMA2).

    Returns (ModuleDict, handles). Detach with: for h in handles: h.remove().
    """
    dim = dim or getattr(model, "hidden_dim", 2048)
    for p in model.model.parameters():
        p.requires_grad_(False)

    # Compute the VIB in fp32 for fp16 backbones AND massive-/large-activation backbones
    # (normalize_input, e.g. VideoLLaMA2 run in bf16): bf16's 8-bit mantissa makes the KL
    # (mu^2 / exp(logvar)) and z blow up to inf even on normalized O(1) inputs -> 0*inf = NaN.
    # Qwen-Omni (bf16, normalize_input=False) stays bf16 and is byte-for-byte unchanged.
    vib_dtype = (torch.float32
                 if (getattr(model, "dtype", None) == torch.float16 or normalize_input)
                 else model.dtype)
    kw = {"normalize_input": normalize_input} if cls is VariationalBottleneck else {}
    bottlenecks = nn.ModuleDict(
        {"audio": cls(dim, **kw), "vision": cls(dim, **kw)}
    ).to(model.device, vib_dtype)

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


def load_attached(model, ckpt_path):
    """Attach a TRAINED bottleneck checkpoint to `model` for inference (eval mode).

    Reads the saved class + dim, attaches via forward hooks (freezing the model), loads
    the weights, switches to eval (deterministic z=mu). Returns (bottlenecks, handles) —
    keep the reference alive for the duration of inference.
    """
    import torch

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cls = {"VariationalBottleneck": VariationalBottleneck,
           "ResidualBottleneck": ResidualBottleneck}.get(ck.get("cls"), VariationalBottleneck)
    bottlenecks, handles = attach_bottlenecks(model, dim=ck.get("dim"), cls=cls,
                                              normalize_input=ck.get("normalize_input", False))
    bottlenecks.load_state_dict(ck["state_dict"])
    bottlenecks.eval()
    return bottlenecks, handles
