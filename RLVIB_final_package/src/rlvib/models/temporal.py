"""Temporal input features -- give the per-modality bottleneck a clock (the deck's tau_t slide).

The plain VIB sees each token without its time, so it cannot represent misalignment. Here we
concatenate a shared sinusoidal phase tau_t = [sin(2 pi t / T), cos(2 pi t / T)] to every token
before the encoder, so audio at t and a frame at t+Delta differ in their clocks.

`TemporalVariationalBottleneck` is a drop-in subclass of VariationalBottleneck: only the encoder
is widened to accept the clock channels; the residual and the zero-init output projection are
unchanged, so it is still exact identity at init. Attach it via
attach_bottlenecks(..., cls=TemporalVariationalBottleneck).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from rlvib.models.bottleneck import VariationalBottleneck


def clock_features(seq_len: int, period: float = 100.0, *, device=None, dtype=None) -> torch.Tensor:
    """(seq_len, 2) sinusoidal phase for token positions 0..seq_len-1.

    `period` T sets the wrap length (in tokens); one shared phase is used for both streams so a
    given absolute position maps to the same clock regardless of modality.
    """
    t = torch.arange(seq_len, device=device, dtype=dtype or torch.float32)
    ph = 2.0 * math.pi * t / period
    return torch.stack((torch.sin(ph), torch.cos(ph)), dim=-1)          # (S, 2)


class TemporalVariationalBottleneck(VariationalBottleneck):
    """VIB with a token-position clock concatenated at the encoder input (n_clock extra channels).

    Output width and residual stay at `dim`, so W_out is zero-init identity exactly as the parent.
    """

    n_clock = 2

    def __init__(self, dim: int = 2048, hidden: int | None = None,
                 normalize_input: bool = False, period: float = 100.0):
        super().__init__(dim, hidden, normalize_input=normalize_input)
        hidden = hidden or dim
        self.period = period
        # widen ONLY the encoder to accept the clock channels; to_mu/to_logvar/out unchanged.
        self.enc = nn.Linear(dim + self.n_clock, hidden)
        nn.init.kaiming_uniform_(self.enc.weight, a=5 ** 0.5)
        nn.init.zeros_(self.enc.bias)

    def _clocked(self, x: torch.Tensor) -> torch.Tensor:
        clk = clock_features(x.shape[-2], self.period, device=x.device, dtype=x.dtype)
        if x.dim() == 3:                                    # (B, S, D)
            clk = clk.unsqueeze(0).expand(x.shape[0], -1, -1)
        return torch.cat((x, clk), dim=-1)

    def forward(self, x):
        if self.bypass:
            return x
        pd = self.enc.weight.dtype
        with torch.autocast(x.device.type, enabled=False):
            xc = torch.nan_to_num(x.to(pd))
            enc_in = self._clocked(F.layer_norm(xc, (xc.shape[-1],)) if self.normalize_input else xc)
            h = self.act(self.enc(enc_in))
            mu = self.to_mu(h)
            logvar = self.to_logvar(h).clamp(-8.0, 8.0)
            if self.training:
                if self._forced_eps is not None and self._forced_eps.shape == mu.shape:
                    eps = self._forced_eps.to(mu.dtype)
                else:
                    eps = torch.randn_like(mu)
                self.last_eps = eps.detach()
                z = mu + eps * torch.exp(0.5 * logvar)
            else:
                z = mu
            self.last_kl = (-0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())).mean()
            delta = self.out(z)
        return x + delta.to(x.dtype)
