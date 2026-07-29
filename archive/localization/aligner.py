"""Trainable audio-visual alignment head on FROZEN Qwen adapter tokens.

The frozen audio and visual adapter tokens are only *partially* comparable (raw
cosine localizes the source ~1/3 of the way; see localize_cosine + the research memo).
`AVAligner` learns two small projections that map both into a shared space where
cosine(audio, visual_patch) genuinely tracks the sounding source. The base model stays
frozen; only f_a/f_v train, contrastively (matched audio<->video vs mismatched).
"""
from __future__ import annotations

import torch.nn as nn


class AVAligner(nn.Module):
    """Two small MLPs projecting audio + visual tokens into a shared cosine space."""

    def __init__(self, dim: int = 2048, proj: int = 512):
        super().__init__()
        self.f_a = nn.Sequential(nn.Linear(dim, proj), nn.GELU(), nn.Linear(proj, proj))
        self.f_v = nn.Sequential(nn.Linear(dim, proj), nn.GELU(), nn.Linear(proj, proj))

    @staticmethod
    def _unit(z):
        return z / (z.norm(dim=-1, keepdim=True) + 1e-6)

    def audio(self, a):
        """(..., dim) audio vector(s) -> unit-norm (..., proj)."""
        return self._unit(self.f_a(a))

    def visual(self, v):
        """(..., dim) visual token(s) -> unit-norm (..., proj)."""
        return self._unit(self.f_v(v))

    def simmap(self, a_vec, V):
        """a_vec (dim,), V (T_v, dim) -> (T_v,) per-patch cosine similarity."""
        return self.visual(V) @ self.audio(a_vec)
