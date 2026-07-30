"""Cross-modal attention fusion BEFORE the frozen LLM (comparison arm vs the VIB/FiLM/LoRA).

At the point where the audio and vision tokens have been merged into the LLM's inputs_embeds
(the only place both streams coexist), apply a small trainable bidirectional cross-attention:

    audio_tokens  +=  W_out^a . MHA(Q=audio,  K/V=vision)     (audio reads the vision stream)
    vision_tokens +=  W_out^v . MHA(Q=vision, K/V=audio)      (vision reads the audio stream)

Both output projections are ZERO-INIT => identity at init, and `bypass` returns the exact
frozen base -- the anchored-DPO reference works unchanged (same contract as the bottleneck).

Wiring (Qwen-Omni thinker): two hooks --
  1. a forward pre-hook on the THINKER captures input_ids -> audio/vision placeholder masks
     (ids from rlvib.eval.attention_av), stashed on the module;
  2. a forward pre-hook (with kwargs) on thinker.model edits inputs_embeds at those positions.
Decode steps (seq len 1, no media positions) are skipped by the mask-length check.

Train with the identical anchored swap-DPO recipe: `train_swap_anchored.py --xattn`
(checkpoints in runs/anchored_<model>_xattn -> EXP=xattn works with all selection tooling).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CrossModalAttention(nn.Module):
    """Bidirectional audio<->vision cross-attention with zero-init output projections."""

    def __init__(self, dim: int = 2048, heads: int = 8):
        super().__init__()
        self.dim = dim
        self.a_from_v = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.v_from_a = nn.MultiheadAttention(dim, heads, batch_first=True)
        for mha in (self.a_from_v, self.v_from_a):
            nn.init.zeros_(mha.out_proj.weight)              # zero-init => identity at init
            nn.init.zeros_(mha.out_proj.bias)
        self.bypass = False
        self.last_kl = None                                  # total_kl() -> 0 (no rate term)
        self._masks = None                                   # (audio_mask, vision_mask) per forward

    def set_masks(self, a_mask, v_mask) -> None:
        self._masks = (a_mask, v_mask)

    def forward(self, embeds: torch.Tensor) -> torch.Tensor:
        """embeds: (B, S, D) merged inputs_embeds; edits the media positions in place-safe copy."""
        if self.bypass or self._masks is None:
            return embeds
        a_mask, v_mask = self._masks
        if a_mask.shape[0] != embeds.shape[1] or not bool(a_mask.any()) or not bool(v_mask.any()):
            return embeds                                    # decode step / unimodal input
        pd = self.a_from_v.out_proj.weight.dtype
        x = embeds[0].to(pd)                                 # (S, D); batch-1 loop everywhere
        aud, vis = x[a_mask].unsqueeze(0), x[v_mask].unsqueeze(0)
        da, _ = self.a_from_v(aud, vis, vis, need_weights=False)
        dv, _ = self.v_from_a(vis, aud, aud, need_weights=False)
        out = embeds.clone()
        out[0, a_mask] = (aud + da)[0].to(embeds.dtype)
        out[0, v_mask] = (vis + dv)[0].to(embeds.dtype)
        return out


def attach_xattn(model, heads: int = 8):
    """Freeze the model; hook the thinker (mask capture) + its decoder (embed edit).
    Returns (ModuleDict({'xattn': module}), handles) -- the attach_bottlenecks contract, so
    set_bypass / total_kl / set_condition / the trainer and eval loops work unchanged."""
    from rlvib.eval.attention_av import audio_token_ids, visual_token_ids

    for p in model.model.parameters():
        p.requires_grad_(False)
    lm = getattr(model.model, "thinker", model.model)
    dim = lm.get_input_embeddings().weight.shape[1]
    a_ids, v_ids = audio_token_ids(model), visual_token_ids(model)
    xdtype = torch.float32 if getattr(model, "dtype", None) == torch.float16 else model.dtype
    mod = CrossModalAttention(dim, heads=heads).to(model.device, xdtype)

    def capture(module, args, kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        if torch.is_tensor(ids) and ids.dim() == 2 and ids.shape[1] > 1:
            row = ids[0]
            a_mask = torch.zeros_like(row, dtype=torch.bool)
            v_mask = torch.zeros_like(row, dtype=torch.bool)
            for i in a_ids:
                a_mask |= row == i
            for i in v_ids:
                v_mask |= row == i
            mod.set_masks(a_mask, v_mask)
        return None

    def edit(module, args, kwargs):
        emb = kwargs.get("inputs_embeds")
        if emb is None or mod.bypass:
            return None
        kwargs = dict(kwargs)
        kwargs["inputs_embeds"] = mod(emb)
        return args, kwargs

    handles = [lm.register_forward_pre_hook(capture, with_kwargs=True),
               lm.model.register_forward_pre_hook(edit, with_kwargs=True)]
    return nn.ModuleDict({"xattn": mod}), handles


def load_attached_xattn(model, ck):
    """Attach + load a trained cross-attention checkpoint for inference."""
    mods, handles = attach_xattn(model, heads=ck.get("heads", 8))
    mods.load_state_dict(ck["state_dict"])
    mods.eval()
    return mods, handles
