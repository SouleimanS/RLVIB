"""Attention to audio / visual tokens, as a fraction of attention to all input tokens.

Reproduces the MoD-DPO++ "Figure 6" probe. For a clip, run ONE forward with eager attention
and measure how much of the answer position's attention mass lands on the audio and/or visual
placeholder tokens versus all input tokens (including text). The VIB / FiLM adapter reshapes
exactly those tokens, so base vs trained shows whether the adapter makes the model ATTEND to
that evidence more.

`--modality audio` isolates the AUDIO tokens (the V->A grounding axis, cleanly token-represented)
-- recommended for Qwen3-Omni, whose combined AV number is dominated by ~1300 vision tokens and is
sensitive to architecture-specific attention (sinks, vision handling) that differs from Qwen2.5.

Qwen-Omni only: needs the model loaded with attn_implementation='eager'. Memory is O(L*H*S^2);
keep S small (low --fps, short clips).
"""
from __future__ import annotations

import torch

# Placeholder (repeated per media chunk) token surface forms, split by modality; detection also
# falls back to config token-index attrs. Audio vs visual are kept separate so the mask can be
# restricted to one modality.
_AUDIO_TOKEN_STRINGS = ("<|AUDIO|>", "<|audio_pad|>")
_VISUAL_TOKEN_STRINGS = ("<|IMAGE|>", "<|image_pad|>", "<|VIDEO|>", "<|video_pad|>")
_AUDIO_CONFIG_ATTRS = ("audio_token_index", "audio_token_id")
_VISUAL_CONFIG_ATTRS = ("image_token_index", "video_token_index", "image_token_id", "video_token_id")


def _configs(model):
    out = []
    base = getattr(model.model, "config", None)
    if base is not None:
        out.append(base)
        for sub in ("thinker_config", "thinker"):
            c = getattr(base, sub, None)
            c = getattr(c, "config", c)
            if c is not None and c is not base:
                out.append(c)
    return out


def _ids_from(model, token_strings, config_attrs) -> set[int]:
    tok = getattr(model, "tokenizer", None) or model.processor.tokenizer
    ids: set[int] = set()
    unk = getattr(tok, "unk_token_id", None)
    for s in token_strings:
        try:
            i = tok.convert_tokens_to_ids(s)
        except Exception:  # noqa: BLE001
            i = None
        if isinstance(i, int) and i >= 0 and i != unk:
            ids.add(i)
    for c in _configs(model):
        for a in config_attrs:
            v = getattr(c, a, None)
            if isinstance(v, int) and v >= 0:
                ids.add(v)
    return ids


def audio_token_ids(model) -> set[int]:
    """Token ids marking audio placeholder positions in input_ids."""
    return _ids_from(model, _AUDIO_TOKEN_STRINGS, _AUDIO_CONFIG_ATTRS)


def visual_token_ids(model) -> set[int]:
    """Token ids marking image/video placeholder positions in input_ids."""
    return _ids_from(model, _VISUAL_TOKEN_STRINGS, _VISUAL_CONFIG_ATTRS)


def av_token_ids(model) -> set[int]:
    """Audio + visual placeholder token ids (the union)."""
    return audio_token_ids(model) | visual_token_ids(model)


def modality_ids(model, modality: str = "av") -> set[int]:
    """`modality` in {'av','audio','vision'} -> the corresponding placeholder token-id set."""
    return {"audio": audio_token_ids, "vision": visual_token_ids}.get(
        modality, av_token_ids)(model)


@torch.no_grad()
def av_attention_fraction(model, messages, av_ids: set[int] | None = None,
                          use_audio_in_video: bool = True, query: str = "last"):
    """Fraction of the answer-position attention mass on the masked tokens / on all input tokens.

    Returns (fraction, n_masked_tokens, n_total_tokens, per_layer_fracs). `query="last"` uses the
    final prompt position (predicts the answer); `query="mean"` averages over all positions (a
    global measure, runs much higher -- dominated by media-attends-to-media). Averages over heads
    then layers; `per_layer_fracs` is the per-layer profile (head-mean) for diagnostics. Requires
    the model loaded with attn='eager'.
    """
    av_ids = av_ids if av_ids is not None else av_token_ids(model)
    inputs = model.build_inputs(messages, use_audio_in_video=use_audio_in_video)
    lm = getattr(model.model, "thinker", model.model)        # Qwen3 -> .thinker; Qwen2.5 -> itself
    out = lm(**inputs, output_attentions=True, use_cache=False)
    attns = getattr(out, "attentions", None)
    if not attns:
        raise RuntimeError("no attentions returned -- load the model with attn='eager' "
                           "(sdpa/flash do not expose attention weights)")
    ids = inputs["input_ids"][0]                             # (S,)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    for i in av_ids:
        mask |= ids == i
    per_layer = []
    for a in attns:                                          # a: (1, H, S, S)
        if query == "mean":
            aq = a[0].float().mean(dim=1).mean(dim=0, keepdim=True)   # (1, S_key) head+query mean
        else:
            aq = a[0, :, -1, :].float()                      # (H, S_key) last-query, per head
        denom = aq.sum(dim=-1) + 1e-9                        # ~1 (causal, normalized)
        per_layer.append((aq[:, mask].sum(dim=-1) / denom).mean().item())   # head-mean
    del out, attns
    frac = sum(per_layer) / len(per_layer)                   # mean over layers
    return frac, int(mask.sum()), int(mask.numel()), per_layer
