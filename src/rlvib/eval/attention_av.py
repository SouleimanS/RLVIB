"""Total attention to audio-visual tokens, as a fraction of attention to all input tokens.

Reproduces the MoD-DPO++ "Figure 6" probe. For a clip, run ONE forward with eager attention
and measure how much of the answer position's attention mass lands on the audio + visual
placeholder tokens versus all input tokens (including text). The VIB / FiLM adapter reshapes
exactly those AV token embeddings, so base vs trained shows whether the adapter makes the model
ATTEND to audio-visual evidence more (the thesis), not just whether the answer changes.

Qwen-Omni only: needs the model loaded with attn_implementation='eager' (sdpa/flash do not
return attention weights). Memory is O(L*H*S^2); keep S small (low --fps, short clips).

  model = get_model("qwen2.5-omni", attn="eager")
  frac, n_av, n_tot = av_attention_fraction(model, model.message(video=v, audio=a, prompt=q))
"""
from __future__ import annotations

import torch

# Audio / image / video PLACEHOLDER token surface forms across Qwen-Omni versions (the repeated
# media tokens, not the bos/eos markers). Detection also falls back to config token-index attrs.
_AV_TOKEN_STRINGS = ("<|AUDIO|>", "<|audio_pad|>", "<|IMAGE|>", "<|image_pad|>",
                     "<|VIDEO|>", "<|video_pad|>")
_AV_CONFIG_ATTRS = ("audio_token_index", "image_token_index", "video_token_index",
                    "audio_token_id", "image_token_id", "video_token_id")


def av_token_ids(model) -> set[int]:
    """Token ids marking audio / image / video placeholder positions in input_ids.

    Tries the tokenizer's special-token surface forms first, then config token-index attrs
    (the thinker may nest its config). Prints nothing; callers should sanity-check it is
    non-empty (otherwise the AV mask is empty and the fraction is 0)."""
    tok = getattr(model, "tokenizer", None) or model.processor.tokenizer
    ids: set[int] = set()
    unk = getattr(tok, "unk_token_id", None)
    for s in _AV_TOKEN_STRINGS:
        try:
            i = tok.convert_tokens_to_ids(s)
        except Exception:  # noqa: BLE001
            i = None
        if isinstance(i, int) and i >= 0 and i != unk:
            ids.add(i)
    cfgs = []
    base = getattr(model.model, "config", None)
    if base is not None:
        cfgs.append(base)
        for sub in ("thinker_config", "thinker"):
            c = getattr(base, sub, None)
            c = getattr(c, "config", c)
            if c is not None and c is not base:
                cfgs.append(c)
    for c in cfgs:
        for a in _AV_CONFIG_ATTRS:
            v = getattr(c, a, None)
            if isinstance(v, int) and v >= 0:
                ids.add(v)
    return ids


@torch.no_grad()
def av_attention_fraction(model, messages, av_ids: set[int] | None = None,
                          use_audio_in_video: bool = True, query: str = "last"):
    """Fraction of the answer-position attention mass on AV tokens / on all input tokens.

    Returns (fraction, n_av_tokens, n_total_tokens). `query="last"` uses the final prompt
    position (the one that predicts the answer); `query="mean"` averages over all positions.
    Averages over heads then layers. Requires the model loaded with attn='eager'.
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
    av_mask = torch.zeros_like(ids, dtype=torch.bool)
    for i in av_ids:
        av_mask |= ids == i
    layer_fracs = []
    for a in attns:                                          # a: (1, H, S, S)
        if query == "mean":
            aq = a[0].float().mean(dim=1)                    # (S_query, S_key) head-mean
            aq = aq.mean(dim=0, keepdim=True)                # (1, S_key) query-mean
        else:
            aq = a[0, :, -1, :].float()                      # (H, S_key) last-query, per head
        denom = aq.sum(dim=-1) + 1e-9                        # ~1 (causal, normalized)
        layer_fracs.append((aq[:, av_mask].sum(dim=-1) / denom).mean().item())
    del out, attns
    frac = sum(layer_fracs) / len(layer_fracs)               # mean over layers
    return frac, int(av_mask.sum()), int(av_mask.numel())
