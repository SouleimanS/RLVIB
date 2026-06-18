"""Modality-conditional DPO (mDPO) over a frozen AV-LLM + trainable bottleneck.

For an audio-dependent question with gold answer letter L, we prefer the policy to
assign higher *relative* log-prob to L WITH the audio than WITHOUT it (audio-drop =
the v0 counterfactual, via use_audio_in_video=False — no ffmpeg needed):

  margin = beta * [ (logp_pol(L|full) - logp_ref(L|full))
                  - (logp_pol(L|drop) - logp_ref(L|drop)) ]
  loss   = -logsigmoid(margin) + beta_kl * KL_rate

Reference = the same model with the bottleneck *bypassed* (identity) — so no second
model is held in memory; only the bottleneck updates. See ib-rl-method-and-framing.md.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from rlvib.models.bottleneck import set_bypass, total_kl


def answer_logp_vec(model, messages, use_audio_in_video: bool = True):
    """First-token log-prob vector over the vocab (keeps grad). One forward; index as
    many candidate answers as needed (e.g. chosen & rejected) from the same result."""
    inputs = model.build_inputs(messages, use_audio_in_video=use_audio_in_video)
    lm = getattr(model.model, "thinker", model.model)  # Qwen3 -> .thinker; Qwen2.5 -> itself
    if getattr(model, "dtype", None) == torch.float16:
        # fp16 backbones (VideoLLaMA2) overflow in attention (Q.K^T > 65504) -> NaN logits;
        # compute the forward in bf16 (same range as fp32). bf16/fp32 backbones unchanged.
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = lm(**inputs).logits[:, -1, :]
    else:
        logits = lm(**inputs).logits[:, -1, :]         # next-token logits at the gen position
    return torch.log_softmax(logits.float(), dim=-1)[0]


def letter_id(model, letter: str) -> int:
    """Token id for a bare MCQ letter (diag confirmed the model emits the bare letter).

    Qwen wrappers expose the tokenizer via `processor.tokenizer`; VideoLLaMA2 holds it
    directly as `model.tokenizer` (its `processor` is a dict of preprocessors)."""
    tok = getattr(model, "tokenizer", None) or model.processor.tokenizer
    return tok(letter, add_special_tokens=False).input_ids[0]


def answer_logprob(model, messages, letter: str, use_audio_in_video: bool = True):
    """log p(first generated token == `letter`) given prompt + media (keeps grad)."""
    return answer_logp_vec(model, messages, use_audio_in_video)[letter_id(model, letter)]


def mdpo_step(model, bottlenecks, optimizer, batch, beta: float = 0.1,
              beta_kl: float = 0.01) -> dict:
    """One optimizer step over a list of {messages, gold_letter}.

    Audio-drop is the counterfactual (full AV vs use_audio_in_video=False). Grads
    accumulate over the batch, then a single optimizer step. Returns mean metrics.
    """
    optimizer.zero_grad()
    losses, margins, kls = [], [], []
    for ex in batch:
        L, msg = ex["gold_letter"], ex["messages"]

        set_bypass(bottlenecks, False)  # policy
        lp_full = answer_logprob(model, msg, L, use_audio_in_video=True)
        kl = total_kl(bottlenecks)
        lp_drop = answer_logprob(model, msg, L, use_audio_in_video=False)

        with torch.no_grad():           # reference = bottleneck bypassed (frozen base)
            set_bypass(bottlenecks, True)
            rp_full = answer_logprob(model, msg, L, use_audio_in_video=True)
            rp_drop = answer_logprob(model, msg, L, use_audio_in_video=False)
            set_bypass(bottlenecks, False)

        margin = beta * ((lp_full - rp_full) - (lp_drop - rp_drop))
        loss = -F.logsigmoid(margin) + beta_kl * kl
        loss.backward()

        losses.append(float(loss.detach()))
        margins.append(float(margin.detach()))
        kls.append(float(kl.detach()) if hasattr(kl, "detach") else float(kl))

    optimizer.step()
    n = max(1, len(batch))
    return {"loss": sum(losses) / n, "margin": sum(margins) / n, "kl": sum(kls) / n}


def dpo_step(model, bottlenecks, optimizer, batch, beta: float = 0.1,
             beta_kl: float = 0.01) -> dict:
    """Contrastive DPO on explicit chosen/rejected letters (one full-AV input each).

    For audio-swap pairs: chosen = the HEARD event, rejected = the SEEN event, both
    scored on the SAME swapped clip -> directly penalizes the visual shortcut. A
    shortcut model puts mass on `rejected`, so `chosen` has room to grow (the gradient
    the audio-drop counterfactual lacked). Reference = bottleneck bypassed. `p_chosen`
    = fraction of the batch where the policy already prefers the heard answer.
    """
    optimizer.zero_grad()
    losses, margins, kls, prefs = [], [], [], []
    for ex in batch:
        c, r = letter_id(model, ex["chosen_letter"]), letter_id(model, ex["rejected_letter"])

        set_bypass(bottlenecks, False)              # policy
        lp = answer_logp_vec(model, ex["messages"])
        kl = total_kl(bottlenecks)
        cp, rp = lp[c], lp[r]

        with torch.no_grad():                       # reference = bottleneck bypassed
            set_bypass(bottlenecks, True)
            lr = answer_logp_vec(model, ex["messages"])
            cr, rr = lr[c], lr[r]
            set_bypass(bottlenecks, False)

        margin = beta * ((cp - cr) - (rp - rr))
        loss = -F.logsigmoid(margin) + beta_kl * kl
        loss.backward()

        losses.append(float(loss.detach()))
        margins.append(float(margin.detach()))
        kls.append(float(kl.detach()) if hasattr(kl, "detach") else float(kl))
        prefs.append(float((cp > rp).detach()))

    optimizer.step()
    n = max(1, len(batch))
    return {"loss": sum(losses) / n, "margin": sum(margins) / n,
            "kl": sum(kls) / n, "p_chosen": sum(prefs) / n}


def anchored_dpo_step(model, bottlenecks, optimizer, swap_batch, anchor_batch,
                      beta: float = 0.1, beta_kl: float = 0.01, lam_anchor: float = 1.0,
                      delta: float = 0.0, lam_kl: float = 1.0) -> dict:
    """Collapse-resistant swap-DPO (see docs/research/dpo-collapse-and-fixes.md).

    Adds the two anchors plain DPO lacked:
      1. mDPO chosen anchor  -log sigmoid(beta*(cp - cr) - delta): pins the chosen letter's
         policy log-prob at/above the frozen base, blocking likelihood displacement.
      2. General KL-to-base   KL(base || policy) on `anchor_batch` (normal, non-swap prompts):
         penalizes the always-on adapter for drifting from the frozen model's answer
         distribution where it should be identity -- the input-blind constraint DPO omits.
    `chosen_minus_ref` (= cp - cr) should stay >= 0; `gen_kl` should stay small.
    """
    optimizer.zero_grad()
    losses, margins, prefs, anchors, cmr, gkls = [], [], [], [], [], []
    for ex in swap_batch:
        c, r = letter_id(model, ex["chosen_letter"]), letter_id(model, ex["rejected_letter"])
        set_bypass(bottlenecks, False)
        lp = answer_logp_vec(model, ex["messages"])
        kl = total_kl(bottlenecks)
        cp, rp = lp[c], lp[r]
        with torch.no_grad():
            set_bypass(bottlenecks, True)
            lr = answer_logp_vec(model, ex["messages"])
            cr, rr = lr[c], lr[r]
            set_bypass(bottlenecks, False)
        margin = beta * ((cp - cr) - (rp - rr))
        anchor = -F.logsigmoid(beta * (cp - cr) - delta)   # pin chosen >= ref (mDPO L_AncPO)
        loss = -F.logsigmoid(margin) + lam_anchor * anchor + beta_kl * kl
        loss.backward()
        losses.append(float(loss.detach()))
        margins.append(float(margin.detach()))
        prefs.append(float((cp > rp).detach()))
        anchors.append(float(anchor.detach()))
        cmr.append(float((cp - cr).detach()))

    for ex in anchor_batch:                                # general KL-to-base anchor
        set_bypass(bottlenecks, False)
        lp_pol = answer_logp_vec(model, ex["messages"])
        with torch.no_grad():
            set_bypass(bottlenecks, True)
            lp_base = answer_logp_vec(model, ex["messages"])
            set_bypass(bottlenecks, False)
        gkl = (lp_base.exp() * (lp_base - lp_pol)).sum()   # KL(base || policy) at the answer pos
        (lam_kl * gkl).backward()
        gkls.append(float(gkl.detach()))

    torch.nn.utils.clip_grad_norm_(bottlenecks.parameters(), max_norm=1.0)  # spike insurance
    optimizer.step()
    n, m = max(1, len(swap_batch)), max(1, len(anchor_batch))
    return {"loss": sum(losses) / n, "margin": sum(margins) / n, "p_chosen": sum(prefs) / n,
            "anchor": sum(anchors) / n, "chosen_minus_ref": sum(cmr) / n, "gen_kl": sum(gkls) / m}
