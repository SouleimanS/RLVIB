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


def answer_logprob(model, messages, letter: str, use_audio_in_video: bool = True):
    """log p(first generated token == `letter`) given prompt + media (keeps grad)."""
    inputs = model.build_inputs(messages, use_audio_in_video=use_audio_in_video)
    lm = getattr(model.model, "thinker", model.model)  # Qwen3 -> .thinker; Qwen2.5 -> itself
    logits = lm(**inputs).logits[:, -1, :]             # next-token logits at the gen position
    logp = torch.log_softmax(logits.float(), dim=-1)
    tid = model.processor.tokenizer(letter, add_special_tokens=False).input_ids[0]
    return logp[0, tid]


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
