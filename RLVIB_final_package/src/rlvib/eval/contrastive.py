"""Audio-aware contrastive decoding (AAD / VCD-style), composed on top of a trained adapter.

Inference-time, training-free: contrast the answer-token logits computed WITH audio against
the same forward WITH THE AUDIO REMOVED, and amplify tokens whose score rises when audio is
present:

    logit_cd = (1 + alpha) * logit_full - alpha * logit_no_audio

with a VCD-style plausibility constraint that keeps only tokens the full pass already finds
plausible (so the contrast can't promote a degenerate token). The attached bottleneck is
active in BOTH passes, so this sits on top of it rather than replacing it.

For the discriminative evals (yes/no, single-letter) the answer is the first generated token,
so we score the first position only: two prefills per item, no decode loop. Works for the
Qwen3-/Qwen2.5-Omni wrappers (which share message / build_inputs / processor, with the LM at
model.model.thinker or model.model). VideoLLaMA2 answers in full sentences, so first-token
scoring is not valid for it and the eval falls back to plain decoding there.

See run_avhbench.py / run_cmm.py --audio-cd.
"""
from __future__ import annotations

import math

import torch


def contrastive_logits(logits_full, logits_noaud, alpha: float = 1.0, plausibility: float = 0.1):
    """(1+alpha)*full - alpha*no_audio, restricted to tokens plausible under `full` (VCD)."""
    lf = logits_full.float()
    ln = logits_noaud.float()
    cd = (1.0 + alpha) * lf - alpha * ln
    if plausibility and plausibility > 0:
        keep = lf >= lf.max(dim=-1, keepdim=True).values + math.log(plausibility)
        cd = cd.masked_fill(~keep, float("-inf"))
    return cd


@torch.no_grad()
def contrastive_answer(model, video=None, audio=None, prompt: str = "", alpha: float = 1.0,
                       use_audio_in_video: bool = True, plausibility: float = 0.1) -> str:
    """Audio-aware contrastive first-token answer for a Qwen-Omni-style wrapper.

    Two prefills through the thinker -- one with audio, one with the audio removed -- combine
    the next-token (answer) logits via contrastive_logits, and decode the argmax token.
    """
    lm = getattr(model.model, "thinker", model.model)
    inp_f = model.build_inputs(model.message(video=video, audio=audio, prompt=prompt),
                               use_audio_in_video=use_audio_in_video)
    inp_n = model.build_inputs(model.message(video=video, prompt=prompt),  # audio removed
                               use_audio_in_video=False)
    lf = lm(**inp_f).logits[:, -1, :]
    ln = lm(**inp_n).logits[:, -1, :]
    cd = contrastive_logits(lf, ln, alpha=alpha, plausibility=plausibility)
    tok_id = int(cd.argmax(dim=-1)[0])
    tokenizer = getattr(model, "tokenizer", None) or model.processor.tokenizer
    return tokenizer.decode([tok_id], skip_special_tokens=True).strip()
