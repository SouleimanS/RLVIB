"""MAD: Modality-Adaptive Decoding (Chung et al., KAIST, arXiv:2601.21181) -- training-free.

Two steps per item (paper Fig. 2 / Alg. 1):
  1. WEIGHT EXTRACTION: append the modality query prompt
        "To answer this question, which modality is needed (audio, video, or both)?"
     and read the model's next-token logits for 'both' / 'video' / 'audio';
     softmax -> (w_av, w_v, w_a).
  2. MODALITY-ADAPTIVE GENERATION (Eq. 9), per decoded token, from 4 input configs
     (vaq clean, v'aq video-perturbed, va'q audio-perturbed, v'a'q both-perturbed):
        logit_MAD = (1 + g*w_av) * L[vaq] - g*w_av * L[v'aq]      # visual CD | audio present
                  + (1 + g*w_av) * L[vaq] - g*w_av * L[va'q]      # audio  CD | visual present
                  + (1 + g*w_v ) * L[va'q] - g*w_v  * L[v'a'q]    # visual CD | audio absent
                  + (1 + g*w_a ) * L[v'aq] - g*w_a  * L[v'a'q]    # audio  CD | visual absent
     with gamma g = 2.5 (paper Sec. 4.1.4), greedy argmax.

Perturbation here = MODALITY REMOVAL (one of the paper's stated corruption choices):
  v' = drop the video (keep the clip's audio track as an audio input)
  a' = use_audio_in_video=False (video without its audio)
Cost: 4 forwards per decoded token + 1 weight pass -> use --limit for large benchmarks.
"""
from __future__ import annotations

import torch

MODALITY_QUERY = ("To answer this question, which modality is needed (audio, video, or both)? "
                  "Answer with one word.")
GAMMA = 2.5


def _tok(model):
    return getattr(model, "tokenizer", None) or model.processor.tokenizer


def _word_ids(model, word: str) -> list[int]:
    """First-token ids for the surface forms of `word` ('both'/'video'/'audio')."""
    tok = _tok(model)
    ids = []
    for s in (word, " " + word, word.capitalize(), " " + word.capitalize()):
        t = tok(s, add_special_tokens=False).input_ids
        if t:
            ids.append(t[0])
    return sorted(set(ids))


@torch.no_grad()
def _first_logits(model, messages, use_audio_in_video: bool = True):
    """Raw next-token logits at the answer position (NOT log-softmaxed -- Eq. 9 mixes logits)."""
    inputs = model.build_inputs(messages, use_audio_in_video=use_audio_in_video)
    lm = getattr(model.model, "thinker", model.model)
    return lm(**inputs).logits[0, -1, :].float(), inputs


def modality_weights(model, video, audio, question, fps=None) -> tuple[float, float, float]:
    """(w_av, w_v, w_a) from the model's own modality self-assessment (paper Eq. 6)."""
    msg = model.message(video=video, audio=audio,
                        prompt=f"{question}\n{MODALITY_QUERY}", fps=fps)
    logits, _ = _first_logits(model, msg, use_audio_in_video=video is not None)
    z = []
    for word in ("both", "video", "audio"):
        ids = _word_ids(model, word)
        z.append(max(float(logits[i]) for i in ids))
    w = torch.softmax(torch.tensor(z), dim=0).tolist()
    return w[0], w[1], w[2]                                  # w_av, w_v, w_a


def _configs(model, video, audio, prompt, fps):
    """The four input configurations. Returns {name: (messages, use_audio_in_video)}.
    Video items: audio rides inside the clip; v' keeps the audio track via an audio-only
    message on the same file. Audio-only items (CMM): the video branches collapse."""
    cfg = {}
    if video is not None:
        cfg["vaq"] = (model.message(video=video, audio=audio, prompt=prompt, fps=fps), True)
        cfg["v_aq"] = (model.message(audio=audio or video, prompt=prompt), False)   # video removed
        cfg["va_q"] = (model.message(video=video, prompt=prompt, fps=fps), False)   # audio removed
        cfg["v_a_q"] = (model.message(prompt=prompt), False)                        # both removed
    else:                                                     # audio-only item
        cfg["vaq"] = (model.message(audio=audio, prompt=prompt), False)
        cfg["v_aq"] = cfg["vaq"]
        cfg["va_q"] = (model.message(prompt=prompt), False)
        cfg["v_a_q"] = cfg["va_q"]
    return cfg


@torch.no_grad()
def mad_answer(model, video=None, audio=None, prompt: str = "", fps=None,
               gamma: float = GAMMA, max_new_tokens: int = 8) -> str:
    """Greedy MAD decoding (Eq. 9). Re-encodes the prompt per step (batch-1, short answers)."""
    w_av, w_v, w_a = modality_weights(model, video, audio, prompt, fps=fps)
    tok = _tok(model)
    eos = tok.eos_token_id
    generated: list[int] = []

    base_cfg = _configs(model, video, audio, prompt, fps)
    for _ in range(max_new_tokens):
        suffix = tok.decode(generated, skip_special_tokens=True) if generated else ""
        logits = {}
        for name, (msgs, uaiv) in base_cfg.items():
            if suffix:                                        # extend the prompt with the partial answer
                msgs = [dict(turn) for turn in msgs]
                msgs[-1] = dict(msgs[-1])
                msgs[-1]["content"] = list(msgs[-1]["content"]) + [{"type": "text", "text": suffix}]
            logits[name], _ = _first_logits(model, msgs, use_audio_in_video=uaiv)
        lm = ((1 + gamma * w_av) * logits["vaq"] - gamma * w_av * logits["v_aq"]     # vis CD | aud present
              + (1 + gamma * w_av) * logits["vaq"] - gamma * w_av * logits["va_q"]   # aud CD | vis present
              + (1 + gamma * w_v) * logits["va_q"] - gamma * w_v * logits["v_a_q"]   # vis CD | aud absent
              + (1 + gamma * w_a) * logits["v_aq"] - gamma * w_a * logits["v_a_q"])  # aud CD | vis absent
        nxt = int(lm.argmax())
        if nxt == eos:
            break
        generated.append(nxt)
    return tok.decode(generated, skip_special_tokens=True).strip()
