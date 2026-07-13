"""MoD-DPO / MoD-DPO++ objective (Chaubey et al., arXiv:2603.03192) on the RLVIB adapter.

Modality-Decoupled DPO adds two STOP-GRADIENT offsets to the DPO margin, computed from
corrupted-input passes of the CURRENT policy (treated as fixed targets within a step):

  margin = tau * logratio_theta(clean)            tau = beta + beta_inv - beta_sens
         - beta      * logratio_ref(clean)
         - beta_inv  * logratio_theta'(IRRELEVANT modality corrupted)   [invariance]
         + beta_sens * logratio_theta'(RELEVANT modality corrupted)     [sensitivity]
         - gamma_lpd * logratio_text(text-only)                         [++ only: LPD]

  loss = -log sigmoid(margin)     with logratio(x) = log p(y_w|x) - log p(y_l|x)

(paper Eqs. 5-12; pi'_theta fixed per step = no_grad here; pi_text = pi_ref on text-only
inputs; batches alternate audio-related and vision-related prompts). Reference = the
adapter bypassed, exactly as in the rest of the repo.

ADAPTATION NOTES (documented in README.md): the paper trains the full LLM on 18k GPT-4o
sentence-level preference pairs; here the SAME objective trains our frozen-backbone adapter
on our AVE letter-answer pairs, so results compare OBJECTIVES at equal capacity/data, not
reproduce the paper's absolute numbers.
"""
from __future__ import annotations

import os
import random

import torch
import torch.nn.functional as F

from rlvib.data.pairs import silence_audio, swap_audio
from rlvib.models.bottleneck import set_bypass, total_kl
from rlvib.train.dpo import answer_logp_vec, letter_id

DEFAULTS = dict(beta=0.1, beta_inv=0.02, beta_sens=0.05, gamma_lpd=0.05)   # paper Sec. 5.2


def build_corruptions(record, items, out_dir: str, rng: random.Random) -> dict:
    """Materialize the two corrupted variants of a swapped clip (cached by filename).

    audio-corrupted  a': the clip with its audio track silenced           (relevant for HEAR)
    video-corrupted  v': ANOTHER clip's video carrying THIS clip's audio  (relevant for SEE)
    -- the paper's "mismatched audiovisual contexts using segments from different files".
    """
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(record["video_path"]))[0]
    a_corr = os.path.join(out_dir, f"{stem}__acorr.mp4")
    if not os.path.exists(a_corr):
        silence_audio(record["video_path"], a_corr)
    donor = rng.choice(items)
    v_corr = os.path.join(out_dir, f"{stem}__vcorr.mp4")
    if not os.path.exists(v_corr):
        swap_audio(donor["video_path"], record["video_path"], v_corr)
    return {"audio_corrupt": a_corr, "video_corrupt": v_corr}


def mod_dpo_step(model, bottlenecks, optimizer, batch, *, beta: float = 0.1,
                 beta_inv: float = 0.02, beta_sens: float = 0.05, gamma_lpd: float = 0.0,
                 beta_kl: float = 0.0) -> dict:
    """One optimizer step. Each ex must carry:
      messages            clean input (a, v, x)
      messages_irr        IRRELEVANT modality corrupted (invariance target)
      messages_rel        RELEVANT modality corrupted   (sensitivity target)
      messages_text       text-only input (LPD; only used when gamma_lpd > 0)
      chosen_letter / rejected_letter
    gamma_lpd = 0 -> MoD-DPO; > 0 -> MoD-DPO++.
    """
    optimizer.zero_grad()
    tau = beta + beta_inv - beta_sens
    losses, margins, prefs, invs, senss = [], [], [], [], []
    for ex in batch:
        c, r = letter_id(model, ex["chosen_letter"]), letter_id(model, ex["rejected_letter"])

        set_bypass(bottlenecks, False)                       # policy, clean (keeps grad)
        lp = answer_logp_vec(model, ex["messages"])
        kl = total_kl(bottlenecks)
        lr_theta = lp[c] - lp[r]

        with torch.no_grad():                                # all fixed targets of this step
            lpi = answer_logp_vec(model, ex["messages_irr"])      # policy, irrelevant corrupted
            lr_irr = lpi[c] - lpi[r]
            lps = answer_logp_vec(model, ex["messages_rel"])      # policy, relevant corrupted
            lr_rel = lps[c] - lps[r]
            set_bypass(bottlenecks, True)                         # reference = bypassed base
            lpr = answer_logp_vec(model, ex["messages"])
            lr_ref = lpr[c] - lpr[r]
            lr_text = torch.zeros(())
            if gamma_lpd > 0:
                lpt = answer_logp_vec(model, ex["messages_text"])  # pi_text = pi_ref, text-only
                lr_text = lpt[c] - lpt[r]
            set_bypass(bottlenecks, False)

        margin = (tau * lr_theta - beta * lr_ref
                  - beta_inv * lr_irr + beta_sens * lr_rel - gamma_lpd * lr_text)
        loss = -F.logsigmoid(margin)
        if beta_kl and torch.is_tensor(kl):
            loss = loss + beta_kl * kl
        loss.backward()

        losses.append(float(loss.detach()))
        margins.append(float(margin.detach()))
        prefs.append(float((lp[c] > lp[r]).detach()))
        invs.append(float(lr_irr))
        senss.append(float(lr_rel))

    torch.nn.utils.clip_grad_norm_(bottlenecks.parameters(), max_norm=1.0)
    optimizer.step()
    n = max(1, len(batch))
    return {"loss": sum(losses) / n, "margin": sum(margins) / n, "p_chosen": sum(prefs) / n,
            "lr_irr": sum(invs) / n, "lr_rel": sum(senss) / n}
