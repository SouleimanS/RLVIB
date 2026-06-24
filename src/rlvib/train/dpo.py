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

from rlvib.models.bottleneck import capture_eps, set_bypass, set_forced_eps, total_kl


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


def grpo_step(model, bottlenecks, optimizer, batch, *, group: int = 8, beta_kl: float = 0.01,
              lam_ref: float = 0.05, r_correct: float = 1.0, r_abstain: float = 0.0,
              r_halluc: float = -1.0, std_norm: bool = True) -> dict:
    """One GRPO step over the trainable bottleneck on yes/no grounding items.

    Each ex = {messages, gold ('yes'|'no'), yes_id, no_id[, abstain_id]}. Per example we draw
    `group` stochastic VIB passes (the model is in train() mode, so z = mu + sigma*eps differs
    each pass -- this IS the exploration; with W_out zero-init the *value* is base at init but the
    gradient w.r.t. the adapter is not), sample an answer from the candidate tokens, score it with
    a verifiable ternary reward, group-normalize to advantages, and apply the GRPO policy-gradient
    to the bottleneck params (grad flows through the realized z via reparameterization). Two extra
    terms regularize, matching docs/research/grpo-vib.md: `beta_kl * KL_VIB` (the bottleneck's own
    compression rate) and `lam_ref * KL(base || policy)` on the answer distribution (the IBL-style
    / GRPO reference-KL anchor against adapter drift off the frozen base -- same form as
    anchored_dpo_step's `gen_kl`).

    Ternary reward (TruthRL): +1 correct, 0 abstain, -1 hallucinate -> because the group advantage
    A_i = (r_i - mean)/std is monotonic in raw reward, abstention always out-advantages a wrong
    "confident" answer, so the policy is pushed to abstain-when-unsure rather than hallucinate.
    Omit `abstain_id` for plain 2-way yes/no (reward collapses to +1/-1).

    Advantage: A_i = r_i - mean(r), optionally /(std(r)+eps). `std_norm=False` is the Dr. GRPO
    ("GRPO Done Right", arXiv:2503.20783) correction -- dividing by std biases the gradient toward
    low-variance (too-easy/too-hard) groups. Length normalization is N/A here (single-token answers).

    KEY RISK (see the memo): if every sample in a group lands the same reward, the advantage is
    ~0 and the policy-gradient term vanishes -- watch `adv_std`. `kl_vib` -> 0 means the sampler is
    collapsing to deterministic (exploration dying). Returns mean metrics.

    Memory: two-phase / fixed-eps. Phase A samples all `group` actions under no_grad (capturing
    each forward's VIB noise); phase B replays each sample's noise WITH grad and backward()s it
    immediately, so peak memory is ONE forward graph regardless of `group` (the gradient is
    identical to building all group graphs at once -- the eps is held fixed per sample).
    """
    optimizer.zero_grad()
    was_training = bottlenecks.training
    bottlenecks.train()                                     # ensure z = mu + sigma*eps (sampling)
    rewards_m, advstd_m, klvib_m, klref_m, pcorrect_m = [], [], [], [], []
    for ex in batch:
        gold = ex["gold"]
        cands = [("yes", ex["yes_id"]), ("no", ex["no_id"])]
        if ex.get("abstain_id") is not None:
            cands.append(("abstain", ex["abstain_id"]))
        ids = [i for _, i in cands]

        # Phase A: sample `group` actions + rewards under no_grad, snapshotting each VIB noise.
        eps_list, actions, rewards = [], [], []
        set_forced_eps(bottlenecks, None)                   # free-running sampling
        with torch.no_grad():
            for _ in range(group):
                set_bypass(bottlenecks, False)
                lp = answer_logp_vec(model, ex["messages"])
                eps_list.append(capture_eps(bottlenecks))
                cat = torch.log_softmax(lp[ids], dim=0)
                a = int(torch.distributions.Categorical(logits=cat).sample())
                actions.append(a)
                label = cands[a][0]
                rewards.append(r_correct if label == gold
                               else (r_abstain if label == "abstain" else r_halluc))

        r = torch.tensor(rewards, dtype=torch.float32, device=lp.device)
        adv = r - r.mean()                                  # group-relative advantage (mean-centered)
        if std_norm:                                        # classic GRPO; Dr. GRPO drops /std (it
            adv = adv / (r.std() + 1e-6)                     # over-weights low-variance groups -> bias)

        with torch.no_grad():                               # reference = bottleneck bypassed (base)
            set_bypass(bottlenecks, True)
            base = answer_logp_vec(model, ex["messages"])
            set_bypass(bottlenecks, False)

        # Phase B: replay each sample's eps WITH grad and backward immediately (one live graph).
        kls_b, krefs_b = [], []
        for i in range(group):
            set_forced_eps(bottlenecks, eps_list[i])
            lp = answer_logp_vec(model, ex["messages"])
            kl_vib = total_kl(bottlenecks)
            if not torch.is_tensor(kl_vib):
                kl_vib = torch.zeros((), device=lp.device)
            cat = torch.log_softmax(lp[ids], dim=0)
            logp = cat[actions[i]]                          # log pi(a_i | x, z_i)  [keeps grad]
            kl_ref = (base.exp() * (base - lp)).sum()       # KL(base || policy) at the answer pos
            # mean over the group: pg = mean_i(-adv_i*logp_i); kl terms averaged too
            loss_i = (-(adv[i].detach() * logp) + beta_kl * kl_vib + lam_ref * kl_ref) / group
            loss_i.backward()
            kls_b.append(float(kl_vib.detach()))
            krefs_b.append(float(kl_ref.detach()))
        set_forced_eps(bottlenecks, None)

        rewards_m.append(float(r.mean()))
        advstd_m.append(float(r.std()))
        klvib_m.append(sum(kls_b) / max(1, len(kls_b)))
        klref_m.append(sum(krefs_b) / max(1, len(krefs_b)))
        pcorrect_m.append(float((r == r_correct).float().mean()))

    torch.nn.utils.clip_grad_norm_(bottlenecks.parameters(), max_norm=1.0)
    optimizer.step()
    set_forced_eps(bottlenecks, None)
    if not was_training:
        bottlenecks.eval()
    n = max(1, len(batch))
    return {"reward": sum(rewards_m) / n, "adv_std": sum(advstd_m) / n,
            "kl_vib": sum(klvib_m) / n, "kl_ref": sum(klref_m) / n,
            "p_correct": sum(pcorrect_m) / n}
