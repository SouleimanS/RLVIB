# GRPO × VIB — can we train the bottleneck with RL instead of swap-DPO?

> Date: 2026-06-24 · Status: design memo + literature review (decision-oriented).
> Method: 5-angle deep-research fan-out (GRPO on tiny adapter / low-entropy policy ·
> VIB+RL double-KL interaction · multimodal-hallucination RL recipes · verifiable-vs-RM
> reward design · GRPO failure modes vs DPO) + a 25-claim adversarial verification pass
> (20 confirmed, 5 killed). Extends `ib-rl-method-and-framing.md` and
> `dpo-collapse-and-fixes.md`.
> Sourcing caveat: many arXiv PDFs returned HTTP 403 through the proxy, so several
> mechanism claims read from author abstracts / extracted quotes rather than full text;
> corroboration was consistent. Key sources are Sept–Oct 2025 and not yet canonical.

## TL;DR decision

Our exact configuration — a tiny zero-init per-modality VIB adapter serving **as** a GRPO
policy on a frozen audio-visual LLM — is **novel and unattested**. No surveyed paper does
precisely this. The adjacent literature points to a clear, lower-risk recipe:

1. **Don't bet on vanilla GRPO over the adapter.** Full-token GRPO under LoRA *failed to beat
   the base*; gains require **token-selective** updates that act as an implicit regularizer in
   low-capacity regimes [S-GRPO/T-SPMO]. Our `W_out`-zero-init adapter is an even lower-capacity,
   more near-deterministic policy, so this risk applies directly.
2. **Prefer IB-as-auxiliary-regularizer over IB-as-policy.** Every precedent (InfoRM, VIB+SAC
   meta-RL) puts the bottleneck on a *latent / reward model* and regularizes the RL objective with
   it; none makes the bottleneck itself the policy. The well-supported move is to keep our
   `β_kl·KL_VIB` term and optionally add an InfoRM/IBL-style latent-outlier penalty to the GRPO
   reward — **not** to reinterpret the VIB sampler as the RL action distribution.
3. **Use an abstention-aware ternary reward, not binary.** Binary correct/incorrect *increases*
   hallucination (rewards always-guessing); a ternary correct(+1)/abstain(0)/hallucinate(−1)
   reward makes abstention strictly preferred over hallucination **by construction** under GRPO's
   group-relative advantage [TruthRL].
4. **GRPO is not a strict upgrade over swap-DPO.** The "online GRPO beats offline DPO" claim was
   *refuted* in verification. Treat GRPO as an alternative with different failure modes, A/B'd
   against the held swap-DPO `broad` recipe — not a presumed win.

The single biggest open risk is structural: a near-deterministic policy + binary verifiable
reward + weak KL is a setup that is **designed to mode-collapse** regardless of optimizer
[arXiv:2510.20817], and produces **zero-variance GRPO groups** (all samples same reward → zero
advantage → no gradient). Both are addressable but must be designed for up front.

---

## 1. Is GRPO viable when only a tiny VIB adapter is trainable? (Q1)

**Yes, but only via token-selective variants — and our zero-init regime sharpens the risk.**

Lee & Tong, *Token-Efficient RL for LLM Reasoning* (ICML 2025, [arXiv:2504.20834](https://arxiv.org/html/2504.20834v1))
trains GRPO with **only a LoRA adapter on a frozen backbone** (single 40 GB GPU), lifting
Qwen2-1.5B SVAMP 46% → >70%. That is the closest structural analog to our plan. The decisive
detail: **full-token GRPO under LoRA failed to improve over the base**; the gains came from
S-GRPO (update 30–50% of tokens) / T-SPMO (<5%), and the authors read selective token-level
optimization as "an implicit regularizer in low-parameter training regimes." *(Verified 3-0. The
stronger inverse — "vanilla GRPO is simply not viable on tiny params" — was **refuted 0-3**, so
the safe statement is the conditional one above, not a blanket no.)*

**Why this bites harder for us than for LoRA.** Our adapter is `y = x + W_out·z` with `W_out`
zero-init (`bottleneck.py:62`), so at init the policy *is* the frozen base — near-deterministic,
and its only stochasticity is the VIB's own `z = μ(x) + σ(x)·ε` (`bottleneck.py:98`). For GRPO's
group-relative advantage `Â_i = (r_i − mean)/std` to be non-degenerate, sampled outputs within a
group must *differ in reward*. If every sample in a group answers the same yes/no, the group has
zero variance and contributes no gradient. **Exploration would have to come entirely from
`σ(x)·ε`** — which means `logvar` must not collapse early (the usual VIB failure where KL→0,
`σ→0`, the sampler goes deterministic). This is the make-or-break unknown to prototype first.

## 2. Do the VIB-KL and GRPO-reference-KL conflict? (Q2)

**No — they are distinct, coexisting regularizers at different points in the pipeline (3-0).**

- The **VIB bottleneck KL** is `KL(N(μ,σ²)‖N(0,I))` on the *latent* (our `last_kl`,
  `bottleneck.py:99–100`) — an encoder-vs-prior compression term.
- **GRPO's KL-to-reference** is `KL(π_θ‖π_ref)` on the *policy's answer distribution*.

InfoRM ([arXiv:2510.13694](https://arxiv.org/abs/2510.13694), extending NeurIPS 2024) adds a VIB
KL to the *reward model's* objective (`max I_pref − β·I_bottleneck`) and benchmarks it *against*
"Standard RM with KL" across swept RL KL-penalty values — i.e. the two KLs are explicitly separate
stages. The VIB+SAC meta-RL work ([PMC9864208](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9864208/))
puts the VIB on a latent *task* representation while SAC's max-entropy term governs the policy —
again structurally distinct. And on the conflict worry itself: in KL-regularized RL, the
reference-KL's effect on diversity is governed by its **coefficient and reward scaling, not KL
direction** — the "reverse-KL = mode-seeking" VI intuition does **not** transfer [arXiv:2510.20817].

> Caveat: none of these precedents puts the VIB on a *per-modality policy adapter* as we do, so
> they are framing analogies, not architectural precedent. Open: does a strong VIB-KL effectively
> substitute for the GRPO reference-KL, and what coefficient ratio avoids both KL collapse (σ→0)
> and mode collapse?

## 3. IB-as-policy vs IB-as-regularizer (Q3)

**Evidence leans decisively to IB-as-auxiliary-regularizer (3-0).** InfoRM's concrete instance is
**IBL** (IB-Level regularization): the RL objective becomes `reward − γ·IBL`, where IBL penalizes
responses with large Mahalanobis distance from the SFT distribution in IB latent space — a
distribution-level regularizer that mitigates reward hacking and enables principled early stopping.
The policy is a *separately trained* LLM; the IB space is **not** the policy. No surveyed source
makes a VIB adapter the GRPO policy on a frozen LLM — that absence is itself a signal that the
established, lower-risk pattern is IB-as-regularizer.

**For RLVIB this suggests a hybrid rather than an either/or:** keep the VIB adapter as the trainable
module whose `z` perturbs the frozen features (as now), drive it with the GRPO policy-gradient, and
*additionally* carry an IBL-style latent-outlier penalty (our `general KL-to-base` anchor in
`anchored_dpo_step` already plays a closely related role — penalizing adapter drift from the frozen
base on broad inputs; `dpo.py:164–172`). The novelty/risk lives in "VIB-sampler = RL action
distribution," which nothing validates.

## 4. Reward design for AV grounding (Q4)

Most actionable part of the research.

- **Verifiable/rule-based rewards work for grounding — no reward model needed.** Perception-R1
  ([arXiv:2504.07954](https://arxiv.org/pdf/2504.07954), NeurIPS 2025) uses a pure rule-based
  reward (Format + Answer) with GRPO for MLLM perception: +4.2% RefCOCO+ grounding, +17.9%
  counting, OCR/detection gains on Qwen2.5-VL-3B. This supports a **yes/no correctness reward on
  AVHBench/CMM** with no learned RM (3-0).
- **But binary backfires on hallucination.** TruthRL ([arXiv:2509.25760](https://arxiv.org/abs/2509.25760),
  Meta): the binary-reward ablation gets the *highest accuracy but highest hallucination* and loses
  the ability to abstain. A **ternary** reward fixes it — because `Â_i = (r_i − mean)/std` is
  monotonic in raw reward, in any group containing both an abstention (r=0) and a hallucination
  (r=−1), **abstention always gets the larger advantage**. True by construction, 28.9% hallucination
  reduction reported (3-0).
- **OmniDPO's data recipe transfers.** OmniDPO ([arXiv:2509.00723](https://arxiv.org/pdf/2509.00723),
  AAAI 2026) builds preference pairs from **noisy/corrupted/masked video and audio variants** to
  force attention onto sensory evidence over text priors — a concrete way to construct grounding
  rewards/preferences. Same-benchmark deltas to beat: avg **+3.48% CMM, +4.23% AVHBench** (AVHBench
  F1 +5.82% on Qwen2.5-Omni). *(Author-self-reported; the replication claim was the lone 2-1 split,
  and OmniDPO fine-tunes the model rather than a frozen adapter.)*

**Recommendation:** ternary AVHBench/CMM reward — `+1` correct yes/no, `0` for an explicit
abstention/"can't tell", `−1` for a confident wrong answer (the hallucination). Watch over-abstention
(partly self-corrects via group-relative credit). Note AVHBench is yes/no, so "abstain" needs a
harness convention (e.g. a third allowed answer, or a low-margin band).

## 5. Pitfalls & the swap-DPO comparison (Q5)

- **Mode collapse is designed-in for our regime (3-0).** "KL-Regularized RL … is Designed to Mode
  Collapse" ([arXiv:2510.20817](https://arxiv.org/html/2510.20817v1)) proves low-regularization +
  equal verifiable rewards specify a *unimodal target by construction* (shown for GRPO and RLOO) —
  a property of the objective, not the algorithm. Near-deterministic frozen policy + binary
  grounding reward + weak KL is exactly this. **Mitigation:** tune KL coefficient and reward
  scaling deliberately; monitor zero-advantage groups.
- **Reward hacking**: answering with no reasoning, and verifier-format exploitation; countered by
  composite penalties (`P_answer`, `P_structural`). *But* the "penalties reduce hacking at no
  accuracy cost" claim was **refuted (1-2)** — treat them as a mitigation with a possible tradeoff
  [arXiv:2509.15557].
- **GRPO ⊁ DPO**: the "online GRPO beats offline DPO for truthfulness" claim was **refuted (1-2)**.
  A/B GRPO against the held swap-DPO `broad` recipe; don't presume a win.

**Refuted claims — do NOT repeat:** vanilla full-token GRPO viable on tiny params (0-3); GRPO
strictly beats DPO (1-2); composite penalties are free (1-2); "VIB gives 200–5000× RL
sample-efficiency" (0-3).

---

## Concrete sketch (illustrative — not yet wired into the trainer)

This mirrors the existing `*_step` functions in `src/rlvib/train/dpo.py` and reuses the real
helpers (`answer_logp_vec`, `letter_id`, `set_bypass`, `total_kl`). It is the **IB-adapter-as-policy
+ IBL-style anchor** hybrid; per §1 it is *unvalidated* and the zero-variance-group / σ-collapse
risks are real. Treat as a starting point for a GPU prototype, not a drop-in.

```python
# illustrative — additive GRPO step over the trainable bottleneck on yes/no grounding items.
# Exploration comes from the VIB sampler z = mu + sigma*eps (model.training=True), so logvar
# must NOT collapse to -inf or the group degenerates (all samples identical -> zero advantage).
import torch, torch.nn.functional as F
from rlvib.models.bottleneck import set_bypass, total_kl

def grpo_step(model, bottlenecks, optimizer, batch, *, group=8, beta_kl=0.01, lam_ref=0.05,
              r_correct=1.0, r_abstain=0.0, r_halluc=-1.0):
    """One GRPO step. Each ex = {messages, gold ('yes'|'no'), yes_id, no_id}.

    Per example: draw `group` stochastic VIB passes -> sample an answer -> ternary reward ->
    group-normalized advantage -> REINFORCE on the sampled-token log-prob (grad flows through
    the realized z via reparameterization). Plus VIB-KL (compression) and a KL-to-base anchor
    on the policy distribution (the IBL-style outlier penalty / GRPO reference-KL)."""
    optimizer.zero_grad()
    metrics = {"reward": [], "adv_std": [], "kl_vib": [], "kl_ref": []}
    for ex in batch:
        yes, no, gold = ex["yes_id"], ex["no_id"], ex["gold"]
        logps, rewards, kls = [], [], []
        for _ in range(group):                      # the GRPO group (stochastic z each pass)
            set_bypass(bottlenecks, False)
            lp = answer_logp_vec(model, ex["messages"])          # policy w/ sampled z
            kls.append(total_kl(bottlenecks))
            # 2-way (or 3-way w/ an abstain token) categorical over the answer tokens:
            cat = torch.log_softmax(torch.stack([lp[yes], lp[no]]), dim=0)
            a = torch.distributions.Categorical(logits=cat.detach()).sample()
            logps.append(cat[a])                                 # log pi(a_i | x, z_i)  [keeps grad]
            ans = "yes" if a == 0 else "no"
            rewards.append(r_correct if ans == gold else r_halluc)  # add r_abstain branch if used
        r = torch.tensor(rewards, device=cat.device)
        adv = (r - r.mean()) / (r.std() + 1e-6)     # group-relative advantage (zero if no variance)
        kl_vib = torch.stack([k if torch.is_tensor(k) else torch.zeros(()) for k in kls]).mean()

        # reference-KL on the answer dist (bottleneck bypassed = frozen base); the IBL/GRPO anchor
        with torch.no_grad():
            set_bypass(bottlenecks, True); base = answer_logp_vec(model, ex["messages"])
            set_bypass(bottlenecks, False)
        kl_ref = (base.exp() * (base - lp)).sum()   # KL(base || policy) at the answer position

        pg = -(adv.detach() * torch.stack(logps)).mean()        # GRPO policy-gradient
        (pg + beta_kl * kl_vib + lam_ref * kl_ref).backward()
        metrics["reward"].append(float(r.mean())); metrics["adv_std"].append(float(r.std()))
        metrics["kl_vib"].append(float(kl_vib.detach())); metrics["kl_ref"].append(float(kl_ref.detach()))

    torch.nn.utils.clip_grad_norm_(bottlenecks.parameters(), max_norm=1.0)
    optimizer.step()
    return {k: sum(v) / max(1, len(v)) for k, v in metrics.items()}
```

**Watch in the metrics:** `adv_std` near 0 means zero-variance groups (no learning signal — raise
`group`, raise the VIB `σ` floor, or pick harder/mixed items per group); `kl_vib`→0 means the
sampler is collapsing to deterministic (kill exploration); `kl_ref` blowing up means the adapter is
drifting off the frozen base (the swap-DPO collapse mode this anchor exists to prevent).

## Open questions (prototype targets, in priority order)

1. **Does `σ(x)·ε` produce enough output variance** for non-degenerate GRPO groups on a zero-init
   adapter, or must we add an explicit entropy/σ-floor term? (Make-or-break.)
2. **Coefficient ratio** `β_kl : λ_ref` that avoids *both* VIB-KL collapse and policy mode collapse
   on a near-deterministic policy.
3. **Abstention encoding for AVHBench** (third token vs low-margin band) and whether ternary reward
   beats binary on V→A hallucination without over-abstaining.
4. **GRPO vs the held swap-DPO `broad` recipe** head-to-head on the corrected harness — is RL worth
   the added instability, or is it best as an *IBL-style anchor* bolted onto swap-DPO?

## Sources (verified)

| # | Source | Role |
|---|--------|------|
| 1 | [arXiv:2504.20834](https://arxiv.org/html/2504.20834v1) — S-GRPO / T-SPMO (ICML 2025) | GRPO on frozen backbone + adapter; token-selective is what works |
| 2 | [arXiv:2510.13694](https://arxiv.org/abs/2510.13694) — InfoRM (ext. NeurIPS 2024) | VIB-as-RM-regularizer; IBL outlier penalty; the two-KL separation |
| 3 | [PMC9864208](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9864208/) — VIB+SAC meta-RL (Sensors 2023) | IB on latent task rep, separate from RL entropy term |
| 4 | [arXiv:2504.07954](https://arxiv.org/pdf/2504.07954) — Perception-R1 (NeurIPS 2025) | rule-based GRPO works for MLLM grounding/perception |
| 5 | [arXiv:2509.25760](https://arxiv.org/abs/2509.25760) — TruthRL (Meta) | ternary > binary reward for hallucination, by GRPO construction |
| 6 | [arXiv:2510.20817](https://arxiv.org/html/2510.20817v1) — KL-Reg RL is Designed to Mode Collapse | our regime is structurally collapse-prone |
| 7 | [arXiv:2509.15557](https://arxiv.org/pdf/2509.15557) | reward-hacking modes + composite penalties (not free) |
| 8 | [arXiv:2509.00723](https://arxiv.org/pdf/2509.00723) — OmniDPO (AAAI 2026) | DPO baseline + same-benchmark targets + noisy-modality pair recipe |

> Verification: 5 angles → 23 sources fetched → 43 claims → 25 verified → 20 confirmed / 5 killed
> → 8 findings (105 agents). Refuted-claim guardrails folded into §1, §4, §5.
