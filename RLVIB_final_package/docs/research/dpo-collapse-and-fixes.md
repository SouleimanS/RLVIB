# Why the swap-DPO bottleneck collapsed — and how to retrain without breaking the model

Deep-research synthesis (5 angles: DPO collapse mechanism · anti-collapse variants · multimodal
preference optimization · capability retention / alignment tax · collapse detection). Sourcing
caveat: agents hit HTTP 403 on direct PDF fetches; mechanism claims read from released code are
high-confidence, exact benchmark decimals medium-confidence.

## TL;DR — root cause and the one-line fix

Our DPO loss is a log-sigmoid of the **difference** `(chosen − rejected)`, so it can be "won" by
pushing **both** answer log-probs down. The **only** regularizer we had was a KL on the *bottleneck
latent* — nothing anchored the model's **output distribution** to the frozen base. The bottleneck
sits **always-on** in the forward path and DPO's objective is **input-blind** (it only touches the
preference pairs), so the freed probability mass collapsed onto a single degenerate token ("no") on
*every* input. This is **likelihood displacement** (Razin et al., ICLR 2025), and its modality form is
named **"Visual/Audio Anchor Collapse"** (ACPO, 2026): eroded chosen-likelihood makes the model abandon
the modality for the language prior. Our yes/no + single-letter-MC answers are the **worst case** (near-
identical chosen/rejected embeddings = high "CHES" = maximal displacement). The fix is to **anchor the
chosen-answer likelihood to the frozen base AND anchor the output distribution to the base on general
inputs** — neither of which our loss did.

## 1. The mechanism (cited)

- **Likelihood displacement.** DPO routinely *lowers* the chosen response's probability; the canonical
  example "prefer *No* over *Never* can sharply increase *Yes*" is our exact failure. Driven by similar
  chosen/rejected embeddings (CHES score); worst on low-edit-distance pairs.
  (Razin et al., ICLR 2025, [2410.08847](https://arxiv.org/abs/2410.08847); Pal et al. DPOP/Smaug,
  [2402.13228](https://arxiv.org/abs/2402.13228).)
- **The squeezing effect.** Off-policy DPO run too long redistributes probability mass onto the already-
  dominant token, hollowing out the rest → a constant-answer policy.
  (Ren & Sutherland, ICLR 2025, [2407.10490](https://arxiv.org/abs/2407.10490).)
- **No absolute anchor → OOD drift.** DPO can assign higher-than-reference probability to OOD responses;
  PPO avoids this with an *explicit* per-token KL. DPO's KL is implicit and only constrains the
  preference pairs — general/held-out inputs are unconstrained.
  (Xu et al., ICML 2024, [2404.10719](https://arxiv.org/abs/2404.10719).)
- **Off-policy "chosen" may be unreachable.** If the audio-consistent answers weren't sampled from the
  base, the implicit KL can't raise them, so DPO satisfies the margin only by crushing the rejected
  neighborhood → collapse. (OPA-DPO, CVPR 2025, [2501.09695](https://arxiv.org/abs/2501.09695).)
- **Always-on adapter + input-blind loss.** A frozen base + tiny adapter does *not* inherently protect
  general ability: the adapter transforms representations for *all* inputs, and nothing in DPO penalizes
  its effect off-target. (Alignment-tax / forgetting literature, below.)
- Alignment tax is normally *modest* (a few points); our 100×-scale collapse needs the displacement +
  squeezing + no-anchor mechanisms, not routine forgetting.
  (Lin et al., EMNLP 2024, [2309.06256](https://arxiv.org/abs/2309.06256).)

## 2. The fixes, ranked (highest leverage first)

1. **mDPO anchor term (the missing ~3 lines).** Add `L_anchor = −log σ(β·(logπ_chosen − logπ_ref,chosen))`
   — a one-sided preference of the chosen answer against the frozen reference that **pins the chosen
   log-prob at or above the base**, directly preventing the collapse. mDPO is *literally our setup*
   (frozen MLLM + modality-conditional DPO); our audio-swap is its "conditional preference" term.
   (Wang et al., mDPO, EMNLP 2024, [2406.11839](https://arxiv.org/abs/2406.11839), verified vs released
   `mdpo_trainer.py`.) Equivalent: **DPOP** `+λ·max(0, logπ_ref,chosen − logπ_chosen)`
   ([2402.13228](https://arxiv.org/abs/2402.13228)), or an **NLL-on-chosen / SFT** term (RPO).
2. **Explicit output-KL / distillation to the *bypassed base* on a mix of GENERAL inputs.** Add
   `λ_kl·KL(π_base(·|x) ‖ π_policy(·|x))` over general AVQA + text prompts where the adapter should be
   identity. Because the base is frozen, `π_base` is **free** (set the bottleneck to bypass) — exact
   target, one extra forward. This is the frozen-model analog of InstructGPT's **PPO-ptx**
   ([2203.02155](https://arxiv.org/abs/2203.02155)) and **Learning-without-Forgetting**
   ([1606.09282](https://arxiv.org/abs/1606.09282)). *This is the constraint DPO structurally lacked.*
3. **ACPO asymmetric gradient.** Scale down **only the rejected** reward's gradient, keep the chosen
   distribution as a gradient-stable anchor — the one method shown to restore chosen-reward *and* keep
   general leaderboards (MMBench/MMStar) intact. ([2603.22165](https://arxiv.org/abs/2603.22165).)
4. **On-policy data + more of it + de-dup.** Sample/score chosen answers reachable by the base
   (OPA-DPO-style), use *far* more than ~200 pairs, diverse; CHES/edit-distance-filter near-identical
   pairs. (Tajwar et al., ICML 2024, [2404.14367](https://arxiv.org/abs/2404.14367); Razin et al.)
5. **Gate the adapter on audio presence** (architectural, durable cure): make it identity on text-only /
   general inputs so it *cannot* steer them.
6. **Post-hoc adapter-scaling dial** (`α·Δadapter`, α=0 → exact base) to trade capability vs. tax and
   recover instantly if a benchmark regresses (WiSE-FT / model-soup analog,
   [2109.01903](https://arxiv.org/abs/2109.01903)).

**Drop-in objective swaps** (one TRL `loss_type` flag — each has the absolute anchor DPO lacks):
- **IPO** ([2310.12036](https://arxiv.org/abs/2310.12036)): regress the log-ratio margin to a *finite*
  target `1/(2β)` instead of +∞, so the KL keeps biting and the policy can't collapse to a deterministic
  answer — the cleanest fix for our high-CHES (near-deterministic) yes/no preferences.
- **KTO** ([2402.01306](https://arxiv.org/abs/2402.01306)): pointwise (binary desirable/undesirable, no
  pairs), anchored to an absolute reference point `z0 = KL(π‖π_ref)`, so you *cannot* win by dragging
  both log-probs down; robust to label imbalance (handles ~90% fewer desirable examples).
- **APO-zero** ([2408.06266](https://arxiv.org/abs/2408.06266)): per-side anchored sigmoids that push
  **chosen ↑ / rejected ↓ in *absolute* terms** (use when the base underperforms the chosen answers —
  likely our case). APO's framing names our failure: DPO leaves the "both ↓" scenario underspecified.

## 3. Monitoring & model-selection protocol (so collapse is caught in steps, not post-hoc)

Log **every step** (all ~free):
- **Answer-label histogram** on a fixed probe set (fraction yes/no, or letter distribution). *The single
  signal that would have caught our 99%-"no" model.* Alarm if any label > ~70–80% or drifts to 100%.
- **Absolute chosen AND rejected log-probs, separately** — not just `rewards/margins`/`accuracies`,
  which look perfect *during* the collapse. Chosen logp trending below the reference = displacement.
- **KL(π ‖ π_ref)** as the budget axis — true quality peaks then declines as KL grows (Gao et al.,
  overoptimization scaling laws, [2210.10760](https://arxiv.org/abs/2210.10760)).

Gate **model selection on a small but DIVERSE held-out OOD suite** (AVHBench/CMM/DAVE slices), **never**
on the in-distribution proxy (`heard_rate`) — capabilities move in opposite directions
(Ivison et al., NeurIPS 2024, [2406.09279](https://arxiv.org/abs/2406.09279)). **Below-chance MCQ is a
hard alarm** (systematic bias, not noise). Stop / roll back the moment any trips.

## 4. Concrete corrected recipe

```
loss = L_pref            # standard DPO: chosen vs rejected answer
     + L_cond            # conditional: logp(ans | TRUE audio) > logp(ans | swapped/muted audio)  [our swap]
     + λ_a · L_anchor    # −logσ(β·(logπ_chosen − logπ_ref,chosen))    pin chosen ≥ base   [fix #1]
     + λ_k · KL(base ‖ policy on a mix of GENERAL inputs)               anchor outputs     [fix #2]
   (+ λ_n · NLL(chosen)) # optional SFT anchor
```
- β ≈ 0.1–0.5 (stronger reference anchoring than we used); λ_a ≈ 1, λ_k tuned so general-input KL stays
  small; mix **5–15% general data**; **many** diverse pairs, on-policy if possible.
- **Select** the checkpoint at the held-out-benchmark peak, not the `heard_rate` peak.
- Keep the α-scaling dial for post-hoc recovery.

## 5. Regime choice (we are open to any)

Try the **anchored mDPO recipe within frozen-base + small adapter first** (cheap, preserves the design).
If it stays fragile, the literature points to **on-policy** optimization (PPO/GRPO with an *explicit* KL
to base) and/or **LoRA** (forgets ~30× less than full FT but not zero) as the more robust regime — these
trade simplicity for the explicit, sampled KL leash DPO lacks. On-policy + explicit KL is the
belt-and-suspenders version of fixes #1–#2.

## 6. Novelty note

No published work isolates **"audio-ignoring collapse under modality-conditional DPO on a frozen
AV-LLM"** — the collapse theory (mDPO, ACPO, Razin) is all in the *vision* literature, and the AV-DPO
works (video-SALMONN 2/o1) treat audio+visual jointly without our audio-swap ablation. The anchored,
collapse-instrumented version of our method is genuine contribution space — provided we *prove* general
capability survives (cite ACPO / BPO / HA-DPO, which actually report retention).
