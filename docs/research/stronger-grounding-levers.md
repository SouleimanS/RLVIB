# Stronger grounding on the working Qwen3 arm — ranked levers

> Date: 2026-06-19 · Status: decision memo (forward plan after the qcond null).
> Method: 5-angle deep-research fan-out (counterfactual signal · anchor/objective hardening ·
> online RL · architecture for audio uptake · audio-necessary data + measurement) + a **3-vote
> adversarial verification** pass on 18 load-bearing claims (2 killed, corrections folded in §9).
> Companion to `../reports/02-model-and-training.md` (v0–v3), `dpo-collapse-and-fixes.md`,
> `query-conditioned-bottleneck.md` (the qcond null this follows up).

## 0. TL;DR — the ranked plan

- **The reframe that changes everything (verified): audio-ignoring lives *inside* the frozen LLM.**
  Omni-LLMs attend *less* to the media tokens under an **audio** query than under a text query, and
  prune audio tokens aggressively [2503.00059; 2605.11605]. Our VIB sits on the adapter outputs
  **before** the Thinker, so it can rewrite vision but **cannot make the LLM use audio** — exactly
  what the interpretability probe shows (generic 59% vision rewrite, audio rate ≈ 0). This is why the
  gain is modest, why it's a vision re-projection, and why query-conditioning was null.
- **And the +6.0 isn't even resolvable at n=300** (SE ≈ ±2.8pp/arm, 95% CI half-width ≈ ±5.5pp;
  detecting a 5–6pp gain at 80% power needs **n ≈ 1000–1500 per arm**) [§2]. We are tuning inside the
  noise band.
- **Ranked by payoff ÷ cost:**
  0. **Measurement first** (prerequisite, eval-only): scale n, Wilson CIs, multi-seed, val/test split,
     and an **audio-necessary split** — biased benchmarks (≈80% single-frame-solvable) mask real audio
     gains; an audio-necessary split should *amplify* them.
  1. **Harden the anchor — add NLL-on-chosen (RPO).** One loss term; provably flips the chosen logp
     from falling to rising; capability-positive. Cheapest lever.
  2. **Stronger counterfactual signal — Hungarian-hard + modality-decoupled + on-policy pairs.**
     Closest precedent (MoD-DPO) reports ~27% AVHBench AV-matching. Data + knobs.
  3. **Cross-attention *into* the LLM (audio-as-K/V), Whisper-Flamingo-style.** The only frozen-backbone
     mechanism with strong evidence of forcing genuine second-modality use — the **root-cause** fix.
  4. **Online RL (GSPO/RLOO) with a modality-counterfactual reward.** Higher ceiling, but
     frozen-adapter-only RL is *unprecedented* (risk).
  5. **Light LoRA on the LLM** — stage-2 refinement only.
- **Win condition (from interpretability):** a lever only counts if it raises **audio** uptake (the
  `vib_saliency` audio rate / audio-ablation ΔAcc), not if it just enlarges the vision rewrite or lifts
  AVHBench while CMM-HR sinks (the yes-bias artifact).

---

## 1. The reframe — why the current ceiling is low (angle 4)

- Omni-LLMs **structurally** down-weight the media stream on audio queries vs text queries
  [2503.00059, ACL'25-F], and "tend to disregard audio" — VideoLLaMA2 prunes audio 1496→10 tokens with
  no loss; decoupled pruning destroys *jointly*-informative tokens [2605.11605; 2602.04804]. So the
  failure is **downstream of our bottleneck**, inside the frozen Thinker.
- Consequence: any lever that only reshapes the *pre-LLM* code or the *offline* signal keeps hitting the
  same wall (a vision re-projection the LLM resolves). To raise genuine audio reliance we must either
  push audio *into the LLM's attention* (lever C) or accept a small unfreeze (lever E) — or pivot the
  headline (out of scope here; the user chose "stronger result").
- **Honest caveat (the central open gap):** *no published frozen-LLM method has been shown to eliminate
  audio-ignoring when audio is the only discriminative cue* (the AVQA-Hard regime). The frozen-LLM
  ceiling may genuinely be low; budget for the light-LoRA fallback.

---

## 2. Measurement first — the prerequisite (angle 5)

Cheap (eval-only) and it gates everything below; do it before chasing gains.

- **Benchmarks are visually biased:** a single muted middle frame answers ≈80% of AVQA and ≈54% of
  Music-AVQA (GPT-4o) [2509.17901]; >90% of AV-localization items are vision-solvable [2409.06709];
  text-only fine-tuning alone lifts omni audio-QA [Omni-R1, 2505.09439]. A gain measured on these is
  largely shortcut.
- **Build/eval an audio-necessary split:** text-shortcut filter (drop items a text-only pass answers)
  → modality-ablation filter (keep only items where **video-only AND audio-only both fail**) → or use
  **DAVE**, which is audio-necessary by dual-ablation design [2503.09321, NeurIPS'25 D&B]. A real audio
  gain should be *larger* here (the subtraction-of-shortcuts effect).
- **Statistics (verified):** n=300 ⇒ SE ≈ ±2.8pp, 95% CI ≈ ±5.5pp; a *difference* SE ≈ ±3.8pp, so the
  +6.0 is statistically indistinguishable from 0 in one run. Detecting 5–6pp at 80% power needs
  **n ≈ 1000–1500 per arm**. Report **Wilson** CIs (preferred for n≤500) + **multi-seed mean±std**
  (across-seed σ ≈ 1.6pp can fully explain a single +6) + **leakage-free val-select / test-report**.
- **Experiment (no training):** rerun selection eval at `LIMIT=1000` on AVHBench/CMM + a DAVE
  audio-necessary slice; pipe every cell through `aggregate_ci.py` (Wilson + across-seed). *This is
  also how we earn the right to call any lever below a "win."*

---

## 3. Lever A — harden the anchor (RPO / APO / χ²) · in-design, cheapest (angle 2)

Our mDPO anchor `−logσ(β(cp−cr)−δ)` **saturates** once satisfied and stops pushing; nothing puts a
hard floor under the chosen logp.

- **RPO — add `+λ·NLL(chosen)` (recommended).** The SFT term *provably redirects the gradient* so the
  chosen logp **rises** rather than falls; reported capability-**positive** vs DPO (MT-Bench 7.381 vs
  7.278; AlpacaEval2 LC 23.28% vs 21.15%) [2405.16436, NeurIPS'24]. One line in `anchored_dpo_step`.
- **APO-zero** — an anchored objective whose **chosen gradient is always positive** regardless of the
  rejected term (clean for our frozen-adapter regime where the reference *is* the bypassed backbone)
  [2408.06266]. **χ²-PO** — a one-line link swap giving **single-policy concentrability** (well-suited to
  our offline pairs) [2407.13399]. **KTO** — pointwise/absolute anchor (removing its asymmetry costs
  9.4/11.0 pts BBH/GSM8K, quantifying vanilla-DPO's capability tax) [2402.01306].
- **Don't lean on DPOP here:** its floor `max(0, log[π_ref/π_θ])` is **trivially satisfied at our
  zero-init start** (adapter ≈ reference), so it barely bites early [2402.13228].
- **Cost:** ~zero (loss term). **Risk:** low. **Experiment:** add a `LAMNLL` knob; run
  `LAMNLL∈{0.1,0.5}` vs the held recipe, same broad anchor, select mid-training, CIs.

## 4. Lever B — stronger counterfactual signal · in-design, high-evidence (angle 1)

Is the audio-swap gradient just too weak/easy?

- **Hungarian-hard swaps:** replace random swaps with **maximally-mismatched** audio (minimize
  caption cosine), the construction that exposes audio-blindness [2604.02605]. Hard > easy pairs
  [DA-DPO 2601.00623].
- **Modality-decoupled negatives (the strongest AV precedent):** MoD-DPO adds *invariance to
  irrelevant-modality corruption* + *sensitivity to relevant-modality corruption* → **~27% relative
  AVHBench AV-matching** [2603.03192, CVPR'26]. Add audio-drop / audio-shuffle negatives alongside the
  swap.
- **On-policy rejected rollouts:** off-policy "chosen" is ~unreachable under the implicit KL; OPA-DPO
  reaches SOTA at **4.8K** on-policy pairs [2501.09695, CVPR'25]. For us "on-policy" = re-roll rejected
  from the current bottleneck every 1–2 rounds. Mix on+off beats pure on-policy [SIMPLEMIX 2505.02363].
- **⚠️ Keep the anchor, don't CHES-filter:** our "heard vs seen" letters are near-identical (high CHES),
  the displacement-prone case [2410.08847] — but *filtering* them would delete exactly our signal, so
  rely on Lever A's anchor instead.
- **Cost:** low–med (extend `data/pairs.py`: Hungarian swap + drop/shuffle negatives + on-policy
  re-roll). **Experiment:** `PAIRS=600`, `EXP=hardneg`, decoupled negatives; CIs vs held recipe.

---

## 5. Lever C — cross-attention *into* the LLM (audio-as-K/V) · the root-cause fix (angle 4)

The strongest evidence-backed way to make a **frozen** backbone genuinely use a second modality:
**Whisper-Flamingo** injects visual K/V via **gated zero-init cross-attention** into a *frozen* Whisper
and trains only the gated layers + projection → SOTA AVSR (0.76% WER, LRS3) [2406.10082]. The direct
analogue: insert gated cross-attention adapters at frozen-Thinker layers with **audio tokens as
keys/values**, so audio is a persistent residual the LLM cannot route around (Flamingo `tanh(α=0)`
init preserves the base at step 0 [2204.14198]). LAVISH (audio→vision latent fusion in a frozen ViT)
is the lighter, pre-LLM cousin [2212.07983, CVPR'23]; a contrastive AV-alignment head can feed it
[MA-AVT 2406.04930; MEERKAT +~37% grounding, 2407.01851].

- **Why it's the right lever:** it targets the *in-LLM* suppression (§1) instead of re-skinning vision.
- **Cost:** med–high (new module + per-layer hooks; still LLM-frozen, keeps the zero-init/bypass
  identity so anchored-DPO + the free reference survive). **Risk:** med. **Experiment:** prototype
  `AudioXAttnAdapter` on ~2–4 Thinker layers, smoke-test identity-at-init, then train with the held
  anchored recipe; gate on the **audio-ablation ΔAcc** (must rise) not just AVHBench.

## 6. Lever D — online RL with a modality-counterfactual reward (angle 3)

- On-policy provably expands coverage beyond offline DPO's fixed pairs [2601.08421]; **GSPO ran on
  Qwen3-Omni-30B-A3B** [OmniVideo-R1, 2602.05847] and **GRPO on Qwen2.5-Omni** gave +5.24pp AVQA in 562
  steps/4.5K samples [EchoInk-R1, 2505.04623]; **RLOO** is a simpler-than-PPO option [2402.14740];
  off-policy **AVATAR** gives 5× sample efficiency [2508.03100].
- **The novel reward:** answer must **change when audio is swapped/removed** (modality-counterfactual
  sensitivity) + a **ternary abstention** reward (correct +/abstain 0/wrong −) [TruthRL 2509.25760],
  with GRPO group-agreement as a free uncertainty signal [TIAR 2605.25850]. No one trains this yet —
  open lane.
- **⚠️ Load-bearing risk:** *no published work does GRPO/RLOO with the LLM frozen and only an adapter
  training* — the KL/reference is defined over the full model but only the bottleneck moves, risking
  reward-variance collapse. Mitigations: reference-free (SimPO-style) objective, or accept light LoRA
  (lever E).
- **Cost:** high (rollouts, ~2–4× wall-clock). **Payoff:** potentially the highest ceiling. **Do after
  B/C show the offline ceiling.** **Experiment:** GRPO/RLOO smoke test on the bottleneck with the
  counterfactual reward; watch advantage variance + CMM guards.

## 7. Lever E — light LoRA on the LLM (angle 4)

LoRA on attn+MLP raises the grounding ceiling modestly but carries alignment-tax (forgetting) and does
**not** by itself fix the in-LLM audio suppression — so it's a **stage-2** refinement on top of C (or a
crutch for D's KL issue), not a first move. **Experiment:** only if frozen plateaus; LoRA rank 16–32 on
attn+MLP, re-check CMM PA/HR hard.

---

## 8. Ranked plan (cheapest-highest-payoff first) + the bar

1. **Measure properly** (eval-only): `LIMIT≥1000`, Wilson CIs, ≥3 seeds, val/test split, **DAVE
   audio-necessary slice**. *Earns the right to trust any gain.*
2. **RPO anchor** (`+λ·NLL(chosen)`): one knob, capability-positive, fixes the saturating anchor.
3. **Hard + decoupled + on-policy negatives** (`PAIRS=600`, MoD-DPO-style): the strongest in-design AV
   precedent (~27%).
4. **Cross-attention-into-LLM adapter**: the root-cause fix for genuine audio uptake.
5. **Online RL + counterfactual reward**: if offline plateaus; mind the frozen-adapter-RL risk.
6. **Light LoRA**: stage-2 only.

Every step is accepted only if, with CIs, it beats base **and** keeps CMM_PA≥0.90, CMM_HR≥0.780, **and
moves the audio-ablation ΔAcc** — a bigger AVHBench with flat audio-uptake or sinking HR is not grounding.

**Novelty kept intact:** modality-counterfactual *RL reward* + audio-as-K/V into a frozen LLM + the
anchored compression, evaluated on an audio-necessary split, is unoccupied and sharpens (not overlaps)
the `ib-rl-method-and-framing.md` positioning.

---

## 9. Verification notes (3-vote adversarial; corrections applied)

18 load-bearing claims; all resolve to real papers. **2 killed (≥2/3 refute):**
- **OmniVideo-R1 [2602.05847] uses GSPO, not GRPO**, on ~101K samples (88K+12K) — the
  online-RL-on-Qwen3-Omni-30B *precedent* stands; the algorithm name and the ≥4.3% Daily-Omni margin
  are corrected/dropped.
- **Omni-R1 [2505.09439] specific numbers were misattributed** (conflated Qwen2-Audio's 30.5→~44.6 with
  Qwen2.5-Omni's ~49.3 baseline) — kept only the *qualitative* finding (text-only tuning helps audio-QA
  ⇒ text shortcuts dominate).

Softened (mechanism verified, specifics not): **RPO** weight is not "η=0.005" (TRL default `rpo_alpha=1`;
MT-Bench/AlpacaEval numbers *are* verified); **LAVISH** dropped "bi-directional" (latent-token attention
bottleneck); **2503.00059** audio-vs-text attention gap verified, layer-specificity not; **APO-zero**
"decoupled" framing fine. Required-n stated as ~1000–1500/arm (votes split 900–1300 vs 1200–1500). All
other claims (MoD-DPO 27%, Whisper-Flamingo, OPA-DPO, DPOP, KTO, χ²-PO, EchoInk-R1, DAVE, OmniHalluc-L
41.55→51.09, the power math) **verified 3/3 or 2/3**. Recency caveat: the 2026 arXiv items
(MoD-DPO, "See and Hear", OmniHalluc-L, OmniVideo-R1) are very new — re-check before paper submission.

---

## Selected bibliography (grouped; ids verified this pass)

**Counterfactual / preference signal** — "Do AV-LLMs Really See and Hear?" (2604.02605); MoD-DPO
(CVPR'26, 2603.03192); OPA-DPO (CVPR'25, 2501.09695); mDPO (EMNLP'24, 2406.11839); SIMPLEMIX
(2505.02363); DA-DPO (2601.00623); MBPO (2506.08022); CHES/displacement (ICLR'25, 2410.08847).

**Anchor / objective** — RPO (NeurIPS'24, 2405.16436); DPOP/Smaug (2402.13228); KTO (2402.01306);
APO (2408.06266); χ²-PO (2407.13399); IPO (2310.12036).

**Online RL** — EchoInk-R1 (2505.04623); OmniVideo-R1/**GSPO** (2602.05847); RLOO (ICLR'25, 2402.14740);
AVATAR (2508.03100); on-policy coverage (2601.08421); TruthRL (2509.25760); TIAR (2605.25850).

**Architecture (audio uptake, frozen)** — Whisper-Flamingo (Interspeech'24, 2406.10082); Flamingo
(NeurIPS'22, 2204.14198); LAVISH (CVPR'23, 2212.07983); MA-AVT (CVPRW'24, 2406.04930); MEERKAT (ECCV'24,
2407.01851); omni audio-ignoring (2503.00059); audio token pruning (2605.11605, 2602.04804).

**Data / measurement** — DAVE (NeurIPS'25, 2503.09321); "Does Audio Matter" + AVQA-Hard (2509.17901);
Omni-R1 (2505.09439); Visual Biases in AVSL (2409.06709); Look-Listen-Answer (NeurIPS'24, 2404.12020);
AVHBench (ICLR'25, 2410.18325); OmniHalluc-L/MPRC (2606.03614).
