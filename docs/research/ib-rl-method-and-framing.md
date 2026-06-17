# Training Method & Framing — Information Bottleneck × Reinforcement Learning

> Date: 2026-06-17 · Extends `docs/research/grounding-audio-to-video.md`.
> Method: 5-angle deep-research fan-out on the **IB×RL intersection + novelty/framing**,
> plus a 14-citation verification pass (all real; 2 metadata corrections folded in — §6).

## 0. TL;DR

- **Conceptual hook (why IB and RL are *one* idea here):** the KL-to-reference penalty in
  RLHF/DPO/GRPO already imposes a **rate**-like constraint at the *policy* level
  (β = rate budget; the optimal policy is the Boltzmann tilt `π_ref·exp(r/β)`). Our move:
  make the information bottleneck **explicit and representational**, on the **audio-visual
  fusion code Z** — so the compression that forces genuine cross-modal grounding happens
  *where the modalities meet*, and **calibrated abstention is read off the bottleneck's rate.**
- **Method (working name RLVIB):** a *variational* bottleneck on the fused AV adapter outputs
  (frozen LLM + encoders), trained **(1) supervised VIB warmup → (2) counterfactual-modality
  preference/RL (DPO→GRPO) → (3) IB-rate + semantic-confidence abstention** on AV-mismatch.
- **Novelty (verified open):** *explicit VIB on the AV fusion bottleneck* + *counterfactual-
  modality RL* + *IB-rate abstention* — no work combines all three. Closest neighbours each
  miss one axis (§3–4).
- **Title front-runner:** **"RLVIB: Reinforcement-Learned Variational Information Bottleneck
  for Grounded and Abstaining Audio-Visual QA."**

---

## 1. The IB ↔ RL bridge (the conceptual spine)

- The KL-regularized objective `max E[r] − β·KL(π‖π_ref)` has the closed-form optimum
  `π*(y|x) ∝ π_ref(y|x)·exp(r(x,y)/β)` and value `β·log Z(x)` — a **free-energy /
  rate-distortion** form; **β is the rate budget**. Increasing β tightens the rate (stay near
  `π_ref`); decreasing it spends rate to chase reward. (Standard; energy-based lens.)
- Galashov et al. (ICLR'19) **connect** KL-regularized RL to the Information Bottleneck and to
  capacity-limited information transfer to a default policy. *Caveat (verified):* this is a
  *connection/analogy*, **not** a proven "KL = IB Lagrangian dual" theorem — state it softly.
  [1905.01240]
- DPO is **implicit MI maximization**: InfoPO (NAACL'25) shows DPO under Bradley-Terry is a
  contrastive InfoNCE lower bound on `I(chosen; label | x)`; an explicit MI estimator beats it
  [2505.08507]. GRPO ≈ DPO contrastively ("It Takes Two…") [2510.00977].
- So the *policy* already does implicit, **behavioral** compression. We add **explicit,
  representational** compression on Z (an orthogonal pressure):
  - **InfoRM** (NeurIPS'24): VIB *in the RLHF loop* is tractable ≤7B, and the IB latent yields
    a reward-hacking / overoptimization signal (Cluster Separation Index) — directly
    repurposable as an **abstention trigger** [2402.09345].
  - **IBRO**: the IB principle gives a **one-line** GRPO add (advantage-weighted token entropy)
    [2507.18391]. **IIB-LPO**: IB as a trajectory filter prevents RLVR exploration collapse
    (both text-only) [2601.05870].

**Takeaway:** "IB + RL" is not two stapled ideas — RLHF *is* an implicit rate machine; we make
the bottleneck explicit, multimodal, and grounding-aware.

---

## 2. The method (RLVIB)

**Architecture.** Replace the v0 deterministic `ResidualBottleneck` with a **variational**
bottleneck `q(z | a, v)` on the per-modality adapter outputs (Qwen3-Omni
`audio_tower.proj2` / `visual.merger`; analogous for Qwen2.5-Omni / VideoLLaMA2); LLM +
encoders frozen. The bottleneck is the **policy's trainable representation** *and* the **IB's Z**.

**Stage 1 — supervised VIB warmup.** Train only the bottleneck on AVQA labels with
`L = L_CE + β·KL(q(z|a,v) ‖ p(z))`, plus:
- a **δ-VAE minimum-rate floor** `KL ≥ δ` so Z doesn't collapse before RL [1901.03416];
- a **modality-leakage IB term** that penalizes modality-specific info in the shared code
  (anti-shortcut) [2506.04870 — *general* multimodal, not AV-specific];
- **cyclical β** annealing to avoid KL-vanishing phase transitions [Fu et al., NAACL'19].

**Stage 2 — counterfactual-modality preference/RL.** Build pairs by modality intervention:
*chosen* = correct answer with intact A+V; *rejected* = the answer under **dropped / shuffled /
swapped-from-another-clip** audio or video (forces genuine fusion). Train the bottleneck with
**conditional DPO first** (plain DPO learns text shortcuts — mDPO), then **GRPO**. Key knobs:
- the **explicit VIB rate on Z is orthogonal** to the policy KL — keep both, on **decoupled β**
  (β_VIB on a slow outer loop; β_DPO **dynamic, batch-level** [2407.08639]);
- consider **χ²** regularization instead of pure KL for the offline pairs (better
  concentrability) [2407.13399];
- for GRPO, add the **IBRO** token-entropy IB term [2507.18391].

**Stage 3 — abstention on AV-mismatch.** Combine three signals into an abstain decision:
the **IB rate `I(X;Z)`** (high forced rate ⇒ hard-to-compress ⇒ uncertain), **InfoRM-style CSI**
(latent outlier) [2402.09345], and a **FISCORE-style semantic-confidence RL reward** (cluster
sampled answers; reward abstention on low-confidence) [2510.24020]. Evaluate on **AVHBench
AV-Matching** and **OMD-Bench** [2603.27187].

**Anti-collapse guardrails (the design must respect these):**
- KL-regularized RL is **structurally mode-collapse-prone** [2510.20817]; stacking a VIB rate
  compounds it → keep the δ min-rate floor + **adaptive entropy coefficient** [2510.10959].
- Watch CSI + policy entropy jointly; if CSI spikes (reward hacking), down-weight the RL reward.

---

## 3. Novelty landscape & the gap

|  | exists? | who | misses |
|---|---|---|---|
| IB on multimodal fusion (no RL) | ✅ | MIB, CIB-VQA, **AdaVIB** (AAAI'25), MCIB-sarcasm, **Vittle** (VIB in a VLM) | no RL; mostly vision-only / classification |
| RL on AV grounding (no IB) | ✅ | **EchoInk-R1**, **AURORA**, **OmniVideo-R1**, AV-Reasoner | plain correctness reward; no IB; no abstention |
| IB × RL together | ✅ | MIB-RL (**robotics** sensors), **IBRO**/**IIB-LPO** (**text** RLVR), **InfoRM** (VIB on the *reward model*) | not AV; not on a fusion bottleneck |
| RL for calibrated abstention | ✅ | GRACE, Abstain-R1 (**text/RAG**) | not multimodal; not IB |
| **explicit VIB on AV fusion + counterfactual RL + IB-rate abstention** | ❌ | — | **the open lane** |

**Defensible contribution:** the *combination* — a VIB on the **audio-visual fusion**
representation, **preference/RL-trained** with **counterfactual-modality** signals, with
**abstention read off the IB rate**. The IB×RL precedent exists only for *text* (IBRO/IIB-LPO)
or *robot sensors* (MIB-RL); extending it to AV-MLLM grounding + abstention is unoccupied.

---

## 4. Scoop risks (all verified REAL) & how we differentiate

- **EchoInk-R1** [2505.04623] — GRPO on **Qwen2.5-Omni** for AV reasoning. ⇒ **RL-alone is taken.**
  Differentiate via the *explicit IB architecture* + abstention, not "GRPO for AVQA."
- **Vittle / VIBT** [2505.13946] — variational IB *inside an MLLM* (vision-only, no RL).
  ⇒ **IB-alone is taken; highest scoop risk.** Differentiate via *audio-visual* + *RL* +
  *counterfactual reward* + *abstention*.
- **OmniVideo-R1** [2602.05847] — RL AV with "modality-attentive fusion" (IB-adjacent, Feb'26);
  **AURORA** [2508.02149] — GRPO for Ref-AVS; **OmniHalluc-L** [2606.03614] — Qwen2.5-Omni
  AV-hallucination *calibration* but **inference-time, no IB/RL** (Jun'26). All adjacent; none
  train an explicit IB + RL + abstention.
- **Implication:** lean the paper on the **triple combination + the IB-rate abstention** (the
  freshest open angle — calibrated AV-mismatch abstention has *no* trained IB/RL solution), and
  **move quickly** — the area is heating up month-over-month.

---

## 5. Title / framing options

1. ⭐ **"RLVIB: Reinforcement-Learned Variational Information Bottleneck for Grounded and
   Abstaining Audio-Visual QA"** — names all three novel axes (RL + VIB + AV grounding) and the
   abstention contribution; acronym fits the "R1"-era naming. *Recommended.*
2. **"When to Trust Both Eyes and Ears: Information-Bottleneck Reinforcement Learning for
   Reliable Audio-Visual Grounding"** — reliability-first hook; foregrounds abstention.
3. **"Making the Bottleneck Explicit: From the RLHF KL to an Audio-Visual Information
   Bottleneck for Grounded QA"** — leads with the IB↔RL conceptual contribution (riskier, more
   theory-forward).
4. **"Grounding Audio in Video via a Reinforcement-Learned Information Bottleneck"** — clean and
   general; drop "abstaining" if abstention becomes a secondary result.

If only **one** keyword survives: keep **Information Bottleneck** as the *mechanism* and frame
RL as the *training signal* — the IB is the more differentiated, less-crowded anchor (RL-for-AV
is already a crowded "R1" field; IB-on-the-AV-fusion-bottleneck is not).

---

## 6. Verification notes (corrections applied)

All 14 spot-checked citations are **real**. Two corrections folded in above:
- `1905.01240` (Galashov, ICLR'19): supports a *connection* to the IB, **not** a "KL = IB
  Lagrangian-dual" theorem — softened to the rate-distortion/free-energy reading.
- `2506.04870` (ICML'25): experiments are on **synthetic vision** disentanglement (DSprites,
  MPI3D), **not** audio-visual — cited only for the modality-leakage IB idea, not as AV evidence.
- `2502.20750` (AdaVIB) confirmed AAAI'25; all scoop-risk papers (Vittle, OmniVideo-R1, AURORA,
  OmniHalluc-L) confirmed real.

---

## 7. Selected bibliography (grouped; ✓ = spot-verified this pass)

**IB ↔ RL / info-theoretic alignment** — Galashov "Information asymmetry in KL-regularized RL"
(ICLR'19, arXiv:1905.01240 ✓); InfoRM (NeurIPS'24, 2402.09345 ✓); IBRO "Revisiting LLM Reasoning
via IB" (2507.18391 ✓); InfoPO (NAACL'25, 2505.08507 ✓); "It Takes Two: GRPO is Secretly DPO"
(2510.00977 ✓); χPO (ICLR'25, 2407.13399 ✓); β-DPO (NeurIPS'24, 2407.08639 ✓); IIB-LPO
(2601.05870 ✓); Catastrophic Goodhart (NeurIPS'24, 2407.14503); KL-RL mode collapse (2510.20817).

**IB in RL representation learning** — IBAC (NeurIPS'19, 1910.12911); DRIBO (ICML'22, 2102.13268);
Dynamics-generalization IB (2008.00614); RPC (NeurIPS'21, 2109.03214); InfoBot (ICLR'19,
1901.10902); MIB-RL multi-sensor (Neural Networks'24, 2410.17551).

**IB for multimodal / MLLM** — Vittle/VIBT (2505.13946 ✓); AdaVIB (AAAI'25, 2502.20750 ✓);
Aligning-MM-reps-via-IB (ICML'25, 2506.04870 ✓ — synthetic vision); MCIB conditional-IB sarcasm
(AAAI'26, 2508.10644); VIB-Probe (2601.05547); Attention Bottlenecks for Multimodal Fusion
(NeurIPS'21, 2107.00135).

**RL for AV grounding (competitors)** — EchoInk-R1 (2505.04623); OmniVideo-R1 (2602.05847 ✓);
AURORA (2508.02149 ✓); AV-Reasoner (2506.05328); Meerkat (ECCV'24, 2407.01851).

**Abstention / reliability** — GRACE (2601.04525); Abstain-R1 (2604.17073); FISCORE
(2510.24020); OmniHalluc-L (2606.03614 ✓); OMD-Bench (2603.27187); ACR "Knowing When to Answer"
(2602.04924).

**Stability / scheduling** — δ-VAE (ICLR'19, 1901.03416); Cyclical annealing (NAACL'19,
N19-1021); adaptive entropy coefficient (2510.10959).
