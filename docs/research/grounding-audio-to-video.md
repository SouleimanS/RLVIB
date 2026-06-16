# Grounding Audio to Video in Multimodal LLMs — Research Synthesis

> Date: 2026-06-16 · Status: design memo + literature review
> Method: 5-angle deep-research fan-out (grounding/bias, information bottleneck,
> RL/abstention, architectures/freezing, datasets/metrics) + a citation-verification
> pass on the 10 load-bearing references. All 10 verify as **real papers**; metadata
> corrections are folded in below and flagged in §10.

## Project framing

Build an audio-visual MLLM that **grounds audio in video** — i.e. answers that
**(a)** genuinely require *both* modalities (no video-only / language-prior
shortcut) and **(c)** **abstain** when evidence is insufficient or the two
modalities are mismatched. Constraints: **frozen LLM** + trainable
bottleneck/projector; compare **Qwen3-Omni** vs **VideoLLaMA2**; **DPO first,
GRPO later**; clean-room rebuild (no reuse of prior corrupted code/AVQA copy).

---

## Locked decisions (2026-06-16)

- **v0 base model: Qwen3-Omni-30B-A3B** (Thinker only — skip the Talker/Code2Wav
  speech stack; this is text-answer QA). Freeze the encoders + Thinker; train the
  fusion bottleneck on the **audio + vision adapter outputs (pre-Thinker)**; add
  LoRA only if v0 plateaus (MoE-LoRA is fiddly — avoid until needed).
  **VideoLLaMA2 = comparison arm (arm 2).**
- **Eval suite:**
  - **AVHBench** [arXiv:2410.18325] — AV cross-modal hallucination + AV-matching
    (matching doubles as an AV-mismatch / abstention probe).
  - **CMM — "The Curse of Multi-Modalities"** [arXiv:2410.12787] — hallucination
    across language/visual/audio; diagnoses unimodal-prior overreliance + spurious
    cross-modal correlation (the anti-shortcut axis).
  - **DAVE** [arXiv:2503.09321] (recommended) — both-modalities-required QA →
    modality-ablation **ΔAcc** headline (AVHBench/CMM are both hallucination-axis
    and don't give ΔAcc directly).
- **Risk to watch:** Qwen3-Omni is Sept-2025-new → tooling maturity; the Step-1
  smoke test gates everything downstream.

---

## Step 1 results — Qwen3-Omni runs (2026-06-16)

Smoke test (`scripts/smoketest_qwen3omni.py`) **passes** on one H200.

- **Loads thinker-only**, ~63.4 GB (the `UNEXPECTED talker.*/code2wav.*` keys are
  the skipped speech stack — expected with `enable_audio_output=False`).
- **Env recipe that works:** transformers 5.12.1, plus `torchvision`, `ffmpeg`,
  `eva-decord` with `FORCE_QWENVL_VIDEO_READER=decord`; cast **float** processor
  outputs to bf16 (leave int ids/grids alone) before `generate`.
- **Bottleneck attach points (confirmed via forward hooks):**
  - audio → `thinker.audio_tower.proj2` (`Linear`) → `(T_a, 2048)` (~85 tokens / 3 s clip)
  - vision → `thinker.visual.merger` (`Qwen3OmniMoeVisionPatchMerger`) → `(T_v, 2048)` (~3456 tokens)
  - Both emit **2048-d** token streams (the Thinker hidden size); features are
    scattered into `inputs_embeds` at the audio/image placeholder positions.
- **Audio is used:** AV vs video-only answers differ under greedy decode
  (`Answer changed when audio dropped -> True`).

**=> the fusion bottleneck is a module over two 2048-d token sequences**
(audio `T_a×2048`, video `T_v×2048`), inserted by wrapping/replacing those two
submodules so their outputs pass through it before entering the Thinker.

---

## Step 2 results — frozen Qwen3-Omni baselines (2026-06-16)

Eval harness (`rlvib.eval.run_{avhbench,cmm,dave}`) works end-to-end; parse rates
~1.0. Numbers below are 100-sample sanity slices unless noted (full runs pending
to tighten CIs); the *patterns* are the signal.

**DAVE — modality-ablation ΔAcc (the grounding headline).** Same MC question,
media swapped per mode (n=100 each; effectively ~4-way, chance 0.25):

| mode | acc |
|---|---|
| audio_visual_alignment (video+audio) | 0.37 |
| visual_only (silent video) | 0.34 |
| audio_only | 0.32 |
| text_only (language prior) | 0.25 |

=> text_only sits at chance (no language shortcut — DAVE isn't text-solvable).
Audio alone helps a little (0.32 > 0.25). **But AV − visual_only is only +0.03**:
with video present the model barely uses audio — a video-dominant shortcut. Growing
**AV − visual_only** is the bottleneck's explicit target.

**AVHBench (100-sample slice).** overall 0.70 | AV-Matching 0.62 (weakest) |
Video-driven Audio Hallucination 0.65 | Audio-driven Video Hallucination 0.88.
=> strong on video, weak on audio + correspondence.

**CMM visual-language (100, video-only slice).** PA 0.98 / HR 0.80 / acc 0.89.
=> nails present objects but hallucinates ~20% of absent ones via spurious
correlation. (Audio subsets pending the full run.)

**Three headroom metrics the bottleneck must move:** DAVE `AV − visual_only`
(~+0.03; audio under-used), AVHBench AV-Matching (0.62), CMM HR (0.80).

---

## Step 4 — bottleneck architecture validated (2026-06-16)

`scripts/smoketest_bottleneck.py` passes on Qwen3-Omni: zero-init residual
bottlenecks on `audio_tower.proj2` + `visual.merger` are **identity at init**
(answer byte-identical), **16.8M trainable params with 0 trainable elsewhere**
(LLM + encoders frozen), and **gradients flow** through the frozen Thinker into the
bottleneck (fc2 grad norms ~4.7 / ~24, LM loss 17.1). The frozen-LLM +
trainable-bottleneck path is green. `attach_bottlenecks(model)` is model-agnostic
(`hidden_dim` per model: 2048 Qwen3, 3584 Qwen2.5 / VideoLLaMA2). Next:
counterfactual-DPO training (Step 5).

---

## 0. TL;DR — decisions

1. **Drop the *video-only* VIB.** A single-stream VIB compresses visual
   distractors but cannot force the answer to need audio, and a "keep-shared"
   variant would *discard audio-unique* info (timbre/pitch). Replace with a
   **conditional, per-modality IB on the fused AV stream** + a
   **synergy-preserving** term.
2. **RL is the grounding *supervision*, IB is the *prior*.** Use
   **conditional/per-modality DPO** (plain multimodal DPO learns text shortcuts),
   with preference pairs built from **modality counterfactuals** (drop / shuffle /
   swap-from-another-clip audio or video). Move to **GRPO** when a programmatic
   grounding+abstention reward is ready.
3. **Abstain on AV-mismatch is your novelty lane** — consider promoting it from a
   *metric* (your current Q4 choice) to a *first-class trained output*.
4. **Base model: VideoLLaMA2 for the v0** (clean, separable audio/video branches);
   Qwen3-Omni as the second arm. Freeze encoders + LLM; train the fusion
   bottleneck on the per-modality connector outputs; optional LoRA on attn+MLP.
5. **Data: abandon AVQA.** Evaluate on **DAVE / OmniVideoBench**
   (grounding-guaranteed), scale-pretrain on **VALOR-1M**, optionally add
   **AVSBench/AVE** for grounding supervision; build an **audio-necessary split**
   via text-shortcut → modality-ablation filtering.
6. **Metrics:** grounding = ΔAcc(AV−V-only, AV−A-only); abstention =
   Coverage@Risk + AURC/E-AURC + Abstain-ECE.

---

## 1. The problem is real and under-measured

- AV-LLMs **routinely ignore audio**, answering from vision alone even when audio
  is present and relevant [Aligned Better Listen Better, arXiv:2504.02061].
- Standard AVQA benchmarks are **visually biased / unimodally solvable**; in some
  AV-localization benchmarks vision-only models *beat* AV baselines
  [Visual Biases in AVSL, arXiv:2409.06709]. MUSIC-AVQA has severe answer skew
  (>90% "yes" on some templates) [DAVE, arXiv:2503.09321].
- Accuracy ≠ grounding: AURA reports models hitting high answer accuracy while
  **reasoning factual-consistency stays low**, i.e. right answers via wrong
  (shortcut) reasoning [AURA, arXiv:2508.07470].
- Three biases co-exist (language-prior + visual-only + audio-only) and **fixing
  one inflates another** — single-modality debiasing is insufficient
  [Look-Listen-Answer, NeurIPS 2024, arXiv:2404.12020; Cross-Modality Bias causal
  view, arXiv:2305.19664].

**Implication:** evaluation must use *both-modalities-required* items and
*process-level* checks, not raw accuracy.

---

## 2. Information Bottleneck: verdict on the hypothesis

**A video-only VIB is principled but insufficient for grounding.** Evidence:

- **Single-stream VIB removes intra-stream distractors** (DRIBO suppresses video
  background shortcuts [arXiv:2102.13268]) — useful, but it cannot make the answer
  *depend on audio*.
- **Vanilla / joint IB discards *synergy*** — the cross-modal binding that *is*
  grounding. MRdIB shows joint-compression IB loses synergistic signal
  [arXiv:2509.20225]; CoMM recovers unique+synergy via Partial Information
  Decomposition and beats contrastive baselines [CoMM, ICLR 2025].
- **"Keep-shared" MV-IB throws away modality-unique info** (audio timbre/pitch
  has no visual correlate) — theoretically baked into MV-IB
  [Federici et al., ICLR 2020, arXiv:2002.07017].
- **Per-modality VIB with *adaptive β* > single joint bottleneck.** OMIB derives a
  per-modality regularization-weight bound [ICML 2025, arXiv:2505.19996]; CAL's
  Asymmetric-IB sub-component sets per-modality compression by *contribution*
  (AVE 74.21 / CREMA-D 79.30 / KS 74.82) [arXiv:2510.26289]; IBMEA shows per-stream
  IB + contrastive fusion is additive [arXiv:2407.19302].
- **β is fragile** (Narrowing-IB: ~3× degradation across β [ICLR 2025,
  arXiv:2502.14889]) and an **RL loop makes it worse** — a fixed β over-compresses
  early / under-compresses late as the reward shifts.
- ⚠️ **Epistemic caveat:** IB compression and generalization are causally decoupled
  in ReLU nets — VIB may help as a **stochastic/noise-injection regularizer**, not
  literal information compression [Saxe et al., 2019]. Do **not** rest the paper's
  claim on the IB *mechanism*; claim the *behavioral* grounding result.

### The reframed objective
Compress video V into code Z keeping only the bits that **add predictive value
given the audio** — informally `min I(Z;V | A)` while `max I(Z;Y | A)` — i.e. a
**conditional IB**, plus a **synergy-preserving** term so joint-only features
survive. Closest prior art is a **conditional IB for shortcut learning scoped to
sarcasm detection** [arXiv:2508.10644] — *not* AV grounding, so this lane is open.
A VIB-inside-an-MLLM precedent exists but on *attention-head outputs* for VLM
hallucination, not a fused-AV grounding bottleneck [VIB-Probe, arXiv:2601.05547].

**IB vs alternatives:** IB and contrastive (InfoNCE) are **complementary, not
substitutes** — InfoNCE aligns but leaves a "modality gap" and doesn't penalize
modality-specific redundancy [Mind-the-Gap, NeurIPS 2022, arXiv:2203.02053; CIBR,
arXiv:2503.24182; Aligning via IB, ICML 2025, arXiv:2506.04870]. Counterfactual
data augmentation operates at the *data* level and can run at *inference* (e.g.
"what if audio absent") — complementary to IB's *loss-level* compression
[Implicit Counterfactual AVS, ICCV 2025]. No head-to-head IB-vs-counterfactual
comparison exists for AV shortcut reduction → combining them is defensible.

---

## 3. RL / preference optimization

- **Plain multimodal DPO learns text shortcuts** — the "unconditional preference
  problem": stripping images barely changes scores [mDPO, EMNLP 2024,
  arXiv:2406.11839]. → use **conditional / per-modality DPO from day one**
  (OmniDPO does per-modality conditional DPO for omni-hallucination
  [AAAI 2026, arXiv:2509.00723]; MoD-DPO enforces invariance to irrelevant-modality
  corruption + sensitivity to relevant-modality [arXiv:2603.03192]).
- **Build preference pairs from modality counterfactuals.** The strongest
  grounding/hallucination methods all share: *rejected = response under a
  corrupted/absent modality; chosen = response that needs the real modality*
  [MFPO arXiv:2410.15334; CHiP ICLR 2025 arXiv:2501.16629; CounterVid
  arXiv:2601.04778; HII-DPO arXiv:2602.10425]. **On-policy** pairs (sampled from
  the current model) beat off-policy ones [OPA-DPO, CVPR 2025, arXiv:2501.09695].
- **DPO vs online RL (GRPO/PPO/RLOO):** DPO = offline, stable, cheap, but bounded
  by its pair distribution; GRPO = online, critic-free (group-mean baseline),
  optimizes *any programmatic reward* and explores [DeepSeek-R1, arXiv:2501.12948;
  RLOO, ICLR 2025, arXiv:2402.14740]. Online DPO and online GRPO converge similarly
  and both beat offline DPO [Bridging Offline/Online, arXiv:2506.21495]. **EchoInk-R1
  already ran GRPO on Qwen2.5-Omni for AVQA** (85.77%) — precedent, not blocker
  [arXiv:2505.04623].
- **Abstention needs a ternary reward** (correct +, wrong −, abstain ~0); strong
  positive abstention reward → over-abstention [TruthRL, arXiv:2509.25760;
  Rewarding Intellectual Humility, arXiv:2601.20126]. GRPO's **group agreement** is
  a free uncertainty signal to modulate the abstention advantage [TIAR,
  arXiv:2605.25850]; over-abstention is also fixable at inference [ReCoVERR, ACL
  2024, arXiv:2402.15610]. Survey: [Know Your Limits, TACL 2025].

**⚠️ Open question for *our* setup:** all cited DPO/GRPO works tune the full model
or LLM-LoRA; **DPO/GRPO with a frozen LLM + trainable-projector-only is
underexplored** — the KL is defined over the full-model distribution, so its
calibration when only the bottleneck moves is unverified. Validate early.

---

## 4. Architecture & freeze/train split (Q2)

**Recommended (both models):** ❄️ freeze all pretrained encoders (audio + vision)
and the LLM backbone; 🔥 train the **AV fusion bottleneck** on the *per-modality
connector outputs* (pre-LLM); optionally add **LoRA on both attn + MLP** of the LLM
(ACL-2024 PEFT study: "both" placement beats single-module [arXiv:2406.05130]).

- **VideoLLaMA2** [arXiv:2406.07476] — CLIP/SigLIP visual encoder (frozen every
  stage) + **STC connector** | **BEATs** audio encoder + **audio-MLP projector**;
  fusion happens *inside* the LLM. → cleanest drop-in: insert the bottleneck fusing
  *STC-output (video) + audio-MLP-output (audio)* just before the LLM.
- **Qwen3-Omni** [arXiv:2509.17765] — **AuT** audio enc + **SigLIP2** vision enc +
  per-encoder adapters → **Thinker** (MoE LLM) + Talker. ⚠️ Their report
  **explicitly warns against co-training encoders+adapters under a frozen LLM**
  (degrades encoder perception) — so operate the bottleneck on *adapter outputs*,
  leave encoders untouched.
- **Bottleneck form:** prefer a Q-Former/Perceiver-style cross-attention (compresses
  tokens + learns temporal alignment) over a plain MLP. **LAVISH** (latent
  bottleneck doing AV fusion+compression in a frozen backbone, CVPR 2023,
  arXiv:2212.07983) is the closest architectural cousin to "VIB on the fused stream."
  Frozen-encoder+frozen-LLM+train-projector is proven [BLIP-2 arXiv:2301.12597;
  SALMONN ICLR 2024; Ultravox]. Deeper option: gated cross-attention adapters at
  multiple LLM layers, zero-init [Flamingo, NeurIPS 2022, arXiv:2204.14198].

---

## 5. Datasets (Q3) — abandon AVQA

AVQA's weakness is **structural**, not just resolution/corruption: VGGSound-sourced
**10-second, label-driven** clips, **8 narrow categories**, large
**unimodally-solvable** fraction. MUSIC-AVQA is closed-set (42 classes/93 words)
with severe answer skew.

| Role | Dataset | Why |
|---|---|---|
| Eval (grounding-guaranteed) | **DAVE** [arXiv:2503.09321] | 2,426 egocentric (Epic-Kitchens/Ego4D); every Q needs both modalities |
| Eval (long video, per-step modality) | **OmniVideoBench** [arXiv:2510.10689] | reasoning traces tagged 54% visual / 46% audio |
| Scale pretrain | **VALOR-1M** [arXiv:2304.08345] | 1M trimodal, human AV captions |
| Grounding supervision | **AVSBench** (ECCV 2022) / **AVE** [arXiv:1803.08842] | pixel-/second-level sounding-object grounding |
| Robustness eval | **FortisAVQA/MAVEN** [arXiv:2504.00487], **AVHBench** [arXiv:2410.18325] | bias-stress + cross-modal hallucination |
| AV-mismatch abstention eval | **OMD-Bench** [arXiv:2603.27187] | the only AV-mismatch calibration probe found |

**Audio-necessary split (cheap → rigorous):** text-shortcut filter (drop any Q a
text-only LLM answers, à la AVUT [arXiv:2503.19951]) → modality-ablation filter
(keep only items where audio-only *and* video-only baselines both fail) →
ideally DAVE-style structural design.

---

## 6. Metrics

- **Grounding:** ΔAcc = Acc(AV) − Acc(V-only) and Acc(AV) − Acc(A-only); large
  positive Δ ⇒ genuine reliance. Optionally **counterfactual consistency** (answer
  flips when the necessary modality is swapped). Process-level: AuraScore-style
  factual-consistency if CoT traces exist [AURA, arXiv:2508.07470].
- **Abstention / selective prediction:** **Coverage@Risk** {5,10,20%},
  **AURC / E-AURC** [Geifman-El-Yaniv, arXiv:1705.08500; Reliable VQA, ECCV 2022,
  arXiv:2204.13631], **Abstain-ECE**; for an explicit IDK head add UAC / MCC.

---

## 7. Recommended design

- **v0 (de-risked, DPO-first):** VideoLLaMA2 frozen; train a cross-attention
  **fusion bottleneck** (no stochastic IB yet) on STC+audio-MLP outputs;
  **conditional/per-modality DPO** with **on-policy counterfactual pairs**
  (audio-dropped / video-dropped / audio-swapped as rejected) + **ternary
  abstention** pairs on mismatched/insufficient items. Eval on DAVE +
  audio-necessary split.
- **v1 (the contribution):** add the **conditional + synergy-preserving VIB** on
  the fused stream with **per-modality adaptive β** (OMIB/CAL-style). Ablate
  VIB-vs-no-VIB — that comparison *is* a result.
- **v2:** swap DPO → **GRPO** with a programmatic reward (modality-counterfactual
  sensitivity + ternary abstention, TIAR-style group-agreement gating). Repeat on
  Qwen3-Omni as the second arm.

---

## 8. Novelty gap (positioning)

Every component exists *separately*; **no found work combines them**:

| Component | Exists? | Gap |
|---|---|---|
| Conditional/per-modality IB for fusion | ✅ (sarcasm only [2508.10644]; OMIB/CAL general) | not for AV-MLLM grounding w/ frozen LLM |
| Counterfactual-modality DPO | ✅ MoD-DPO / OmniDPO | vision/omni-leaning; not audio-specific; not IB-coupled |
| GRPO on AV-MLLM | ✅ EchoInk-R1 | plain reward; no IB; no abstention |
| RL calibrated abstention | ✅ TruthRL / TIAR | text-only; not AV-*mismatch* |
| AV-mismatch abstention | ⚠️ OMD-Bench (eval only) | no *training method* |

**Defensible contribution:** *a conditional/synergy IB on the fused AV stream,
trained with counterfactual-modality RL, that also produces calibrated abstention
on audio-video mismatch* — with the AV-mismatch-abstention training method being
the sharpest unclaimed piece.

---

## 9. Risks / open questions

- Frozen-LLM + projector-only DPO/GRPO calibration (see §3) — **validate first**.
- β scheduling under a moving RL reward; risk of modality/posterior collapse —
  monitor per-modality rate; anneal β [Modality Collapse, ICML 2022,
  arXiv:2206.04496].
- Shared-shortcut failure: if a spurious cue appears in *both* modalities, IB
  encodes rather than discards it — counterfactual *swap* (not just drop) is the
  stronger probe.
- Over-abstention from the abstention reward — cap with coverage constraint +
  must-answer positives.
- Counterfactual realism: silence ≠ noise ≠ swapped-clip; swap-from-another-clip is
  the strongest "must-change" intervention.

---

## 10. Verification notes (citation corrections applied)

All 10 spot-checked references are **real**. Corrections folded in above:
- `2508.10644` full title ends **"… in Sarcasm Detection"** (MUStARD++) — domain-specific, **does not** cover AV grounding (good for novelty).
- `2601.05547` (VIB-Probe): VIB on **attention-head outputs** of the full model, not purely the "visual stream."
- `2509.00723` (OmniDPO): **AAAI 2026**, not 2025.
- `2510.26289`: framework is **CAL** (Contribution-Guided Asymmetric Learning); "AIB" is a sub-component.
- `2503.09321` (DAVE): **NeurIPS 2025 venue unconfirmed** — treat as arXiv/under-review; sample count + design confirmed.
- `2505.19996` (OMIB): ICML 2025 confirmed; bound is a **per-modality regularization-weight** bound (not a channel-rate bound).
- Truncated titles (now corrected): EchoInk-R1 ("**Exploring** …"), MoD-DPO (full subtitle), OMD-Bench (full subtitle).
- `2406.11839` (mDPO) and `2406.05130` (PEFT-MLLM) fully verified.

Caveat: citations *not* in the 10-item spot-check (canonical IB/RL/architecture
papers and the broader 2026 bench/method list) are agent-sourced; canonical ones
are High-confidence, very-recent 2026 entries should be re-checked before being
cited in a paper.

---

## Selected bibliography (grouped)

**Grounding / bias / benchmarks** — Look-Listen-Answer (NeurIPS 2024,
arXiv:2404.12020); FortisAVQA/MAVEN (arXiv:2504.00487); Aligned Better Listen
Better (ICLR 2025, arXiv:2504.02061); Curse of Multi-Modalities (arXiv:2410.12787);
AVHBench (ICLR 2025, arXiv:2410.18325); AURA (arXiv:2508.07470); AVTrustBench
(arXiv:2501.02135); Visual Biases in AVSL (arXiv:2409.06709); MLLMs Modality Bias
(arXiv:2505.18657).

**Information bottleneck** — Deep VIB (ICLR 2017, arXiv:1612.00410); MV-IB
(ICLR 2020, arXiv:2002.07017); MBT/Attention Bottlenecks (NeurIPS 2021,
arXiv:2107.00135); DRIBO (ICML 2022, arXiv:2102.13268); MIB (TMM 2022,
arXiv:2210.17444); DBF (ACL 2023, arXiv:2305.14652); CIB-VQA (IJCV 2023,
arXiv:2209.06954); IBMEA (ACM MM 2024, arXiv:2407.19302); OMIB (ICML 2025,
arXiv:2505.19996); CAL/Asymmetric-IB (arXiv:2510.26289); MRdIB (arXiv:2509.20225);
CoMM (ICLR 2025); Narrowing-IB (ICLR 2025, arXiv:2502.14889); Conditional-IB
(sarcasm) (arXiv:2508.10644); VIB-Probe (arXiv:2601.05547); Saxe et al. (2019).

**RL / preference / abstention** — DeepSeek-R1/GRPO (arXiv:2501.12948); RLOO
(ICLR 2025, arXiv:2402.14740); Bridging Offline/Online (arXiv:2506.21495); mDPO
(EMNLP 2024, arXiv:2406.11839); OmniDPO (AAAI 2026, arXiv:2509.00723); MoD-DPO
(arXiv:2603.03192); CHiP (ICLR 2025, arXiv:2501.16629); OPA-DPO (CVPR 2025,
arXiv:2501.09695); MFPO (arXiv:2410.15334); CounterVid (arXiv:2601.04778); EchoInk-R1
(arXiv:2505.04623); TruthRL (arXiv:2509.25760); Rewarding Intellectual Humility
(arXiv:2601.20126); TIAR (arXiv:2605.25850); ReCoVERR (ACL 2024, arXiv:2402.15610);
Know Your Limits (TACL 2025).

**Architecture / PEFT** — Qwen3-Omni (arXiv:2509.17765); Qwen2.5-Omni
(arXiv:2503.20215); VideoLLaMA2 (arXiv:2406.07476); BLIP-2 (arXiv:2301.12597);
SALMONN (ICLR 2024, arXiv:2310.13289); Flamingo (NeurIPS 2022, arXiv:2204.14198);
LAVISH (CVPR 2023, arXiv:2212.07983); PEFT-MLLM (ACL 2024, arXiv:2406.05130).

**Datasets / metrics** — AVQA (ACM MM 2022); MUSIC-AVQA (CVPR 2022,
arXiv:2203.14072); DAVE (arXiv:2503.09321); Pano-AVQA (ICCV 2021, arXiv:2110.05122);
AVSBench (ECCV 2022); AVE (ECCV 2018, arXiv:1803.08842); VALOR (arXiv:2304.08345);
AVSD (CVPR 2019, arXiv:1901.09107); OmniVideoBench (arXiv:2510.10689); AVUT
(EMNLP 2025, arXiv:2503.19951); OMD-Bench (arXiv:2603.27187); Selective
Classification (arXiv:1705.08500); Reliable VQA (ECCV 2022, arXiv:2204.13631).
