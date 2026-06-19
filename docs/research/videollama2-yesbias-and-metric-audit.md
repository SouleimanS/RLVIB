# VideoLLaMA2's yes-bias collapse: backbone diagnosis, metric audit, and ranked fixes

Deep-research synthesis (5 angles: DPO→hallucination mechanism · multimodal/anchored DPO mitigations ·
yes/no benchmark metric pathologies · VideoLLaMA2 design + cross-backbone transfer · small-sample eval
statistics + IB/frozen-adapter grounding). **Follow-up to `dpo-collapse-and-fixes.md`:** that memo
explained why *naive* swap-DPO collapses and motivated our **anchored** recipe (chosen-likelihood term +
KL-to-base). This memo explains why that anchored recipe **grounded Qwen3-Omni but still collapsed
VideoLLaMA2.1-7B-AV**, asks whether the collapse is partly a **measurement artifact**, and gives the
specific changes to try.

Sourcing caveat: arXiv/ACL/OpenReview PDFs returned HTTP 403 to direct fetches; every load-bearing claim
below was verified through WebSearch server-side snippets and official GitHub READMEs, and the seven most
load-bearing citations (incl. the three 2026 papers) were re-verified individually — all exist and say
what is attributed. Exact per-model benchmark decimals remain medium-confidence; mechanisms are
high-confidence.

---

## TL;DR

1. **The collapse may be partly our metric, not the model.** Our VideoLLaMA2 **base** CMM hallucination-
   resistance is **HR ≈ 0.01–0.09**. But CMM's own paper reports open audio-visual models with **HR ≈
   34–59 %**, and AVHBench measures VideoLLaMA2 at a **62.4 % yes-rate** — biased, but nowhere near the
   ~100 % yes-rate of a genuinely yes-collapsed model (the *original* Video-LLaMA, which AVHBench clocks
   at ~50 % acc / 100 % yes). A base HR 5–30× below the published range is a red flag that our HR harness
   (prompt template or answer parsing) is manufacturing extra "yes". **Audit HR before concluding VL2 is a
   null** — if parsing is the culprit, both the base and the bottleneck HR numbers are wrong and the "VIB
   destroys HR" story largely dissolves.

2. **If the collapse survives the audit, it is over-determined by this backbone.** VideoLLaMA2's audio-
   visual head **inherits its (yes-biased) vision-language base LM and barely uses audio** (its audio
   tokens prune from 1,496 → 10 with no loss; it aligns audio and vision *separately* and "tends to
   disregard audio"). The audio-swap preference signal is too weak to exploit on this backbone, so DPO
   takes the easy gradient — **sharpening the base LM's yes-prior** instead of adding audio grounding.
   Likelihood displacement is governed by the **base model's** embedding geometry, so the identical recipe
   can ground one backbone and invert on another.

3. **Our training log is consistent with this, not against it.** `p_chosen ≈ 1.0` is *preference-margin
   accuracy* (chosen reward > rejected reward), **not** absolute chosen likelihood — DPO routinely
   satisfies the margin while the absolute chosen log-prob *falls* (likelihood displacement / DPOP). And
   `frac_yes ≈ 0.53` was the *training-time anchor* balance, which does not constrain *eval-time* HR on the
   held-out hallucination probes.

4. **Our anchor was the right instinct but too soft.** A `−log σ(β(c_w − c_r) − δ)` term saturates once
   satisfied and stops pushing; nothing in the recipe puts a **hard floor** under the chosen (or the
   "no"-answer) likelihood. The literature's working fixes are a **hard positive-reward anchor** (mDPO),
   a **DPOP λ-penalty**, or — cheapest — an explicit **NLL/SFT term on the chosen answer**, which is shown
   to flip the chosen log-prob from *decreasing* to *increasing* during training.

**Priority of fixes:** (1) audit the HR metric + report Wilson CIs and yes-rate; (2) harden the chosen
anchor (add NLL-on-chosen and/or an mDPO/DPOP hard floor) and explicitly protect "no"-answer likelihood;
(3) add a modality-conditional negative (corrupted audio); (4) make the chosen data on-policy; (5)
CHES-filter / raise edit-distance of swap pairs; (6) early-stop on val, tune β. Details + evidence in §4.

---

## 1. Did the model collapse, or did our metric? (the audit — angle 3)

### 1.1 HR floors mechanically under any yes-bias — by design
CMM defines two metrics over paired probes: **PA** (perception accuracy) = accuracy on the *existent*
("yes") probe; **HR** (hallucination resistance) = accuracy on the *non-existent* ("no") probe ("HR
assesses its resistance to hallucinations by correctly identifying the **absence** of non-existent objects
or events"). A model that leans "yes" scores **high PA and near-zero HR automatically** — the PA≫HR gap
*is* CMM's headline diagnostic of "overreliance on unimodal priors." So a low HR is expected for a
yes-biased model; the question is whether *our* HR is **too** low to be real.
(CMM, Leng et al., [2410.12787](https://arxiv.org/abs/2410.12787); repo `DAMO-NLP-SG/CMM`.)

### 1.2 The discrepancy that triggers the audit
- **Our measurement:** VideoLLaMA2 base HR ≈ **0.01–0.09** (val 0.091 / test 0.012 on 150-example halves).
- **CMM paper, open AV models:** HR ≈ **34.5 % (Qwen2-Audio), 39 % (Audio-Flamingo), 52 % (GAMA-IT)**;
  "Visual-Audio LMMs generally achieve PA over 80." (medium-confidence decimals, read from the results
  table via snippet; the *range* is firm.)
- **AVHBench, VideoLLaMA2:** **74.2 % accuracy, 62.4 % yes-rate** on audio-driven video hallucination —
  i.e. a *moderately* yes-leaning but far-from-degenerate model. A truly collapsed model looks like the
  original **Video-LLaMA: ~50 % acc, ~100 % yes**.
  (AVHBench, Sung-Bin et al., ICLR 2025, [2410.18325](https://arxiv.org/abs/2410.18325); repo
  `kaist-ami/AVHBench`.)

A base HR of 1–9 % is inconsistent with a 62 %-yes-rate model and sits far below CMM's published open-model
band. The most likely explanation is **our harness**: a prompt that elicits more "yes" than CMM's protocol,
or — more likely — an **answer parser that mis-scores "no" responses** (e.g., the model says "No, there is
no bell" but the parser fails to extract "no" and records it as wrong). That single bug would deflate HR on
*both* base and bottleneck rows and is exactly the kind of thing that produces an implausible floor.

### 1.3 The concrete audit (run on the cluster, reads existing JSONs — no GPU)
```python
import json, collections
d = json.load(open("runs/cmm_videollama2.json"))           # base; repeat for *_broad_step30.json
rec = d["records"]
no = [r for r in rec if r.get("answer") == "no"]           # the HR subset
print("n_no =", len(no), " yes-rate(all) =",
      sum(r["pred"]=="yes" for r in rec)/len(rec))
print("HR =", sum(r["pred"]=="no" for r in no)/max(len(no),1))
for r in no[:12]:                                            # eyeball the raw outputs
    print(repr(r.get("pred")), "<=", repr(r.get("raw", r.get("output",""))[:120]))
```
This returns three things we need: (a) **`n_no`** — the true denominator of HR, which decides whether the
val/test swing is noise (§1.4); (b) the **overall yes-rate**, to compare against AVHBench's 62.4 %; and (c)
**a dozen raw "no"-labeled outputs**, to see whether the model is actually saying "no" while the parser
scores it wrong. If (c) shows correct "no" text scored as wrong → it's a parsing bug, fix the parser and
re-derive every HR number. If the model genuinely emits "yes" almost always (overall yes-rate ≫ 62 %) →
the collapse is real and §2/§3 apply.

### 1.4 The val/test swing is n-dependent — report a CI, not a point
Whether base HR 0.091 vs 0.012 is "just noise" depends entirely on **`n_no`** per half (the HR denominator,
*not* 150):
- at **n≈150**: 0.091 (≈14/150) vs 0.012 (≈2/150) → z ≈ 3.0, **p ≈ 0.002, non-overlapping Wilson CIs — not
  noise** (this kills the earlier "within sampling noise" claim);
- at **n≈75**: z ≈ 2.2, p ≈ 0.03 — marginal;
- at **n≈40**: z ≈ 1.6, p ≈ 0.10 — **consistent with noise**.

So do **not** assert the swing is statistically insignificant until `n_no` is known. Regardless of the
verdict, near-floor proportions on small n must be reported as **Wilson or Clopper–Pearson intervals**, not
point estimates — the Wald interval is invalid near 0 (it runs negative), and the **rule of three** says
even *0 correct* in n=75 only bounds true HR below ≈ 0.04. Concretely, a Wilson 95 % CI for p̂=0.05 spans
≈[0.009, 0.236] at n=20 and ≈[0.020, 0.122] at n=80.
(Brown, Cai & DasGupta, *Stat. Sci.* 2001; rule of three, Hanley & Lippman-Hand, *JAMA* 1983; Wilson 1927.)

### 1.5 What the field does so a near-floor metric can't embarrass you
- **Always report the yes-rate (answer distribution) and F1/balanced-accuracy alongside accuracy.** Every
  benchmark here builds this in — POPE reports a "Yes ratio" + F1 (with "yes" as positive class) precisely
  because under a skewed answer distribution accuracy misleads; CMM splits PA/HR; AVHBench reports a "Yes
  (%)" column per task. A floored HR with yes-rate ≈ 1.0 should be read as "degenerate-yes," and the
  *informative* number is the yes-rate. (POPE, Li et al., EMNLP 2023,
  [2305.10355](https://arxiv.org/abs/2305.10355).)
- **MLLM yes-bias is general and quantified:** across balanced yes/no probes, open MLLM say-yes rates run
  0.51–0.81 (mean ≈ 0.69) vs GPT-4o ≈ 0.46, and say-yes rate correlates with hallucination at r ≈ −0.99 —
  so the recommended target is *balancing the answer distribution*, not just raising accuracy.
  (PhD, Liu et al., [2403.11116](https://arxiv.org/abs/2403.11116).)
- **Even the "ground truth" can be noisy and asymmetric:** RePOPE found POPE's labels have ~9.3 % wrong-yes
  vs ~1.7 % wrong-no, enough to reorder model rankings — so treat small HR/F1 gaps near the floor with
  suspicion. (RePOPE, Neuhaus & Hein, [2504.15707](https://arxiv.org/abs/2504.15707).)
- **Attach a bootstrap CI** (resample the eval set with replacement, 1,000–10,000×, report 2.5–97.5
  percentiles) to every headline number — and prefer a **paired** test (same items) over comparing two
  disjoint halves. NLP benchmarks are routinely underpowered (Card et al., *With Little Power*,
  [2010.06595](https://arxiv.org/abs/2010.06595)); single-split rankings often fail to reproduce under
  other splits (Gorman & Bedrick, ACL 2019).

---

## 2. If the collapse is real: why *this* backbone (angle 1 + 4)

The prior memo established the general mechanism. The new, backbone-specific evidence explains why our
*already-anchored* recipe transferred to Qwen3-Omni but not VideoLLaMA2.

### 2.1 VideoLLaMA2's AV head inherits its (yes-biased) base LM
The single most on-point finding: *"the AVLLM's audio behavior strongly matches its vision-language base
model, indicating limited additional alignment to audio supervision … current AVLLMs typically initialize
from a pretrained LVLM checkpoint and add audio adapters … the model inherits strong priors toward visual
information."* AV-LLMs encode correct audio semantics in *intermediate* layers but **deeper fusion layers
privilege vision and suppress audio**, so the audio rarely surfaces at the output.
(*Do Audio-Visual LLMs Really See and Hear?*, Selvakumar et al., Univ. Maryland, Apr 2026,
[2604.02605](https://arxiv.org/abs/2604.02605).)
**Consequence for us:** a preference meant to *add* audio grounding has little audio signal to latch onto,
so it instead nudges the inherited base-LM distribution — whose prior is "yes."

### 2.2 VideoLLaMA2 specifically under-uses audio
- Its audio tokens are largely **prunable noise**: FastAV cuts VideoLLaMA2's audio tokens from **1,496 → 10
  with no degradation** (sometimes improving). If 99 %+ of audio tokens are droppable, there is almost no
  audio gradient for a swap-preference to exploit. (FastAV, Jan 2026,
  [2601.13143](https://arxiv.org/abs/2601.13143).)
- It aligns vision-language and audio-language **separately** (BEATs encoder → a **2-layer MLP** projector,
  `mm_projector_a`), which yields "less coordinated audio-video representations" that "tend to disregard
  audio" because the visual stream has higher information density. (Dolphin, ICLR 2025,
  [2504.02061](https://arxiv.org/abs/2504.02061); VideoLLaMA2, Cheng et al.,
  [2406.07476](https://arxiv.org/abs/2406.07476).)

### 2.3 DPO outcomes are base-model-dependent, so a recipe can invert across backbones
Likelihood displacement is driven by the **base model's** hidden-embedding geometry (quantified by the CHES
score) — the canonical example, "preferring *No* over *Never* sharply increases P(*Yes*)," is base-distribution-
dependent, so an identical preference set can ground one backbone and flip another toward a degenerate "yes."
(Razin et al., ICLR 2025, [2410.08847](https://arxiv.org/abs/2410.08847).) Independently, DPO is highly
sensitive to the reference/initial policy and its implicit rewards are mis-scaled relative to the base
(motivating Cal-DPO, [2412.14516](https://arxiv.org/abs/2412.14516)); the same preference data transfers
only partially across model families (MIA-DPO; *Is DPO Superior to PPO?*,
[2404.10719](https://arxiv.org/abs/2404.10719)). **These reconcile cleanly:** transfer degrades gracefully
on a backbone whose base distribution is far from a displacement trap (Qwen3-Omni) but can *invert* on one
sitting near it (VideoLLaMA2's yes-prior).

### 2.4 The architectural asymmetry behind the different calibration
Qwen3-Omni's audio pathway is trained end-to-end on **~20 M hours** of supervised audio (a from-scratch AuT
encoder) and time-*fuses* audio+video (TMRoPE), whereas VideoLLaMA2.1-AV **freezes BEATs and tunes only a
2-layer projector** and aligns the modalities separately. A backbone with a heavily-trained, well-fused
audio path has real audio signal for a tiny adapter to amplify; one with a thin, separable audio path does
not. (Qwen3-Omni, [2509.17765](https://arxiv.org/abs/2509.17765); Qwen2.5-Omni,
[2503.20215](https://arxiv.org/abs/2503.20215).)

### 2.5 The clean single story
VideoLLaMA2's audio tokens are too weak/noisy for the swap-preference to exploit (§2.1–2.2) → DPO finds the
easier gradient and **sharpens the inherited base-LM yes-prior** (§2.3) → eval-time "no"-answer likelihood
erodes → **HR floors** while AVHBench (yes-heavy) barely moves. Our soft anchor slowed but did not stop this
(§4). The **falsifiable prediction**: on the failing VL2 run, the *absolute* chosen-answer log-prob trend is
flat/decreasing and the swap pairs have **high CHES** (near-identical chosen/rejected embeddings); on
Qwen3-Omni they do not.

---

## 3. Why our anchored recipe didn't catch it

Our recipe already has (a) a chosen-likelihood term `−log σ(β(c_w − c_r) − δ)` and (b) a KL-to-base on
general inputs. Two gaps:

1. **The chosen anchor is *soft*.** A log-sigmoid margin term **saturates** once the margin clears δ and
   stops pushing the chosen up — it does not put a **hard floor** under the absolute chosen likelihood.
   mDPO's anchor instead enforces a **positive reward** on the chosen ("avoiding the decrease in their
   likelihood — an intrinsic problem of relative preference optimization"); DPOP adds a one-sided penalty
   `λ·max(0, log[π_ref(y_w)/π_θ(y_w)])` that activates *only* when the chosen drops below the reference.
   (mDPO, EMNLP 2024, [2406.11839](https://arxiv.org/abs/2406.11839); DPOP/Smaug,
   [2402.13228](https://arxiv.org/abs/2402.13228).)
2. **KL-to-base protects *general* inputs, not the "no"-answer on hallucination probes.** The yes/no
   absence probes are exactly where the swap signal pushes toward "yes," and they are not "general inputs,"
   so nothing in the recipe directly protects the model's ability to say **"no."** The broad anchor's
   visual-presence "do you SEE X" negatives were meant to, but on VL2 they were too few/soft.

---

## 4. Ranked, evidence-annotated fixes

Each fix: **what**, **evidence** (citation + strength), **frozen-adapter fit**, **expected effect/risk**.
Ordered by expected value-per-cost for our frozen-backbone + tiny-per-modality-adapter constraint.

**1 — Audit the HR metric, then report Wilson CIs + yes-rate (do this first).**
*What:* run §1.3; if "no" responses are mis-parsed, fix the parser and re-derive all HR; switch reporting to
Wilson CI + yes-rate. *Evidence:* our 1–9 % vs published 34–59 % vs 62.4 % AVHBench yes-rate
([2410.12787](https://arxiv.org/abs/2410.12787), [2410.18325](https://arxiv.org/abs/2410.18325)); POPE/PhD
yes-rate practice ([2305.10355](https://arxiv.org/abs/2305.10355), [2403.11116](https://arxiv.org/abs/2403.11116));
binomial-CI stats (Brown-Cai-DasGupta 2001). **Strength: High.** *Fit:* pure eval-side, no retrain.
*Effect:* may **dissolve or sharply rescope the null** for free; at minimum makes every number defensible.
*Risk:* none.

**2 — Add an NLL/SFT term on the chosen answer (cheapest real anti-displacement lever).**
*What:* add `α · CE(chosen_answer)` to the loss (the "DPO+NLL" trick). *Evidence:* Iterative RPO shows the
chosen-sequence log-prob **decreases without** the NLL term and **increases with** it, and calls the term
"crucial." (NeurIPS 2024, [2404.19733](https://arxiv.org/abs/2404.19733).) **Strength: High (mechanism),
Medium (for *hallucination* specifically — demonstrated on text reasoning).** *Fit:* trivial — one extra
cross-entropy on the adapter-modified forward; frozen base unchanged. *Effect:* directly pins absolute
chosen likelihood up. *Risk:* α too large weakens preference learning; tune.

**3 — Harden the chosen anchor to a positive-reward floor (mDPO / DPOP).**
*What:* replace/augment the soft margin with mDPO's positive-reward anchor or DPOP's
`λ·max(0, log[π_ref(y_w)/π_θ(y_w)])`. *Evidence:* mDPO raises Bunny MMHalBench 2.28→2.96 / cuts halluc.
0.56→0.42 and is trained with **LoRA on a frozen-ish backbone**; DPOP makes the chosen log-prob
non-decreasing at λ=5. ([2406.11839](https://arxiv.org/abs/2406.11839),
[2402.13228](https://arxiv.org/abs/2402.13228); ACPO's stop-gradient asymmetric scaling is the 2026 variant,
[2603.22165](https://arxiv.org/abs/2603.22165).) **Strength: High, multimodal-validated.** *Fit:* loss-only,
reference = our frozen base — ideal. *Effect:* prevents the likelihood-displacement collapse at the source.
*Risk:* λ too large stalls preference learning.

**4 — Explicitly protect "no"-answer likelihood (target HR directly).**
*What:* add **audio-absence and visual-absence negatives** (ground-truth answer = "no") to the anchor, under
the NLL/hard-floor of fix 2/3, so "no" likelihood is pinned, not just "general" outputs. *Evidence:* the
gap in §3.2; CMM attributes hallucination to unaddressed absence probes
([2410.12787](https://arxiv.org/abs/2410.12787)); image-conditional negatives are the standard VLM-halluc.
fix (POVID, [2403.08730 / repo]; HALVA adds a **KL-to-base** regularizer *with* a contrastive term and
*retains* general performance, ICLR 2025). **Strength: Medium-High.** *Fit:* data + loss only. *Effect:*
the most direct lever on HR itself. *Risk:* over-weighting "no" could depress PA — guard both.

**5 — Make the chosen data on-policy.**
*What:* sample/lightly-edit the chosen "audio-consistent" answers **from the frozen base itself** rather than
using off-policy templated text. *Evidence:* OPA-DPO — the KL-to-reference **impedes learning from off-policy
data**; on-policy data let 4.8k samples cut LLaVA-1.5 hallucination 13.26 % (AMBER).
([2501.09695](https://arxiv.org/abs/2501.09695), CVPR 2025 Oral.) **Strength: High, VLM-specific.** *Fit:* a
data-pipeline change; frozen base = the reference, so this is natural. *Effect:* lets the anchor actually
raise the chosen instead of only crushing the rejected. *Risk:* more pipeline work; modest.

**6 — CHES-filter / raise edit-distance of the swap pairs.**
*What:* drop preference pairs whose chosen/rejected are near-identical (high CHES / low edit distance), which
are the displacement-triggering ones. *Evidence:* CHES predicts displacement and filtering high-CHES pairs
beats adding an SFT term (Razin, [2410.08847](https://arxiv.org/abs/2410.08847)); low edit distance is
DPOP's trigger ([2402.13228](https://arxiv.org/abs/2402.13228)). **Strength: High (mechanism).** *Fit:*
data-side, no retrain infra change. *Effect:* removes the worst-case pairs (our yes/no + single-letter answers
are near-worst-case). *Risk:* shrinks the training set.

**7 — Early-stop on val; tune β / fewer steps.**
*What:* select on the val half early; the squeezing effect worsens with **over-training** and off-policy data.
*Evidence:* Ren & Sutherland squeezing ([2407.10490](https://arxiv.org/abs/2407.10490)); DPO's β is the
KL-strength knob ([2305.18290](https://arxiv.org/abs/2305.18290)). **Strength: Medium.** *Fit:* free. *Effect:*
limits drift. *Risk:* under-training loses the (small) gain.

**Method positioning (not a fix, for the paper):** the IB + frozen-adapter framing is well-grounded — VIB
buys robustness by discarding nuisance info (Alemi et al., [1612.00410](https://arxiv.org/abs/1612.00410));
frozen-backbone + small trained module is the standard PEFT regime (prefix-tuning
[2101.00190](https://arxiv.org/abs/2101.00190); adapters [1902.00751](https://arxiv.org/abs/1902.00751));
KL-to-frozen-reference is the canonical drift control (Stiennon et al.,
[2009.01325](https://arxiv.org/abs/2009.01325); DPO's implicit-KL view); and others already bottleneck/edit
the intermediate token stream of frozen MLLMs to change grounding (e.g.,
[2507.15652](https://arxiv.org/abs/2507.15652), [2502.03628](https://arxiv.org/abs/2502.03628)).

---

## 5. Conflicts and falsifiable predictions

- **Anchor vs reference-free.** SimPO ([2405.14734](https://arxiv.org/abs/2405.14734)) *removes* the
  reference/KL; mDPO/HALVA/OPA-DPO *rely* on it. Resolution: OPA-DPO shows **KL helps iff the data is
  on-policy** — so keep the KL anchor *and* do fix 5, or the anchor fights learning.
- **Transfer "works" vs "inverts."** Partial cross-model transfer (MIA-DPO) vs outright inversion (Razin)
  — not contradictory: graceful on aligned backbones, flippable near a displacement trap.
- **Audio tokens: signal or noise?** FastAV (prune 99 %+ of VL2 audio, no loss) vs the premise that DPO
  should *strengthen* audio grounding — the cleanest single story (§2.5) is that VL2's audio is too weak to
  ground, so DPO defaults to the yes-prior.
- **Falsifiable test of the whole diagnosis:** log the *absolute* chosen-answer log-prob trajectory and the
  swap-pair CHES on both backbones. The diagnosis predicts: VL2 = decreasing chosen log-prob + high CHES;
  Qwen3-Omni = stable chosen log-prob + lower CHES. If VL2's chosen log-prob is *stable* yet HR still floors,
  the cause is the metric (§1), not displacement.

---

## Sources (verified this round)

Mechanism: Razin et al. *Unintentional Unalignment* ICLR 2025 [2410.08847] · Pal et al. *DPOP/Smaug*
[2402.13228] · Ren & Sutherland *squeezing* ICLR 2025 [2407.10490] · Sharma et al. *Sycophancy*
[2310.13548] · Perez et al. [2212.09251].
Mitigations: mDPO EMNLP 2024 [2406.11839] · Iterative RPO (DPO+NLL) NeurIPS 2024 [2404.19733] · APO
[2408.06266] · ACPO 2026 [2603.22165] · OPA-DPO CVPR 2025 [2501.09695] · HALVA ICLR 2025 · POVID
[2403.08730] · RLHF-V [2312.00849] · HA-DPO [2311.16839] · CSR [2405.14622] · TPO [2412.14487] · SimPO
[2405.14734] · Cal-DPO [2412.14516].
Backbone: *Do AV-LLMs Really See and Hear?* 2026 [2604.02605] · FastAV 2026 [2601.13143] · Dolphin ICLR
2025 [2504.02061] · VideoLLaMA2 [2406.07476] · Qwen2.5-Omni [2503.20215] · Qwen3-Omni [2509.17765] · *Is
DPO Superior to PPO?* [2404.10719].
Metrics & stats: CMM [2410.12787] · AVHBench [2410.18325] · POPE [2305.10355] · RePOPE [2504.15707] · PhD
[2403.11116] · Card et al. *With Little Power* [2010.06595] · Gorman & Bedrick ACL 2019 · Brown-Cai-DasGupta
*Stat. Sci.* 2001.
Method: VIB [1612.00410] · prefix-tuning [2101.00190] · adapters [1902.00751] · Stiennon KL-to-reference
[2009.01325] · DPO [2305.18290].
