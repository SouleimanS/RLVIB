# Presenting the VIB results + a strengthening plan — deep-research synthesis

> Date: 2026-07-01 · Method: 5-angle deep-research fan-out (presentation craft · publishability
> of frozen-adapter alignment · negative-results/mechanism-mismatch reporting · stronger objectives
> · counterfactuals/probes/benchmarks), 25 sources fetched, 56 claims extracted, **25 verified with a
> 3-vote adversarial pass → 21 confirmed, 4 refuted**. Confirms/extends `dpo-collapse-and-fixes.md`,
> `query-conditioned-bottleneck.md`, `film-multistage-recipe.md`.

## 0. TL;DR

- **Present the +5.2-pt AVHBench gain as a real, calibrated result — do not over-hedge it into a
  non-result.** It is statistically significant on the best backbone (Qwen3-Omni; p<1e-4, paired
  McNemar, n=5002) with CMM capability preserved. Under-claiming a genuine effect damages credibility
  as much as hype [Bowman, *The Dangers of Underclaiming*, ACL 2022, arXiv:2110.08300].
- **Fully disclose the mixed profile** (Qwen2.5 marginal, VideoLLaMA2 null) and the **mechanistic
  twist** (the unconditional adapter grounds by *rewriting vision*, not attending to audio) — but
  frame the null as *backbone-inappropriateness*, not method failure, and the twist as an honest
  **Clever-Hans-style insight** [Anders et al., Information Fusion 2022, arXiv:1912.11425].
- **This honest/mixed/mechanistically-surprising profile is a direct fit for dedicated top venues**:
  NeurIPS/ICLR **ICBINB** ("I Can't Believe It's Not Better") and the ACL/EMNLP **Insights from
  Negative Results** workshop. A main-venue framing rests on the significant best-backbone gain +
  the mechanistic contribution + a confirmed FiLM result.
- **If judged too weak, the ranked next moves** (all validated in 2024-2026 lit): (1) **confirm the
  prompt-aware FiLM variant at scale** — the *only* on-thesis mechanism; (2) **stronger objective:
  curriculum GRPO** (validated on Qwen2.5-Omni, one of your exact backbones); (3) **AVCD-style
  contrastive decoding** (training-free, composable); (4) **add MMAU**; (5) **harder counterfactuals**.

---

## 1. How to present (Q1)

**1.1 Calibrate, don't under-hedge (high).** The +5.2 gain is a real effect; report it as such.
Bowman [2110.08300] shows anti-hype routinely overshoots into claiming a system is *less* capable than
it is, which harms credibility. So: state the significant gain plainly, *and* disclose the null/marginal
backbones and the mechanism — he condemns over- and under-claiming equally.

**1.2 The VideoLLaMA2 null is backbone-inappropriateness, not method failure (high).** Bowman names
"misleading presentations of valid negative results from weak/dated baselines" as a core under-claiming
failure (his adversarial-SQuAD example: error rates halved by stronger models). A backbone that *barely
uses audio* has no audio signal to ground — frame it that way; it makes the audio-usage of the backbone
a *prerequisite*, which is itself a finding.

**1.3 Report per-category on AVHBench's four native tasks (high).** AVHBench [KAIST, ICLR 2025,
arXiv:2410.18325] is three judgment tasks (Audio-driven Video Hall., Video-driven Audio Hall.,
Audio-visual Matching) + a captioning task; the official repo reports per sub-task. Your gains map onto
two of them (AV-Matching +11, Video-driven-Audio +3), so lead with the per-category breakdown. Keep
**AVHBench and CMM reporting distinct** (PA/HR belong to CMM, not AVHBench).

**1.4 The problem is well-motivated (high).** AVHBench establishes cross-modal hallucination as a real,
benchmarked, still-active (MoD-DPO, AVCD, 2025-26) problem: AV-LLMs "struggle with hallucinations caused
by cross-interactions between modalities." Improving grounding on it is not niche.

**1.5 Frame the mechanistic twist as a Clever-Hans finding (medium).** "Strong held-out accuracy can
coexist with an unintended mechanism" — and Clever-Hans is *distinguished from overfitting by its
invisibility to held-out accuracy* [1912.11425]. Present the attention audit as a principled mechanistic
check (the SpRAy precedent: systematic attribution over a corpus, not an anecdote) that revealed the
adapter grounds via the *vision* stream. This "right answer, wrong (named) mechanism" twist is the
paper's most interesting content, and it directly motivates FiLM (the on-thesis fix).

**1.6 Venue (high).** Workshop fit is squarely established: **ICBINB** (NeurIPS 2020-23 / ICLR 2025-26;
mission = "slow science… pushes back against leaderboard-ism… share surprising or negative results";
2023 theme explicitly invited foundation-model failure modes) and the **Insights from Negative Results
in NLP** workshop (EMNLP/NAACL series; "invites unexpected or negative results that highlight
methodological issues", ~64% acceptance). A main-venue attempt would rest on the significant gain + the
mechanistic contribution + a confirmed FiLM result — *the sources cannot adjudicate the main-venue bar*
(open question).

**1.7 Report adversarial-probe accuracy with its construction bias (medium).** Absolute accuracy on
intentionally-hard swap probes is not a clean capability score; contextualize it.

### Suggested talk spine (maps to your deck)
collapse (naïve DPO) → two-anchor fix → **significant, capability-preserving gain (DPO@60, paired stats)**
→ per-category AVHBench → **mechanistic twist** (grounds by rewriting vision — Clever-Hans) → **FiLM**
(the on-thesis fix: raises attention to AV tokens) → honest cross-backbone table (Qwen3 win, Qwen2.5
marginal, VideoLLaMA2 backbone-null) → next steps.

---

## 2. Strengthening plan (Q2) — ranked, with commands (all code already in-repo)

**#1 — Confirm FiLM at scale (the central open question).** FiLM is the only variant that raises
attention to AV tokens; whether that on-thesis mechanism also delivers a benchmark gain is *the*
unresolved question, and a clean win would let you reframe the paper around FiLM.
```bash
# full AVHBench + CMM on the val-selected FiLM step (160), then paired significance:
python -u -m rlvib.eval.run_avhbench --model qwen3-omni --fps 2 --limit 0 \
    --qa-json data/AVHBench/qa.json --video-root data/AVHBench/videos \
    --bottleneck runs/anchored_qwen3-omni_film/bottleneck_step160.pt \
    --out runs/avhbench_qwen3-omni_film_sysfull_step160.json
EXP=film BENCH=cmm bash scripts/eval_one.sh qwen3-omni 160
python scripts/paired_stats.py --model qwen3-omni --exp film --step 160 --suffix sysfull --dev 300
python scripts/paired_stats.py --model qwen3-omni --exp film --step 160 --vs broad:60 --suffix sysfull --dev 300
```

**#2 — Stronger objective: curriculum GRPO (high; validated on your backbone).** SARI [arXiv:2504.15900]
extends GRPO to a Qwen2.5-Omni-based audio LLM (SFT → curriculum-guided GRPO), reaching 67.08% MMAU
test-mini. Your GRPO code exists (`train_grpo.py`, `grpo_step`) but hit signal-collapse on *easy* yes/no
items (adv_std=0). The fix (your own diagnosis + SARI's curriculum) is **harder items → reward variance**:
train GRPO on the audio-**swap** counterfactuals, not the easy probe.
```bash
python scripts/train_grpo.py --help          # confirm args
# run GRPO on the swap items (harder -> reward variance), curriculum easy->hard:
python -u scripts/train_grpo.py --model qwen3-omni --group 4 --swap 1 --epochs 2   # adjust to its flags
```
*(mDPO [EMNLP 2024, arXiv:2406.11839] is the closest validated prior art and it validates your existing
design — image-swap conditional pairs + a reward anchor forcing positive chosen-reward — so cite it as
support for swap-DPO+chosen-anchor, not as a change to make.)*

**#3 — AVCD-style contrastive decoding (high; training-free, composable).** AVCD [NeurIPS 2025,
arXiv:2505.20862] dynamically identifies the *less-dominant modality* via attention and masks it to build
contrastive negatives (~2% VideoLLaMA2, ~7% video-SALMONN on AVHBench). Your repo has a simpler fixed-
modality VCD (`--audio-cd`); run it on the best checkpoint first (stacks at inference, no retraining):
```bash
python -u -m rlvib.eval.run_avhbench --model qwen3-omni --fps 2 --limit 0 \
    --bottleneck runs/anchored_qwen3-omni_film/bottleneck_step160.pt --audio-cd 1.0 \
    --qa-json data/AVHBench/qa.json --video-root data/AVHBench/videos \
    --out runs/avhbench_qwen3-omni_film_cd_step160.json
```
Upgrading `contrastive.py` to *dynamic* AVCD (attention-chosen modality) is a real, on-target build — flag
it if `--audio-cd` helps.

**#4 — Add MMAU (audio-understanding axis).** Shows the adapter helps audio *reasoning*, not just
hallucination; SARI's exact benchmark. Harness is built; needs the audios downloaded.
```bash
CKPTS="broad:60 film:160" bash scripts/launch_mmau.sh qwen3-omni
```

**#5 — Harder counterfactuals (medium; a small code change).** mDPO/AHA-style *counterfactual hard
negatives* strengthen a weak preference signal. Current swaps are different-category (clean mismatch);
adding **same-category / acoustically-confusable** swaps + the Tier-C abstention items (`data/pairs.py`)
forces finer audio discrimination. Not yet wired — I can add a `--hard-swap` option.

---

## 3. Do NOT cite (refuted in the verification pass)
- ✗ "GRPO gives +16.35% over Qwen2-Audio-7B" as evidence it beats your +5.2 (refuted 0-3).
- ✗ "AVHBench authors show LoRA+audio-alignment works via attention-refinement, supporting our attention
  thesis" (refuted 0-3) — the benchmark authors do **not** independently name attention-refinement; do not
  claim external support for the attention mechanism from AVHBench.
- ✗ "ablation-mismatch is a canonical accepted contribution type at Insights" (refuted 1-2) — soften.

## 4. Open questions (unresolved by the sources)
1. Does the significant gain + mechanistic finding clear a **main-venue** bar, or is a workshop the
   realistic target? (depends on effect size + FiLM confirmation).
2. Can **FiLM** be confirmed at scale — and if so, reframe the paper around it as the primary contribution?
3. Would a **corpus-level attribution audit** (SpRAy spirit, over the full held-out set) more rigorously
   establish the "grounds via vision" claim than the single-backbone attention probe?
4. Which *single* strengthening change most improves **audio grounding specifically** (vs audio QA
   accuracy)? — the lit validates the directions but does not rank them for the grounding metric.

## Sources (primary, verified)
Bowman, Dangers of Underclaiming (ACL'22, arXiv:2110.08300) · AVHBench (ICLR'25, arXiv:2410.18325) ·
Clever-Hans / SpRAy (Information Fusion'22, arXiv:1912.11425) · mDPO (EMNLP'24, arXiv:2406.11839) ·
AVCD (NeurIPS'25, arXiv:2505.20862) · SARI curriculum-GRPO (arXiv:2504.15900) · ICBINB workshop
(neurips.cc/…/66506) · Insights from Negative Results (insights-workshop.github.io; aclweb.org portal).
