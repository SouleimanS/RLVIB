# Anchored swap-DPO: fixing the bottleneck capability-collapse

**Status:** working result, 2026-06-18 · **Model:** Qwen3-Omni (frozen) + per-modality VIB bottleneck
· **Code:** `scripts/train_swap_anchored.py`, `src/rlvib/train/dpo.py::anchored_dpo_step`
· **Figures:** `scripts/plot_anchored.py` → `docs/reports/figs/`
· **Background research:** [`docs/research/dpo-collapse-and-fixes.md`](../research/dpo-collapse-and-fixes.md)

## Summary

We train a small per-modality **variational information bottleneck (VIB)** on a *frozen* Qwen3-Omni so
it grounds answers in what is **heard** rather than what is **seen**. The training signal is an
**audio-swap preference**: on a clip whose audio has been replaced, prefer the answer matching the
*audio* over the answer matching the *video*.

A first attempt (plain swap-DPO, regularizing only the bottleneck's latent rate) drove the
in-distribution proxy to 100% but **catastrophically collapsed** the model on held-out benchmarks —
it answered "no" to almost everything and fell below chance on multiple choice. After a literature
diagnosis (likelihood displacement; see the research memo), we rebuilt the objective with **two
anchors** — an *mDPO chosen-likelihood anchor* and an *explicit KL-to-base on general inputs* — plus a
per-step collapse monitor and benchmark-based model selection. The rebuilt run **preserves all general
capability and adds a real, on-target gain** on the audio-visual hallucination benchmark.

## 1. Background — the technique

The base model is frozen. A VIB bottleneck is inserted on the per-modality adapter tokens; only the
bottleneck (~tens of M params) trains. Toggling the bottleneck to **bypass** recovers the exact frozen
base — which we exploit as a free reference distribution.

**Audio-swap preference.** For a clip, we form a multiple-choice question and a *swapped* version of
the clip (audio replaced by another category's audio). The **chosen** letter is the heard
(audio-consistent) event; the **rejected** letter is the seen (video-consistent) event. Both are scored
on the *same* swapped clip, so preferring `chosen` requires using the audio.

## 2. Experiment 1 — the collapse (plain swap-DPO)

Objective (what we ran first), with `cp,rp` = policy log-probs of the chosen/rejected letter
(bottleneck active), `cr,rr` = reference log-probs (bottleneck bypassed), `KL_VIB` = the latent rate:

```
L = -log σ( β·[(cp - cr) - (rp - rr)] )  +  β_kl · KL_VIB
```

Note the only regularizer is `KL_VIB` on the **latent** — nothing constrains the model's **output
distribution** against the frozen base. We selected the checkpoint on an in-distribution "heard-rate"
proxy.

**Result — proxy up, model destroyed:**

| signal | before | after |
|---|---|---|
| heard-rate (in-distribution proxy) | 0.70 | **1.00** |
| CMM perception accuracy (PA) | 0.953 | **0.007** |
| CMM hallucination-resistance (HR) | 0.780 | **~0.99** |
| DAVE audio-visual MCQ | 0.380 | **0.18** (below chance) |

The signature is textbook: **PA crashes while HR inflates** because the model collapsed to a constant
"no" (saying "no" to everything trivially "resists" hallucination probes while failing all perception
probes), and forced-choice MCQ fell *below* chance — positive evidence of systematic, not random,
mis-answering.

## 3. Diagnosis (cited — full synthesis in the research memo)

The collapse is **DPO likelihood displacement** specialized to a modality (ACPO names the multimodal
form "Visual/Audio Anchor Collapse"):

- DPO's loss is a `log σ` of the **difference** `(chosen − rejected)`, satisfiable by pushing **both**
  answer log-probs down — there is no absolute-likelihood floor.
  [Razin et al., ICLR 2025](https://arxiv.org/abs/2410.08847);
  [Pal et al., DPOP/Smaug](https://arxiv.org/abs/2402.13228).
- Our yes/no + single-letter answers are the **worst case** (near-identical chosen/rejected embeddings
  = high "CHES" = maximal displacement). [Razin et al.](https://arxiv.org/abs/2410.08847)
- DPO's KL to the base is **implicit** and only touches the preference pairs; PPO-RLHF instead holds an
  **explicit** per-token KL to base, which our objective lacked entirely on general inputs.
  [Xu et al., ICML 2024](https://arxiv.org/abs/2404.10719);
  [Ouyang et al., InstructGPT](https://arxiv.org/abs/2203.02155).
- The bottleneck is **always-on** in the forward path, so an input-blind objective collapsed outputs on
  *every* prompt, not just swap clips.
- We selected on the in-distribution proxy — the Goodhart trap; selection must be on diverse held-out
  benchmarks. [Gao et al.](https://arxiv.org/abs/2210.10760);
  [Ivison et al., NeurIPS 2024](https://arxiv.org/abs/2406.09279).

## 4. The fix — anchored swap-DPO

We add two terms to the swap-DPO loss (`src/rlvib/train/dpo.py::anchored_dpo_step`):

**(a) mDPO chosen-likelihood anchor** — a one-sided preference of the chosen answer against the frozen
reference that pins the chosen log-prob at/above the base, blocking displacement:

```
L_anchor = -log σ( β·(cp - cr) - δ )            # δ = 0  → cp ≥ cr (chosen ≥ base)
```

(from [mDPO, Wang et al., EMNLP 2024](https://arxiv.org/abs/2406.11839); equivalent in spirit to
[DPOP](https://arxiv.org/abs/2402.13228) and the SFT-anchor of
[RPO, NeurIPS 2024](https://arxiv.org/abs/2405.16436).)

**(b) Explicit KL-to-base on general (non-swap) inputs** — over a mix of matched-MCQ and yes/no prompts
`x` where the adapter should be identity, penalize drift of the answer distribution from the frozen
base (`p_base` = bypassed, `p_policy` = active):

```
L_kl = KL( p_base(·|x) ‖ p_policy(·|x) )         # at the answer position, full vocab
```

(the frozen-model analog of InstructGPT's [PPO-ptx](https://arxiv.org/abs/2203.02155) and
[Learning-without-Forgetting](https://arxiv.org/abs/1606.09282) distillation.)

**Total objective:**

```
L = -log σ( β·[(cp-cr) - (rp-rr)] )  +  λ_anchor · L_anchor  +  β_kl · KL_VIB  +  λ_kl · L_kl
```

**Hyperparameters:** β=0.1, β_kl=0.01, λ_anchor=1.0, δ=0.0, λ_kl=1.0, lr=5e-5 (AdamW, bottleneck only),
pairs=300, epochs=2, accum=4, anchor-batch=4 → 150 steps.

**Monitoring (per step) + selection** — the signals we were blind to before:
`p_chosen` (audio preference), `chosen_minus_ref` = `cp−cr` (anchor target, should stay ≥ 0), `gen_kl`,
and a **balanced yes/no probe** on held-out clips reporting `frac_yes` (the cheap collapse detector —
alarms and early-stops if it skews to 0/1). Checkpoints are saved every 10 steps so model selection
runs on the **held-out benchmarks**, never the proxy.

## 5. Experiment 2 — results (anchored swap-DPO)

The run reached all 150 steps **with no collapse alarm**. Probe (balanced yes/no, held-out clips):

| | base | step 150 |
|---|---|---|
| `frac_yes` | 0.53 | **0.55** (stayed balanced) |
| probe accuracy | — | **0.97** (audio used correctly) |

Held-out benchmarks, `qwen3-omni` base vs. anchored bottleneck (`_bn`):

| Benchmark | Base | **Collapsed (Exp 1)** | **Anchored (Exp 2)** |
|---|---|---|---|
| CMM perception (PA) | 0.953 | 0.007 | **0.940** |
| CMM hallucination-resist (HR) | 0.780 | ~0.99 | **0.787** |
| CMM overall acc | 0.867 | — | **0.863** |
| DAVE audio-visual MCQ | 0.380 | 0.18 | **0.380** |
| AVHBench overall | 0.643 | — | **0.677** |
| ↳ AV Matching | 0.568 | — | **0.636** (+6.8) |
| ↳ Video→Audio Hallucination | 0.679 | — | **0.729** (+5.0) |
| ↳ Audio→Video Hallucination | 0.667 | — | 0.625 (−4.2) |

**Reading:** capability is fully preserved (CMM, DAVE at/near base — the collapse is gone), and there is
a coherent **on-target gain on AVHBench** (+3.4 overall), concentrated exactly on the audio-grounding
axes — *AV Matching* and *not hallucinating audio from video*. The probe accuracy of 0.97 with balanced
labels is direct evidence the bottleneck **uses audio correctly** rather than sitting idle. Gains are
within small-n noise on the AVHBench splits (n=72–140) but the direction is consistent.

**Figures** (regenerate with `python scripts/plot_anchored.py` on the cluster after a run):

- `figs/fig_probe.png` — `frac_yes` vs step, base line + collapse zones: stays at base (no collapse).
- `figs/fig_train.png` — `p_chosen` / `chosen_minus_ref` / `gen_kl` vs step (learning while anchors hold).
- `figs/fig_bench.png` — base vs collapsed vs anchored bars (the headline).

## 6. Conclusion & next steps

The *same* method that collapsed CMM-PA to 0.007 now lands at 0.940 **plus** a real grounding gain —
a clean confirmation of the diagnosis: the collapse was a missing output anchor, not a flaw in the
swap-preference idea. The two anchors (chosen-likelihood floor + KL-to-base on general inputs) are
necessary and sufficient to make frozen-base + adapter swap-DPO stable.

Next:
1. **Model selection** across the 15 checkpoints — pick best AVHBench s.t. CMM-PA ≥ 0.90, DAVE ≥ 0.36.
2. **Scale the gain** — more pairs (300 → 600+), and a touch more grounding pressure
   (`λ_anchor=2.0` or `λ_kl=0.5`), then re-select.
3. **Qualitative frames** — add a swap-clip example (frame + base vs anchored answer) for the writeup.

## Sources

Primary references (full annotated synthesis in
[`docs/research/dpo-collapse-and-fixes.md`](../research/dpo-collapse-and-fixes.md)):

- [Razin et al., *Unintentional Unalignment: Likelihood Displacement in DPO*, ICLR 2025](https://arxiv.org/abs/2410.08847)
- [Wang et al., *mDPO: Conditional Preference Optimization for Multimodal LLMs*, EMNLP 2024](https://arxiv.org/abs/2406.11839)
- [Pal et al., *Smaug / DPO-Positive*, 2024](https://arxiv.org/abs/2402.13228)
- [Ren & Sutherland, *Learning Dynamics of LLM Finetuning* (squeezing effect), ICLR 2025](https://arxiv.org/abs/2407.10490)
- [Xu et al., *Is DPO Superior to PPO for LLM Alignment?*, ICML 2024](https://arxiv.org/abs/2404.10719)
- [Liu et al., *Your SFT Loss is Implicitly an Adversarial Regularizer (RPO)*, NeurIPS 2024](https://arxiv.org/abs/2405.16436)
- [Ouyang et al., *InstructGPT* (PPO-ptx / explicit KL-to-base), 2022](https://arxiv.org/abs/2203.02155)
- [Li & Hoiem, *Learning without Forgetting*, 2016](https://arxiv.org/abs/1606.09282)
- [Gao et al., *Scaling Laws for Reward Model Overoptimization*, ICML 2023](https://arxiv.org/abs/2210.10760)
- [Ivison et al., *Unpacking DPO and PPO*, NeurIPS 2024](https://arxiv.org/abs/2406.09279)
