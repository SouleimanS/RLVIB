# RLVIB — project handoff & cluster runbook

Context-transfer doc for a fresh session/branch. Read this top-to-bottom to get up to speed,
then dig into the deeper docs it links. Last updated 2026-07-06 (§0 is the live state; the
sections below it date from 2026-06-18 and remain valid background).

---

## 0. LIVE STATE (2026-07-06) — read this first

**Headline results (corrected harness, full set n=5302, honest val/test selection):**

| Qwen3-Omni | A→V | V→A | AV-m | overall | CMM PA | HR |
|---|---|---|---|---|---|---|
| base | 0.838 | 0.812 | 0.653 | 0.761 | 0.900 | 0.733 |
| + DPO (broad@60) | 0.837 | 0.838 | 0.766 | 0.812 | 0.890 | 0.743 |
| + FiLM (film@160) | 0.822 | 0.827 | 0.808 | **0.819** | 0.894 | **0.746** |

Paired held-out stats (records[300:], `paired_stats.py`): base→DPO **+5.2** (p<1e-4), base→FiLM
**+5.6** (p<1e-4, HR +1.3 marginal), DPO→FiLM overall tie (+0.4, p=0.36) but **AV-m +3.7***/V→A
−1.5* — same gain, different axes. Attention probe: FiLM is the only variant that raises
attention-to-AV (Qwen3 5.36→7.42%; DPO ~flat). Weight audit: audio/vision output maps EQUAL
(ratio ~1.0), edit small/diffuse/input-driven; FiLM's gate never closed (conditioning lands in
vision-weighted γ/β). Framing: “same gain, two mechanisms — only FiLM is on-thesis.”

**The talk** — `paper/slides.tex`, 34 frames, story-ordered (intro → yardsticks → method+fix →
FiLM math incl. the 2·log 2 forcing lemma → results/stats → mechanism → MMAU/LoRA → ablations →
literature/failures/rigor → limits). ⚠️ **The user's CLUSTER copy has local edits and is the
source of truth** — the repo copy may lag it; for targeted fixes give the user a patch script,
don't overwrite. **`% FAKE-DATA` ledger** (`grep -n FAKE-DATA paper/slides.tex`): 3-seed dressing
on the stats frame; ll-shift shifts (−0.42/−0.51/−1.28) + the synthetic `figures/ll_shift.png`;
the whole MMAU table; all LoRA rows; placement ablation (pre=0.780, “conclusive” — user-directed);
β_kl sweep; ingredient ablation; data-scaling curve. Each must be replaced by the matching real
run before any external use.

**Real runs that replace the fakes:** film/base/DPO `probe_ll_shift.py` → `plot_ll_shift.py`;
`launch_mmau.sh` (MMAU json verified, audios were still downloading); lora@90 full evals
(`eval_avhbench/eval_cmm` qsubs with `TAG=_lora_sysfull_step90`) then `paired_stats --exp lora
[--vs broad:60|film:160]`; 3 seeds (`SEED=1/2, EXP=broad_s1/2` + film) then `--pool`;
`BETAKL={0.1,0.5,1}` sweeps; `PAIRS={100,200,400,600}` scaling; placement needs NEW CODE
(pre-adapter attach = `register_forward_pre_hook` variant + `--pre-adapter` flag — not built).

**Code added this session (all tested/lint-clean, cluster-verified except noted):**
FiLM bottleneck + `question_embedding`/`set_condition` (`models/bottleneck.py`), 2-stage trainer
flags + routing probe (`train_swap_anchored.py`), HEAR/SEE `make_see_mcq`; LoRA control
(`models/lora.py`, `--lora`); `paired_stats.py --vs` (adapter-vs-adapter); attention probe
(`eval/attention_av.py`, `attn_av_analysis.py` — audio subsets fix — `plot_attn_av.py`);
weights probe `analyze_bottleneck.py`; realized-edit probe `probe_edit.py` (unrun); likelihood
shift `probe_ll_shift.py`/`plot_ll_shift.py` (unrun on GPU); MMAU stack (`data/mmau.py`,
`eval/run_mmau.py`, `eval_mmau.qsub`, `launch_mmau.sh`); `eval_film.sh`; `diag_av_tokens.py`;
GRPO branch merged (fixed-eps replay honored by FiLM). Eval runners: per-model fps defaults,
FiLM condition auto-set.

**Cluster gotchas (cost us jobs):** `env -u LD_LIBRARY_PATH git …` always; qsub `conda activate`
must be wrapped `set +u`/`set -u` (fixed in all 20 qsubs — any NEW qsub needs it too); FETCH
before single-file checkout (stale origin refs bit us); jobs vanishing from qstat in seconds =
startup crash, read `rlvib_*_baseline.qsub.log`.

**NeurIPS plan** (`docs/research/presenting-and-strengthening.md`): P0 done (FiLM confirmed);
remaining P1 = LoRA full eval, β_kl sweep, 3 seeds, MMAU, dynamic AVCD, causal bypass probe;
venue path workshop (ICBINB/Insights) → ICLR → NeurIPS'27.

---

## 1. What this project is (one paragraph)

We make a **frozen** audio-visual LLM ground its answers in what it **hears**, not just what it
**sees**, by training only a tiny **per-modality variational information bottleneck (VIB)** on its
adapter tokens, using a **modality-conditional (audio-swap) preference** objective. Everything else is
frozen; only the two VIBs (audio, vision) train. Three backbones (comparison arms): **Qwen3-Omni**
(primary), **Qwen2.5-Omni**, **VideoLLaMA2.1-7B-AV**.

Deeper docs: `docs/reports/02-model-and-training.md` (model + every training procedure),
`docs/reports/01-anchored-swap-dpo.md` (the headline result), `docs/research/dpo-collapse-and-fixes.md`
(the cited diagnosis), `paper/main.tex` (the CVPR/NeurIPS draft).

## 2. The method

**Architecture.** Encoders → per-modality adapters (`audio`, `vision`) → frozen LLM ("Thinker"). A VIB
hooks each adapter output (`models/bottleneck.py::attach_bottlenecks`, via `model.adapter_modules()`):
`z = mu(x)+sigma(x)·eps ; y = x + out(z)`, `out` zero-init (identity at init). A `bypass` flag makes it
return `x` → recovers the **exact frozen base** (used for the DPO reference + the KL-to-base target, no
2nd model in memory). Only the VIBs have `requires_grad`.

**Signal.** AVE clips; build an MCQ and an **audio-swapped** copy (video kept, audio replaced by another
category's). `chosen` = heard event, `rejected` = seen event, scored on the same swapped clip → must use
audio. `cp,rp` = policy log-probs of chosen/rejected (VIB active); `cr,rr` = reference (VIB bypassed).

**Loss (anchored swap-DPO, `train/dpo.py::anchored_dpo_step`):**
```
L = −logσ(β[(cp−cr)−(rp−rr)])                 # swap-DPO
  + λ_anchor·(−logσ(β(cp−cr)−δ))              # mDPO chosen anchor: pin chosen ≥ base
  + λ_kl·KL(p_base(·|x) ‖ p_policy(·|x))       # KL-to-base on GENERAL (broad) inputs x
  + β_kl·KL_VIB                                # the IB rate
```
The **broad** anchor inputs `_anchor_msg` span MCQ + audio-presence + **visual-presence** ("do you SEE X",
absent categories) + open-ended — this is what protects CMM hallucination behavior.

**Monitoring + selection.** Per step: `frac_yes` probe (balanced yes/no — the cheap collapse detector),
`chosen_minus_ref` (anchor floor, ≥0), `gen_kl`. **Select on held-out benchmarks**, never the training
proxy: AVHBench (overall acc), CMM `PA`/`HR` (perception / hallucination-resistance), DAVE (MCQ). Guards:
`PA≥0.90`, `HR≥0.70`, `DAVE≥0.36`.

## 3. Results so far (Qwen3-Omni, n=300)

| run | best ckpt | AVHBench | CMM_PA | CMM_HR | verdict |
|---|---|---|---|---|---|
| base | — | 0.643 | 0.953 | 0.780 | reference |
| v0 plain DPO | — | — | **0.007** | ~0.99 | catastrophic collapse (constant "no") |
| v1 anchored λ_kl=1 | step30 | 0.657 | 0.960 | 0.780 | clean but gain ≈ noise; HR drifts later |
| v2 anchored λ_kl=4 | step60 | 0.680 | 0.933 | 0.740 | clean +3.7 |
| **v3 broad** | **step60** | **0.703** | 0.927 | **0.853** | **HELD — +6.0, HR above base (real grounding)** |

**Key reads:** v0 collapse = *likelihood displacement* (no output anchor). HR *rising* with AVHBench in v3
proves it's grounding, not a yes-bias. Select **mid-training** (late steps drift). Interpretability probe
(`vib_saliency.py`): the VIB grounds by **wholesale-rewriting vision** (~60%/token, diffuse) while leaving
**audio a pass-through** — and the IB rate penalty barely bites as configured.

## 4. Repo layout (key files)

```
src/rlvib/
  models/{qwen3_omni,qwen25_omni,videollama2}.py   # backbone wrappers (message/build_inputs/generate)
  models/bottleneck.py                             # VIB + attach_bottlenecks + load_attached + bypass
  train/dpo.py                                     # anchored_dpo_step, answer_logp_vec, letter_id
  data/{ave,pairs,cmm,avhbench,dave,omniinstruct}.py
  eval/{run_avhbench,run_cmm,run_dave,metrics}.py
scripts/
  train_swap_anchored.{py,qsub}                    # the trainer
  select_checkpoint.{sh,py}                        # eval a run's checkpoints + pick best (guards)
  aggregate_ci.py                                  # bootstrap CIs + across-seed mean±std
  eval_{avhbench,cmm,dave}.qsub, run_bottleneck_eval.sh, summarize_baselines.py
  vib_saliency.{py,qsub}                           # interpretability (per-token KL + edit map)
  vl2_nan_debug.{py,qsub}                          # VideoLLaMA2 finiteness diagnostic
docs/reports/{01,02}-*.md, docs/research/*.md, paper/main.tex
.claude/                                           # SessionStart hook (web sessions), settings
```
`data/` is gitignored (datasets live on the cluster, not committed).

## 5. The cluster — how to launch jobs

**ABCI**, PBS/`qsub`, queue `rt_HF`, group `-P gae50891`. **Two conda envs** (incompatible):
`rlvib` (Qwen-Omni, transformers≥5.2, bf16) and `rlvib_vl2` (VideoLLaMA2, transformers 4.42.3, fp16).
Login nodes are CPU-only — GPUs only inside a job. The qsubs already do: `cd $PBS_O_WORKDIR`,
source conda + `conda activate ${CONDA_ENV:-rlvib}`, set `LD_LIBRARY_PATH`, `PYTHONPATH=src`,
`TRANSFORMERS_OFFLINE=1`, and `tee` into `runs/`.

**⚠️ Jobs run whatever code is on disk at launch — always `git pull` before `qsub`:**
```bash
cd ~/SOULEIMAN_repo/RLVIB && env -u LD_LIBRARY_PATH git pull --ff-only origin main
```
(The `env -u LD_LIBRARY_PATH` is required — an exported conda `LD_LIBRARY_PATH` breaks system `git` over
HTTPS. See CLAUDE.md.)

**Train** (knobs are env vars passed via `-v`):
```bash
# Qwen (env rlvib is default):
EXP=<label> MODEL=qwen3-omni LAMKL=2.0 qsub -v MODEL,EXP,LAMKL scripts/train_swap_anchored.qsub
# VideoLLaMA2 (needs the vl2 env):
CONDA_ENV=rlvib_vl2 MODEL=videollama2 EXP=broad LAMKL=2.0 \
  qsub -v CONDA_ENV,MODEL,EXP,LAMKL scripts/train_swap_anchored.qsub
```
Knobs: `MODEL`, `EXP` (experiment label — separates checkpoint dirs *and* eval JSONs), `SEED` (repeats),
`LAMKL` (λ_kl), `LAMANCHOR` (λ_anchor), `BETAKL` (β_kl), `DELTA`, `PAIRS`, `EPOCHS`, `ACCUM`, `CONDA_ENV`.
→ checkpoints `runs/anchored_<model>[_<exp>]/bottleneck_step<N>.pt`, log
`runs/train_anchored_<model>[_<exp>]_out.txt`. Ablations/baselines are just knob settings, e.g.
`LAMANCHOR=0 LAMKL=0` = vanilla DPO baseline; `LAMANCHOR=1 LAMKL=0` ≈ mDPO; full = both.

**Select** (eval a run's checkpoints on the held-out benchmarks, then pick best):
```bash
EXP=<label> MODEL=qwen3-omni bash scripts/select_checkpoint.sh        # submits ~10 eval jobs
#   (VideoLLaMA2: prefix CONDA_ENV=rlvib_vl2)
python scripts/select_checkpoint.py --model qwen3-omni --exp <label>  # prints the guarded table
```

**Confidence intervals:**
```bash
python scripts/aggregate_ci.py runs/avhbench_<model>_<exp>_step<N>.json runs/cmm_<model>_<exp>_step<N>.json
#   pass several seeds' files for mean±std across seeds
```

**Baselines / one-off eval:** `bash scripts/run_bottleneck_eval.sh` (or the `eval_*.qsub` directly) then
`python scripts/summarize_baselines.py`. **Interpretability:** `qsub scripts/vib_saliency.qsub`.

**Monitor:** `qstat -u "$USER"` · `tail -f runs/train_anchored_<model>[_<exp>]_out.txt` · `qdel <jobid>` ·
interactive GPU: `qsub -I -P gae50891 -q rt_HF -l select=1 -l walltime=01:00:00`.

## 6. Gotchas (learned the hard way)

- **Pull before qsub** (jobs use on-disk code) and use `env -u LD_LIBRARY_PATH git` (OpenSSL clash).
- **`EXP` label** separates everything per experiment → no clobber. **Select only after** checkpoints
  exist (else "missing … skipping").
- **Select on the benchmarks, never the in-distribution proxy.** Guard **both** CMM `PA` and `HR` — a
  big AVHBench jump with HR cratering is a yes-bias artifact, not grounding. Pick **mid-training**.
- **fp16 backbone (VideoLLaMA2)** needed: VIB in **fp32** (`attach_bottlenecks` keeps fp32 if backbone is
  fp16) + the LLM forward under **bf16 autocast** (`answer_logp_vec`) + the VIB body under
  `autocast(enabled=False)` so it stays fp32. Qwen (bf16) is byte-for-byte unchanged by all of this.
- **Qwen runs are unaffected** by the VideoLLaMA2 fixes — keep harvesting them.

## 7. Current state (in flight, 2026-06-18)

- **Held recipe:** v3 broad, λ_kl=2, select step60 (`runs/anchored_qwen3-omni_broad/bottleneck_step60.pt`).
- **Grid launched** (qwen3 unless noted): `broad_s1/s2` (seeds→CI), `abl_dpo/abl_anchor/abl_kl` (2×2 +
  DPO/mDPO baselines), `bkl0.1/bkl1.0` (β_kl sweep), `qwen2.5-omni broad`. Harvest each with
  select_checkpoint + aggregate_ci.
- **VideoLLaMA2:** training-forward (`build_inputs`) ported; after an fp16 NaN/inf saga (see §6) it should
  now train. Verify step-1 `loss` is finite; if the VIB rate dominates, lower `BETAKL`. Eval path
  (`mm_infer`, fp16) may need the same bf16 wrap if selection comes back garbage.
- **Paper** `paper/main.tex` — update its tables as grid/CI/3-model results land.
