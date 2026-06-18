# RLVIB — project handoff & cluster runbook

Context-transfer doc for a fresh session/branch. Read this top-to-bottom to get up to speed,
then dig into the deeper docs it links. Last updated 2026-06-18.

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
