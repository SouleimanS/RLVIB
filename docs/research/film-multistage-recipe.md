# Verifying Qwen3-Omni + the prompt-aware FiLM bottleneck — recipe & run guide

> Date: 2026-06-29 · Status: implemented + run guide.
> Basis: a 3-angle deep-research fan-out — (A) a faithfulness audit of the retrain/eval pipeline,
> (B) the FiLM module design, (C) the multi-stage training recipe — synthesised here. Extends
> `query-conditioned-bottleneck.md` (the GO decision + FiLM scope-1 design), `dpo-collapse-and-fixes.md`
> (the anchored objective), and the unconditional-rewrite finding in `../reports/02-model-and-training.md`.

Two goals: **(1)** make sure the Qwen3-Omni numbers are *accurate* by retraining + re-benchmarking
under a trustworthy harness, and **(2)** add a **prompt-aware (FiLM) bottleneck** trained the right
way (multi-stage), so grounding is *question-routed* rather than a static vision rewrite.

---

## Part A — verify the Qwen3 numbers (retrain + re-benchmark)

The objective/data/loss are correct (audit confirmed: swap contrast `chosen=heard / rejected=seen`
is the right direction, the KL(base‖policy) anchor and bypass-reference are right, eval is
deterministic). Three things threatened *number validity*; all three are addressed:

| # | Issue | Fix (this branch) |
|---|---|---|
| **M1** | **Selection-on-test bias.** The default `select_checkpoint.py` picks `argmax(AVHBench)` over the **full** reported set and reports that same number — an optimistic bias. | Use `select_holdout.py` (selects on a fixed val half, reports on the disjoint test half). It already exists; it is now the documented selection step (commands below). The step60 "broad" pick must be **re-derived on val**. |
| **M2** | **fps not pinned.** `eval_one.sh` forced `--fps 2` for qwen3, but the qsub/selection path used the processor default → selection and final eval could silently differ (qwen3 starves → yes-bias at fps<2). | `run_avhbench.py` / `run_cmm.py` now **default fps per-model** (qwen3→2, qwen2.5→1) *regardless of launcher*; explicit `--fps` still overrides. So every path is pinned. |
| **M3** | **No exact checkpoint reproducibility** (un-seeded VIB reparam + CUDA nondeterminism). | Inherent; eval is deterministic (z=μ, greedy). Report **mean±std across seeds**, not "same checkpoint twice" (`SEED=0/1/2`, `aggregate_ci.py`/`paired_stats.py`). |

Also note **S1 (set expectations):** the docs' old `0.703 / PA 0.927 / HR 0.853` figures are
**pre-protocol-fix** (before system-prompt removal + `"Answer yes or no."`). A fresh run reproduces
the **corrected-harness** magnitudes (base A→V ≈ 0.84), not those. And **N4 (faithfulness):** with the
*unconditional* VIB the gain is mechanistically a **vision-stream rewrite**, not audio grounding —
which is exactly what Part B fixes.

### A.1 Commands (ABCI; run from the repo root on the login node, git with `LD_LIBRARY_PATH` cleared)

```bash
cd ~/SOULEIMAN_repo/RLVIB
env -u LD_LIBRARY_PATH git pull origin claude/blissful-ride-n9dras

# (0) one-time leakage check (AVE train vs AVHBench/CMM clips) -- needs data/ (cluster)
source /home/aab11336im/anaconda3/etc/profile.d/conda.sh && conda activate rlvib
PYTHONPATH=src python scripts/check_overlap.py        # expect "clean: no AVE clip ids ..."

# (a) RETRAIN Qwen3-Omni, unconditional VIB, "broad" recipe (lam_kl=2) -- this is Stage 1
EXP=broad MODEL=qwen3-omni LAMKL=2.0 PAIRS=300 EPOCHS=2 SEED=0 \
  qsub -v EXP,MODEL,LAMKL,PAIRS,EPOCHS,SEED scripts/train_swap_anchored.qsub
#   -> runs/anchored_qwen3-omni_broad/bottleneck_step{10..150}.pt   (frac_yes must stay ~0.4-0.6)

# (b) selection grid (LIMIT=300) then HONEST val/test selection (fixes M1)
EXP=broad MODEL=qwen3-omni LIMIT=300 STEPS="30 60 90 120 150" FPS=2 bash scripts/select_checkpoint.sh
python scripts/select_holdout.py --model qwen3-omni --exp broad      # -> HONEST step (on val)

# (c) full re-benchmark of base + the val-selected step <S>, then read the table
qsub -I -P gae50891 -q rt_HF -l select=1 -l walltime=06:00:00        # interactive GPU node
#   inside the node (eval_one.sh pins fps=2 and resumes if cut off):
bash scripts/eval_one.sh qwen3-omni                                  # base, full AVHBench+CMM
bash scripts/eval_one.sh qwen3-omni <S>                              # the broad checkpoint @ step<S>
python scripts/make_table.py                                        # corrected-harness numbers
# significance on the held-out tail:
python scripts/paired_stats.py --model qwen3-omni --exp broad --step <S> --suffix sysfull --dev 300
```

For CIs, repeat (a) with `SEED=1 EXP=broad_s1` / `SEED=2 EXP=broad_s2` and pool with `paired_stats.py`.

---

## Part B — the prompt-aware FiLM bottleneck (what was added)

`FiLMVariationalBottleneck` (in `src/rlvib/models/bottleneck.py`), a subclass of the VIB that
conditions on a pooled **question** embedding `q`:

```
h          = GELU(enc(x))
gamma,beta = film(q).chunk(2)        # film zero-init  -> gamma=beta=0 at init
h'         = (1 + clamp(gamma)) * h + beta
mu,logvar  = to_mu(h'), to_logvar(h')
z          = mu + sigma*eps (train) | mu (eval)
g          = sigmoid(gate(q))        # gate bias +4 -> g~=0.98 (open) at init
y          = x + g * out(z)          # out zero-init -> y == x at init, for ANY q,gamma,beta,g
```

Key properties (verified in `tests/test_bottleneck_film.py`):
- **Identity-at-init for any `q`** (out is zero-init) ⇒ `bypass`→exact-base and the anchored-DPO
  reference are untouched; attaching the untrained module reproduces the frozen base exactly.
- **`q` comes from the frozen LLM's own embedding table** — `question_embedding(model, question)`
  tokenizes the question text only, looks it up in `thinker.get_input_embeddings()`, mean-pools.
  No new encoder, no new deps, `q` lives in the model's token space.
- **Out-of-band plumbing:** the adapter hook only sees the adapter output `x`, never the question,
  so `set_condition(bottlenecks, q_emb)` projects `q` once (shared `q_proj`) and stashes it on both
  modality bottlenecks before each forward. It is a **no-op for a plain VIB**, so train/eval loops
  call it unconditionally. Wired into `dpo.anchored_dpo_step`, `run_avhbench.py`, `run_cmm.py`,
  and the trainer's probes.
- **Per-modality output gate `g(q)`** routes by question: keep audio when asked about sound, vision
  when asked about looks. Started "open" (bias +4) so the gate learns to *close the irrelevant*
  stream rather than open audio from a dead 0. The live gate is exposed for the gate-usage term and
  the routing probe.

---

## Part C — the multi-stage training recipe (the "do it right")

**Two-stage, warm-started.** The empirical pathology to break: the unconditional VIB grounds by
**rewriting ~59% of every vision token** while the **audio module is ≈ pass-through**. A gate trained
on HEAR questions only has no reason to ever route differently — it will reproduce the vision-rewrite
shortcut. The cure is fundamentally a **data-composition** fix.

### Stage 1 — unconditional VIB (the grounding direction)
The existing anchored recipe, **unchanged** (`= Part A (a)`): `lr 5e-5, β 0.1, λ_anchor 1, λ_kl 2
(broad), β_kl 0.01`, ~150 steps. Cheap, validated, non-collapsing. Select on the held-out benchmark.
Output: `runs/anchored_qwen3-omni_broad/bottleneck_step<S>.pt`.

### Stage 2 — FiLM + gate, question-routed
Warm-start the FiLM module from the Stage-1 core (`--init-from`; FiLM/gate keep their identity
init, non-strict load), then learn routing.

- **THE crux — HEAR vs SEE on the SAME swapped clip.** For each swapped clip (seen = A, heard = B):
  - **HEAR** "which do you HEAR?": chosen = B (heard), rejected = A (seen) → suppress vision.
  - **SEE** "which do you SEE?": chosen = A (seen), rejected = B (heard) → suppress audio.
  The gold answer **flips with the question word**, and the only input distinguishing the two
  forwards is `q`. A question-independent gate cannot satisfy both → the unique minimiser is a
  `g(q)` that routes by question type. `make_see_mcq` (in `ave.py`) builds the SEE counterpart on
  the same clip (no new ffmpeg); `--see-frac 1.0` makes it 1:1 HEAR/SEE. **Without SEE batches no
  gate term reliably works** — this is the load-bearing anti-pass-through mechanism.
- **Phase 2a (core-frozen warmup, `--warmup-steps 80`):** train only `film/gate/q_proj` on the
  fixed, already-grounded edit (lowest collapse risk window). **Phase 2b:** unfreeze, joint.
- **Losses (Stage 2):** `lr 3e-5` (+10% linear warmup), `λ_anchor 1.5` (extra displacement
  insurance while the gate moves), `λ_kl 2` (anchor set already spans HEAR/SEE/general),
  **`β_kl 0.05`** (5× — real compression pressure doubles as a query→answer shortcut guard),
  optional **`λ_gate 0.05`** gate-usage hinge (push the question-relevant gate above 0.6;
  symmetry-breaking insurance, default off — the SEE data does the real work).

### Commands

```bash
# Stage 1 = Part A (a). Pick the val-selected step <S> as the warm-start.
# Stage 2 (interactive recommended -- many flags; eval_one-style env):
qsub -I -P gae50891 -q rt_HF -l select=1 -l walltime=06:00:00
source /home/aab11336im/anaconda3/etc/profile.d/conda.sh && conda activate rlvib
export PYTHONPATH=$PWD/src CUDA_VISIBLE_DEVICES=0 TRANSFORMERS_OFFLINE=1 \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u scripts/train_swap_anchored.py --film \
  --init-from runs/anchored_qwen3-omni_broad/bottleneck_step<S>.pt \
  --save-dir runs/anchored_qwen3-omni_film \
  --pairs 400 --epochs 3 --accum 4 --see-frac 1.0 --warmup-steps 80 \
  --warmup-frac 0.1 --lr 3e-5 --lam-anchor 1.5 --lam-kl 2 --beta-kl 0.05 \
  --lam-gate 0.05 --gate-target 0.6

# ... or as a batch job (same recipe via env vars):
EXP=film FILM=1 INIT_FROM=runs/anchored_qwen3-omni_broad/bottleneck_step<S>.pt \
  PAIRS=400 EPOCHS=3 SEEFRAC=1.0 WARMUPSTEPS=80 WARMUPFRAC=0.1 LR=3e-5 \
  LAMANCHOR=1.5 LAMKL=2.0 BETAKL=0.05 LAMGATE=0.05 \
  qsub -v EXP,FILM,INIT_FROM,PAIRS,EPOCHS,SEEFRAC,WARMUPSTEPS,WARMUPFRAC,LR,LAMANCHOR,LAMKL,BETAKL,LAMGATE \
  scripts/train_swap_anchored.qsub

# Evaluate a FiLM checkpoint (load_attached restores the FiLM module; the runners auto-detect
# q_proj and set the question condition per item):
EXP=film bash scripts/eval_one.sh qwen3-omni <S2>
```

---

## Part D — monitors & acceptance

- **Collapse (keep):** `frac_yes ∈ [0.15, 0.85]` (alarm), hard-stop <0.05/>0.95; `chosen_minus_ref ≥ 0`.
- **Routing probe (new, printed every eval as `[route ...]`):** forwards a fixed set of swapped
  clips under HEAR and SEE and reports the relative edit `‖edit‖/‖x‖` per modality × question-type
  plus `d_audio = rel_a|H − rel_a|S` and `d_vision = rel_v|S − rel_v|H`.
  **Success = `d_audio > 0` and `d_vision > 0`** (each modality edited more on the question that needs
  it), with `rel_a|H` clearly above the Stage-1 audio baseline (≈0). A `ROUTING FAIL` flag prints
  when `d_audio ≤ 0` → raise `--lam-gate`, keep `--see-frac` balanced, or extend `--warmup-steps`.
- **Faithfulness gate (selection, unchanged):** accept only if AVHBench rises **and** CMM-HR ≥ base
  (HR cratering while AVHBench rises = the yes-bias artifact, not grounding), PA ≥ ~0.90. Select on
  the held-out split, never on the full reported set.

The headline result for the FiLM claim is the routing-probe trajectory: `d_audio`/`d_vision` rising
above 0 — direct evidence grounding became **question-routed**, not a static vision rewrite.

---

## Files changed

- `src/rlvib/models/bottleneck.py` — `FiLMVariationalBottleneck`, `question_embedding`,
  `set_condition`; `attach_bottlenecks` (cond_dim + shared `q_proj`); `set_bypass`/`load_attached`
  (FiLM-aware).
- `src/rlvib/data/ave.py` — `make_see_mcq` (SEE counterpart; reuses the swapped clip).
- `src/rlvib/train/dpo.py` — `anchored_dpo_step` sets the condition per forward; optional gate-usage
  hinge (`lam_gate`/`gate_target`); returns per-modality gate means.
- `scripts/train_swap_anchored.py` — `--film/--init-from/--see-frac/--warmup-steps/--warmup-frac/`
  `--lam-gate/--gate-target`; HEAR+SEE batch building; core freeze/unfreeze; LR warmup; routing
  probe; checkpoint records `cls`+`cond_dim`. **Stage-1 path (no `--film`) is unchanged** (same RNG
  stream, same checkpoint format).
- `scripts/train_swap_anchored.qsub` — optional FiLM env-var passthrough (empty ⇒ Stage-1 default).
- `src/rlvib/eval/run_avhbench.py`, `run_cmm.py` — per-model fps default (M2); set the FiLM
  condition per item when a FiLM checkpoint is attached.
- `tests/test_bottleneck_film.py` — identity-at-init (cond/uncond, 2D/3D), `set_condition` no-op,
  gradient flow, eval determinism.

## References
FiLM (Perez et al., AAAI'18, arXiv:1709.07871); identity-at-init `γ=1+Δγ` (Temporal-FiLM,
arXiv:1909.06628); gate-init/stability (Gu et al., ICML'20); mDPO conditional preference
(EMNLP'24, arXiv:2406.11839); query-conditioning ≠ faithful grounding (FPVG, EMNLP'23-F,
arXiv:2305.15015; "wrong reasons", ACL'20, arXiv:2004.05704); modality-collapse gate regularization
(arXiv:2505.15417). Full bibliography in `query-conditioned-bottleneck.md`.
