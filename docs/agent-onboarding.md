# RLVIB — agent onboarding (whole-project + repo map)

Read this first if you're a new agent picking up RLVIB. It's the practical map: what the project
is, how the repo is laid out, how to run the evals, the cluster setup, and — most importantly — the
state and hard-won fixes from the most recent work. For the *research/method* depth, follow the
pointers to `docs/HANDOFF.md`, `docs/reports/`, and `docs/research/`; this file does not duplicate
them.

---

## 1. What the project is

RLVIB trains a **tiny variational information bottleneck (VIB) adapter on top of a FROZEN
audio-visual LLM** to improve **audio grounding** (the model's tendency to hallucinate sounds /
over-affirm presence). The adapter is trained with a **"swap-DPO"** objective that uses **two
anchors** (an mDPO chosen-anchor + a KL-to-base on broad inputs) to prevent the naive-DPO failure
mode of **collapsing to a constant "no."** It is a clean-room rebuild of an earlier project
(`av_ib`); the old code and the local AVQA dataset were corrupted and are **not** reused.

The scientific result the paper reports is the **base → `+ours` delta** on audio-visual
hallucination benchmarks under one consistent harness — not absolute SOTA. A yes-biased frozen base
is the *correct, expected* starting point (documented behaviour of Omni models), and the VIB reduces
that bias / improves grounding.

**Method one-liner** (see `docs/HANDOFF.md` for the full loss):
- VIB: `z = μ(x) + σ(x)⊙ε ; y = x + W_out·z`, with `W_out` zero-init so the untrained adapter is a
  no-op (base predictions for free). One bottleneck per modality, wrapping the adapter outputs.
- Loss = swap-DPO `−logσ(β[(cp−cr)−(rp−rr)])` + `λ_anchor`·mDPO-chosen-anchor + `λ_kl`·KL-to-base on
  GENERAL ("broad") inputs + `β_kl`·KL_VIB. Code: `src/rlvib/train/dpo.py`.
- **"broad"** = the held/best recipe (v3): broadened KL-to-base anchor (MCQ + audio-presence +
  visual-presence + open-ended), `λ_kl=2`, **select mid-training (step60)**. Checkpoints live at
  `runs/anchored_<model>_broad/bottleneck_step<N>.pt`.

---

## 2. Repository layout

```
src/rlvib/
  models/            # model wrappers, all share: message(), generate(), adapter_modules(), hidden_dim
    __init__.py      #   get_model(name); registry + aliases
    qwen3_omni.py    #   Qwen3-Omni-30B-A3B-Instruct (MoE), hidden 2048 — the v0 base
    qwen25_omni.py   #   Qwen2.5-Omni-7B (Thinker-only class), hidden 3584
    videollama2.py   #   VideoLLaMA2.1-7B-AV, hidden 3584 — DIFFERENT env (rlvib_vl2, transformers 4.42)
    api_models.py    #   GeminiModel / OpenAIModel — benchmark-only, no VIB
    bottleneck.py    #   the VIB module + load_attached(model, ckpt) -> hooks on adapter_modules()
    aligner.py
  data/
    avhbench.py      #   AVHBenchDataset; BINARY_TASKS (3 yes/no tasks) + AV Captioning
    cmm.py           #   CMMDataset; AUDIO_SUBSETS
    ave.py, ...      #   training data loaders
  eval/
    run_avhbench.py  #   AVHBench yes/no eval (the headline grounding probe)
    run_cmm.py       #   CMM eval (PA / HR capability guards)
    run_dave.py      #   DAVE MCQ diagnostic
    metrics.py       #   parse_yes_no (negation-aware), accuracy, parse_choice  [torch-free, unit-tested]
    contrastive.py   #   audio-aware contrastive decoding (optional; --audio-cd)
    timeout.py
  train/             #   dpo.py (anchored swap-DPO), trainer, anchors
scripts/             # qsub job scripts + python helpers + the interactive runners (see §4)
docs/                # HANDOFF.md (method+status), reports/ (01 swap-dpo, 02 model+training), research/
tests/               # torch-free unit tests (metrics, contrastive, model message())
environment.yml      # conda env `rlvib` (Qwen stack). VideoLLaMA2 uses a separate env.
CLAUDE.md            # project instructions (cluster activation, qsub conventions, gotchas)
```

Every model wrapper exposes the SAME interface so the eval/train code is model-agnostic:
`message(video, audio, prompt, fps=None)`, `generate(messages, use_audio_in_video, max_new_tokens)`,
`adapter_modules() -> {"audio": ..., "vision": ...}` (the VIB attach points), `device`, `dtype`,
`hidden_dim`.

---

## 3. Models & benchmarks

**Models** (`get_model("<name>")`):
| name | model | env | notes |
|---|---|---|---|
| `qwen3-omni` | Qwen3-Omni-30B-A3B-Instruct | `rlvib` | MoE; **must run on ONE GPU** (see §5); the v0 base |
| `qwen2.5-omni` | Qwen2.5-Omni-7B | `rlvib` | Thinker-only class |
| `videollama2` | VideoLLaMA2.1-7B-AV | `rlvib_vl2` | transformers 4.42; own frame sampler |
| `gemini`, `gpt4o` | API | `rlvib` | benchmark-only, no VIB, cost money |

**Benchmarks** (`data/` must be populated — gitignored; see the qsub headers for download cmds):
- **AVHBench** (arXiv:2410.18325) — `data/AVHBench/{qa.json, videos/}`. 3 binary yes/no tasks +
  captioning. **5302** binary QA total: A→V "Audio-driven Video Hallucination" (~1136, asks about
  video), **V→A "Video-driven Audio Hallucination" (~2290, the audio-grounding probe)**, "AV
  Matching" (~1876). This is the headline benchmark.
- **CMM** (arXiv:2410.12787) — `data/CMM/all_data_final_reorg.json`. Metrics **PA** (perception acc,
  on yes-items) and **HR** (hallucination resistance, on no-items). Capability guards: PA≥0.90,
  HR≥0.70.
- **DAVE** — MCQ diagnostic (near-chance for these models).

---

## 4. How to run evals (the practical part)

GPUs only exist inside ABCI jobs (§5). Two ways to run:

**A) Interactively, one config at a time — `scripts/eval_one.sh` (preferred, resumable).**
Runs ONE model/checkpoint's full AVHBench + CMM on a GPU node; resumes on re-run (walltime cutoffs
are fine); writes canonical `runs/{avhbench,cmm}_<model>_sysfull[_broad_sysfull_step<N>].json`.
```bash
bash scripts/eval_one.sh qwen3-omni            # base, full AVHBench+CMM
bash scripts/eval_one.sh qwen3-omni 60         # broad checkpoint @ step60
BENCH=avhbench bash scripts/eval_one.sh qwen3-omni     # just AVHBench
N=500 bash scripts/eval_one.sh qwen3-omni      # partial (limit 500) instead of full
CONDA_ENV=rlvib_vl2 bash scripts/eval_one.sh videollama2 60   # VideoLLaMA2 (its env)
YN_SUFFIX="..." bash scripts/eval_one.sh qwen3-omni          # A/B the answer suffix (see §6)
```
Env overrides: `N` (limit, 0=full), `FPS`, `BENCH` (avhbench|cmm|both), `EXP` (default broad),
`CONDA_ENV`, `YN_SUFFIX`. It self-activates conda, pins to one GPU, asserts CUDA.

**B) Quick smoke — `scripts/smoke.sh`** (`--no-resume`, throwaway `runs/smoke_*` names, N=150
default): `N=150 STEPS='60 150' MODELS='qwen2.5-omni qwen3-omni' bash scripts/smoke.sh`.

**C) Batch matrix — `scripts/launch_all_evals.sh`** submits the whole campaign via qsub. NOTE: the
batch path historically broke on Qwen (see §5); the interactive `eval_one.sh` is the reliable path.

**Tabulate results — `scripts/make_table.py`** reads every `runs/*.json` (handles `_full`,
`_sysfull`, and partial `smoke_avh_/smoke_cmm_`), prefers full>smoke per cell, and flags partial
rows with `src`/`n` columns. Just `python scripts/make_table.py`.

The eval entry points print a live tqdm: AVHBench `acc/AdV/VdA/AVm`, CMM `acc/PA/HR`. CMM goes
through `scripts/run_cmm_autoskip.py` which steps over clips that wedge the decoder.

---

## 5. Cluster / environment (ABCI) + hard gotchas

ABCI cluster, PBS/qsub, group `gae50891`, queue `rt_HF` (8× H200 ~141 GB/node). Login nodes are
CPU-only — `torch.cuda.is_available()` is False there (expected). Env `rlvib` (Qwen) / `rlvib_vl2`
(VideoLLaMA2). Verified: transformers **5.12.1**, torch **2.12.1+cu130**.

Activate (login isn't preconfigured for conda):
```bash
source /home/aab11336im/anaconda3/etc/profile.d/conda.sh && conda activate rlvib
```
Interactive GPU node: `qsub -I -P gae50891 -q rt_HF -l select=1 -l walltime=06:00:00`.

**Gotchas that cost real time — do not relearn these:**
1. **`LD_LIBRARY_PATH` breaks `git` over HTTPS.** Exporting `$CONDA_PREFIX/lib` puts conda's OpenSSL
   ahead of the system one → `git-remote-https: symbol lookup error … EVP_md2 … OPENSSL_3.0.0`. Run
   git as `env -u LD_LIBRARY_PATH git …`. The eval qsubs had this export **removed** because it also
   crashed the Qwen *batch* jobs (same OpenSSL clash in the media/import path); torch finds its CUDA
   libs via RPATH and doesn't need it.
2. **qwen3-omni (30B MoE) must run on a SINGLE GPU.** `device_map="auto"` shards it across GPUs and
   the MoE `grouped_mm` then fails per-item (`tensors on cuda:0 vs cuda:1`) → all-zeros. The
   interactive runners pin to one GPU (`CUDA_VISIBLE_DEVICES%%,*`); the qsubs set
   `CUDA_VISIBLE_DEVICES=0`. The 30B-A3B fits on one H200 (~64 GB).
3. **Out of points** → no GPU at all (batch or interactive). Each `rt_HF` job grabs a whole 8-GPU
   node but evals use 1 GPU, so the campaign burns points fast; you can run up to 8 configs in
   parallel on one node with `CUDA_VISIBLE_DEVICES=1 … &` per run.
4. `data/` is gitignored — videos/JSON are downloaded on the cluster, not committed.

---

## 6. ⭐ Latest-session findings & fixes (the qwen3 + harness saga) — READ THIS

The most recent work was getting the **AVHBench numbers to match the published literature** (the
MoD-DPO Table-1 Qwen3-Omni base row: A→V 0.835 / V→A 0.765 / AV-m 0.585). qwen3 was giving a
yes-biased ~0.66. The chain of bugs/fixes, all now in the code:

1. **device sharding** → all-zeros. Fixed: single-GPU pin (§5.2).
2. **Wrong system prompt.** An earlier change added Qwen2.5's *"You are Qwen, a virtual human…"*
   system prompt to **both** wrappers. That's correct for **qwen2.5** (official + validated) but
   **wrong for qwen3** — Qwen3-Omni's eval protocol is **"No system prompt for any evaluation
   benchmark"** (verified against the Qwen3-Omni README), and that string isn't even a Qwen3 prompt.
   Fixed: `qwen3_omni.message()` is **user-only**; `qwen25_omni.message()` **keeps** the system
   prompt. (`tests/test_model_message.py` locks this.)
3. **fps is per-model.** Default `qwen2.5-omni=1` (its standalone-validated value), `qwen3-omni=2`
   (its original/training default; fps=1 starves qwen3's video task). Override with `FPS=`.
   VideoLLaMA2 ignores fps (own sampler).
4. **transformers version ruled out** — 5.2.x ≡ 5.12.1 on AVHBench (the `mrope_section` rope warning
   is benign; qwen3 grounds perfectly — the smoke test has it read on-screen text).
5. **⭐ THE DOMINANT LEVER: the answer-format suffix.** AVHBench never published its eval wrapper.
   The original suffix `" Answer with a single word: Yes or No."` elicited *extra* yes-bias from
   qwen3. The **AVHBench co-author lab convention** (`kaistmm/AVCD`) is **`"Answer yes or no."`**.
   Switching to it lifted qwen3 **A→V 0.636 → 0.802**, V→A 0.711 → 0.777, overall 0.659 → 0.736 —
   i.e. onto the published table within noise (V→A even *above* theirs). It is now the **default**
   and is **configurable via `--yn-suffix` / `YN_SUFFIX=`**. (MAD's variant
   `"Answer only 'Yes' or 'No'. Do not include any explanation."` was tested and is *worse*: 0.750.)

**Consequence:** the suffix change moves **every model's** AVHBench numbers, so any AVHBench JSON
produced before this fix is stale — the whole campaign must be re-run on the corrected harness. CMM
is unaffected (no suffix there). The yes-bias itself is documented expected base behaviour (OmniDPO
states the Omni base "exhibits a strong bias toward answering 'yes'").

**Caveat on the held-checkpoint numbers in the older docs** (e.g. qwen3 broad@60 = AVHBench 0.703 /
PA 0.927 / HR 0.853 at n=300 in `docs/reports/02`): those predate the suffix + system-prompt fixes,
so they are **old-protocol** and not comparable to fresh `_sysfull` runs. Re-measure before citing.

---

## 7. Current campaign state & what's left

- **qwen3-omni base**: full `_sysfull` run on the corrected harness (A→V ~0.80). Done/validated.
- **qwen2.5-omni**: only smokes so far (base + broad@60 + broad@150) — need full `_sysfull` runs on
  the corrected suffix (the old "good" 0.805 used the old suffix; re-run).
- **gemini, gpt4o**: full base (`_full`). Unaffected by the Qwen fixes.
- **videollama2**: base + broad@60/@150 not yet completed interactively.
- **Trained checkpoints to eval per model**: `broad@60`, `broad@150` (and whatever
  `runs/anchored_<model>_broad/bottleneck_step*.pt` exist).
- **To finish**: run `eval_one.sh` for each (model × {base, 60, 150}) on the corrected harness, then
  `make_table.py`. Then compute paired significance with `scripts/paired_stats.py` (base vs +ours).

---

## 8. Git & working conventions

- **Branch**: develop on `claude/blissful-ride-n9dras`. **Never push elsewhere without explicit
  permission. No PRs unless asked.**
- Commit message footer (required):
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>` and the `Claude-Session:`
  line. **Do not** put the model identifier in commits/PRs/code/artifacts.
- Pull on the cluster with `env -u LD_LIBRARY_PATH git pull origin claude/blissful-ride-n9dras`.
- Tests are torch-free where possible (`tests/`); ruff line-length 100; the SessionStart hook runs
  ruff + pytest in web sessions (the GPU stack stays cluster-only, so model tests `importorskip`
  torch).

## 9. Deeper docs
- `docs/HANDOFF.md` — method/loss, training status, the cp/rp/cr/rr DPO notation, results table.
- `docs/reports/01-anchored-swap-dpo.md`, `02-model-and-training.md` — the v1→v3 recipe evolution
  and why "broad" is held.
- `docs/research/` — DPO-collapse analysis, grounding framing, videollama2 fp16/bf16 + yes-bias
  audit, query-conditioned-bottleneck (the proposed next step: a prompt-aware FiLM bottleneck).
