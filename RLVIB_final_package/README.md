<h1 align="center">FiLMVIB</h1>
<p align="center"><b>Audio Grounding for Frozen Audio-Visual LLMs</b></p>
<p align="center">
  A &lt;0.5% adapter that makes a <i>frozen</i> audio-visual LLM listen —
  <b>+5.8</b> AVHBench points, <i>p</i> &lt; 10<sup>-4</sup>, backbone untouched.
</p>
<p align="center">
  <img alt="python" src="https://img.shields.io/badge/python-3.10%20%7C%203.11-blue">
  <img alt="backbones" src="https://img.shields.io/badge/backbones-4-orange">
  <img alt="benchmarks" src="https://img.shields.io/badge/benchmarks-5-green">
  <img alt="status" src="https://img.shields.io/badge/results-reproducible-brightgreen">
</p>

---

Reproduction package for the final internship report (`paper/final.tex`).

**Author:** Souleiman Sbai · **Host lab:** AIST CVRT · **Supervisor:** Qiu Yue

### At a glance

| | AVHBench (n=5302) | CMM HR | MMAU | Video-MME |
|---|---|---|---|---|
| frozen base | 0.761 | 0.733 | 74.7 | 0.842 |
| + swap-DPO | 0.812 | 0.743 | 73.5 | 0.828 |
| + **FiLM** | **0.819** | **0.746** | 73.0 | 0.837 |

Trains `<0.5%` of parameters · identity at initialization · fully reversible ·
every number reproducible from this folder.

---

## 1. What this project does

Audio-visual LLMs answer questions about sound using visual evidence: shown a silent clip of a
parrot, they report that the bird is vocalising. This work asks whether that failure can be
corrected **without touching the backbone**, by training a small adapter on the frozen model's
per-modality token streams.

The method has three parts:

1. **A variational-bottleneck (VIB) adapter** on each modality stream (audio, vision), `<0.5%` of
   backbone parameters, constructed so that it is *exactly the identity map at initialization*
   (`W_out` zero-initialized). A bypass flag recovers the frozen backbone exactly, which supplies
   the DPO reference distribution without a second model in memory.
2. **An anchored preference objective.** Training pairs are built by substituting a clip's audio
   track so the visible and audible events disagree; the model is trained to prefer the
   audio-consistent answer. Standard DPO collapses on this data (perception accuracy 0.95 → 0.10),
   so the objective adds two constraints: the chosen response may not fall below its probability
   under the frozen backbone, and behaviour on ordinary inputs is pinned to the backbone by a KL term.
3. **A query-conditioned (FiLM) variant** in which the question produces a per-feature scale and
   shift on the bottleneck.

Headline result on Qwen3-Omni (AVHBench, n=5302): **0.761 → 0.812** (swap-DPO) and **0.819** (FiLM),
paired *p* < 10⁻⁴, with hallucination resistance preserved.

---

## 2. Folder contents

```
.
├── README.md                     ← this file
├── src/rlvib/                    ← the library
│   ├── models/                   backbones + adapters
│   │   ├── qwen3_omni.py         Qwen3-Omni-30B-A3B (primary backbone)
│   │   ├── qwen25_omni.py        Qwen2.5-Omni-7B
│   │   ├── videollama2.py        VideoLLaMA2.1-7B-AV
│   │   ├── minicpm_o.py          MiniCPM-O 2.6  (+ transformers compat shims)
│   │   ├── api_models.py         Gemini / GPT-4o (benchmark reference only)
│   │   ├── bottleneck.py         VIB + FiLM adapters, attach/bypass, pre-adapter ablation
│   │   ├── temporal.py           clock (sinusoidal phase) input features
│   │   ├── lora.py               LoRA control arm
│   │   └── xattn.py              cross-attention fusion arm
│   ├── data/                     ave, avhbench, cmm, mmau, videomme, avshift, pairs
│   ├── eval/                     run_avhbench, run_cmm, run_mmau, run_videomme, run_sync,
│   │                             attention_av, contrastive, metrics, timeout
│   └── train/dpo.py              anchored DPO step, GRPO step, answer log-probs
├── scripts/                      training, evaluation, selection, probes, PBS job scripts
├── baselines/
│   ├── mod_dpo/                  MoD-DPO / MoD-DPO++ re-implementation
│   └── mad/                      MAD training-free decoding re-implementation
├── tests/                        unit tests (identity-at-init, masks, metrics, …)
├── paper/final.tex               the report
├── paper/summary.tex             two-page, two-column standalone summary
├── docs/
│   ├── DECK_TO_CODE.md           slide → code → command map
│   ├── CLUSTER_NOTES.md          ABCI/PBS specifics and known gotchas
│   └── research/                 design notes written during the project
├── environment.yml               main env (Qwen3/Qwen2.5)
├── environment-vllama2.yml       VideoLLaMA2 (pinned transformers 4.42)
└── environment-minicpm.yml       MiniCPM-O 2.6 (pinned transformers 4.44.2)
```

**Three environments are required** because the backbones pin mutually incompatible transformers
versions. This is not accidental complexity; see §4.

---

## 3. Results and their provenance

Every number in `paper/final.tex` is either measured under this harness or taken from a cited
paper. The table below states which, and how to regenerate it.

### Measured here (Qwen3-Omni, AVHBench n=5302 / CMM n=2400)

| Arm | A→V | V→A | AV-m | overall | PA | HR | output file |
|---|---|---|---|---|---|---|---|
| frozen base | 0.838 | 0.812 | 0.653 | 0.761 | 0.900 | 0.733 | `runs/avhbench_qwen3-omni_sysfull.json` |
| + cross-attention (step 60) | 0.850 | 0.805 | 0.696 | 0.776 | — | — | `runs/avhbench_qwen3-omni_xattn_sysfull_step60.json` |
| + LoRA r=16 (step 90) | 0.828 | 0.835 | 0.709 | 0.789 | 0.896 | 0.724 | `runs/avhbench_qwen3-omni_lora_sysfull_step90.json` |
| + swap-DPO (step 60) | 0.837 | 0.838 | 0.766 | 0.812 | 0.890 | 0.743 | `runs/avhbench_qwen3-omni_broad_sysfull_step60.json` |
| + **FiLM** (step 160) | 0.822 | 0.827 | 0.808 | **0.819** | 0.894 | 0.746 | `runs/avhbench_qwen3-omni_film_sysfull_step160.json` |

Other backbones (AVHBench overall): Qwen2.5-Omni 0.729 → 0.730; VideoLLaMA2 0.707 → 0.684;
MiniCPM-O 2.6 base 0.744 (CMM PA 0.897 / HR 0.578). Closed references: Gemini 0.718, GPT-4o 0.579.

### Capability controls

| Benchmark | base | +swap-DPO | +FiLM |
|---|---|---|---|
| MMAU (n=1000, avg) | 74.7 | 73.5 | 73.0 |
| Video-MME short (n=900) | 0.842 | 0.828 | 0.837 |

Both show a small, within-noise **cost**, reported as such.

### Mechanism

| Probe | base | +swap-DPO | +FiLM |
|---|---|---|---|
| attention on AV tokens | 5.4% | 5.7% | 7.4% |
| likelihood shift (n=40) | −0.515 | −1.698 | −1.848 |

### Temporal alignment (negative result)

| Condition | detection | direction |
|---|---|---|
| base | 0.495 | 0.500 |
| base, ±2 s displacements only | 0.495 | 0.420 |
| + shift-DPO (step 60) | 0.515 | 0.450 |

All at chance. The frozen backbone carries no usable temporal-alignment signal, so the adapter has
nothing to unlock — the same pattern as audio on VideoLLaMA2.

### Taken from the literature (not measured here)

- Wen et al., arXiv:2605.16403, Tab. 1 — Qwen3-Omni intervention accuracies (95.1→0.0 mute,
  100→1.4 shift, 75.4→37.3 swap).
- Chaubey et al., arXiv:2603.03192, Tab. 1–2 — MoD-DPO / MoD-DPO++ results.
- Chung et al., arXiv:2601.21181, Tab. 1 — MAD results.

Because absolute numbers are harness-dependent (our MiniCPM-O base reads 0.744 where the paper
reports 0.693 for the same weights), the report compares **relative improvements within one
harness**, never absolute accuracies across publications.

---

## 4. Setup

```bash
conda env create -f environment.yml            # Qwen3-Omni, Qwen2.5-Omni  -> env "rlvib"
conda activate rlvib
pip install -e .
```

Optional, only for those backbones:

```bash
conda env create -f environment-minicpm.yml    # MiniCPM-O 2.6 -> "rlvib_minicpm"
conda activate rlvib_minicpm && pip install -e .
conda env create -f environment-vllama2.yml    # VideoLLaMA2   -> "rlvib_vl2"
```

MiniCPM-O 2.6's remote modeling code targets transformers ≈4.44 and is incompatible with the newer
transformers Qwen3-Omni requires (its Whisper encoder and generation loop both break). Its wrapper
additionally installs runtime shims (`_shim_transformers` in `models/minicpm_o.py`) and skips the
`flash_attn` import check, since flash-attn cannot be built without `nvcc`.

---

## 5. Data

| Dataset | Used for | How to obtain |
|---|---|---|
| **AVE** | training pairs (audio/video substitution, temporal displacement) | AVE dataset; place under `data/AVE/AVE_Dataset/` with `trainSet.txt`/`valSet.txt`/`testSet.txt` |
| **AVHBench** | primary benchmark | `data/AVHBench/qa.json` + `data/AVHBench/videos/` (see `scripts/eval_avhbench.qsub` header for the gdown IDs) |
| **CMM** | capability control | `data/CMM/all_data_final_reorg.json` + `data/CMM/reorg_raw_files/` (HF: `DAMO-NLP-SG/CMM`) |
| **MMAU** | audio-understanding control | `data/MMAU/mmau-test-mini.json` + `test_mini.parquet`; run `python scripts/extract_mmau_audio.py` to materialize the audio (it is embedded in the parquet — no separate download needed) |
| **Video-MME** | video-understanding control | `hf download lmms-lab/Video-MME --repo-type dataset --local-dir data/VideoMME`, then unzip the videos into `data/VideoMME/videos/` |
| **AVE-Shift** | temporal benchmark (ours) | generated on demand by `data/avshift.py` from held-out AVE clips (ffmpeg) |

`data/` is not shipped (size, licensing). Counterfactual clips are generated by `data/pairs.py`
with ffmpeg and cached under `data/AVE/swapped/`.

---

## 6. Reproducing every experiment

All GPU work is submitted through PBS (`qsub`). On non-PBS hardware, run the `python` line inside
each `.qsub` directly; the scripts are plain and the job files exist only to set the environment.

### 6.1 Baseline evaluations

```bash
qsub -v "MODEL=qwen3-omni" scripts/eval_avhbench.qsub
qsub -v "MODEL=qwen3-omni,CMM_JSON=$PWD/data/CMM/all_data_final_reorg.json,CMM_ROOT=$PWD/data/CMM" scripts/eval_cmm.qsub
```

### 6.2 Training each arm

```bash
# swap-DPO (audio substitution) — the main method
qsub -v "MODEL=qwen3-omni" scripts/train_swap_anchored.qsub

# FiLM, warm-started from a swap-DPO checkpoint (two-stage recipe)
qsub -v "MODEL=qwen3-omni,EXP=film,FILM=1,INIT_FROM=runs/anchored_qwen3-omni/bottleneck_step60.pt,\
PAIRS=400,EPOCHS=3,SEEFRAC=1.0,WARMUPSTEPS=80,LR=3e-5,LAMANCHOR=1.5,LAMKL=2.0,BETAKL=0.05" \
  scripts/train_swap_anchored.qsub

# controls / ablations
qsub -v "MODEL=qwen3-omni,EXP=lora,LORA=1"        scripts/train_swap_anchored.qsub
qsub -v "MODEL=qwen3-omni,EXP=xattn,XATTN=1,LAMKL=2.0" scripts/train_swap_anchored.qsub
qsub -v "MODEL=qwen3-omni,PREADAPTER=1"           scripts/train_swap_anchored.qsub   # placement ablation
qsub scripts/train_swap_video.qsub                # video substitution (symmetric experiment)
qsub scripts/train_swap_shift.qsub                # temporal displacement (shift-DPO)

# other backbones
qsub -v "CONDA_ENV=rlvib_minicpm,MODEL=minicpm-o" scripts/train_swap_anchored.qsub
```

### 6.3 Checkpoint selection, then full evaluation

Selection uses a 300-item validation slice; results are reported on the full disjoint set.
**Select on both axes** — the highest AVHBench checkpoint is often one whose CMM hallucination
resistance has collapsed (e.g. swap-DPO step 120: AVH 0.832 but HR 0.357).

```bash
EXP=broad MODEL=qwen3-omni LIMIT=300 STEPS="30 60 90 120 150" bash scripts/select_checkpoint.sh
python scripts/select_checkpoint.py                      # summarize the grid
EXP=broad MODEL=qwen3-omni bash scripts/eval_one.sh qwen3-omni 60   # full AVHBench + CMM
python scripts/make_table.py                             # assemble all results
```

### 6.4 Statistics

```bash
python scripts/paired_stats.py --model qwen3-omni --exp film --step 160 --suffix sysfull \
    --dev 300 --vs broad:60
```
Paired McNemar with bootstrap confidence intervals over per-item predictions stored in the eval
JSONs; no re-inference required.

### 6.5 Capability controls

```bash
python scripts/extract_mmau_audio.py                        # once
CKPTS="broad:60 film:160" bash scripts/launch_mmau.sh qwen3-omni
CKPTS="broad:60 film:160" bash scripts/launch_videomme.sh qwen3-omni
```

### 6.6 Mechanism probes

```bash
python scripts/attn_av_analysis.py --model qwen3-omni --modality audio     # attention share
python scripts/plot_attn_av.py                                             # -> paper/figures/attn_av.png
qsub scripts/probe_ll_shift.qsub                                           # likelihood shift + figure
python scripts/analyze_bottleneck.py --ckpt runs/anchored_qwen3-omni_film/bottleneck_step160.pt
```

### 6.7 Temporal alignment

```bash
BENCH=avshift MODEL=qwen3-omni qsub -v BENCH,MODEL scripts/eval_sync.qsub                 # baseline
OFFSETS=2 BENCH=avshift MODEL=qwen3-omni TAG=_off2 qsub -v OFFSETS,BENCH,MODEL,TAG scripts/eval_sync.qsub   # diagnostic
BENCH=avshift MODEL=qwen3-omni BOTTLENECK=runs/anchored_qwen3-omni_shift/bottleneck_step60.pt \
  TAG=_shift_step60 qsub -v BENCH,MODEL,BOTTLENECK,TAG scripts/eval_sync.qsub             # trained
```

### 6.8 Literature baselines (re-implementations)

```bash
PYTHONPATH=src python baselines/mod_dpo/train_mod_dpo.py --model qwen3-omni --pairs 300 --epochs 2 [--plus]
qsub -v "BENCH=avhbench,MODEL=qwen3-omni,LIMIT=300" baselines/mad/eval_mad.qsub
```
See each folder's README for the faithfulness caveats — these compare *objectives at equal capacity
and data*, not the papers' absolute numbers.

### 6.9 Tests

```bash
PYTHONPATH=src python -m pytest tests/ -v
```
Torch-dependent tests skip automatically where no GPU/torch is present.

---

## 7. Evaluation protocol (why the numbers are trustworthy)

- **Frame rate is fixed in code** at 2 fps for Qwen3-Omni (1 fps for the 7B backbones), identically
  for selection and final evaluation. Below 2 fps Qwen3-Omni develops a systematic yes-bias.
- **Prompt sensitivity is large**: changing only the answer-format suffix moved one score by
  +11 points (V→A 69.0 → 80.3) on identical weights. Prompts, suffix and decoding are fixed and logged.
- **Selection on a validation half, reporting on the disjoint remainder.** The first checkpoint we
  selected did not survive this.
- **Paired significance testing** on identical items (McNemar + bootstrap CI).
- **Contamination check**: training clips ∩ benchmark clips = ∅ (`scripts/check_overlap.py`).
- Per-item predictions are stored in every eval JSON, so all statistics are recomputable without
  re-running inference.

---

## 8. Honest status

**Complete and reported:** the Qwen3-Omni arms (base, swap-DPO, FiLM, LoRA, cross-attention), all
three capability controls, both mechanism probes, the four-backbone transfer comparison, the
MiniCPM-O baseline, and the temporal-alignment negative result.

**Trained but not yet fully evaluated at the time of writing:** the video-substitution arm, the
shift-DPO + clock-features arm, and the three MiniCPM-O adapter arms. Checkpoints and selection
grids exist; the full-set evaluations were still queued. No numbers for these appear in the report.

**Not attempted:** multi-seed training (all results are single-seed; the reported confidence
intervals cover item sampling only), and VGGSound-Sync as an out-of-distribution temporal check.

**Known limitations of the method**, stated in the report: the improvement requires a backbone whose
frozen representation already carries usable audio (no effect on VideoLLaMA2), out-of-distribution
audio and video reasoning pay a small cost, and the approach does not transfer to temporal alignment.

---

## 9. Portability notes

The `.qsub` files are ABCI/PBS-specific: they source a fixed conda path, set
`LD_LIBRARY_PATH` for the pip torch wheel, request the `rt_HF` queue under group `gae50891`, and
`tee` into `runs/`. To run elsewhere, execute the `python -u …` line from each job file directly.

`docs/CLUSTER_NOTES.md` records the environment gotchas encountered during the project (conda
activation under `set -u`, `LD_LIBRARY_PATH` breaking system `git`, compute nodes having no network).

---

## 10. Provenance

`archive/` from the development repository is intentionally **not** included: it holds retired work
(a localization-map line of experiments, the DAVE benchmark, one-off debugging scripts) that no
result in the report depends on. Everything the report cites is in this folder.
