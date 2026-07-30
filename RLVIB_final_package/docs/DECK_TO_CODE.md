# Deck → code map & launch guide

Every claim in `paper/phd_talk.tex` has code in this repo. This maps each slide/experiment to its
source and the command that produces it. Retired work (not in the deck) lives in `archive/`.

## Environment

```bash
conda env create -f environment.yml        # first time
source /home/aab11336im/anaconda3/etc/profile.d/conda.sh && conda activate rlvib   # ABCI
export PYTHONPATH="$PWD/src:$PYTHONPATH"    # or: pip install -e .
```
GPUs only inside a `qsub` job (`rt_HF`). Login nodes are CPU-only and are the only nodes with
network. Frame rate is pinned in code (2 fps Qwen3, 1 fps 7B) — see the eval runners.

## Slide → code → launch

| Deck slide / claim | Code | Launch |
|---|---|---|
| AV-LLM anatomy; 4 backbones | `src/rlvib/models/{qwen3_omni,qwen25_omni,videollama2,minicpm_o,api_models}.py`, factory `models/__init__.py` | `get_model("qwen3-omni" \| "minicpm-o" \| …)` |
| AVHBench (target) | `data/avhbench.py`, `eval/run_avhbench.py` | `qsub -v MODEL=qwen3-omni scripts/eval_avhbench.qsub` |
| CMM (capability guard) | `data/cmm.py`, `eval/run_cmm.py` | `qsub -v MODEL=qwen3-omni scripts/eval_cmm.qsub` |
| Experimental protocol (2 fps, selection) | fps pinned in `eval/run_avhbench.py`/`run_cmm.py`; `scripts/select_holdout.py`, `select_checkpoint.sh` | `python scripts/select_holdout.py --model qwen3-omni --exp film` |
| VIB background + the adapter | `models/bottleneck.py` (`VariationalBottleneck`) | attached by the trainer |
| Swap trick (manufacture conflict) | `data/pairs.py` (`swap_audio`, `make_swap_examples`), `data/ave.py` | built inside training |
| Naïve DPO collapse | `train/dpo.py`, `scripts/train_swap.py` | `python scripts/train_swap.py …` (the baseline that collapses) |
| The fix: engine + 2 rails | `scripts/train_swap_anchored.py` (`anchored_dpo_step` in `train/dpo.py`) | `qsub -v MODEL=qwen3-omni scripts/train_swap_anchored.qsub` |
| FiLM (question-steered) | `models/bottleneck.py` (`FiLMVariationalBottleneck`, `question_embedding`, `set_condition`) | `EXP=film FILM=1 INIT_FROM=… qsub -v EXP,FILM,INIT_FROM,… scripts/train_swap_anchored.qsub` |
| Main results table | `eval/run_avhbench.py` + `run_cmm.py`; `scripts/make_table.py` | `python scripts/make_table.py` |
| Paired statistics | `scripts/paired_stats.py` | `python scripts/paired_stats.py --model qwen3-omni --exp film --step 160 --vs broad:60` |
| MMAU guard | `data/mmau.py`, `eval/run_mmau.py` | `bash scripts/launch_mmau.sh` (or `qsub scripts/eval_mmau.qsub`) |
| Video-MME guard | `data/videomme.py`, `eval/run_videomme.py` | `CKPTS="film:160" bash scripts/launch_videomme.sh qwen3-omni` |
| Attention-to-AV probe | `eval/attention_av.py`, `scripts/attn_av_analysis.py`, `plot_attn_av.py` | `python scripts/attn_av_analysis.py --model qwen3-omni --modality audio` |
| Likelihood-shift probe | `scripts/probe_ll_shift.py`, `plot_ll_shift.py` | `python scripts/probe_ll_shift.py --model qwen3-omni --bottleneck …` |
| Transfer across backbones | the four model wrappers + same recipe | run the trainer/evals per `--model` |
| **MiniCPM-O 2.6 backbone** | `models/minicpm_o.py` (registered `minicpm-o`) | `qsub -v MODEL=minicpm-o scripts/train_swap_anchored.qsub`; `qsub -v MODEL=minicpm-o scripts/eval_avhbench.qsub` |
| vs MoD-DPO++ (baseline) | `baselines/mod_dpo/` | `PYTHONPATH=src python baselines/mod_dpo/train_mod_dpo.py --model minicpm-o --plus` |
| **Swap-video (A→V, marginal)** | `data/pairs.py` (`swap_video`, `make_video_swap_examples`); trainer `--swap video` | `SWAP=video MODEL=qwen3-omni qsub -v SWAP,MODEL scripts/train_swap_anchored.qsub` |
| Cross-attention arm | `models/xattn.py` | `EXP=xattn XATTN=1 qsub -v EXP,XATTN,MODEL scripts/train_swap_anchored.qsub` |
| LoRA control | `models/lora.py` | `LORA=1 qsub -v LORA,MODEL scripts/train_swap_anchored.qsub` |
| MAD (training-free baseline) | `baselines/mad/` | `qsub -v BENCH=avhbench,MODEL=qwen3-omni baselines/mad/eval_mad.qsub` |
| **Next axis: shift-DPO** | `data/pairs.py` (`shift_audio`, `make_shift_examples`); trainer `--swap shift` | `SWAP=shift MODEL=qwen3-omni qsub -v SWAP,MODEL scripts/train_swap_anchored.qsub` |
| **Temporal input features (clock τ_t)** | `models/temporal.py` (`TemporalVariationalBottleneck`, `clock_features`) | `SWAP=shift TEMPORAL=1 MODEL=qwen3-omni qsub -v SWAP,TEMPORAL,MODEL scripts/train_swap_anchored.qsub`; unit-tested in `tests/test_temporal.py` |
| **Temporal-alignment benchmark** (AVE-Shift + VGGSound-Sync) | `data/avshift.py`, `eval/run_sync.py` | `BENCH=avshift qsub -v BENCH,MODEL scripts/eval_sync.qsub` ; `BENCH=vggsync …` |

## Smoke tests (fast, GPU)

```bash
qsub scripts/gpucheck.qsub                                  # CUDA sanity
qsub scripts/smoketest_bottleneck.qsub                      # attach + identity-at-init on the real model
PYTHONPATH=src python -m rlvib.eval.run_sync --bench avshift --model qwen3-omni --limit 8
```

## Notes for first cluster run

- `models/minicpm_o.py` — the two adapter attach-point module paths (`adapter_modules()`) and the
  `build_inputs`/`chat` route are resolved defensively; confirm shapes on the first attach (same
  contract as the cross-attention arm). Print `model.named_modules()` if attachment complains.
- `TemporalVariationalBottleneck` — selected by the trainer's `--temporal` flag (mutually
  exclusive with `--film`/`--lora`/`--xattn`); pair with `--swap shift` for shift-DPO.
- VGGSound-Sync needs `data/VGGSoundSync/` populated (manifest + `videos/`); AVE-Shift materializes
  its shifted clips on demand under `data/AVE/shifted/`.
