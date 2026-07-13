# MoD-DPO / MoD-DPO++ baseline (Chaubey et al., arXiv:2603.03192)

Modality-Decoupled DPO: standard DPO plus two **stop-gradient** margin offsets from
corrupted-input passes of the current policy — **invariance** to corrupting the
prompt-*irrelevant* modality (−β_inv) and **sensitivity** to corrupting the *relevant* one
(+β_sens), with τ = β + β_inv − β_sens on the policy term, alternating audio-related and
vision-related prompts. **++** additionally subtracts a γ_LPD **text-only** logratio
(language-prior debiasing, π_text = π_ref). Paper defaults kept: β=0.1, β_inv=0.02,
β_sens=0.05, γ_LPD=0.05.

## Faithful vs adapted (read before citing numbers)

| | paper | here |
|---|---|---|
| trainable params | full LLM (LLaMA-Factory) | our frozen-backbone **VIB adapter** (capacity-matched to our other arms) |
| preference data | 18.1k GPT-4o sentence pairs (MSR-VTT/VALOR/AudioCaps) | our **AVE letter pairs** (HEAR = audio-related, SEE = vision-related) |
| answers scored | sentence log-probs | single-letter log-probs (repo convention) |
| corruption a′ / v′ | mismatched/degraded segments | **silenced audio** / **mismatched video** via ffmpeg (`pairs.py`) — the paper's "segments from different files" |

So results compare **objectives at equal capacity and data** against our anchored recipe —
they do not reproduce the paper's absolute numbers.

## Run

```bash
# train (cluster, GPU):
PYTHONPATH=src python baselines/mod_dpo/train_mod_dpo.py --model qwen3-omni --pairs 300 --epochs 2          # MoD-DPO
PYTHONPATH=src python baselines/mod_dpo/train_mod_dpo.py --model qwen3-omni --pairs 300 --epochs 2 --plus   # MoD-DPO++

# checkpoints land in runs/anchored_<model>_moddpo[pp]/ -> ALL existing tooling works:
EXP=moddpo MODEL=qwen3-omni LIMIT=300 STEPS="30 60 90 120 150" bash scripts/select_checkpoint.sh
python scripts/select_holdout.py --model qwen3-omni --exp moddpo
bash scripts/eval_one.sh qwen3-omni <S>   # with EXP=moddpo
python scripts/paired_stats.py --model qwen3-omni --exp moddpo --step <S> --suffix sysfull --dev 300 --vs broad:60
```
