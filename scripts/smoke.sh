#!/bin/bash
# Quick interactive smoke test of the AVHBench + CMM evals -- run ON a GPU compute node.
# Self-contained (sources conda, activates the env, sets paths), so you paste ONE short line
# instead of a multi-line block that mangles the terminal:
#
#     N=150 bash scripts/smoke.sh
#     N=150 MODELS='qwen2.5-omni qwen3-omni' bash scripts/smoke.sh
#     CONDA_ENV=rlvib_vl2 MODELS=videollama2 N=150 bash scripts/smoke.sh   # VideoLLaMA2 (its env)
#
# Writes runs/smoke_{avh,cmm}_<model>.json. Qwen models use FPS (default 1, to match the
# standalone harness); VideoLLaMA2 has its own sampler so FPS is not passed to it.
set -euo pipefail
cd "$(dirname "$0")/.."

N="${N:-150}"
MODELS="${MODELS:-qwen2.5-omni}"
FPS="${FPS:-1}"
AVHBENCH_QA="${AVHBENCH_QA:-data/AVHBench/qa.json}"
AVHBENCH_VIDEOS="${AVHBENCH_VIDEOS:-data/AVHBench/videos}"
CMM_JSON="${CMM_JSON:-data/CMM/all_data_final_reorg.json}"
CMM_ROOT="${CMM_ROOT:-data/CMM}"

source /home/aab11336im/anaconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-rlvib}"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -c "import torch; assert torch.cuda.is_available(), \
  'NO GPU -- start an interactive session first: qsub -I -P gae50891 -q rt_HF -l select=1 -l walltime=01:00:00'"

for M in $MODELS; do
    FPS_ARG=()
    case "$M" in qwen3-omni|qwen2.5-omni) [ -n "$FPS" ] && FPS_ARG=(--fps "$FPS");; esac
    echo "===== $M / AVHBench (N=$N) ====="
    python -u -m rlvib.eval.run_avhbench --model "$M" \
        --qa-json "$AVHBENCH_QA" --video-root "$AVHBENCH_VIDEOS" \
        --limit "$N" --no-resume "${FPS_ARG[@]}" --out "runs/smoke_avh_${M}.json"
    echo "===== $M / CMM (N=$N) ====="
    python -u -m rlvib.eval.run_cmm --model "$M" \
        --json-path "$CMM_JSON" --data-root "$CMM_ROOT" \
        --limit "$N" --no-resume "${FPS_ARG[@]}" --out "runs/smoke_cmm_${M}.json"
done
echo "=== smoke done -> runs/smoke_{avh,cmm}_<model>.json ==="
