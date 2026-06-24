#!/bin/bash
# Interactive smoke test of AVHBench + CMM (run ON a GPU node). One pasteable line per run:
#     N=150 bash scripts/smoke.sh                                  # base only, qwen2.5
#     N=150 STEPS='60 150' bash scripts/smoke.sh                   # base + broad@60 + broad@150
#     N=150 MODELS='qwen2.5-omni qwen3-omni' STEPS='60 150' bash scripts/smoke.sh
#     CONDA_ENV=rlvib_vl2 MODELS=videollama2 STEPS='60 150' N=150 bash scripts/smoke.sh
#
# Writes runs/smoke_{avh,cmm}_<model>[_<exp>_step<s>].json. Qwen models pass FPS (default 1, to
# match the standalone harness); VideoLLaMA2 has its own sampler so FPS is not passed to it.
set -euo pipefail
cd "$(dirname "$0")/.."

N="${N:-150}"
MODELS="${MODELS:-qwen2.5-omni}"
STEPS="${STEPS:-}"                 # extra trained checkpoints to run, e.g. '60 150' (family $EXP)
EXP="${EXP:-broad}"
AVHBENCH_QA="${AVHBENCH_QA:-data/AVHBench/qa.json}"
AVHBENCH_VIDEOS="${AVHBENCH_VIDEOS:-data/AVHBench/videos}"
CMM_JSON="${CMM_JSON:-data/CMM/all_data_final_reorg.json}"
CMM_ROOT="${CMM_ROOT:-data/CMM}"

source /home/aab11336im/anaconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-rlvib}"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export TRANSFORMERS_OFFLINE=1
# Pin to ONE GPU. device_map='auto' otherwise shards the 30B-MoE qwen3 across GPUs and its MoE
# grouped_mm then fails ("tensors on cuda:0 vs cuda:1"). %%,* keeps just the first visible GPU.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -c "import torch; assert torch.cuda.is_available(), \
  'NO GPU -- start an interactive session first: qsub -I -P gae50891 -q rt_HF -l select=1 -l walltime=01:00:00'"

run_one() {                       # $1=model  $2=bottleneck-or-empty  $3=output tag
    local M="$1" BN="$2" TAG="$3" FPS_ARG=() BN_ARG=() _fps=""
    case "$M" in                  # per-model fps default (override FPS=...): 2.5=1, qwen3=2
        qwen2.5-omni) _fps="${FPS:-1}";;
        qwen3-omni)   _fps="${FPS:-2}";;
    esac
    [ -n "$_fps" ] && FPS_ARG=(--fps "$_fps")
    [ -n "$BN" ] && BN_ARG=(--bottleneck "$BN")
    echo "===== ${M}${TAG} / AVHBench (N=$N) ====="
    python -u -m rlvib.eval.run_avhbench --model "$M" \
        --qa-json "$AVHBENCH_QA" --video-root "$AVHBENCH_VIDEOS" \
        --limit "$N" --no-resume "${FPS_ARG[@]}" "${BN_ARG[@]}" --out "runs/smoke_avh_${M}${TAG}.json"
    echo "===== ${M}${TAG} / CMM (N=$N) ====="
    python -u -m rlvib.eval.run_cmm --model "$M" \
        --json-path "$CMM_JSON" --data-root "$CMM_ROOT" \
        --limit "$N" --no-resume "${FPS_ARG[@]}" "${BN_ARG[@]}" --out "runs/smoke_cmm_${M}${TAG}.json"
}

for M in $MODELS; do
    run_one "$M" "" ""                                    # base
    for s in $STEPS; do                                   # trained checkpoints (if present)
        ckpt="runs/anchored_${M}_${EXP}/bottleneck_step${s}.pt"
        if [ -f "$ckpt" ]; then run_one "$M" "$ckpt" "_${EXP}_step${s}"
        else echo "skip ${M} ${EXP} step${s} (missing $ckpt)"; fi
    done
done
echo "=== smoke done -> runs/smoke_{avh,cmm}_<model>*.json ==="
