#!/bin/bash
# Run ONE full AVHBench + CMM eval directly on an interactive GPU node (no qsub). RESUMES on
# re-run, so a walltime cutoff mid-run is fine -- just run the same line again to continue.
# Canonical outputs under runs/ (same names the launcher uses), so they're directly comparable.
#
#   bash scripts/eval_one.sh qwen2.5-omni           # base (AVHBench + CMM, full)
#   bash scripts/eval_one.sh qwen2.5-omni 60        # broad checkpoint @ step60
#   bash scripts/eval_one.sh qwen2.5-omni 150
#   BENCH=avhbench bash scripts/eval_one.sh qwen3-omni        # just AVHBench
#   N=300         bash scripts/eval_one.sh qwen2.5-omni       # partial (limit 300), not full
#   CONDA_ENV=rlvib_vl2 bash scripts/eval_one.sh videollama2 60
set -euo pipefail
cd "$(dirname "$0")/.."

M="${1:?usage: eval_one.sh <model> [step]   (step omitted = base)}"
STEP="${2:-}"
EXP="${EXP:-broad}"
N="${N:-0}"                 # 0 = full set
FPS="${FPS:-1}"
BENCH="${BENCH:-both}"      # avhbench | cmm | both
AVHBENCH_QA="${AVHBENCH_QA:-data/AVHBench/qa.json}"
AVHBENCH_VIDEOS="${AVHBENCH_VIDEOS:-data/AVHBench/videos}"
CMM_JSON="${CMM_JSON:-data/CMM/all_data_final_reorg.json}"
CMM_ROOT="${CMM_ROOT:-data/CMM}"

source /home/aab11336im/anaconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-rlvib}"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -c "import torch; assert torch.cuda.is_available(), 'NO GPU -- run inside an interactive session'"

BN_ARG=(); TAG="_sysfull"
if [ -n "$STEP" ]; then
    CKPT="runs/anchored_${M}_${EXP}/bottleneck_step${STEP}.pt"
    [ -f "$CKPT" ] || { echo "missing checkpoint: $CKPT"; exit 1; }
    BN_ARG=(--bottleneck "$CKPT"); TAG="_${EXP}_sysfull_step${STEP}"
fi
FPS_ARG=(); case "$M" in qwen3-omni|qwen2.5-omni) [ -n "$FPS" ] && FPS_ARG=(--fps "$FPS");; esac

run_avh() {                                                  # resumes from the --out JSON
    echo "===== ${M}${TAG} / AVHBench (limit=$N) ====="
    python -u -m rlvib.eval.run_avhbench --model "$M" \
        --qa-json "$AVHBENCH_QA" --video-root "$AVHBENCH_VIDEOS" \
        --limit "$N" "${FPS_ARG[@]}" "${BN_ARG[@]}" --out "runs/avhbench_${M}${TAG}.json"
}
run_cmm() {                                                  # autoskip wrapper: resumes + steps over hanging clips
    echo "===== ${M}${TAG} / CMM (limit=$N) ====="
    python -u scripts/run_cmm_autoskip.py --out "runs/cmm_${M}${TAG}.json" \
        --json "$CMM_JSON" --root "$CMM_ROOT" \
        -- --model "$M" --limit "$N" "${FPS_ARG[@]}" "${BN_ARG[@]}"
}

case "$BENCH" in avhbench|both) run_avh;; esac
case "$BENCH" in cmm|both) run_cmm;; esac
echo "=== done -> runs/avhbench_${M}${TAG}.json , runs/cmm_${M}${TAG}.json ==="
