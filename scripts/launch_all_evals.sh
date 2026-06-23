#!/bin/bash
# Launch full-set (LIMIT=0) AVHBench + CMM for EVERY model and EVERY trained checkpoint
# (all steps -- not just the holdout pick), in parallel via qsub. Run on a LOGIN node.
#
# Uses the corrected eval protocol (the official Qwen system prompt now lives in the model
# wrappers). Outputs are tagged `_sysfull` so they do NOT touch the old, pre-system-prompt
# `_full` JSONs, and AVHBench runs with NO_RESUME=1 so records are never mixed across protocols
# (CMM gets a fresh `_sysfull` filename, so it starts clean too).
#
#   bash scripts/launch_all_evals.sh                          # all models, every checkpoint
#   MODELS='qwen2.5-omni' bash scripts/launch_all_evals.sh    # one model
#   MODELS='qwen3-omni qwen2.5-omni' bash scripts/launch_all_evals.sh   # skip videollama2 (fix
#                                                              #   doesn't touch it -- saves GPU h)
#   EXPGLOB='broad*' bash scripts/launch_all_evals.sh         # restrict trained families
#   FORCE=1 bash scripts/launch_all_evals.sh                  # ignore the already-submitted guard
#
# IDEMPOTENT: each (model, base|checkpoint) drops a sentinel under runs/.alleval_<mark>_submitted/
# and is skipped on a re-run, so a second invocation never double-submits onto a live JSON.
set -uo pipefail                       # NOT -e: keep going if a model has no checkpoints/data
cd "$(dirname "$0")/.."

MODELS="${MODELS:-qwen3-omni qwen2.5-omni videollama2}"
EXPGLOB="${EXPGLOB:-*}"                 # which trained variants to sweep (every step within each)
MARK="${MARK:-sysfull}"                # tag marker -> runs/{avhbench,cmm}_<model>_<mark>*.json
WALL="${WALL:-12:00:00}"
CMM_JSON="${CMM_JSON:-$PWD/data/CMM/all_data_final_reorg.json}"
CMM_ROOT="${CMM_ROOT:-$PWD/data/CMM}"

# VideoLLaMA2 lives in its own conda env (transformers 4.42); everything else in rlvib.
env_for() { case "$1" in videollama2) echo "${VL2_ENV:-rlvib_vl2}";; *) echo "${CONDA_ENV:-rlvib}";; esac; }

GUARD="runs/.alleval_${MARK}_submitted"
mkdir -p "$GUARD"
submitted() { [ "${FORCE:-0}" != 1 ] && [ -e "$GUARD/$1" ]; }
mark() { : > "$GUARD/$1"; }

submit_pair() {                        # $1=MODEL  $2=TAG (leading _)  $3=BOTTLENECK abs path (opt)
    local M="$1" TAG="$2" BN="${3:-}" ENV
    ENV="$(env_for "$M")"
    qsub -v "MODEL=$M,LIMIT=0,TAG=$TAG,CONDA_ENV=$ENV,NO_RESUME=1${BN:+,BOTTLENECK=$BN}" \
        -l walltime="$WALL" scripts/eval_avhbench.qsub
    qsub -v "MODEL=$M,LIMIT=0,TAG=$TAG,CMM_JSON=$CMM_JSON,CMM_ROOT=$CMM_ROOT,CONDA_ENV=$ENV${BN:+,BOTTLENECK=$BN}" \
        -l walltime="$WALL" scripts/eval_cmm.qsub
}

echo "=== BASES (AVHBench + CMM, full set) ==="
for M in $MODELS; do
    key="${M}__base"
    if submitted "$key"; then echo "   skip $M base (already submitted; FORCE=1 to override)"; continue; fi
    echo ">> $M base"
    submit_pair "$M" "_${MARK}" && mark "$key"
done

echo "=== TRAINED (every checkpoint step) ==="
for M in $MODELS; do
    for ckpt in runs/anchored_${M}_${EXPGLOB}/bottleneck_step*.pt; do
        [ -f "$ckpt" ] || continue                          # glob with no match -> the literal; skip
        exp=$(basename "$(dirname "$ckpt")"); exp="${exp#anchored_${M}_}"
        step=$(basename "$ckpt" .pt); step="${step#bottleneck_step}"
        key="${M}__${exp}_step${step}"
        if submitted "$key"; then echo "   skip $M/$exp step$step (already submitted)"; continue; fi
        echo ">> $M/$exp step$step"
        submit_pair "$M" "_${exp}_${MARK}_step${step}" "$PWD/$ckpt" && mark "$key"
    done
done

echo
echo "submitted. watch the queue:   qstat -u \"\$USER\""
echo "live tqdm per job:            tail -f runs/avhbench_baseline*${MARK}*_out.txt"
echo "                             tail -f runs/cmm_baseline*${MARK}*_out.txt"
echo "results:                      runs/avhbench_<model>_${MARK}*.json  runs/cmm_<model>_${MARK}*.json"
echo "(to re-submit cleanly:  rm -rf $GUARD)"
