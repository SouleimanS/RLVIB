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
#   EXPGLOB='*'      bash scripts/launch_all_evals.sh         # all trained families (incl. abl_*)
#   STEPS='60 90'    bash scripts/launch_all_evals.sh         # only these checkpoint steps
#   ALL_STEPS=1      bash scripts/launch_all_evals.sh         # every saved step (heavy!)
#   FORCE=1          bash scripts/launch_all_evals.sh         # ignore the already-submitted guard
#
# IDEMPOTENT: each (model, base|checkpoint) drops a sentinel under runs/.alleval_<mark>_submitted/
# and is skipped on a re-run, so a second invocation never double-submits onto a live JSON.
set -uo pipefail                       # NOT -e: keep going if a model has no checkpoints/data
cd "$(dirname "$0")/.."

MODELS="${MODELS:-qwen3-omni qwen2.5-omni videollama2}"
EXPGLOB="${EXPGLOB:-broad*}"           # trained family to sweep; broad = the main run (abl_* are
                                       #   ablations, excluded by default). EXPGLOB='*' = all.
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

# Which step(s) per experiment: default = the holdout-selected pick (the "relevant" one behind
# the paper table); STEPS='60 90' = those exact steps; ALL_STEPS=1 = every saved checkpoint.
steps_for() {                          # $1=MODEL $2=EXP $3=DIR  ->  echoes space-separated steps
    local M="$1" exp="$2" dir="$3" c b
    if [ -n "${STEPS:-}" ]; then
        echo "${STEPS//,/ }"
    elif [ "${ALL_STEPS:-0}" = 1 ]; then
        for c in "$dir"bottleneck_step*.pt; do
            [ -f "$c" ] || continue
            b=$(basename "$c" .pt); echo "${b#bottleneck_step}"
        done
    else
        python scripts/select_holdout.py --model "$M" --exp "$exp" 2>/dev/null \
            | grep -oP 'HONEST.*?step\K[0-9]+' | head -1 || true
    fi
}

mode="holdout-selected step"
[ -n "${STEPS:-}" ] && mode="steps ${STEPS}"
[ "${ALL_STEPS:-0}" = 1 ] && mode="every step"
echo "=== TRAINED (family '$EXPGLOB'; $mode) ==="
for M in $MODELS; do
    for dir in runs/anchored_${M}_${EXPGLOB}/; do
        [ -d "$dir" ] || continue                           # glob with no match -> literal; skip
        exp=$(basename "$dir"); exp="${exp#anchored_${M}_}"
        steps=$(steps_for "$M" "$exp" "$dir")
        [ -n "$steps" ] || { echo "   skip $M/$exp (no step selected; set STEPS=NN or ALL_STEPS=1)"; continue; }
        for step in $steps; do
            ckpt="${dir}bottleneck_step${step}.pt"
            [ -f "$ckpt" ] || { echo "   skip $M/$exp step$step (missing $ckpt)"; continue; }
            key="${M}__${exp}_step${step}"
            submitted "$key" && { echo "   skip $M/$exp step$step (already submitted)"; continue; }
            echo ">> $M/$exp step$step"
            submit_pair "$M" "_${exp}_${MARK}_step${step}" "$PWD/$ckpt" && mark "$key"
        done
    done
done

echo
echo "submitted. watch the queue:   qstat -u \"\$USER\""
echo "live tqdm per job:            tail -f runs/avhbench_baseline*${MARK}*_out.txt"
echo "                             tail -f runs/cmm_baseline*${MARK}*_out.txt"
echo "results:                      runs/avhbench_<model>_${MARK}*.json  runs/cmm_<model>_${MARK}*.json"
echo "(to re-submit cleanly:  rm -rf $GUARD)"
