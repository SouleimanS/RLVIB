#!/bin/bash
# Launch the multi-seed anchored "broad" training campaign (5 seeds x 3 models by default).
# Seed N -> EXP "broad" (N=0) or "broad_s{N}"; checkpoints under runs/anchored_<model>_<EXP>.
# Skips a (model,seed) whose checkpoint dir already has checkpoints, so it only fills gaps
# (you already have qwen3 seeds 0/1/2 = broad/broad_s1/broad_s2, qwen2.5 + vl2 seed 0 = broad).
#
#   bash scripts/launch_seeds.sh
#   MODELS="qwen2.5-omni videollama2" SEEDS="1 2 3 4" bash scripts/launch_seeds.sh
set -euo pipefail
cd "$(dirname "$0")/.."

MODELS="${MODELS:-qwen3-omni qwen2.5-omni videollama2}"
SEEDS="${SEEDS:-0 1 2 3 4}"

for M in $MODELS; do
    ENV=rlvib
    [ "$M" = videollama2 ] && ENV=rlvib_vl2
    for S in $SEEDS; do
        EXP=broad
        [ "$S" -ne 0 ] && EXP="broad_s${S}"
        DIR="runs/anchored_${M}_${EXP}"
        if ls "$DIR"/bottleneck_step*.pt >/dev/null 2>&1; then
            echo "skip  ${M} seed${S}  (exists: $DIR)"
            continue
        fi
        echo ">> train ${M} seed${S}  (EXP=${EXP}, env=${ENV})"
        MODEL=$M EXP=$EXP SEED=$S CONDA_ENV=$ENV PAIRS=300 EPOCHS=2 \
            LAMKL=2.0 LAMANCHOR=1.0 BETAKL=0.01 \
            qsub -v MODEL,EXP,SEED,CONDA_ENV,PAIRS,EPOCHS,LAMKL,LAMANCHOR,BETAKL \
            -l walltime=03:00:00 scripts/train_swap_anchored.qsub
    done
done
echo "submitted.  watch: qstat -u \"\$USER\""
