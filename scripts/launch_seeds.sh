#!/bin/bash
# Launch the multi-seed anchored "broad" training campaign (5 seeds x 3 models by default).
# Seed N -> EXP "broad" (N=0) or "broad_s{N}"; checkpoints under runs/anchored_<model>_<EXP>.
# Skips a (model,seed) whose checkpoint dir already has checkpoints, so it only fills gaps
# (you already have qwen3 seeds 0/1/2 = broad/broad_s1/broad_s2, qwen2.5 + vl2 seed 0 = broad).
#
#   bash scripts/launch_seeds.sh
#   MODELS="qwen2.5-omni videollama2" SEEDS="1 2 3 4" bash scripts/launch_seeds.sh
#   MODELS=qwen3-omni SEEDS=3 bash scripts/launch_seeds.sh                 # one missing seed
#   # a non-default config under a fixed name (e.g. 1000 pairs / 3 epochs):
#   MODELS=qwen3-omni SEEDS=0 PAIRS=1000 EPOCHS=3 EXP_NAME=broad1k WALL=12:00:00 bash scripts/launch_seeds.sh
# Override PAIRS/EPOCHS/WALL to change the recipe; EXP_NAME forces a fixed run name
# (else it derives broad / broad_s{N} from the seed). Skips a run that already has checkpoints.
set -euo pipefail
cd "$(dirname "$0")/.."

MODELS="${MODELS:-qwen3-omni qwen2.5-omni videollama2}"
SEEDS="${SEEDS:-0 1 2 3 4}"
PAIRS="${PAIRS:-300}"
EPOCHS="${EPOCHS:-2}"
WALL="${WALL:-03:00:00}"
EXP_NAME="${EXP_NAME:-}"           # force a fixed EXP (e.g. broad1k); else derive from the seed

for M in $MODELS; do
    ENV=rlvib
    [ "$M" = videollama2 ] && ENV=rlvib_vl2
    for S in $SEEDS; do
        if [ -n "$EXP_NAME" ]; then
            EXP="$EXP_NAME"
        else
            EXP=broad
            [ "$S" -ne 0 ] && EXP="broad_s${S}"
        fi
        DIR="runs/anchored_${M}_${EXP}"
        if ls "$DIR"/bottleneck_step*.pt >/dev/null 2>&1; then
            echo "skip  ${M} seed${S}  (exists: $DIR -- rm its bottleneck_step*.pt to retrain)"
            continue
        fi
        echo ">> train ${M} seed${S}  (EXP=${EXP}, pairs=${PAIRS}, epochs=${EPOCHS}, env=${ENV})"
        MODEL=$M EXP=$EXP SEED=$S CONDA_ENV=$ENV PAIRS=$PAIRS EPOCHS=$EPOCHS \
            LAMKL=2.0 LAMANCHOR=1.0 BETAKL=0.01 \
            qsub -v MODEL,EXP,SEED,CONDA_ENV,PAIRS,EPOCHS,LAMKL,LAMANCHOR,BETAKL \
            -l walltime="$WALL" scripts/train_swap_anchored.qsub
    done
done
echo "submitted.  watch: qstat -u \"\$USER\""
