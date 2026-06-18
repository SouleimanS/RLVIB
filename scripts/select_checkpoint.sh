#!/bin/bash
# Model selection across anchored swap-DPO checkpoints (see docs/reports/01-anchored-swap-dpo.md).
# Evaluates each checkpoint on AVHBench (the signal) + CMM (the capability guard) [+ DAVE],
# tagging outputs per step so they don't collide. Run on the LOGIN node, then summarize:
#   bash scripts/select_checkpoint.sh                     # steps 30..150 by 30, AVHBench+CMM
#   STEPS="90 120 150" WITH_DAVE=1 bash scripts/select_checkpoint.sh
#   python scripts/select_checkpoint.py                   # once the jobs finish
set -euo pipefail
cd "$(dirname "$0")/.."

LIMIT="${LIMIT:-300}"
STEPS="${STEPS:-30 60 90 120 150}"
MODEL="${MODEL:-qwen3-omni}"
CKPT_DIR="${CKPT_DIR:-$PWD/runs/anchored_$MODEL}"
CMM_JSON="${CMM_JSON:-$PWD/data/CMM/all_data_final_reorg.json}"
CMM_ROOT="${CMM_ROOT:-$PWD/data/CMM}"
DAVE_SPLIT="${DAVE_SPLIT:-ego4d}"
WITH_DAVE="${WITH_DAVE:-0}"

for S in $STEPS; do
    BN="$CKPT_DIR/bottleneck_step${S}.pt"
    if [ ! -f "$BN" ]; then echo "!! missing $BN -- skipping"; continue; fi
    TAG="_step${S}"
    echo ">> checkpoint step $S  ($BN)"
    qsub -v "MODEL=$MODEL,LIMIT=$LIMIT,BOTTLENECK=$BN,TAG=$TAG" scripts/eval_avhbench.qsub
    qsub -v "MODEL=$MODEL,LIMIT=$LIMIT,CMM_JSON=$CMM_JSON,CMM_ROOT=$CMM_ROOT,BOTTLENECK=$BN,TAG=$TAG" scripts/eval_cmm.qsub
    if [ "$WITH_DAVE" = 1 ]; then
        qsub -v "MODEL=$MODEL,LIMIT=$LIMIT,MODE=audio_visual_alignment,DAVE_SPLIT=$DAVE_SPLIT,BOTTLENECK=$BN,TAG=$TAG" scripts/eval_dave.qsub
    fi
done

echo "submitted selection grid over steps: $STEPS  (AVHBench+CMM$([ "$WITH_DAVE" = 1 ] && echo '+DAVE'))"
echo "Watch:      qstat -u \"\$USER\""
echo "Summarize:  python scripts/select_checkpoint.py"
