#!/bin/bash
# Evaluate qwen3-omni WITH vs WITHOUT the trained swap-DPO bottleneck on every benchmark.
# This is the transfer test: does the AVE-trained bottleneck move AVHBench / CMM / DAVE?
# Run from the repo root on the LOGIN node:
#   bash scripts/run_bottleneck_eval.sh                 # base + bottleneck, LIMIT=300
#   LIMIT=0 bash scripts/run_bottleneck_eval.sh         # full
#   COND=bn bash scripts/run_bottleneck_eval.sh         # only the bottleneck runs (reuse old base)
# Then:  qstat -u "$USER"   ...   python scripts/summarize_baselines.py
set -euo pipefail
cd "$(dirname "$0")/.."

LIMIT="${LIMIT:-300}"
BN="${BN:-$PWD/runs/bottleneck_swap.pt}"
DAVE_SPLIT="${DAVE_SPLIT:-ego4d}"
CMM_JSON="${CMM_JSON:-$PWD/data/CMM/all_data_final_reorg.json}"
CMM_ROOT="${CMM_ROOT:-$PWD/data/CMM}"
COND="${COND:-base bn}"

for C in $COND; do
    EXTRA=""
    [ "$C" = bn ] && EXTRA=",BOTTLENECK=$BN"
    echo ">> condition=$C  (bottleneck=$([ "$C" = bn ] && echo "$BN" || echo none))"
    qsub -v "MODEL=qwen3-omni,LIMIT=$LIMIT$EXTRA" scripts/eval_avhbench.qsub
    qsub -v "MODEL=qwen3-omni,LIMIT=$LIMIT,CMM_JSON=$CMM_JSON,CMM_ROOT=$CMM_ROOT$EXTRA" scripts/eval_cmm.qsub
    # DAVE: the audio-vs-visual gap is the headline signal, so run both of those modes.
    for M in audio_visual_alignment visual_only; do
        qsub -v "MODEL=qwen3-omni,LIMIT=$LIMIT,MODE=$M,DAVE_SPLIT=$DAVE_SPLIT$EXTRA" scripts/eval_dave.qsub
    done
done

echo "submitted [$COND] x {AVHBench, CMM, DAVE(av,visual)}. Watch: qstat -u \"\$USER\""
echo "Summarize (base vs *_bn side by side): python scripts/summarize_baselines.py"
