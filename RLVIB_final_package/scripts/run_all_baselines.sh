#!/bin/bash
# Launch the full eval matrix: every model x every benchmark, each in its own env.
# Run from the repo root on the LOGIN node:
#   bash scripts/run_all_baselines.sh            # full (LIMIT=0)
#   LIMIT=300 bash scripts/run_all_baselines.sh  # faster, consistent pass
# Then:  qstat -u "$USER"   ...   python scripts/summarize_baselines.py
set -euo pipefail
cd "$(dirname "$0")/.."

LIMIT="${LIMIT:-0}"
DAVE_SPLIT="${DAVE_SPLIT:-ego4d}"
CMM_JSON="$PWD/data/CMM/all_data_final_reorg.json"
CMM_ROOT="$PWD/data/CMM"

for MODEL in qwen3-omni qwen2.5-omni videollama2; do
    ENV=rlvib
    [ "$MODEL" = videollama2 ] && ENV=rlvib_vl2
    echo ">> $MODEL  (env=$ENV, LIMIT=$LIMIT)"

    qsub -v "CONDA_ENV=$ENV,MODEL=$MODEL,LIMIT=$LIMIT" scripts/eval_avhbench.qsub
    qsub -v "CONDA_ENV=$ENV,MODEL=$MODEL,LIMIT=$LIMIT,CMM_JSON=$CMM_JSON,CMM_ROOT=$CMM_ROOT" scripts/eval_cmm.qsub

    # DAVE ablation modes; VideoLLaMA2's mm_infer needs a media input, so it skips
    # text_only (no media) and audio_only (bare wav) for now.
    MODES="audio_visual_alignment visual_only audio_only text_only"
    [ "$MODEL" = videollama2 ] && MODES="audio_visual_alignment visual_only"
    for M in $MODES; do
        qsub -v "CONDA_ENV=$ENV,MODEL=$MODEL,LIMIT=$LIMIT,MODE=$M,DAVE_SPLIT=$DAVE_SPLIT" scripts/eval_dave.qsub
    done
done

echo "All jobs submitted. Watch: qstat -u \"\$USER\"  |  Summarize: python scripts/summarize_baselines.py"
