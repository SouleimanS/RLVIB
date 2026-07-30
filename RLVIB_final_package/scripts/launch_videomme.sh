#!/bin/bash
# Submit Video-MME (w/o subs) for base + trained checkpoints. The runner auto-detects a
# FiLM bottleneck and sets the question condition. Needs data/VideoMME/ populated first
# (see scripts/eval_videomme.qsub header for the download).
#
#   bash scripts/launch_videomme.sh                                     # base qwen3-omni, short
#   CKPTS="film:160" bash scripts/launch_videomme.sh qwen3-omni         # base + FiLM@160
#   CKPTS="broad:60 film:160" DURATIONS=short bash scripts/launch_videomme.sh qwen3-omni
#
# Outputs: runs/videomme_<model>[_<exp>_step<n>].json -- compare results.overall.acc and
# the per-duration entries base vs trained (the adapter must not cost video QA accuracy).
set -euo pipefail
cd "$(dirname "$0")/.."

MODELS="${MODELS:-qwen3-omni}"
[ $# -gt 0 ] && MODELS="$*"
CKPTS="${CKPTS:-}"           # space-separated "exp:step" pairs, e.g. "broad:60 film:160"
DURATIONS="${DURATIONS:-short}"

for M in $MODELS; do
    echo ">> Video-MME base: $M (durations=$DURATIONS)"
    qsub -v "MODEL=$M,DURATIONS=$DURATIONS" scripts/eval_videomme.qsub
    for cs in $CKPTS; do
        exp="${cs%%:*}"; step="${cs##*:}"
        bn="runs/anchored_${M}_${exp}/bottleneck_step${step}.pt"
        if [ -f "$bn" ]; then
            echo ">> Video-MME $M $exp step$step"
            qsub -v "MODEL=$M,DURATIONS=$DURATIONS,BOTTLENECK=$bn,TAG=_${exp}_step${step}" \
                scripts/eval_videomme.qsub
        else
            echo "   skip $M $exp step$step (missing $bn)"
        fi
    done
done
echo "watch: qstat -u \"\$USER\"   outputs: runs/videomme_<model>[_<exp>_step<n>].json"
