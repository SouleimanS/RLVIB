#!/bin/bash
# Submit MMAU (audio understanding: Sound/Music/Speech) eval for base + trained checkpoints,
# for the given models. The runner auto-detects a FiLM bottleneck and sets the condition.
# Needs the MMAU audios under data/MMAU/ (see scripts/eval_mmau.qsub header).
#
#   bash scripts/launch_mmau.sh                                       # base, both Qwen models
#   CKPTS="broad:60" bash scripts/launch_mmau.sh                      # base + broad@60, both
#   CKPTS="broad:60 film:90" bash scripts/launch_mmau.sh qwen3-omni   # base + 2 trained, qwen3 only
#
# When done:  python scripts/make_table.py  reads nothing MMAU; inspect runs/mmau_<model>*.json
# (each has results.{sound,music,speech,overall}). Compare base vs trained per model.
set -euo pipefail
cd "$(dirname "$0")/.."

MODELS="${MODELS:-qwen2.5-omni qwen3-omni}"
[ $# -gt 0 ] && MODELS="$*"
CKPTS="${CKPTS:-}"           # space-separated "exp:step" pairs, e.g. "broad:60 film:90"

for M in $MODELS; do
    echo ">> MMAU base: $M"
    qsub -v "MODEL=$M" scripts/eval_mmau.qsub
    for cs in $CKPTS; do
        exp="${cs%%:*}"; step="${cs##*:}"
        bn="runs/anchored_${M}_${exp}/bottleneck_step${step}.pt"
        if [ -f "$bn" ]; then
            echo ">> MMAU $M $exp step$step"
            qsub -v "MODEL=$M,BOTTLENECK=$bn,TAG=_${exp}_step${step}" scripts/eval_mmau.qsub
        else
            echo "   skip $M $exp step$step (missing $bn)"
        fi
    done
done
echo "watch: qstat -u \"\$USER\"   outputs: runs/mmau_<model>[_<exp>_step<n>].json"
