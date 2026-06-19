#!/bin/bash
# Full-set (LIMIT=0) eval of a model's BASE and (optionally) one selected checkpoint, tagged
# "_full" so it does NOT clobber the 300-subset selection JSONs. Report afterwards on the
# held-out (full minus the 300 dev) with:
#     python scripts/paired_stats.py --model <M> --exp <EXP> --step <STEP> --suffix full --dev 300
#
#   MODEL=qwen3-omni BASE_ONLY=1 bash scripts/full_eval.sh            # base only (once per model)
#   MODEL=qwen3-omni EXP=broad STEP=60 bash scripts/full_eval.sh      # one selected checkpoint
#   WITH_DAVE=0 ... bash scripts/full_eval.sh                          # skip DAVE
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-qwen3-omni}"
ENV="${CONDA_ENV:-rlvib}"
[ "$MODEL" = videollama2 ] && ENV="${CONDA_ENV:-rlvib_vl2}"
CMM_JSON="${CMM_JSON:-$PWD/data/CMM/all_data_final_reorg.json}"
CMM_ROOT="${CMM_ROOT:-$PWD/data/CMM}"
WITH_DAVE="${WITH_DAVE:-1}"
DAVE_SPLIT="${DAVE_SPLIT:-ego4d}"
WALL="${WALL:-12:00:00}"
ONLY="${ONLY:-}"   # empty = avhbench+cmm(+dave); else a subset, e.g. ONLY=cmm or ONLY="cmm dave"
                   # (a CMM/DAVE job resumes its _full JSON, so this is the safe way to finish
                   #  just one benchmark without resubmitting -- and re-duplicating -- the others)
want() { [ -z "$ONLY" ] && return 0; case " $ONLY " in *" $1 "*) return 0;; *) return 1;; esac; }

submit() {  # $1=TAG  $2=BOTTLENECK(optional)
    local TAG="$1" BN="${2:-}"
    if want avhbench; then
        qsub -v "MODEL=$MODEL,LIMIT=0,TAG=$TAG,CONDA_ENV=$ENV${BN:+,BOTTLENECK=$BN}" \
            -l walltime="$WALL" scripts/eval_avhbench.qsub
    fi
    if want cmm; then
        qsub -v "MODEL=$MODEL,LIMIT=0,TAG=$TAG,CMM_JSON=$CMM_JSON,CMM_ROOT=$CMM_ROOT,CONDA_ENV=$ENV${BN:+,BOTTLENECK=$BN}" \
            -l walltime="$WALL" scripts/eval_cmm.qsub
    fi
    if [ "$WITH_DAVE" = 1 ] && want dave; then
        qsub -v "MODEL=$MODEL,LIMIT=0,TAG=$TAG,MODE=audio_visual_alignment,DAVE_SPLIT=$DAVE_SPLIT,CONDA_ENV=$ENV${BN:+,BOTTLENECK=$BN}" \
            -l walltime="$WALL" scripts/eval_dave.qsub
    fi
}

if [ "${BASE_ONLY:-0}" = 1 ]; then
    echo ">> FULL base eval: $MODEL  (tag _full)"
    submit "_full"
else
    : "${EXP:?set EXP (e.g. broad / broad_s1)}"
    : "${STEP:?set STEP (the select_holdout pick for this seed)}"
    BN="$PWD/runs/anchored_${MODEL}_${EXP}/bottleneck_step${STEP}.pt"
    [ -f "$BN" ] || { echo "missing checkpoint $BN"; exit 1; }
    echo ">> FULL eval: $MODEL $EXP step$STEP  (tag _${EXP}_full_step${STEP})"
    submit "_${EXP}_full_step${STEP}" "$BN"
fi
echo "submitted. these JSONs already use the fixed parser (no rederive needed). When done, report"
echo "on the held-out (full minus the 300 dev):"
echo "  python scripts/paired_stats.py --model $MODEL --exp ${EXP:-broad} --step ${STEP:-STEP} --suffix full --dev 300"
