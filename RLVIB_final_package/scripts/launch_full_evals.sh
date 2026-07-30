#!/bin/bash
# Launch full-set (LIMIT=0) evals for ALL available models on the LOGIN node:
#   - each model's BASE
#   - every trained checkpoint in the chosen family (default broad*), with the step SELECTED
#     on the 300-subset via select_holdout
# Each item submits AVHBench + CMM (+ DAVE) through full_eval.sh, tagged _full so it does NOT
# clobber the 300-subset selection JSONs. Reports later with paired_stats --suffix full --dev 300.
#
#   bash scripts/launch_full_evals.sh
#   EXPGLOB='*' bash scripts/launch_full_evals.sh        # every trained variant (abl/bkl/seeds/...)
#   WITH_DAVE=0 MODELS='qwen2.5-omni' bash scripts/launch_full_evals.sh
#   FORCE=1 bash scripts/launch_full_evals.sh            # ignore the already-submitted guard
#
# IDEMPOTENT: each item drops a sentinel in runs/.fulleval_submitted/ once submitted, and is
# skipped on a re-run. This is what stops a second launcher run from putting a duplicate job on
# a JSON a live job is still writing (the bug that risked jamming runs/*_full*.json). To force a
# clean resubmit of everything: FORCE=1 ... or rm -rf runs/.fulleval_submitted.
set -uo pipefail                       # NOT -e: keep going when an item lacks data
cd "$(dirname "$0")/.."

export WITH_DAVE="${WITH_DAVE:-1}"
MODELS="${MODELS:-qwen3-omni qwen2.5-omni videollama2}"
EXPGLOB="${EXPGLOB:-broad*}"           # which trained variants to include

GUARD_DIR="runs/.fulleval_submitted"
mkdir -p "$GUARD_DIR"
submitted() { [ "${FORCE:-0}" != 1 ] && [ -e "$GUARD_DIR/$1" ]; }  # already launched this item?
mark() { : > "$GUARD_DIR/$1"; }                                    # remember we launched it

echo "=== BASES (full set) ==="
for M in $MODELS; do
    if submitted "${M}__full"; then echo "   skip $M base (already submitted; FORCE=1 to override)"; continue; fi
    echo ">> $M base"
    if MODEL=$M BASE_ONLY=1 bash scripts/full_eval.sh; then mark "${M}__full"; else echo "   skip ($M base)"; fi
done

echo "=== TRAINED (family '$EXPGLOB'; step selected on the 300-subset) ==="
for M in $MODELS; do
    for dir in runs/anchored_${M}_${EXPGLOB}/; do
        [ -d "$dir" ] || continue
        exp=$(basename "$dir"); exp="${exp#anchored_${M}_}"
        step=$(python scripts/select_holdout.py --model "$M" --exp "$exp" 2>/dev/null \
               | grep -oP 'HONEST.*?step\K[0-9]+' | head -1 || true)
        if [ -z "${step:-}" ]; then
            echo "   skip $M/$exp (no selected step -- run select_checkpoint.sh + select_holdout on the 300-subset first)"
            continue
        fi
        key="${M}__${exp}_full_step${step}"
        if submitted "$key"; then echo "   skip $M/$exp step$step (already submitted; FORCE=1 to override)"; continue; fi
        echo ">> $M/$exp step$step"
        if MODEL=$M EXP="$exp" STEP="$step" bash scripts/full_eval.sh; then mark "$key"; else echo "   skip ($M/$exp)"; fi
    done
done

echo "=== submitted. watch: qstat -u \"\$USER\" ==="
echo "When done, report on the held-out (full minus the 300 dev), e.g.:"
echo "  python scripts/paired_stats.py --model qwen3-omni --exp broad --step 60 --suffix full --dev 300"
