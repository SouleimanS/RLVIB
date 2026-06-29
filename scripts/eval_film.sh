#!/bin/bash
# Full-set AVHBench + CMM eval of a trained FiLM checkpoint (+ the base row for comparison),
# submitted as batch jobs. The runners auto-detect the FiLM bottleneck (q_proj) and set the
# per-question condition, so nothing extra is needed here. DAVE is skipped (ONLY avhbench cmm).
#
#   STEP=90 bash scripts/eval_film.sh qwen3-omni              # FiLM step90 + base
#   STEP=90 NO_BASE=1 bash scripts/eval_film.sh qwen3-omni    # FiLM only (base already done)
#   STEP=90 EXP=film bash scripts/eval_film.sh qwen2.5-omni   # (EXP defaults to 'film')
#
# When the jobs finish:  python scripts/make_table.py   (the _film_full_step* rows appear)
# and  python scripts/paired_stats.py --model <M> --exp film --step <STEP> --suffix full --dev 300
set -euo pipefail
cd "$(dirname "$0")/.."

M="${1:-qwen3-omni}"
EXP="${EXP:-film}"
STEP="${STEP:?set STEP=<the FiLM checkpoint step, e.g. runs/anchored_${M}_${EXP}/bottleneck_step90.pt -> 90>}"
BN="runs/anchored_${M}_${EXP}/bottleneck_step${STEP}.pt"
[ -f "$BN" ] || { echo "missing FiLM checkpoint: $BN  (is the Stage-2 run done? check runs/anchored_${M}_${EXP}/)"; exit 1; }

echo "=== FiLM eval: $M $EXP step$STEP (AVHBench + CMM, full) ==="
ONLY="avhbench cmm" WITH_DAVE=0 MODEL="$M" EXP="$EXP" STEP="$STEP" bash scripts/full_eval.sh

if [ "${NO_BASE:-0}" != 1 ]; then
    echo "=== base eval: $M (AVHBench + CMM, full) -- for the base-vs-FiLM comparison ==="
    ONLY="avhbench cmm" WITH_DAVE=0 MODEL="$M" BASE_ONLY=1 bash scripts/full_eval.sh
fi
echo "watch: qstat -u \"\$USER\"   then:  python scripts/make_table.py"
