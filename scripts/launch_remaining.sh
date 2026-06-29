#!/bin/bash
# Launch the REMAINING evals to complete the corrected-harness campaign, all in parallel on the
# 8 GPUs of one rt_HF (H200) node. Run INSIDE an interactive GPU job, e.g.:
#
#   qsub -I -P gae50891 -q rt_HF -l select=1 -l walltime=06:00:00
#   cd "$PBS_O_WORKDIR"          # (or the repo root)
#   bash scripts/launch_remaining.sh
#
# Why eval_one.sh and not launch_all_evals.sh: eval_one encodes the correct PER-MODEL fps
# (qwen2.5=1, qwen3=2) and the corrected yn-suffix automatically. launch_all_evals.sh hard-pins
# FPS=1 for every Qwen, which starves qwen3's video task and reintroduces yes-bias (see CLAUDE/
# agent-onboarding §6). Each run is RESUMABLE: a walltime cutoff is fine, just re-run this script.
#
# Remaining cells (gemini/gpt4o are API-only, no VIB -> already complete; qwen2.5-omni is also
# done -- its base/broad@60/broad@150 full-set runs just live under smoke_* filenames):
#   qwen3-omni   : broad@60, broad@150            (base already _sysfull)
#   videollama2  : base, broad@60, broad@150      (own env rlvib_vl2)
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs/launchlogs
EXP="${EXP:-broad}"

# Unify naming: some full-set runs were written under smoke_* filenames (e.g. qwen2.5-omni's
# base/broad@60/broad@150 are real full runs, just misnamed). make_table ranks sysfull>full>smoke,
# so copy those smoke_* JSONs to the canonical sysfull names and the cells report as a complete
# source instead of 'smoke'. Non-destructive: cp only, and never clobbers an existing sysfull file.
# Override with UNIFY_MODEL= / UNIFY_STEPS= ; set UNIFY_MODEL= empty to skip.
_cp_pair() {  # $1=src smoke json   $2=dst sysfull json
    [ -f "$1" ] || { echo "  unify: skip (no $1)"; return 0; }
    [ -e "$2" ] && { echo "  unify: keep existing $2"; return 0; }
    cp "$1" "$2" && echo "  unify: $1 -> $2"
}
unify_names() {
    local M="${UNIFY_MODEL-qwen2.5-omni}" s
    [ -n "$M" ] || { echo "unify_names: UNIFY_MODEL empty -> skip"; return 0; }
    echo "=== unify naming for $M (smoke_* full runs -> sysfull) ==="
    for s in ${UNIFY_STEPS:-base 60 150}; do
        if [ "$s" = base ]; then
            _cp_pair "runs/smoke_avh_${M}.json"               "runs/avhbench_${M}_sysfull.json"
            _cp_pair "runs/smoke_cmm_${M}.json"               "runs/cmm_${M}_sysfull.json"
        else
            _cp_pair "runs/smoke_avh_${M}_${EXP}_step${s}.json" "runs/avhbench_${M}_${EXP}_sysfull_step${s}.json"
            _cp_pair "runs/smoke_cmm_${M}_${EXP}_step${s}.json" "runs/cmm_${M}_${EXP}_sysfull_step${s}.json"
        fi
    done
}
unify_names

echo "remaining-eval launcher: 5 configs across GPUs 0-4. logs -> runs/launchlogs/*.log"

# GPU  model         step   env            log
CUDA_VISIBLE_DEVICES=0                      bash scripts/eval_one.sh qwen3-omni   60  >runs/launchlogs/q3_60.log    2>&1 &
CUDA_VISIBLE_DEVICES=1                      bash scripts/eval_one.sh qwen3-omni   150 >runs/launchlogs/q3_150.log   2>&1 &
CUDA_VISIBLE_DEVICES=2 CONDA_ENV=rlvib_vl2  bash scripts/eval_one.sh videollama2      >runs/launchlogs/vl2_base.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 CONDA_ENV=rlvib_vl2  bash scripts/eval_one.sh videollama2  60  >runs/launchlogs/vl2_60.log   2>&1 &
CUDA_VISIBLE_DEVICES=4 CONDA_ENV=rlvib_vl2  bash scripts/eval_one.sh videollama2  150 >runs/launchlogs/vl2_150.log  2>&1 &

echo "launched 5 background runs. live progress:  tail -f runs/launchlogs/*.log"
echo "waiting for all to finish (re-run this script after a walltime cutoff to resume)..."
wait
echo "=== all remaining evals finished -> tabulating ==="
python scripts/make_table.py
