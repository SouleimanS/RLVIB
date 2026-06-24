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
