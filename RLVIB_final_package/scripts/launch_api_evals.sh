#!/bin/bash
# Full-set closed-API baseline evals (gemini, gpt4o) on AVHBench (5302) + CMM (2400),
# DETACHED via nohup so they survive an SSH drop. Run on a login node WITH internet, in the
# rlvib env, keys exported. These are CPU/network only (no GPU, no qsub).
#
#   export GEMINI_API_KEY=...  OPENAI_API_KEY=...
#   LIMIT=20 MODELS=gemini bash scripts/launch_api_evals.sh    # CHEAP smoke check first (~40 calls)
#   bash scripts/launch_api_evals.sh                            # full both models (~15k calls)
#
# CMM reads runs/cmm_skip_clips.txt (the same clips the local models skip) so every model is
# scored on the same item set. run_{avhbench,cmm} resume their _full JSON, so a smoke run at
# LIMIT=20 is just continued by the full run -- nothing is wasted or double-charged.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
mkdir -p runs

MODELS="${MODELS:-gemini gpt4o}"
LIMIT="${LIMIT:-0}"                # 0 = full set; set e.g. LIMIT=20 for a cheap smoke check
TAG="${TAG:-_full}"
AVHQA="${AVHQA:-$PWD/data/AVHBench/qa.json}"
AVHVID="${AVHVID:-$PWD/data/AVHBench/videos}"
CMMJSON="${CMMJSON:-$PWD/data/CMM/all_data_final_reorg.json}"
CMMROOT="${CMMROOT:-$PWD/data/CMM}"

for m in $MODELS; do
    [ "$m" = gemini ] && : "${GEMINI_API_KEY:?export GEMINI_API_KEY (or GOOGLE_API_KEY) for gemini}"
    [ "$m" = gpt4o ]  && : "${OPENAI_API_KEY:?export OPENAI_API_KEY for gpt4o}"
done

launch() {  # $1=module  $2=outbase  $3...=extra args
    local mod="$1" out="$2"; shift 2
    nohup python -u -m "$mod" --limit "$LIMIT" "$@" --out "runs/${out}.json" \
        > "runs/${out}.log" 2>&1 &
    echo "   -> runs/${out}.json   (PID $!, log runs/${out}.log)"
}

echo "=== API evals (LIMIT=$LIMIT, models: $MODELS) ==="
for m in $MODELS; do
    echo ">> $m AVHBench"
    launch rlvib.eval.run_avhbench "avhbench_${m}${TAG}" --model "$m" --qa-json "$AVHQA" --video-root "$AVHVID"
    echo ">> $m CMM"
    launch rlvib.eval.run_cmm "cmm_${m}${TAG}" --model "$m" --json-path "$CMMJSON" --data-root "$CMMROOT"
done

echo
echo "launched detached (safe to drop SSH)."
echo "  watch:    tail -f runs/cmm_${MODELS%% *}${TAG}.log"
echo "  progress: python scripts/check_full_jsons.py runs/*_gemini${TAG}.json runs/*_gpt4o${TAG}.json"
echo "  stop all: pkill -f 'rlvib.eval.run_'"
