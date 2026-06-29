#!/bin/bash
# Interactive launcher for GRPO training on a GPU node. The python entrypoint can't self-activate
# conda (it's already running in whatever interpreter you invoked), so this wrapper does it -- same
# pattern as eval_one.sh -- then forwards ALL args to train_grpo.py. Avoids the "ImportError:
# Qwen3OmniMoeForConditionalGeneration" you get from the system python3.9 when rlvib isn't active.
#
#   bash scripts/train_grpo.sh --model qwen3-omni --pairs 300 --epochs 2 --group 8
#   CONDA_ENV=rlvib_vl2 bash scripts/train_grpo.sh --model videollama2 ...
# (GPU pin + expandable_segments are set inside train_grpo.py; --std-adv reverts to classic GRPO.)
set -uo pipefail
cd "$(dirname "$0")/.."

source /home/aab11336im/anaconda3/etc/profile.d/conda.sh
set +u                                   # conda activate.d hooks aren't `set -u`-safe
conda activate "${CONDA_ENV:-rlvib}"
set -u
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export TRANSFORMERS_OFFLINE=1
python -c "import torch; assert torch.cuda.is_available(), 'NO GPU -- run inside an interactive session'"

python -u scripts/train_grpo.py "$@"
