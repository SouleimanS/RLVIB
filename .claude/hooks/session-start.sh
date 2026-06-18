#!/bin/bash
# RLVIB - Claude Code (web) session setup.
#
# The full GPU runtime (torch / transformers, see environment.yml) is ABCI-cluster only;
# web sessions are for editing, linting and the pure-Python tests. So we install only the
# light surface: the package (editable, so `import rlvib` works) + ruff + pytest + the
# stdlib-adjacent deps the torch-free modules use. No torch -> fast, cacheable startup.
set -euo pipefail

# Only run in the remote (web) environment; local/cluster sessions use the conda env.
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "${CLAUDE_PROJECT_DIR:-.}"

python -m pip install --quiet ruff pytest numpy tqdm
python -m pip install --quiet -e .

echo "rlvib web session ready (ruff + pytest + editable install; GPU stack stays cluster-only)."
