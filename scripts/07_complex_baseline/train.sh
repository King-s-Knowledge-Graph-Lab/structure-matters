#!/usr/bin/env bash
# Starts (or resumes) the training of ComplEx on CoDEx-L (configs/baselines/complex_codex-l.yaml).
# Expect roughly 4-8 hours on an RTX A2000 (estimate, not measured). Run setup_libkge.sh first.
#   nohup bash scripts/07_complex_baseline/train.sh > train.log 2>&1 &      then:  tail -f train.log
# LIBKGE_ROOT as in setup_libkge.sh (default <repo>/artifacts/libkge).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
LIBKGE_ROOT="${LIBKGE_ROOT:-$REPO/artifacts/libkge}"
cd "$LIBKGE_ROOT"
source venv_kge/bin/activate
if [ -d run ]; then
  kge resume run            # continues from the last checkpoint if the run was interrupted
else
  kge start "$REPO/configs/baselines/complex_codex-l.yaml" --folder run
fi
echo "Done. Best model: $LIBKGE_ROOT/run/checkpoint_best.pt"
echo "To use it in analysis/query_level/complex_ranks.py, copy it to <checkpoint_root>/complex/codex-l-complex.pt"
