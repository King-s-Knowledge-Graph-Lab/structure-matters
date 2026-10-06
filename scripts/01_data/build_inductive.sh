#!/usr/bin/env bash
# Mine the inductive path datasets (GraIL splits) of one KG into
# <data_root>/path_datasets/<dataset>/<version>/ (data_root from configs/paths.yaml).
#   scripts/01_data/build_inductive.sh wn18rr [v1 v2 v3 v4] [-- --seed 42]
set -euo pipefail
cd "$(dirname "$0")/../.."
dataset=${1:?usage: $0 <dataset> [versions] [-- mine_paths options]}; shift
versions=()
while [[ $# -gt 0 && $1 != "--" ]]; do versions+=("$1"); shift; done
[[ ${1:-} == "--" ]] && shift
[[ ${#versions[@]} -eq 0 ]] && versions=(v1 v2 v3 v4)
for v in "${versions[@]}"; do
  PYTHONPATH=src python -m pathe.build.mine_paths inductive "$dataset" --version "$v" "$@"
done
