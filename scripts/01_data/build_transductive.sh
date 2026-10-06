#!/usr/bin/env bash
# Mine the transductive path dataset of one KG into <data_root>/path_datasets/<dataset>/
# (data_root from configs/paths.yaml). The KG itself is downloaded by PyKEEN.
#   scripts/01_data/build_transductive.sh fb15k237 [--seed 42] [--num_paths 20 --num_steps 10]
# The released 20_10 files were mined without a seed: a rebuild gives other paths, and only the
# released files reproduce the reported numbers exactly.
set -euo pipefail
cd "$(dirname "$0")/../.."
dataset=${1:?usage: $0 <dataset> [mine_paths options]}; shift
PYTHONPATH=src python -m pathe.build.mine_paths transductive "$dataset" "$@"
