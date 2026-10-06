#!/usr/bin/env bash
# One-off setup for training ComplEx on CoDEx-L with LibKGE. Does NOT start training.
# Run from anywhere:  bash scripts/07_complex_baseline/setup_libkge.sh
# Locations (defaults follow configs/paths.yaml; override with the environment variables):
#   LIBKGE_ROOT  LibKGE checkout, its venv and the run folder   (default <repo>/artifacts/libkge)
#   PATHE_DATA   PathE's CoDEx-Large path dataset                (default <repo>/artifacts/data/path_datasets/codex-large)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
LIBKGE_ROOT="${LIBKGE_ROOT:-$REPO/artifacts/libkge}"
PATHE_DATA="${PATHE_DATA:-$REPO/artifacts/data/path_datasets/codex-large}"
LIBKGE_COMMIT=9bca31fefbfc41d48fdc03396acd0cf42eee9dc4
mkdir -p "$LIBKGE_ROOT" && cd "$LIBKGE_ROOT"

# 1. separate environment, so that LibKGE's dependencies do not touch the PathE environment
PY="$(command -v python3.10 || command -v python3)"   # LibKGE needs Python >= 3.7; 3.10 is what the PathE environment uses
"$PY" -m venv venv_kge
source venv_kge/bin/activate
pip install --upgrade pip
# same torch build as the PathE environment
pip install torch==2.1.1 --index-url https://download.pytorch.org/whl/cu118

# 2. LibKGE at the commit that was used, with its hyper-parameter-search imports made optional
if [ ! -d kge/.git ]; then
  git clone https://github.com/uma-pi1/kge.git
  git -C kge checkout "$LIBKGE_COMMIT"
  git -C kge apply "$HERE/libkge_optional_search.patch"
fi
pip install -e kge
pip install "numpy<2"        # torch 2.1.1 needs NumPy 1.x

# 3. CoDEx-L in LibKGE format, written from the exact files PathE uses (same entities, same splits, same labels)
python "$HERE/make_libkge_dataset.py" "$PATHE_DATA" kge/data/codex-l
( cd kge/data && python preprocess/preprocess_default.py codex-l )

# 4. checks
python - <<'PY'
import torch, yaml
print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available(), "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU")
d = yaml.safe_load(open("kge/data/codex-l/dataset.yaml"))["dataset"]
assert d["num_entities"] == 77951 and d["num_relations"] == 69, d
print("dataset ok:", d["num_entities"], "entities,", d["num_relations"], "relations")
PY
echo "Setup finished. Start the training with:  bash $HERE/train.sh"
