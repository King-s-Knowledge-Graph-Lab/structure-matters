#!/usr/bin/env bash
# Build the release packages of docs/ARTIFACTS.md from <data_root> and <checkpoint_root>
# (configs/paths.yaml) into OUT_DIR, and write scripts/00_setup/artifacts.txt.
#   scripts/00_setup/pack_artifacts.sh OUT_DIR
# The archives are byte-reproducible (sorted members, fixed owner and time, gzip -n), so packing
# the same files again gives the same md5. The checkpoints must be the release copies written by
# scripts/00_setup/clean_checkpoint.py. The three public LibKGE checkpoints are not packed; they
# are listed in scripts/00_setup/artifacts_external.txt.
set -euo pipefail
cd "$(dirname "$0")/../.."
out=${1:?usage: $0 OUT_DIR}
mkdir -p "$out"
out=$(cd "$out" && pwd)
read -r data_root checkpoint_root < <(PYTHONPATH=src python -c \
  'from pathe.paths import load_paths as l; p = l(); print(p["data_root"], p["checkpoint_root"])')
manifest=scripts/00_setup/artifacts.txt

pack() {  # pack NAME ROOT_KEY ROOT_DIR FILE...   (files relative to ROOT_DIR)
  local name=$1 key=$2 dir=$3; shift 3
  (cd "$dir" && printf '%s\n' "$@" | LC_ALL=C sort | tar --create --no-recursion \
    --files-from=- --owner=0 --group=0 --numeric-owner --mtime='2026-10-05 00:00:00Z' \
    --format=gnu --mode='u+rw,go+r,go-w') | gzip -n -9 > "$out/$name"
  printf '%s %s %s %s\n' "$(md5sum < "$out/$name" | cut -c1-32)" \
    "$(stat -c %s "$out/$name")" "$key" "$name" >> "$manifest.tmp"
  echo "$name: $# files"
}

files() {  # files DIR FIND_ARGS...   (regular files, relative to DIR)
  local dir=$1; shift
  (cd "$dir" && find "$@")
}

: > "$manifest.tmp"
pack pathe-checkpoints.tar.gz checkpoint_root "$checkpoint_root" \
  $(files "$checkpoint_root" link_prediction relation_prediction inductive ablations -type f -name '*.ckpt')
pack complex-codex-l.tar.gz checkpoint_root "$checkpoint_root" complex/codex-l-complex.pt
pack complex-entity-matrices.tar.gz data_root "$data_root" \
  $(files "$data_root" path_datasets/{fb15k237,wn18rr,yago}/train -type f -name '*-complex*')
for kg in fb15k237 codex-large wn18rr yago; do
  pack paths-$kg.tar.gz data_root "$data_root" \
    $(files "$data_root" path_datasets/$kg/{train,val,test} -type f ! -name '20_20_*' ! -name '*-complex*')
done
pack paths-relation-prediction.tar.gz data_root "$data_root" \
  $(files "$data_root" path_datasets/{fb15k237,wn18rr}/{train,val,test} -type f -name '20_20_er_paths.csv')
pack paths-inductive.tar.gz data_root "$data_root" \
  $(files "$data_root" path_datasets/{fb15k237,wn18rr}/v{1,2,3,4} -type f)

{ echo "# md5 size root package   (written by scripts/00_setup/pack_artifacts.sh)"
  cat "$manifest.tmp"; } > "$manifest"
rm "$manifest.tmp"
cat "$manifest"
