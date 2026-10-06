#!/usr/bin/env bash
# Download, verify and unpack the artifacts of docs/ARTIFACTS.md into <data_root> and
# <checkpoint_root> (configs/paths.yaml), then link the checkpoints for runner.py test.
#   scripts/00_setup/fetch_artifacts.sh [SOURCE] [PACKAGE ...]
# SOURCE is the base URL of the release (http://, https://) or a local folder holding the packages;
# default $PATHE_ARTIFACTS. PACKAGE is a name of scripts/00_setup/artifacts.txt, or libkge-complex
# for the three public LibKGE checkpoints (always fetched from LibKGE); default all of them.
# Downloads are kept in $PATHE_DOWNLOADS (default artifacts/downloads) and not fetched again if
# their md5 matches.
set -euo pipefail
cd "$(dirname "$0")/../.."
source=${1:-${PATHE_ARTIFACTS:-}}
[ -n "$source" ] || { sed -n '2,9p' "$0"; exit 2; }
[ $# -gt 0 ] && shift
manifest=scripts/00_setup/artifacts.txt
external=scripts/00_setup/artifacts_external.txt
downloads=${PATHE_DOWNLOADS:-artifacts/downloads}
mkdir -p "$downloads/libkge"
read -r data_root checkpoint_root < <(PYTHONPATH=src python -c \
  'from pathe.paths import load_paths as l; p = l(); print(p["data_root"], p["checkpoint_root"])')

wanted=("$@")
for p in "${wanted[@]}"; do
  [ "$p" = libkge-complex ] || grep -q " $p\$" "$manifest" || { echo "unknown package: $p (see $manifest)"; exit 2; }
done
want() { [ ${#wanted[@]} -eq 0 ] || [[ " ${wanted[*]} " == *" $1 "* ]]; }
md5() { md5sum < "$1" | cut -c1-32; }
root_of() {
  case $1 in data_root) echo "$data_root";; checkpoint_root) echo "$checkpoint_root";;
    *) echo "unknown root $1" >&2; return 1;; esac
}
get() {  # get URL_OR_FILE DEST
  case $1 in
    http://*|https://*) curl -fL --retry 3 -o "$2.part" "$1" && mv "$2.part" "$2";;
    *) cp "$1" "$2";;
  esac
}

status=0
while read -r sum size key name; do
  case $sum in \#*|"") continue;; esac
  want "$name" || continue
  file=$downloads/$name
  if [ ! -f "$file" ] || [ "$(md5 "$file")" != "$sum" ]; then
    echo "fetching $name ($((size / 1048576)) MB)"
    get "${source%/}/$name" "$file" || { echo "could not fetch $name"; status=1; continue; }
  fi
  [ "$(md5 "$file")" = "$sum" ] || { echo "md5 mismatch: $name (expected $sum); not unpacked"; status=1; continue; }
  root=$(root_of "$key") || { status=1; continue; }
  mkdir -p "$root"
  tar -xzf "$file" -C "$root" --no-same-owner
  echo "unpacked $name into $root"
done < "$manifest"

if want libkge-complex; then
  while read -r sum size key rel url; do
    case $sum in \#*|"") continue;; esac
    root=$(root_of "$key") || { status=1; continue; }
    target=$root/$rel
    if [ -f "$target" ] && [ "$(md5 "$target")" = "$sum" ]; then continue; fi
    file=$downloads/libkge/$(basename "$rel")
    if [ ! -f "$file" ] || [ "$(md5 "$file")" != "$sum" ]; then
      echo "fetching $(basename "$rel") ($((size / 1048576)) MB) from LibKGE"
      get "$url" "$file" || { echo "could not fetch $url"; status=1; continue; }
    fi
    [ "$(md5 "$file")" = "$sum" ] || { echo "md5 mismatch: $url (expected $sum)"; status=1; continue; }
    mkdir -p "$(dirname "$target")"
    cp "$file" "$target"
    echo "copied $(basename "$rel") into $(dirname "$target")"
  done < "$external"
fi

PYTHONPATH=src python scripts/00_setup/link_runs.py
exit $status
