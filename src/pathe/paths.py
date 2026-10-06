"""
Locations of data, checkpoints, runs and results, read from ``configs/paths.yaml``.

The repository root is ``$PATHE_ROOT`` if set, otherwise the first directory at or
above the working directory that contains ``configs/paths.yaml``, otherwise the
checkout that holds this package.
"""
import os
from pathlib import Path

import yaml

KEYS = ("data_root", "checkpoint_root", "runs_root", "results_root")


def repo_root() -> Path:
    if os.environ.get("PATHE_ROOT"):
        return Path(os.environ["PATHE_ROOT"])
    cwd = Path.cwd().resolve()
    for d in (cwd, *cwd.parents):
        if (d / "configs" / "paths.yaml").is_file():
            return d
    return Path(__file__).resolve().parents[2]


def load_paths(config=None) -> dict:
    """Return ``{key: absolute Path}`` for the keys of ``configs/paths.yaml``."""
    root = repo_root()
    config = Path(config) if config else root / "configs" / "paths.yaml"
    with open(config) as f:
        paths = yaml.safe_load(f)
    missing = [k for k in KEYS if k not in paths]
    if missing:
        raise KeyError(f"{config} lacks {missing}")
    return {k: (Path(v) if os.path.isabs(v) else root / v) for k, v in paths.items()}
