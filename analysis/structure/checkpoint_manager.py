"""
Checkpoint manager for graph metrics computation.

Tracks completion state per (dataset, category) and provides
save/load helpers for intermediate results and path-mining checkpoints.
"""

import json
import os
import pickle
import logging
import time
import threading
from pathlib import Path

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages checkpoint state for the graph metrics pipeline."""

    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.state_file = self.output_dir / "checkpoint_state.json"
        self._lock = threading.Lock()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._state = self._load_state()

    def _load_state(self) -> dict:
        if self.state_file.exists():
            with open(self.state_file, "r") as f:
                return json.load(f)
        return {}

    def _save_state(self):
        tmp = self.state_file.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(self._state, f, indent=2)
        tmp.rename(self.state_file)

    def _key(self, dataset: str, category: str) -> str:
        return f"{dataset}::{category}"

    def is_completed(self, dataset: str, category: str) -> bool:
        with self._lock:
            return self._state.get(self._key(dataset, category)) == "completed"

    def mark_in_progress(self, dataset: str, category: str):
        with self._lock:
            self._state[self._key(dataset, category)] = "in_progress"
            self._save_state()

    def mark_completed(self, dataset: str, category: str):
        with self._lock:
            self._state[self._key(dataset, category)] = "completed"
            self._save_state()

    # ---- Result persistence ----

    def _results_path(self, dataset: str, category: str) -> Path:
        d = self.output_dir / "results"
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{dataset}__{category}.json"

    def save_results(self, dataset: str, category: str, results: dict):
        """Save metric results dict as JSON and mark category completed."""
        path = self._results_path(dataset, category)
        serialisable = _make_serialisable(results)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(serialisable, f, indent=2, default=str)
        tmp.rename(path)
        self.mark_completed(dataset, category)
        logger.info("Saved results for %s / %s -> %s", dataset, category, path)

    def load_results(self, dataset: str, category: str) -> dict | None:
        path = self._results_path(dataset, category)
        if path.exists():
            with open(path, "r") as f:
                return json.load(f)
        return None

    # ---- Path-mining entity-level checkpoints ----

    def paths_checkpoint_path(self, dataset: str) -> Path:
        d = self.output_dir / "path_analysis"
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{dataset}_paths_checkpoint.pkl"

    def save_paths_checkpoint(self, dataset: str, entity_paths: dict, entity_directions: dict):
        path = self.paths_checkpoint_path(dataset)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump({"paths": entity_paths, "directions": entity_directions}, f,
                        protocol=pickle.HIGHEST_PROTOCOL)
        tmp.rename(path)

    def load_paths_checkpoint(self, dataset: str) -> tuple[dict, dict] | None:
        path = self.paths_checkpoint_path(dataset)
        if path.exists():
            with open(path, "rb") as f:
                data = pickle.load(f)
            return data["paths"], data["directions"]
        return None


def _make_serialisable(obj):
    """Recursively convert numpy/non-JSON types to Python builtins."""
    import numpy as np
    if isinstance(obj, dict):
        return {str(k): _make_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serialisable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if obj is None or isinstance(obj, (int, float, str, bool)):
        return obj
    return str(obj)
