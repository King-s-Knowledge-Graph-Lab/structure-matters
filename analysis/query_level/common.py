"""Locations and file names shared by the Section 7 scripts (roots from configs/paths.yaml)."""
from pathlib import Path

from pathe.paths import load_paths

KGS = ["fb15k237", "codex-large", "wn18rr", "yago"]
TITLES = {"fb15k237": "FB15k-237", "codex-large": "CoDEx-Large", "wn18rr": "WN18RR", "yago": "YAGO3-10"}
# released LibKGE ComplEx checkpoints, under <checkpoint_root>/complex/
COMPLEX_CHECKPOINTS = {"fb15k237": "fb15k-237-complex.pt", "codex-large": "codex-l-complex.pt",
                       "wn18rr": "wnrr-complex.pt", "yago": "yago3-10-complex.pt"}


def add_root_args(ap):
    ap.add_argument("--data-root", default=None, help="default: data_root of configs/paths.yaml")
    ap.add_argument("--results-root", default=None, help="default: results_root of configs/paths.yaml")
    return ap


def roots(args):
    """(data_root, results_root, all roots of configs/paths.yaml)"""
    p = load_paths()
    data = Path(args.data_root) if args.data_root else p["data_root"]
    results = Path(args.results_root) if args.results_root else p["results_root"]
    return data, results, p


def dataset_dir(data_root, kg):
    return Path(data_root) / "path_datasets" / kg


def ranks_file(results_root, model, kg):
    """model: pathe, complex or popularity"""
    return Path(results_root) / "link_prediction" / "ranks" / f"{model}_{kg}.csv"


def idmaps_dir(results_root):
    return Path(results_root) / "link_prediction" / "idmaps"


def ceiling_file(results_root, kg, ext="csv"):
    return Path(results_root) / "query_level" / "ceiling" / f"ceiling_{kg}.{ext}"


def query_level_dir(results_root):
    return Path(results_root) / "query_level"
