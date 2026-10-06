"""Reported PathE results, read from configs/structure/performance.yaml (the one place they are kept)."""
from pathlib import Path

import yaml

DEFAULT = Path(__file__).resolve().parents[2] / "configs" / "structure" / "performance.yaml"


def load_performance(path=DEFAULT) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def transductive(path=DEFAULT) -> dict:
    """{KG: {metric: value}}, e.g. {"FB15k-237": {"LP_MRR": 0.216, ...}, ...}"""
    return load_performance(path)["transductive"]


def inductive_h10(path=DEFAULT) -> dict:
    """{split: Hits@10} with split names as "FB15k-237_V1"."""
    return load_performance(path)["inductive_H10"]
