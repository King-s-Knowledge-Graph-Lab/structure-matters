"""
Category 8: Inductive Split-Specific Properties (M52-M56).

Loads Teru (GrAIL) inductive splits for FB15k-237 and WN18RR (V1-V4)
via PyKEEN's ``InductiveFB15k237`` and ``InductiveWN18RR`` classes.

Each version exposes:
  - ``transductive_training`` : TriplesFactory for the training graph
  - ``inductive_inference``   : TriplesFactory for the inference graph
  - ``inductive_validation``  : TriplesFactory for inference validation triples
  - ``inductive_testing``     : TriplesFactory for inference test triples

Metrics are computed for both the training and inference graphs of each
version.
"""

import logging
import numpy as np
from collections import Counter
from pathlib import Path

import pandas as pd

from graph_utils import compute_shared_data

logger = logging.getLogger(__name__)

# PyKEEN dataset classes and their version strings
INDUCTIVE_SPECS = {
    "FB15k-237": {
        "loader": "pykeen.datasets.inductive.ilp_teru.InductiveFB15k237",
        "versions": ["v1", "v2", "v3", "v4"],
    },
    "WN18RR": {
        "loader": "pykeen.datasets.inductive.ilp_teru.InductiveWN18RR",
        "versions": ["v1", "v2", "v3", "v4"],
    },
}


def _load_inductive_dataset(loader_path: str, version: str):
    """Dynamically import and instantiate an inductive dataset."""
    module_path, cls_name = loader_path.rsplit(".", 1)
    import importlib
    mod = importlib.import_module(module_path)
    cls = getattr(mod, cls_name)
    return cls(version=version)


def _compute_split_metrics(ds, dataset_base: str, version: str) -> dict:
    """Compute M52-M56 for a single inductive split version.

    Parameters
    ----------
    ds : InductiveDataset
        PyKEEN inductive dataset instance.
    dataset_base : str
        Base dataset name (e.g. "FB15k-237").
    version : str
        Version string (e.g. "v1").
    """
    results = {}

    train_tf = ds.transductive_training
    inf_tf = ds.inductive_inference

    train_triples = train_tf.mapped_triples.numpy().astype(np.int64)
    inf_triples = inf_tf.mapped_triples.numpy().astype(np.int64)

    num_train_e = train_tf.num_entities
    num_inf_e = inf_tf.num_entities
    num_train_r = train_tf.num_relations
    num_inf_r = inf_tf.num_relations

    # ---- Entity and relation label sets for overlap computation ----
    # Use the label-to-id mappings so overlap is computed on *string labels*
    train_entity_labels = set(train_tf.entity_to_id.keys())
    inf_entity_labels = set(inf_tf.entity_to_id.keys())
    train_relation_labels = set(train_tf.relation_to_id.keys())
    inf_relation_labels = set(inf_tf.relation_to_id.keys())

    # M52: Entity Overlap
    entity_overlap = len(train_entity_labels & inf_entity_labels)
    results["M52_entity_overlap_count"] = entity_overlap
    results["M52_entity_overlap_frac_of_train"] = (
        float(entity_overlap / len(train_entity_labels)) if train_entity_labels else 0.0
    )
    results["M52_entity_overlap_frac_of_inf"] = (
        float(entity_overlap / len(inf_entity_labels)) if inf_entity_labels else 0.0
    )

    # M53: Relation Overlap
    relation_overlap = len(train_relation_labels & inf_relation_labels)
    results["M53_relation_overlap_count"] = relation_overlap
    results["M53_relation_overlap_frac_of_inf"] = (
        float(relation_overlap / len(inf_relation_labels)) if inf_relation_labels else 0.0
    )

    # M54: Unique Relations in Inference Graph
    unique_in_inf = inf_relation_labels - train_relation_labels
    results["M54_unique_relations_in_inf"] = len(unique_in_inf)

    # M55: Average Degree Comparison
    results["M55_train_num_entities"] = num_train_e
    results["M55_inf_num_entities"] = num_inf_e
    results["M55_train_num_relations"] = num_train_r
    results["M55_inf_num_relations"] = num_inf_r
    results["M55_train_num_triples"] = int(train_tf.num_triples)
    results["M55_inf_num_triples"] = int(inf_tf.num_triples)

    results["M55_train_avg_degree"] = (
        float(2 * train_tf.num_triples / num_train_e) if num_train_e > 0 else 0.0
    )
    results["M55_inf_avg_degree"] = (
        float(2 * inf_tf.num_triples / num_inf_e) if num_inf_e > 0 else 0.0
    )

    # M56: Context Uniqueness in Inference Graph
    if len(inf_triples) > 0 and num_inf_e > 0 and num_inf_r > 0:
        shared_data = compute_shared_data(inf_triples, num_inf_e, num_inf_r)
        full_contexts = shared_data["full_contexts"]
        context_tuples = [tuple(row) for row in full_contexts]
        context_counts = Counter(context_tuples)
        num_distinct = len(context_counts)
        num_unique = sum(1 for c in context_counts.values() if c == 1)
        results["M56_inf_distinct_context_fraction"] = float(num_distinct / num_inf_e)
        results["M56_inf_unique_context_fraction"] = float(num_unique / num_inf_e)
    else:
        results["M56_inf_distinct_context_fraction"] = 0.0
        results["M56_inf_unique_context_fraction"] = 0.0

    # Also compute context uniqueness for the training graph
    if len(train_triples) > 0 and num_train_e > 0 and num_train_r > 0:
        shared_train = compute_shared_data(train_triples, num_train_e, num_train_r)
        full_ctx_train = shared_train["full_contexts"]
        ctx_tuples_train = [tuple(row) for row in full_ctx_train]
        ctx_counts_train = Counter(ctx_tuples_train)
        num_distinct_t = len(ctx_counts_train)
        num_unique_t = sum(1 for c in ctx_counts_train.values() if c == 1)
        results["M56_train_distinct_context_fraction"] = float(num_distinct_t / num_train_e)
        results["M56_train_unique_context_fraction"] = float(num_unique_t / num_train_e)
    else:
        results["M56_train_distinct_context_fraction"] = 0.0
        results["M56_train_unique_context_fraction"] = 0.0

    return results


def compute(output_dir="output", **kwargs):
    """Compute M52-M56 for all inductive splits (FB15k-237 and WN18RR, V1-V4)."""
    logger.info("Computing Category 8: Inductive Splits (M52-M56)")

    all_results = {}

    for ds_base, spec in INDUCTIVE_SPECS.items():
        loader_path = spec["loader"]
        for version in spec["versions"]:
            split_name = f"{ds_base}_{version}"
            logger.info("Processing inductive split: %s", split_name)

            try:
                ds = _load_inductive_dataset(loader_path, version)
                metrics = _compute_split_metrics(ds, ds_base, version)
                all_results[split_name] = metrics
                logger.info(
                    "  %s: train=%d ent / %d triples, inf=%d ent / %d triples, "
                    "entity_overlap=%d, inf_unique_ctx=%.3f",
                    split_name,
                    metrics["M55_train_num_entities"],
                    metrics["M55_train_num_triples"],
                    metrics["M55_inf_num_entities"],
                    metrics["M55_inf_num_triples"],
                    metrics["M52_entity_overlap_count"],
                    metrics["M56_inf_unique_context_fraction"],
                )
            except Exception as e:
                logger.warning("Failed to compute metrics for %s: %s", split_name, e)
                all_results[split_name] = {"error": str(e)}

    # Save per-split CSV
    out_dir = Path(output_dir) / "inductive_splits"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for split_name, metrics in all_results.items():
        row = {"split": split_name}
        row.update(metrics)
        rows.append(row)
    if rows:
        pd.DataFrame(rows).to_csv(out_dir / "inductive_metrics.csv", index=False)

    logger.info("Inductive split metrics saved to %s/", out_dir)
    return all_results
