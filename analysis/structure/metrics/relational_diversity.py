"""
Category 3: Relational Diversity and Distribution (M16-M23).
"""

import logging
import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy
from pathlib import Path

logger = logging.getLogger(__name__)


def gini_coefficient(values):
    """Compute Gini coefficient matching [Gini1912]."""
    values = np.array(values, dtype=np.float64)
    if np.all(values == 0):
        return 0.0
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    index = np.arange(1, n + 1)
    return float((2.0 * np.sum(index * sorted_vals)) / (n * np.sum(sorted_vals)) - (n + 1.0) / n)


def _gini_mad(values):
    """Mean-absolute-difference formulation for cross-check."""
    values = np.array(values, dtype=np.float64)
    if values.mean() == 0:
        return 0.0
    n = len(values)
    return float(np.sum(np.abs(values[:, None] - values[None, :])) / (2 * n * n * values.mean()))


def compute(dataset, G, G_undirected, triples, num_entities, num_relations,
            output_dir="output", dataset_name="", **shared):
    """Compute M16-M23 relational diversity metrics."""
    logger.info("Computing Category 3: Relational Diversity (M16-M23)")

    relation_counts = shared["relation_counts"]
    entity_relations = shared["entity_relations"]
    results = {}

    # M16: Relation Frequency Distribution
    rel_freq_df = pd.DataFrame({
        "relation_id": range(num_relations),
        "count": relation_counts,
        "fraction": relation_counts / len(triples),
    }).sort_values("count", ascending=False).reset_index(drop=True)

    out_dir = Path(output_dir) / "relation_distributions"
    out_dir.mkdir(parents=True, exist_ok=True)
    rel_freq_df.to_csv(out_dir / f"{dataset_name}_relation_freq.csv", index=False)

    # M17: Relation Entropy
    relation_probs = relation_counts / relation_counts.sum()
    relation_entropy = float(scipy_entropy(relation_probs, base=2))
    results["M17_relation_entropy"] = relation_entropy

    # M18: Normalised Relation Entropy
    max_entropy = np.log2(num_relations) if num_relations > 0 else 0
    results["M18_normalised_relation_entropy"] = float(
        relation_entropy / max_entropy if max_entropy > 0 else 0
    )

    # M19: Relation Gini Coefficient
    relation_gini = gini_coefficient(relation_counts)
    relation_gini_check = _gini_mad(relation_counts)
    assert abs(relation_gini - relation_gini_check) < 1e-8, \
        f"Gini cross-check failed: {relation_gini} vs {relation_gini_check}"
    results["M19_relation_gini"] = relation_gini

    # M20: Top-k Relation Concentration
    sorted_fractions = np.sort(relation_counts / relation_counts.sum())[::-1]
    cumulative = np.cumsum(sorted_fractions)
    for k in [1, 2, 3, 5, 10]:
        results[f"M20_top{k}_concentration"] = float(
            cumulative[min(k, len(cumulative)) - 1]
        )

    # M21: Number of Rare Relations
    total_triples = len(triples)
    for threshold_pct in [0.001, 0.005, 0.01]:
        threshold = total_triples * threshold_pct
        label = f"{threshold_pct*100:.1f}pct"
        results[f"M21_rare_relations_{label}"] = int(np.sum(relation_counts < threshold))

    # M22: Relation Type Diversity per Entity
    rel_diversity = np.array([
        len(entity_relations.get(e, set())) for e in range(num_entities)
    ])
    results["M22_avg_rel_diversity"] = float(np.mean(rel_diversity))
    results["M22_median_rel_diversity"] = float(np.median(rel_diversity))
    results["M22_std_rel_diversity"] = float(np.std(rel_diversity))
    results["M22_min_rel_diversity"] = int(np.min(rel_diversity))
    results["M22_max_rel_diversity"] = int(np.max(rel_diversity))

    # M23: Normalised Relation Type Diversity per Entity
    if num_relations > 0:
        normalised_diversity = rel_diversity / num_relations
        results["M23_avg_normalised_diversity"] = float(np.mean(normalised_diversity))
    else:
        results["M23_avg_normalised_diversity"] = 0.0

    return results
