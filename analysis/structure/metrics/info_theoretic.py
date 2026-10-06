"""
Category 9: Information-Theoretic Properties (M57-M60).

Optimised with numpy-based groupby for speed.
"""

import logging
import numpy as np
from scipy.stats import entropy as scipy_entropy

logger = logging.getLogger(__name__)


def _conditional_entropy_fast(group_ids: np.ndarray, target_ids: np.ndarray,
                              total: int) -> float:
    """
    Compute H(target | group) = sum_g p(g) * H(target | group=g).

    Uses lexsort + diff-based grouping.  Groups with only 1 distinct target
    (entropy=0) are skipped entirely for speed — this is the common case
    for M60 where most (h, t) pairs have only one relation.
    """
    # Sort by (group, target)
    order = np.lexsort((target_ids, group_ids))
    g_sorted = group_ids[order]
    t_sorted = target_ids[order]

    # Boundaries where (group, target) changes
    diff_mask = np.empty(len(g_sorted), dtype=bool)
    diff_mask[0] = True
    diff_mask[1:] = (g_sorted[1:] != g_sorted[:-1]) | (t_sorted[1:] != t_sorted[:-1])
    boundaries = np.nonzero(diff_mask)[0]
    boundaries = np.append(boundaries, len(g_sorted))

    # Counts per (group, target) pair
    pair_counts = np.diff(boundaries)
    pair_groups = g_sorted[boundaries[:-1]]

    # Group-level boundaries
    g_diff_mask = np.empty(len(pair_groups), dtype=bool)
    g_diff_mask[0] = True
    g_diff_mask[1:] = pair_groups[1:] != pair_groups[:-1]
    g_boundaries = np.nonzero(g_diff_mask)[0]
    g_boundaries = np.append(g_boundaries, len(pair_counts))

    # Number of distinct targets per group
    group_sizes = np.diff(g_boundaries)

    # For groups with exactly 1 distinct target, entropy = 0 -> skip
    multi_mask = group_sizes > 1
    multi_indices = np.nonzero(multi_mask)[0]

    H = 0.0
    for idx in multi_indices:
        start = g_boundaries[idx]
        end = g_boundaries[idx + 1]
        counts = pair_counts[start:end]
        g_total = counts.sum()
        probs = counts / g_total
        H += (g_total / total) * float(scipy_entropy(probs, base=2))

    return H


def compute(dataset, G, G_undirected, triples, num_entities, num_relations, **shared):
    """Compute M57-M60 information-theoretic metrics."""
    logger.info("Computing Category 9: Info-Theoretic (M57-M60)")

    total_degrees = shared["total_degrees"]
    results = {}
    total = len(triples)

    heads = triples[:, 0].astype(np.int64)
    rels = triples[:, 1].astype(np.int64)
    tails = triples[:, 2].astype(np.int64)

    # M57: Degree Distribution Entropy
    degree_counts = np.bincount(total_degrees.astype(int))
    degree_probs = degree_counts[degree_counts > 0] / num_entities
    results["M57_degree_entropy"] = float(scipy_entropy(degree_probs, base=2))

    # M58: Conditional Entropy H(T|R)
    results["M58_H_tail_given_rel"] = float(_conditional_entropy_fast(rels, tails, total))

    # M59: Conditional Entropy H(H|R)
    results["M59_H_head_given_rel"] = float(_conditional_entropy_fast(rels, heads, total))

    # M60: Conditional Entropy H(R|H,T)
    # Encode (head, tail) as a single int64 key
    pair_keys = heads * (np.int64(num_entities) + 1) + tails
    results["M60_H_rel_given_head_tail"] = float(
        _conditional_entropy_fast(pair_keys, rels, total)
    )

    return results
