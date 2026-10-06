"""
Category 4: Relational Context Uniqueness (M24-M30).
"""

import logging
import hashlib
import numpy as np
import pandas as pd
from collections import Counter
from scipy.stats import entropy as scipy_entropy
from pathlib import Path

logger = logging.getLogger(__name__)


def compute(dataset, G, G_undirected, triples, num_entities, num_relations,
            output_dir="output", dataset_name="", **shared):
    """Compute M24-M30 relational context uniqueness metrics."""
    logger.info("Computing Category 4: Context Uniqueness (M24-M30)")

    full_contexts = shared["full_contexts"]
    results = {}

    # M25: Proportion of Entities with Unique Relational Contexts
    context_tuples = [tuple(row) for row in full_contexts]
    context_counts = Counter(context_tuples)

    num_distinct_contexts = len(context_counts)
    distinct_fraction = num_distinct_contexts / num_entities

    num_unique = sum(1 for count in context_counts.values() if count == 1)
    unique_fraction = num_unique / num_entities

    results["M25_distinct_fraction"] = float(distinct_fraction)
    results["M25_unique_fraction"] = float(unique_fraction)

    # M26: Context Compression Ratio (same as distinct_fraction)
    results["M26_compression_ratio"] = float(distinct_fraction)

    # M27: Largest Equivalence Class Size
    largest_class = max(context_counts.values())
    results["M27_largest_class_size"] = int(largest_class)

    # M28: Average Equivalence Class Size
    avg_class_size = num_entities / num_distinct_contexts
    results["M28_avg_class_size"] = float(avg_class_size)

    # M29: Entropy of Equivalence Class Size Distribution
    class_sizes = np.array(list(context_counts.values()))
    class_probs = class_sizes / class_sizes.sum()
    class_entropy = float(scipy_entropy(class_probs, base=2))
    results["M29_class_entropy"] = class_entropy

    # M30: Proportion of Entities in the Largest Equivalence Class
    results["M30_largest_class_fraction"] = float(largest_class / num_entities)

    # Save context class details
    out_dir = Path(output_dir) / "context_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build mapping: context -> list of entity ids
    context_to_entities = {}
    for entity_id, ctx in enumerate(context_tuples):
        ctx_hash = hashlib.md5(str(ctx).encode()).hexdigest()[:16]
        if ctx_hash not in context_to_entities:
            context_to_entities[ctx_hash] = []
        context_to_entities[ctx_hash].append(entity_id)

    rows = []
    for ctx_hash, entities in context_to_entities.items():
        examples = entities[:5]
        rows.append({
            "context_hash": ctx_hash,
            "num_entities": len(entities),
            "example_entity_ids": str(examples),
        })
    ctx_df = pd.DataFrame(rows).sort_values("num_entities", ascending=False).reset_index(drop=True)
    ctx_df.to_csv(out_dir / f"{dataset_name}_context_classes.csv", index=False)

    return results
