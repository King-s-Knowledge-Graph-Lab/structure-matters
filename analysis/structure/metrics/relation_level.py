"""
Category 7: Relation-Level Graph Properties (M47-M51).
"""

import logging
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.stats import entropy as scipy_entropy
from pathlib import Path

logger = logging.getLogger(__name__)


def compute(dataset, G, G_undirected, triples, num_entities, num_relations,
            output_dir="output", dataset_name="", **shared):
    """Compute M47-M51 relation-level metrics."""
    logger.info("Computing Category 7: Relation-Level (M47-M51)")

    total_degrees = shared["total_degrees"]
    edge_pairs = shared["edge_pairs"]
    typed_edges = shared["typed_edges"]
    results = {}

    # Pre-compute per-relation entity sets and triples
    relation_entities = defaultdict(set)
    relation_triples = defaultdict(list)
    for h, r, t in triples:
        h, r, t = int(h), int(r), int(t)
        relation_entities[r].add(h)
        relation_entities[r].add(t)
        relation_triples[r].append((h, t))

    # M47: Relation-Specific Average Degree
    relation_avg_degrees = []
    for r in range(num_relations):
        ents = relation_entities.get(r, set())
        if ents:
            degrees_r = [total_degrees[e] for e in ents]
            relation_avg_degrees.append(float(np.mean(degrees_r)))
        else:
            relation_avg_degrees.append(0.0)
    results["M47_mean_rel_avg_degree"] = float(np.mean(relation_avg_degrees))
    results["M47_std_rel_avg_degree"] = float(np.std(relation_avg_degrees))

    # M48: Relation Co-occurrence Matrix
    logger.info("Computing relation co-occurrence matrix (%d x %d)...", num_relations, num_relations)
    # Vectorised via set intersection
    rel_ent_list = [relation_entities.get(r, set()) for r in range(num_relations)]
    cooccurrence = np.zeros((num_relations, num_relations), dtype=np.int32)
    for r1 in range(num_relations):
        for r2 in range(r1, num_relations):
            overlap = len(rel_ent_list[r1] & rel_ent_list[r2])
            cooccurrence[r1, r2] = overlap
            cooccurrence[r2, r1] = overlap

    # Save co-occurrence matrix
    out_dir = Path(output_dir) / "relation_cooccurrence"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{dataset_name}_cooccurrence.npy", cooccurrence)

    # Summary stats
    off_diag = cooccurrence[np.triu_indices(num_relations, k=1)]
    results["M48_avg_offdiag_cooccurrence"] = float(np.mean(off_diag)) if len(off_diag) > 0 else 0.0
    results["M48_frac_nonzero_offdiag"] = float(np.mean(off_diag > 0)) if len(off_diag) > 0 else 0.0

    # M49: Relation Co-occurrence Entropy
    upper_tri = cooccurrence[np.triu_indices(num_relations, k=1)]
    upper_tri_nonzero = upper_tri[upper_tri > 0]
    if len(upper_tri_nonzero) > 0:
        probs = upper_tri_nonzero / upper_tri_nonzero.sum()
        cooccurrence_entropy = float(scipy_entropy(probs, base=2))
    else:
        cooccurrence_entropy = 0.0
    results["M49_cooccurrence_entropy"] = cooccurrence_entropy

    # M50: Relation Graph Density
    num_cooccurring = int(np.sum(upper_tri > 0))
    max_pairs = num_relations * (num_relations - 1) / 2
    results["M50_relation_graph_density"] = float(num_cooccurring / max_pairs) if max_pairs > 0 else 0.0

    # M51: Relation-Specific Reciprocity
    relation_recip = {}
    relation_typed_recip = {}
    for r in range(num_relations):
        r_triples = relation_triples.get(r, [])
        if not r_triples:
            relation_recip[r] = 0.0
            relation_typed_recip[r] = 0.0
            continue
        recip_count = sum(1 for h, t in r_triples if (t, h) in edge_pairs)
        typed_count = sum(1 for h, t in r_triples if (t, r, h) in typed_edges)
        relation_recip[r] = recip_count / len(r_triples)
        relation_typed_recip[r] = typed_count / len(r_triples)

    results["M51_mean_rel_recip"] = float(np.mean(list(relation_recip.values())))
    results["M51_std_rel_recip"] = float(np.std(list(relation_recip.values())))
    results["M51_mean_rel_typed_recip"] = float(np.mean(list(relation_typed_recip.values())))

    # Save per-relation data
    out_dir2 = Path(output_dir) / "relation_distributions"
    out_dir2.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in range(num_relations):
        rows.append({
            "relation_id": r,
            "avg_degree": relation_avg_degrees[r],
            "untyped_reciprocity": relation_recip.get(r, 0.0),
            "typed_reciprocity": relation_typed_recip.get(r, 0.0),
        })
    pd.DataFrame(rows).to_csv(out_dir2 / f"{dataset_name}_relation_details.csv", index=False)

    # Stash relation_typed_recip for Category 10 (M63)
    results["_relation_typed_recip"] = relation_typed_recip

    return results
