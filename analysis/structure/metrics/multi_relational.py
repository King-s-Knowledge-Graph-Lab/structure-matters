"""
Category 10: Multi-Relational Graph-Specific Properties (M61-M64).
"""

import logging
import random
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


def compute(dataset, G, G_undirected, triples, num_entities, num_relations,
            relation_typed_recip=None, **shared):
    """Compute M61-M64 multi-relational metrics."""
    logger.info("Computing Category 10: Multi-Relational (M61-M64)")

    relation_counts = shared["relation_counts"]
    results = {}

    # M61: Relation Multiplicity
    pair_relations = defaultdict(set)
    for h, r, t in triples:
        pair_relations[(int(h), int(t))].add(int(r))

    multiplicities = [len(rels) for rels in pair_relations.values()]
    results["M61_avg_multiplicity"] = float(np.mean(multiplicities))
    results["M61_max_multiplicity"] = int(np.max(multiplicities))
    results["M61_fraction_multi"] = float(np.mean(np.array(multiplicities) > 1))

    # M62: Meta-Path Diversity (length 2 – exact)
    incoming_rels = defaultdict(set)
    outgoing_rels = defaultdict(set)
    for h, r, t in triples:
        outgoing_rels[int(h)].add(int(r))
        incoming_rels[int(t)].add(int(r))

    metapaths_2 = set()
    for m in range(num_entities):
        in_r = incoming_rels.get(m, set())
        out_r = outgoing_rels.get(m, set())
        for r1 in in_r:
            for r2 in out_r:
                metapaths_2.add((r1, r2))

    num_metapaths_2 = len(metapaths_2)
    results["M62_num_metapaths_2"] = num_metapaths_2
    results["M62_metapath_2_ratio"] = float(num_metapaths_2 / (num_relations ** 2)) if num_relations > 0 else 0.0

    # M62: Meta-Path Diversity (length 3 – sampled)
    logger.info("Sampling length-3 meta-paths (100,000 walks)...")
    rng = random.Random(42)
    metapaths_3 = set()
    triples_list = triples.tolist()
    num_triples = len(triples_list)

    for _ in range(100_000):
        idx = rng.randint(0, num_triples - 1)
        h1, r1, t1 = int(triples_list[idx][0]), int(triples_list[idx][1]), int(triples_list[idx][2])
        out_edges = list(G.out_edges(t1, data=True))
        if not out_edges:
            continue
        _, t2, data2 = rng.choice(out_edges)
        r2 = data2["relation"]
        out_edges2 = list(G.out_edges(t2, data=True))
        if not out_edges2:
            continue
        _, t3, data3 = rng.choice(out_edges2)
        r3 = data3["relation"]
        metapaths_3.add((r1, r2, r3))

    num_metapaths_3 = len(metapaths_3)
    results["M62_num_metapaths_3"] = num_metapaths_3
    results["M62_metapath_3_ratio"] = float(num_metapaths_3 / (num_relations ** 3)) if num_relations > 0 else 0.0

    # M63: Relation Symmetry Score
    if relation_typed_recip is not None:
        symmetry_scores = list(relation_typed_recip.values())
    else:
        # Fallback: compute from scratch
        typed_edges_set = shared.get("typed_edges", set())
        symmetry_scores = []
        rel_triple_map = defaultdict(list)
        for h, r, t in triples:
            rel_triple_map[int(r)].append((int(h), int(t)))
        for r in range(num_relations):
            r_triples = rel_triple_map.get(r, [])
            if not r_triples:
                symmetry_scores.append(0.0)
                continue
            typed_count = sum(1 for h, t in r_triples if (t, r, h) in typed_edges_set)
            symmetry_scores.append(typed_count / len(r_triples))

    avg_symmetry = float(np.mean(symmetry_scores))
    weighted_symmetry = 0.0
    for r in range(num_relations):
        p_r = relation_counts[r] / len(triples)
        s_r = symmetry_scores[r] if r < len(symmetry_scores) else 0.0
        weighted_symmetry += p_r * s_r

    results["M63_avg_symmetry"] = avg_symmetry
    results["M63_weighted_symmetry"] = float(weighted_symmetry)

    # M64: Relation Composition Patterns (same as M62 length-2 count)
    results["M64_num_composition_patterns"] = num_metapaths_2

    return results
