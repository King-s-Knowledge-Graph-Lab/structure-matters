"""
Unit tests on small synthetic graphs with known properties.

Validates hand-rolled computations (Gini, reciprocity, context uniqueness,
path structure) before running the main pipeline.
"""

import numpy as np
import networkx as nx
from scipy.stats import entropy as scipy_entropy

from graph_utils import (
    build_graph,
    mine_paths_for_entity,
    verify_path,
    extract_relation_sequence,
    extract_entity_sequence,
)


def _make_triples_factory_stub(triples_array, num_entities, num_relations):
    """Lightweight stub mimicking TriplesFactory for unit-test graph building."""
    import torch

    class _Stub:
        def __init__(self, triples, n_ent, n_rel):
            self.mapped_triples = torch.tensor(triples, dtype=torch.long)
            self.num_entities = n_ent
            self.num_relations = n_rel
            self.num_triples = len(triples)

    return _Stub(triples_array, num_entities, num_relations)


# ---------------------------------------------------------------------------
# Gini coefficient
# ---------------------------------------------------------------------------

def _gini_coefficient(values):
    values = np.array(values, dtype=np.float64)
    if np.all(values == 0):
        return 0.0
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    index = np.arange(1, n + 1)
    return float((2.0 * np.sum(index * sorted_vals)) / (n * np.sum(sorted_vals)) - (n + 1.0) / n)


def _gini_mad(values):
    values = np.array(values, dtype=np.float64)
    if values.mean() == 0:
        return 0.0
    n = len(values)
    return float(np.sum(np.abs(values[:, None] - values[None, :])) / (2 * n * n * values.mean()))


def test_gini():
    uniform = [1, 1, 1, 1]
    assert abs(_gini_coefficient(uniform)) < 1e-10, "Gini of uniform must be 0"
    assert abs(_gini_mad(uniform)) < 1e-10

    skewed = [0, 0, 0, 4]
    g = _gini_coefficient(skewed)
    assert abs(g - 0.75) < 1e-10, f"Gini of [0,0,0,4] should be 0.75, got {g}"
    assert abs(g - _gini_mad(skewed)) < 1e-10, "Gini cross-check failed"

    moderate = [1, 2, 3, 4]
    g1 = _gini_coefficient(moderate)
    g2 = _gini_mad(moderate)
    assert abs(g1 - g2) < 1e-10, f"Gini cross-check failed: {g1} vs {g2}"
    assert 0 < g1 < 1, f"Expected Gini in (0,1), got {g1}"

    print("  [PASS] Gini coefficient tests")


# ---------------------------------------------------------------------------
# Reciprocity
# ---------------------------------------------------------------------------

def test_reciprocity():
    # Symmetric graph: every (h, r, t) has (t, r, h)
    triples = np.array([
        [0, 0, 1], [1, 0, 0],
        [1, 0, 2], [2, 0, 1],
        [0, 0, 2], [2, 0, 0],
    ])
    typed_edges = set(map(tuple, triples.tolist()))
    typed_reciprocal = sum(1 for h, r, t in triples if (t, r, h) in typed_edges)
    typed_reciprocity = typed_reciprocal / len(triples)
    assert abs(typed_reciprocity - 1.0) < 1e-10, \
        f"Symmetric graph typed reciprocity should be 1.0, got {typed_reciprocity}"

    # Asymmetric graph
    triples2 = np.array([[0, 0, 1], [1, 0, 2]])
    typed_edges2 = set(map(tuple, triples2.tolist()))
    typed_r2 = sum(1 for h, r, t in triples2 if (t, r, h) in typed_edges2) / len(triples2)
    assert abs(typed_r2 - 0.0) < 1e-10, f"Asymmetric typed reciprocity should be 0, got {typed_r2}"

    print("  [PASS] Reciprocity tests")


# ---------------------------------------------------------------------------
# Context uniqueness
# ---------------------------------------------------------------------------

def test_context_uniqueness():
    # Graph where every entity has a distinct neighbourhood
    # 0 --r0--> 1, 1 --r1--> 2  (chain: each has unique context)
    num_entities, num_relations = 3, 2
    triples = np.array([[0, 0, 1], [1, 1, 2]])

    incoming = np.zeros((num_entities, num_relations), dtype=np.int32)
    outgoing = np.zeros((num_entities, num_relations), dtype=np.int32)
    for h, r, t in triples:
        outgoing[h, r] += 1
        incoming[t, r] += 1
    full = np.concatenate([incoming, outgoing], axis=1)

    from collections import Counter
    context_tuples = [tuple(row) for row in full]
    context_counts = Counter(context_tuples)
    unique_fraction = sum(1 for c in context_counts.values() if c == 1) / num_entities
    assert abs(unique_fraction - 1.0) < 1e-10, \
        f"All contexts distinct -> unique_fraction=1.0, got {unique_fraction}"

    # Two entities with identical context
    # 0 --r0--> 2, 1 --r0--> 2 (entities 0 and 1 both have out-r0 * 1, everything else 0)
    triples2 = np.array([[0, 0, 2], [1, 0, 2]])
    num_entities2, num_relations2 = 3, 1
    incoming2 = np.zeros((num_entities2, num_relations2), dtype=np.int32)
    outgoing2 = np.zeros((num_entities2, num_relations2), dtype=np.int32)
    for h, r, t in triples2:
        outgoing2[h, r] += 1
        incoming2[t, r] += 1
    full2 = np.concatenate([incoming2, outgoing2], axis=1)
    context_tuples2 = [tuple(row) for row in full2]
    context_counts2 = Counter(context_tuples2)
    unique_fraction2 = sum(1 for c in context_counts2.values() if c == 1) / num_entities2
    # Entity 2 has unique context (in-r0=2), entities 0 and 1 share context -> unique = 1/3
    assert abs(unique_fraction2 - 1.0 / 3.0) < 1e-10, \
        f"Expected 1/3 unique, got {unique_fraction2}"

    print("  [PASS] Context uniqueness tests")


# ---------------------------------------------------------------------------
# Path structure
# ---------------------------------------------------------------------------

def test_path_structure():
    # Small graph: 0 --r0--> 1 --r1--> 2 --r0--> 3
    stub = _make_triples_factory_stub(
        [[0, 0, 1], [1, 1, 2], [2, 0, 3]], num_entities=4, num_relations=2
    )
    G, _ = build_graph(stub)

    import random as _rng
    _rng.seed(42)
    paths, directions = mine_paths_for_entity(G, entity=0, num_paths=4, max_length=20)

    for p in paths:
        verify_path(p)
        rel_seq = extract_relation_sequence(p)
        ent_seq = extract_entity_sequence(p)
        assert len(rel_seq) == (len(p) - 1) // 2
        assert len(ent_seq) == (len(p) + 1) // 2

    print("  [PASS] Path structure tests")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_all_tests():
    """Run all unit tests. Raises on failure."""
    print("Running unit tests on synthetic graphs...")
    test_gini()
    test_reciprocity()
    test_context_uniqueness()
    test_path_structure()
    print("All unit tests passed.\n")


if __name__ == "__main__":
    run_all_tests()
