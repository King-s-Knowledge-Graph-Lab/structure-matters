"""
Category 1: Basic Graph Statistics (M1-M6).
"""

import logging
import numpy as np
import networkx as nx

logger = logging.getLogger(__name__)


def compute(dataset, G, G_undirected, triples, num_entities, num_relations, **shared):
    """Compute M1-M6 basic graph statistics."""
    logger.info("Computing Category 1: Basic Stats (M1-M6)")

    num_train = len(triples)
    num_valid = dataset.validation.num_triples
    num_test = dataset.testing.num_triples
    num_total = num_train + num_valid + num_test

    # M4: Graph Density
    full_density = num_train / (num_entities * num_relations * num_entities)
    simple_density = nx.density(G_undirected)

    # M5 & M6: Ratios
    r_to_e_ratio = num_relations / num_entities
    t_to_e_ratio = num_train / num_entities

    return {
        "M1_num_entities": num_entities,
        "M2_num_relations": num_relations,
        "M3_num_train": num_train,
        "M3_num_valid": num_valid,
        "M3_num_test": num_test,
        "M3_num_total": num_total,
        "M4_full_density": float(full_density),
        "M4_simple_density": float(simple_density),
        "M5_relation_to_entity_ratio": float(r_to_e_ratio),
        "M6_triple_to_entity_ratio": float(t_to_e_ratio),
    }
