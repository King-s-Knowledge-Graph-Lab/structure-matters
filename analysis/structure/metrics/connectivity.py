"""
Category 6: Connectivity and Global Structure (M37-M46).
"""

import logging
import random
import pickle
import multiprocessing as mp
import numpy as np
import networkx as nx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# M41/M42 – parallel sampling helpers
# ---------------------------------------------------------------------------

_conn_G = None

def _init_conn_worker(G_bytes):
    global _conn_G
    _conn_G = pickle.loads(G_bytes)


def _diameter_worker(args):
    """BFS from a single source, return max distance."""
    source = args
    lengths = dict(nx.single_source_shortest_path_length(_conn_G, source))
    return max(lengths.values()) if lengths else 0


def _sp_worker(args):
    """Compute shortest path for a list of (src, tgt) pairs."""
    pairs = args
    dists = []
    for src, tgt in pairs:
        try:
            d = nx.shortest_path_length(_conn_G, src, tgt)
            dists.append(d)
        except nx.NetworkXNoPath:
            pass
    return dists


def _approximate_diameter_parallel(G_sub, num_samples=200, num_workers=8):
    """Lower bound on diameter via sampled BFS, parallelised."""
    nodes = list(G_sub.nodes())
    rng = random.Random(42)
    sources = [rng.choice(nodes) for _ in range(num_samples)]

    G_bytes = pickle.dumps(G_sub, protocol=pickle.HIGHEST_PROTOCOL)
    max_dist = 0

    with mp.Pool(num_workers, initializer=_init_conn_worker, initargs=(G_bytes,)) as pool:
        for d in pool.imap_unordered(_diameter_worker, sources):
            if d > max_dist:
                max_dist = d
    return max_dist


def _avg_sp_parallel(G_sub, num_samples=10000, num_workers=8):
    """Parallelised sampled average shortest path."""
    nodes = list(G_sub.nodes())
    rng = random.Random(42)
    pairs = [tuple(rng.sample(nodes, 2)) for _ in range(num_samples)]

    chunk_size = max(1, len(pairs) // num_workers)
    chunks = [pairs[i:i + chunk_size] for i in range(0, len(pairs), chunk_size)]

    G_bytes = pickle.dumps(G_sub, protocol=pickle.HIGHEST_PROTOCOL)
    all_dists = []

    with mp.Pool(num_workers, initializer=_init_conn_worker, initargs=(G_bytes,)) as pool:
        for dists in pool.imap_unordered(_sp_worker, chunks):
            all_dists.extend(dists)

    if all_dists:
        return float(np.mean(all_dists)), float(np.std(all_dists))
    return 0.0, 0.0


# ---------------------------------------------------------------------------
# Main compute function
# ---------------------------------------------------------------------------

def compute(dataset, G, G_undirected, triples, num_entities, num_relations,
            num_workers=8, **shared):
    """Compute M37-M46 connectivity and global structure metrics."""
    logger.info("Computing Category 6: Connectivity (M37-M46)")

    results = {}
    edge_pairs = shared["edge_pairs"]
    typed_edges = shared["typed_edges"]

    # M37: Connected Components (Undirected)
    num_components = nx.number_connected_components(G_undirected)
    results["M37_num_connected_components"] = num_components

    # M38: Largest Connected Component
    largest_cc = max(nx.connected_components(G_undirected), key=len)
    lcc_size = len(largest_cc)
    results["M38_lcc_size"] = lcc_size
    results["M38_lcc_fraction"] = float(lcc_size / num_entities)

    # M39: Strongly Connected Components
    num_scc = nx.number_strongly_connected_components(G)
    results["M39_num_scc"] = num_scc

    # M40: Largest SCC
    largest_scc = max(nx.strongly_connected_components(G), key=len)
    lscc_size = len(largest_scc)
    results["M40_lscc_size"] = lscc_size
    results["M40_lscc_fraction"] = float(lscc_size / num_entities)

    # M41: Diameter (always approximate -- exact is O(V*(V+E)), too slow)
    lcc_subgraph = G_undirected.subgraph(largest_cc)
    logger.info("Approximating diameter via sampled BFS (200 sources, parallel)...")
    diameter = _approximate_diameter_parallel(
        lcc_subgraph, num_samples=200, num_workers=num_workers,
    )
    results["M41_diameter"] = diameter
    results["M41_diameter_exact"] = False

    # M42: Average Shortest Path Length (sampled, parallel)
    logger.info("Estimating average shortest path length (10,000 samples)...")
    avg_sp, std_sp = _avg_sp_parallel(
        lcc_subgraph, num_samples=10000, num_workers=num_workers,
    )
    results["M42_avg_shortest_path"] = avg_sp
    results["M42_std_shortest_path"] = std_sp

    # M43: Average Clustering Coefficient
    # For large graphs, use sampling to avoid O(V * d^2) cost
    if num_entities > 20000:
        logger.info("Estimating average clustering coefficient (5000 samples)...")
        rng_nodes = list(G_undirected.nodes())
        import random as _rng
        _rng.seed(42)
        sample_nodes = _rng.sample(rng_nodes, min(5000, len(rng_nodes)))
        avg_clustering = nx.average_clustering(G_undirected, nodes=sample_nodes)
    else:
        logger.info("Computing average clustering coefficient (exact)...")
        avg_clustering = nx.average_clustering(G_undirected)
    results["M43_avg_clustering"] = float(avg_clustering)

    # M44: Global Clustering Coefficient (Transitivity)
    transitivity = nx.transitivity(G_undirected)
    results["M44_transitivity"] = float(transitivity)

    # M45: Reciprocity
    reciprocal_count = sum(1 for h, r, t in triples if (int(t), int(h)) in edge_pairs)
    untyped_reciprocity = reciprocal_count / len(triples)
    results["M45_untyped_reciprocity"] = float(untyped_reciprocity)

    typed_reciprocal = sum(1 for h, r, t in triples if (int(t), int(r), int(h)) in typed_edges)
    typed_reciprocity = typed_reciprocal / len(triples)
    results["M45_typed_reciprocity"] = float(typed_reciprocity)

    # Cross-check with NetworkX on simple directed graph
    G_simple_directed = nx.DiGraph()
    for h, r, t in triples:
        G_simple_directed.add_edge(int(h), int(t))
    try:
        nx_reciprocity = nx.overall_reciprocity(G_simple_directed)
    except Exception:
        nx_reciprocity = None
    results["M45_nx_reciprocity_crosscheck"] = float(nx_reciprocity) if nx_reciprocity is not None else None

    # M46: Degree Assortativity
    assortativity = nx.degree_assortativity_coefficient(G_undirected)
    results["M46_assortativity"] = float(assortativity)

    return results
