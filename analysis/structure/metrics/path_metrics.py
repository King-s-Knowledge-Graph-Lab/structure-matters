"""
Category 5: Path Availability and Richness (M31-M36).

Path mining is the heaviest computation in the pipeline.  It is fully
parallelised via multiprocessing (see graph_utils.mine_paths_parallel).
Reachability estimation (M36) is also parallelised.
"""

import logging
import random
import multiprocessing as mp
import numpy as np
import networkx as nx
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from graph_utils import (
    mine_paths_parallel,
    extract_relation_sequence,
    extract_entity_sequence,
    verify_path,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# M36 reachability – parallel helpers
# ---------------------------------------------------------------------------

def _reachability_worker(args):
    """Estimate reachability for a subset of random pairs."""
    pairs, k_values = args
    # graph is inherited via fork from the global _reach_G
    counts = {k: 0 for k in k_values}
    for src, tgt in pairs:
        try:
            sp = nx.shortest_path_length(_reach_G, src, tgt)
            for k in k_values:
                if sp <= k:
                    counts[k] += 1
        except nx.NetworkXNoPath:
            pass
    return counts, len(pairs)


_reach_G = None

def _init_reach_worker(G_bytes):
    import pickle
    global _reach_G
    _reach_G = pickle.loads(G_bytes)


def estimate_reachability_parallel(G, num_samples=10000, k_values=None,
                                   num_workers=8, seed=42):
    """Parallel estimation of k-hop reachability."""
    import pickle
    if k_values is None:
        k_values = [2, 3, 5, 10]

    entities = list(G.nodes())
    rng = random.Random(seed)
    pairs = [tuple(rng.sample(entities, 2)) for _ in range(num_samples)]

    chunk_size = max(1, len(pairs) // num_workers)
    chunks = [pairs[i:i + chunk_size] for i in range(0, len(pairs), chunk_size)]
    chunk_args = [(chunk, k_values) for chunk in chunks]

    G_bytes = pickle.dumps(G, protocol=pickle.HIGHEST_PROTOCOL)
    totals = {k: 0 for k in k_values}
    total_pairs = 0

    with mp.Pool(num_workers, initializer=_init_reach_worker, initargs=(G_bytes,)) as pool:
        for counts, n in pool.imap_unordered(_reachability_worker, chunk_args):
            for k in k_values:
                totals[k] += counts[k]
            total_pairs += n

    return {k: totals[k] / total_pairs for k in k_values}


# ---------------------------------------------------------------------------
# Main compute function
# ---------------------------------------------------------------------------

def compute(dataset, G, G_undirected, triples, num_entities, num_relations,
            output_dir="output", dataset_name="", num_workers=16,
            checkpoint_mgr=None, **shared):
    """Compute M31-M36 path metrics."""
    logger.info("Computing Category 5: Path Metrics (M31-M36)")

    results = {}

    # ---- Mine paths (parallelised with checkpointing) ----
    entity_paths, entity_dirs = mine_paths_parallel(
        G, num_entities,
        num_paths=8, max_length=20,
        num_workers=num_workers,
        checkpoint_mgr=checkpoint_mgr,
        dataset_name=dataset_name,
        base_seed=42,
    )

    # Verify path structure (sample check on first 1000 entities)
    verified = 0
    for e in range(min(1000, num_entities)):
        for p in entity_paths.get(e, []):
            verify_path(p)
            verified += 1
    logger.info("Verified %d paths from first 1000 entities", verified)

    # ---- M31: Average Number of Unique Paths per Entity ----
    paths_found = np.array([len(entity_paths.get(e, [])) for e in range(num_entities)])
    results["M31_avg_paths"] = float(np.mean(paths_found))
    results["M31_median_paths"] = float(np.median(paths_found))
    results["M31_std_paths"] = float(np.std(paths_found))

    # ---- M32: Path Coverage ----
    for k in [1, 2, 4, 8]:
        results[f"M32_coverage_geq_{k}"] = float(np.mean(paths_found >= k))

    # ---- M33: Average Path Length ----
    all_path_lengths = []
    for e in range(num_entities):
        for p in entity_paths.get(e, []):
            num_entity_nodes = (len(p) + 1) // 2
            all_path_lengths.append(num_entity_nodes)

    if all_path_lengths:
        results["M33_avg_path_length"] = float(np.mean(all_path_lengths))
        results["M33_median_path_length"] = float(np.median(all_path_lengths))
    else:
        results["M33_avg_path_length"] = 0.0
        results["M33_median_path_length"] = 0.0

    # ---- M34: Path Diversity (distinct relation sequences) ----
    diversities = []
    for e in range(num_entities):
        paths = entity_paths.get(e, [])
        if not paths:
            continue
        rel_sequences = set()
        for p in paths:
            rel_sequences.add(extract_relation_sequence(p))
        diversities.append(len(rel_sequences))
    results["M34_avg_path_diversity"] = float(np.mean(diversities)) if diversities else 0.0

    # ---- M35: Average Distinct Relation Types in Paths per Entity ----
    rel_in_paths = []
    for e in range(num_entities):
        paths = entity_paths.get(e, [])
        if not paths:
            continue
        rel_types = set()
        for p in paths:
            rel_types.update(extract_relation_sequence(p))
        rel_in_paths.append(len(rel_types))
    results["M35_avg_rel_types_in_paths"] = float(np.mean(rel_in_paths)) if rel_in_paths else 0.0

    # ---- M36: k-hop Reachability (parallelised) ----
    logger.info("Estimating k-hop reachability (10,000 samples, parallel)...")
    reachability = estimate_reachability_parallel(
        G, num_samples=10000, k_values=[2, 3, 5, 10],
        num_workers=min(num_workers, 8), seed=42,
    )
    for k, frac in reachability.items():
        results[f"M36_reachability_{k}hop"] = float(frac)

    # ---- Save per-entity path summary ----
    out_dir = Path(output_dir) / "path_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for e in range(num_entities):
        paths = entity_paths.get(e, [])
        if paths:
            lengths = [(len(p) + 1) // 2 for p in paths]
            avg_len = float(np.mean(lengths))
        else:
            avg_len = 0.0
        rows.append({"entity_id": e, "num_paths_found": len(paths), "avg_path_length": avg_len})
    pd.DataFrame(rows).to_csv(out_dir / f"{dataset_name}_paths_per_entity.csv", index=False)

    return results, entity_paths
