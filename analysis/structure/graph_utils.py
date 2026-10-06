"""
Graph construction and path-mining utilities.

Provides functions for building NetworkX graphs from PyKEEN TriplesFactories
and for mining entity-relation paths via parallel random walks.
"""

import random
import logging
import pickle
import multiprocessing as mp
from collections import defaultdict
from pathlib import Path
from functools import partial

import numpy as np
import networkx as nx
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph(triples_factory):
    """
    Build a NetworkX directed multigraph from a PyKEEN TriplesFactory.

    Returns:
        G: nx.MultiDiGraph with edge attribute 'relation' (integer ID)
        triples: numpy array of shape (num_triples, 3) with columns [head, relation, tail]
    """
    G = nx.MultiDiGraph()
    triples = triples_factory.mapped_triples.numpy().astype(np.int64)
    num_entities = triples_factory.num_entities
    G.add_nodes_from(range(num_entities))
    for h, r, t in triples:
        G.add_edge(int(h), int(t), relation=int(r))
    return G, triples


def build_undirected_graph(triples_factory):
    """Build undirected simple graph (no self-loops, no parallel edges)."""
    G_undirected = nx.Graph()
    triples = triples_factory.mapped_triples.numpy().astype(np.int64)
    num_entities = triples_factory.num_entities
    G_undirected.add_nodes_from(range(num_entities))
    for h, r, t in triples:
        G_undirected.add_edge(int(h), int(t))
    return G_undirected


# ---------------------------------------------------------------------------
# Shared data computation
# ---------------------------------------------------------------------------

def compute_shared_data(triples: np.ndarray, num_entities: int, num_relations: int) -> dict:
    """Pre-compute arrays and lookup structures shared across metric categories."""
    heads = triples[:, 0].astype(np.int64)
    rels = triples[:, 1].astype(np.int64)
    tails = triples[:, 2].astype(np.int64)

    # Degree arrays (vectorised)
    out_degrees = np.bincount(heads, minlength=num_entities).astype(np.int64)
    in_degrees = np.bincount(tails, minlength=num_entities).astype(np.int64)
    total_degrees = in_degrees + out_degrees

    relation_counts = np.bincount(rels, minlength=num_relations)

    # Edge lookup sets (still need Python sets for O(1) membership tests)
    edge_pairs = set(zip(heads.tolist(), tails.tolist()))
    typed_edges = set(zip(heads.tolist(), rels.tolist(), tails.tolist()))

    entity_relations = defaultdict(set)
    for h, r, t in zip(heads.tolist(), rels.tolist(), tails.tolist()):
        entity_relations[h].add(r)
        entity_relations[t].add(r)

    # Context matrices (vectorised with np.add.at)
    incoming_contexts = np.zeros((num_entities, num_relations), dtype=np.int32)
    outgoing_contexts = np.zeros((num_entities, num_relations), dtype=np.int32)
    np.add.at(outgoing_contexts, (heads, rels), 1)
    np.add.at(incoming_contexts, (tails, rels), 1)
    full_contexts = np.concatenate([incoming_contexts, outgoing_contexts], axis=1)

    return {
        "in_degrees": in_degrees,
        "out_degrees": out_degrees,
        "total_degrees": total_degrees,
        "relation_counts": relation_counts,
        "edge_pairs": edge_pairs,
        "typed_edges": typed_edges,
        "entity_relations": dict(entity_relations),
        "incoming_contexts": incoming_contexts,
        "outgoing_contexts": outgoing_contexts,
        "full_contexts": full_contexts,
    }


# ---------------------------------------------------------------------------
# Path mining – single entity
# ---------------------------------------------------------------------------

def _random_walk(G, start_entity, max_length, direction):
    """
    Perform a single loop-free random walk producing an entity-relation path.

    Returns list [entity, relation, entity, ...] or None if walk fails.
    """
    visited = {start_entity}
    path = [start_entity]
    current = start_entity

    for _ in range(max_length - 1):
        if direction == "outgoing":
            neighbors = []
            for _, neighbor, data in G.out_edges(current, data=True):
                if neighbor not in visited:
                    neighbors.append((neighbor, data["relation"]))
        else:
            neighbors = []
            for predecessor, _, data in G.in_edges(current, data=True):
                if predecessor not in visited:
                    neighbors.append((predecessor, data["relation"]))

        if not neighbors:
            break

        next_node, relation = random.choice(neighbors)
        path.append(int(relation))
        path.append(int(next_node))
        visited.add(next_node)
        current = next_node

    if len(path) < 3:
        return None

    if direction == "incoming":
        path = path[::-1]

    return path


def mine_paths_for_entity(G, entity, num_paths=8, max_length=20, max_attempts_multiplier=10):
    """
    Mine unique loop-free random-walk entity-relation paths for a single entity.

    Returns (paths, directions).
    """
    paths = []
    directions = []
    seen_paths = set()
    max_attempts = num_paths * max_attempts_multiplier

    for _ in range(max_attempts):
        if len(paths) >= num_paths:
            break
        direction = random.choice(["outgoing", "incoming"])
        path = _random_walk(G, entity, max_length, direction)
        if path is not None:
            path_key = tuple(path)
            if path_key not in seen_paths:
                seen_paths.add(path_key)
                paths.append(path)
                directions.append(direction)

    return paths, directions


# ---------------------------------------------------------------------------
# Path mining – parallel worker
# ---------------------------------------------------------------------------

_worker_graph = None  # module-level for fork-inherited data


def _init_worker(graph_pickle_bytes):
    """Initialise the global graph object in each worker process."""
    global _worker_graph
    _worker_graph = pickle.loads(graph_pickle_bytes)


def _worker_mine_chunk(args):
    """Worker function: mine paths for a chunk of entities."""
    entity_chunk, num_paths, max_length, seed = args
    random.seed(seed)
    results = {}
    for entity in entity_chunk:
        paths, directions = mine_paths_for_entity(
            _worker_graph, entity, num_paths=num_paths, max_length=max_length
        )
        results[entity] = (paths, directions)
    return results


def mine_paths_parallel(
    G,
    num_entities: int,
    num_paths: int = 8,
    max_length: int = 20,
    num_workers: int = 16,
    checkpoint_mgr=None,
    dataset_name: str = "",
    checkpoint_every: int = 5000,
    base_seed: int = 42,
):
    """
    Mine paths for all entities using multiprocessing.

    Supports resumption via checkpoint_mgr.
    Returns (entity_paths_dict, entity_directions_dict).
    """
    entity_paths = {}
    entity_dirs = {}

    # Try to resume from checkpoint
    if checkpoint_mgr is not None:
        loaded = checkpoint_mgr.load_paths_checkpoint(dataset_name)
        if loaded is not None:
            entity_paths, entity_dirs = loaded
            logger.info(
                "Resumed path checkpoint for %s: %d / %d entities done",
                dataset_name, len(entity_paths), num_entities,
            )

    remaining = [e for e in range(num_entities) if e not in entity_paths]
    if not remaining:
        logger.info("All entities already have paths for %s", dataset_name)
        return entity_paths, entity_dirs

    logger.info(
        "Mining paths for %d remaining entities (%s) with %d workers",
        len(remaining), dataset_name, num_workers,
    )

    # Serialise graph once for all workers
    graph_bytes = pickle.dumps(G, protocol=pickle.HIGHEST_PROTOCOL)

    chunk_size = max(1, len(remaining) // (num_workers * 4))
    chunks = [remaining[i : i + chunk_size] for i in range(0, len(remaining), chunk_size)]

    # Assign deterministic seeds per chunk
    rng = np.random.RandomState(base_seed)
    chunk_args = [
        (chunk, num_paths, max_length, int(rng.randint(0, 2**31)))
        for chunk in chunks
    ]

    processed_since_ckpt = 0
    with mp.Pool(num_workers, initializer=_init_worker, initargs=(graph_bytes,)) as pool:
        with tqdm(total=len(remaining), desc=f"  Path mining ({dataset_name})", unit="entity") as pbar:
            for result_dict in pool.imap_unordered(_worker_mine_chunk, chunk_args):
                for entity, (paths, dirs) in result_dict.items():
                    entity_paths[entity] = paths
                    entity_dirs[entity] = dirs
                pbar.update(len(result_dict))
                processed_since_ckpt += len(result_dict)

                if checkpoint_mgr is not None and processed_since_ckpt >= checkpoint_every:
                    checkpoint_mgr.save_paths_checkpoint(dataset_name, entity_paths, entity_dirs)
                    processed_since_ckpt = 0

    # Final checkpoint save
    if checkpoint_mgr is not None:
        checkpoint_mgr.save_paths_checkpoint(dataset_name, entity_paths, entity_dirs)

    return entity_paths, entity_dirs


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def extract_relation_sequence(entity_relation_path: list) -> tuple:
    """Extract relation-only sequence from [e, r, e, r, ..., e]."""
    return tuple(entity_relation_path[i] for i in range(1, len(entity_relation_path), 2))


def extract_entity_sequence(entity_relation_path: list) -> tuple:
    """Extract entity-only sequence from [e, r, e, r, ..., e]."""
    return tuple(entity_relation_path[i] for i in range(0, len(entity_relation_path), 2))


def verify_path(path: list) -> bool:
    """Verify alternating entity-relation structure."""
    assert len(path) >= 3, f"Path too short: {path}"
    assert len(path) % 2 == 1, f"Path must have odd length: len={len(path)}"
    return True
