#!/usr/bin/env python3
"""
Main orchestrator for graph structural-property computation.

Usage:
    python compute_metrics.py [--datasets DS1,DS2,...] [--categories CAT1,CAT2,...]
                              [--workers N] [--output-dir DIR] [--skip-unit-tests]
                              [--skip-plots]

Computes all metrics (M1-M64) from the specification for each KG benchmark
dataset, with checkpointing, progress reporting, and multiprocessing.
"""

import argparse
import gc
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import performance

import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from checkpoint_manager import CheckpointManager
from graph_utils import build_graph, build_undirected_graph, compute_shared_data

from metrics import (
    basic_stats,
    degree_dist,
    relational_diversity,
    context_uniqueness,
    path_metrics,
    connectivity,
    relation_level,
    info_theoretic,
    multi_relational,
    inductive,
)
from metrics.unit_tests import run_all_tests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_LOADERS = {
    "FB15k-237": "pykeen.datasets.FB15k237",
    "WN18RR": "pykeen.datasets.WN18RR",
    "CoDEx-Large": "pykeen.datasets.CoDExLarge",
    "YAGO3-10": "pykeen.datasets.YAGO310",
}

PERFORMANCE_DATA = performance.transductive()  # configs/structure/performance.yaml

# Categories (in execution order within each dataset)
CATEGORY_ORDER = [
    "basic_stats",
    "degree_dist",
    "relational_diversity",
    "context_uniqueness",
    "info_theoretic",
    "multi_relational",
    "relation_level",
    "path_metrics",
    "connectivity",
]

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(output_dir: str):
    log_dir = Path(output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "computation.log"

    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode="a"),
        ],
    )


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(name: str):
    """Dynamically load a PyKEEN dataset by name."""
    module_path, cls_name = DATASET_LOADERS[name].rsplit(".", 1)
    import importlib
    mod = importlib.import_module(module_path)
    cls = getattr(mod, cls_name)
    return cls()


# ---------------------------------------------------------------------------
# Per-dataset pipeline
# ---------------------------------------------------------------------------

def process_dataset(
    dataset_name: str,
    ckpt: CheckpointManager,
    output_dir: str,
    num_workers: int,
    categories: list[str] | None = None,
):
    """Run all metric categories for a single dataset."""
    logger = logging.getLogger("orchestrator")
    logger.info("=" * 70)
    logger.info("Processing dataset: %s", dataset_name)
    logger.info("=" * 70)

    t0 = time.time()

    # ---- Load dataset ----
    logger.info("Loading dataset %s from PyKEEN...", dataset_name)
    dataset = load_dataset(dataset_name)
    num_entities = dataset.num_entities
    num_relations = dataset.num_relations
    logger.info(
        "  Entities=%d, Relations=%d, Train triples=%d",
        num_entities, num_relations, dataset.training.num_triples,
    )

    # ---- Build graphs ----
    logger.info("Building graphs...")
    G, triples = build_graph(dataset.training)
    G_undirected = build_undirected_graph(dataset.training)
    logger.info("  MultiDiGraph: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())

    # ---- Compute shared data ----
    logger.info("Computing shared data (degrees, contexts, lookup sets)...")
    shared = compute_shared_data(triples, num_entities, num_relations)

    # ---- Common kwargs for all category compute() calls ----
    common_kwargs = dict(
        dataset=dataset,
        G=G,
        G_undirected=G_undirected,
        triples=triples,
        num_entities=num_entities,
        num_relations=num_relations,
        output_dir=output_dir,
        dataset_name=dataset_name,
        num_workers=num_workers,
        **shared,
    )

    # ---- Run categories ----
    cats_to_run = categories if categories else CATEGORY_ORDER
    all_results = {}
    relation_typed_recip = None

    for cat_name in tqdm(cats_to_run, desc=f"Categories ({dataset_name})", unit="cat"):
        if ckpt.is_completed(dataset_name, cat_name):
            logger.info("  Skipping %s (already completed)", cat_name)
            cached = ckpt.load_results(dataset_name, cat_name)
            if cached:
                all_results.update(cached)
                # Recover relation_typed_recip if available
                if "_relation_typed_recip" in cached:
                    relation_typed_recip = cached["_relation_typed_recip"]
            continue

        ckpt.mark_in_progress(dataset_name, cat_name)
        logger.info("  Running category: %s", cat_name)
        cat_t0 = time.time()

        try:
            if cat_name == "basic_stats":
                res = basic_stats.compute(**common_kwargs)

            elif cat_name == "degree_dist":
                res = degree_dist.compute(**common_kwargs)

            elif cat_name == "relational_diversity":
                res = relational_diversity.compute(**common_kwargs)

            elif cat_name == "context_uniqueness":
                res = context_uniqueness.compute(**common_kwargs)

            elif cat_name == "info_theoretic":
                res = info_theoretic.compute(**common_kwargs)

            elif cat_name == "multi_relational":
                res = multi_relational.compute(
                    relation_typed_recip=relation_typed_recip,
                    **common_kwargs,
                )

            elif cat_name == "relation_level":
                res = relation_level.compute(**common_kwargs)
                # Stash typed recip for multi_relational
                if "_relation_typed_recip" in res:
                    relation_typed_recip = res.pop("_relation_typed_recip")

            elif cat_name == "path_metrics":
                res, _entity_paths = path_metrics.compute(
                    checkpoint_mgr=ckpt, **common_kwargs,
                )

            elif cat_name == "connectivity":
                res = connectivity.compute(**common_kwargs)

            else:
                logger.warning("Unknown category: %s", cat_name)
                continue

            elapsed = time.time() - cat_t0
            logger.info("  Completed %s in %.1fs", cat_name, elapsed)
            all_results.update(res)
            ckpt.save_results(dataset_name, cat_name, res)

        except Exception:
            logger.exception("  FAILED category %s", cat_name)
            ckpt.mark_in_progress(dataset_name, cat_name)
            raise

    # ---- Save per-entity degree CSV ----
    deg_dir = Path(output_dir) / "degree_distributions"
    deg_dir.mkdir(parents=True, exist_ok=True)
    deg_df = pd.DataFrame({
        "entity_id": range(num_entities),
        "in_degree": shared["in_degrees"],
        "out_degree": shared["out_degrees"],
        "total_degree": shared["total_degrees"],
    })
    deg_df.to_csv(deg_dir / f"{dataset_name}_degrees.csv", index=False)

    elapsed_total = time.time() - t0
    logger.info(
        "Finished dataset %s in %.1fs (%.1f min)",
        dataset_name, elapsed_total, elapsed_total / 60,
    )

    # Free large objects
    del G, G_undirected, triples, shared
    gc.collect()

    return all_results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_results(all_ds_results: dict, output_dir: str, ckpt: CheckpointManager):
    """Merge per-dataset results into summary CSV/JSON."""
    logger = logging.getLogger("orchestrator")
    logger.info("Aggregating results across all datasets...")

    rows = []
    for ds_name, metrics in all_ds_results.items():
        row = {"dataset": ds_name}
        # Add performance data
        if ds_name in PERFORMANCE_DATA:
            for k, v in PERFORMANCE_DATA[ds_name].items():
                row[f"perf_{k}"] = v
        # Add computed metrics (skip internal keys)
        for k, v in metrics.items():
            if not k.startswith("_"):
                row[k] = v
        rows.append(row)

    df = pd.DataFrame(rows)
    out = Path(output_dir)
    df.to_csv(out / "summary_metrics.csv", index=False)

    # JSON version
    json_data = {}
    for row in rows:
        ds = row.pop("dataset")
        json_data[ds] = row
    with open(out / "summary_metrics.json", "w") as f:
        json.dump(json_data, f, indent=2, default=str)

    logger.info("Summary saved to %s/summary_metrics.csv and .json", output_dir)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def generate_plots(output_dir: str, dataset_names: list[str]):
    """Generate comparative plots across datasets."""
    logger = logging.getLogger("orchestrator")
    logger.info("Generating plots...")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid")

    plot_dir = Path(output_dir) / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    colours = sns.color_palette("Set2", n_colors=len(dataset_names))
    ds_colours = dict(zip(dataset_names, colours))

    # ---- 1. Degree distributions ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ds_name in dataset_names:
        deg_file = Path(output_dir) / "degree_distributions" / f"{ds_name}_degrees.csv"
        if not deg_file.exists():
            continue
        df = pd.read_csv(deg_file)
        for ax, col, title in zip(
            axes,
            ["in_degree", "out_degree", "total_degree"],
            ["In-Degree", "Out-Degree", "Total Degree"],
        ):
            vals = df[col].values
            vals = vals[vals > 0]
            ax.hist(vals, bins=100, alpha=0.5, label=ds_name, color=ds_colours[ds_name],
                    density=True, log=True)
            ax.set_xlabel(title)
            ax.set_ylabel("Density (log)")
            ax.set_xscale("log")
            ax.legend(fontsize=8)
    fig.suptitle("Degree Distributions (log-log)")
    fig.tight_layout()
    fig.savefig(plot_dir / "degree_distributions.png", dpi=150)
    plt.close(fig)

    # ---- 2. Relation frequency ----
    fig, ax = plt.subplots(figsize=(10, 6))
    for ds_name in dataset_names:
        rel_file = Path(output_dir) / "relation_distributions" / f"{ds_name}_relation_freq.csv"
        if not rel_file.exists():
            continue
        df = pd.read_csv(rel_file)
        fracs = df["fraction"].values
        ax.plot(range(len(fracs)), fracs, label=ds_name, color=ds_colours[ds_name])
    ax.set_xlabel("Relation rank")
    ax.set_ylabel("Fraction of triples")
    ax.set_yscale("log")
    ax.set_title("Relation Frequency Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "relation_frequency.png", dpi=150)
    plt.close(fig)

    # ---- 3. Context class sizes ----
    fig, ax = plt.subplots(figsize=(10, 6))
    for ds_name in dataset_names:
        ctx_file = Path(output_dir) / "context_analysis" / f"{ds_name}_context_classes.csv"
        if not ctx_file.exists():
            continue
        df = pd.read_csv(ctx_file)
        sizes = df["num_entities"].values
        ax.plot(range(len(sizes)), sizes, label=ds_name, color=ds_colours[ds_name])
    ax.set_xlabel("Class rank")
    ax.set_ylabel("Class size")
    ax.set_yscale("log")
    ax.set_title("Relational Context Equivalence Class Sizes")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "context_class_sizes.png", dpi=150)
    plt.close(fig)

    # ---- 4. Summary bar chart of key metrics ----
    summary_file = Path(output_dir) / "summary_metrics.csv"
    if summary_file.exists():
        sdf = pd.read_csv(summary_file)
        key_metrics = [
            "M25_unique_fraction", "M17_relation_entropy",
            "M43_avg_clustering", "M45_typed_reciprocity",
            "M31_avg_paths", "M46_assortativity",
        ]
        available = [m for m in key_metrics if m in sdf.columns]
        if available:
            fig, axes = plt.subplots(1, len(available), figsize=(4 * len(available), 5))
            if len(available) == 1:
                axes = [axes]
            for ax, metric in zip(axes, available):
                vals = sdf.set_index("dataset")[metric].dropna()
                vals.plot(kind="bar", ax=ax, color=[ds_colours.get(d, "grey") for d in vals.index])
                ax.set_title(metric.replace("_", " "), fontsize=9)
                ax.set_ylabel("")
                ax.tick_params(axis="x", rotation=45, labelsize=8)
            fig.suptitle("Key Metrics Comparison")
            fig.tight_layout()
            fig.savefig(plot_dir / "key_metrics_comparison.png", dpi=150)
            plt.close(fig)

    logger.info("Plots saved to %s/", plot_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compute graph structural metrics.")
    parser.add_argument(
        "--datasets", type=str, default=None,
        help="Comma-separated dataset names (default: all four)",
    )
    parser.add_argument(
        "--categories", type=str, default=None,
        help="Comma-separated category names to run (default: all)",
    )
    parser.add_argument(
        "--workers", type=int, default=16,
        help="Number of worker processes for parallel tasks (default: 16)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(Path(__file__).resolve().parents[2] / "results" / "structure"),
        help="Output directory (default: <repo>/results/structure)",
    )
    parser.add_argument(
        "--skip-unit-tests", action="store_true",
        help="Skip unit-test validation before running",
    )
    parser.add_argument(
        "--skip-plots", action="store_true",
        help="Skip plot generation at the end",
    )
    parser.add_argument(
        "--skip-inductive", action="store_true",
        help="Skip inductive split metrics (Category 8)",
    )
    args = parser.parse_args()

    setup_logging(args.output_dir)
    logger = logging.getLogger("orchestrator")

    # Reproducibility
    random.seed(42)
    np.random.seed(42)

    # Unit tests
    if not args.skip_unit_tests:
        run_all_tests()

    # Parse dataset selection
    if args.datasets:
        dataset_names = [d.strip() for d in args.datasets.split(",")]
        for d in dataset_names:
            if d not in DATASET_LOADERS:
                logger.error("Unknown dataset: %s. Available: %s", d, list(DATASET_LOADERS.keys()))
                sys.exit(1)
    else:
        dataset_names = list(DATASET_LOADERS.keys())

    # Parse category selection
    cats = None
    if args.categories:
        cats = [c.strip() for c in args.categories.split(",")]

    ckpt = CheckpointManager(output_dir=args.output_dir)

    logger.info(
        "Starting pipeline: datasets=%s, workers=%d, output=%s",
        dataset_names, args.workers, args.output_dir,
    )

    # ---- Process each dataset sequentially (to manage memory) ----
    all_ds_results = {}

    for ds_name in tqdm(dataset_names, desc="Datasets", unit="ds"):
        results = process_dataset(
            ds_name, ckpt, args.output_dir, args.workers, categories=cats,
        )
        all_ds_results[ds_name] = results

    # ---- Inductive splits (Category 8) ----
    if not args.skip_inductive:
        if cats is None or "inductive" in cats:
            if not ckpt.is_completed("_global", "inductive"):
                ckpt.mark_in_progress("_global", "inductive")
                try:
                    ind_results = inductive.compute(output_dir=args.output_dir)
                    ckpt.save_results("_global", "inductive", ind_results)
                    logger.info("Inductive split metrics completed.")
                except Exception:
                    logger.exception("Inductive split metrics FAILED")
            else:
                logger.info("Inductive splits already completed, skipping.")

    # ---- Aggregation ----
    aggregate_results(all_ds_results, args.output_dir, ckpt)

    # ---- Plots ----
    if not args.skip_plots:
        try:
            generate_plots(args.output_dir, dataset_names)
        except Exception:
            logger.exception("Plot generation failed (non-fatal)")

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
