#!/usr/bin/env python3
"""
Correlation analysis: linking graph structural properties to model performance.

Implements all 7 parts of the correlation analysis specification for Section 7
of the TGDK paper. Reads pre-computed structural metrics from
output/summary_metrics.csv and output/inductive_splits/inductive_metrics.csv,
then produces tables, plots, and structured outputs for paper drafting.

Usage:
    python correlation_analysis.py [--output-dir DIR]
"""

import argparse
import json
import logging
import os
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import (
    kendalltau,
    linregress,
    pearsonr,
    spearmanr,
    zscore,
)

import performance

warnings.filterwarnings("ignore", category=RuntimeWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# COLUMN_MAP: CSV column name  →  internal short name used in this script
# ──────────────────────────────────────────────────────────────────────────────
COLUMN_MAP = {
    # Performance (strip perf_ prefix)
    "perf_LP_MRR": "LP_MRR",
    "perf_LP_H10": "LP_H10",
    "perf_LP_H1": "LP_H1",
    "perf_RP_MRR": "RP_MRR",
    "perf_RP_H10": "RP_H10",
    "perf_Effi_LP": "Effi_LP",
    "perf_params_M_LP": "params_M_LP",
    "perf_Effi_RP": "Effi_RP",
    "perf_params_M_RP": "params_M_RP",
    # Structural metrics needing renaming
    "M3_num_train": "M3_train_triples",
    "M5_relation_to_entity_ratio": "M5_r_to_e_ratio",
    "M6_triple_to_entity_ratio": "M6_t_to_e_ratio",
    "M8_median_total_degree": "M8_median_total",
    "M10_degree_skewness": "M10_skewness",
    "M13_low_degree_leq_1": "M13_low_deg_1",
    "M13_low_degree_leq_2": "M13_low_deg_2",
    "M13_low_degree_leq_3": "M13_low_deg_3",
    "M13_low_degree_leq_5": "M13_low_deg_5",
    "M13_low_degree_leq_10": "M13_low_deg_10",
    "M18_normalised_relation_entropy": "M18_normalised_entropy",
    "M20_top1_concentration": "M20_top1_conc",
    "M20_top2_concentration": "M20_top2_conc",
    "M20_top3_concentration": "M20_top3_conc",
    "M20_top5_concentration": "M20_top5_conc",
    "M20_top10_concentration": "M20_top10_conc",
    "M23_avg_normalised_diversity": "M23_avg_norm_diversity",
    "M27_largest_class_size": "M27_largest_class",
    "M32_coverage_geq_1": "M32_coverage_1",
    "M32_coverage_geq_2": "M32_coverage_2",
    "M32_coverage_geq_4": "M32_coverage_4",
    "M32_coverage_geq_8": "M32_coverage_8",
    "M35_avg_rel_types_in_paths": "M35_avg_rel_in_paths",
    "M36_reachability_2hop": "M36_reach_2hop",
    "M36_reachability_3hop": "M36_reach_3hop",
    "M36_reachability_5hop": "M36_reach_5hop",
    "M36_reachability_10hop": "M36_reach_10hop",
    "M50_relation_graph_density": "M50_rel_graph_density",
    "M58_H_tail_given_rel": "M58_H_t_given_r",
    "M59_H_head_given_rel": "M59_H_h_given_r",
    "M60_H_rel_given_head_tail": "M60_H_r_given_ht",
    "M62_num_metapaths_2": "M62_metapaths_2_count",
    "M62_metapath_2_ratio": "M62_metapaths_2_ratio",
    "M62_num_metapaths_3": "M62_metapaths_3_count",
    "M62_metapath_3_ratio": "M62_metapaths_3_ratio",
}

INDUCTIVE_COLUMN_MAP = {
    "M55_inf_num_relations": "num_relations_inf",
    "M55_inf_num_entities": "num_entities_inf",
    "M55_inf_num_triples": "num_triples_inf",
    "M55_inf_avg_degree": "avg_degree_inf",
    "M56_inf_distinct_context_fraction": "distinct_context_frac_inf",
    "M56_inf_unique_context_fraction": "unique_context_frac_inf",
    "M55_train_avg_degree": "avg_degree_train",
    "M56_train_distinct_context_fraction": "distinct_context_frac_train",
    "M56_train_unique_context_fraction": "unique_context_frac_train",
}

# ──────────────────────────────────────────────────────────────────────────────
# DISPLAY_NAMES: internal key  →  human-readable label for publication
# ──────────────────────────────────────────────────────────────────────────────
DISPLAY_NAMES = {
    # Basic statistics
    "M1_num_entities": "Number of Entities",
    "M2_num_relations": "Number of Relations",
    "M3_train_triples": "Training Triples",
    "M3_num_total": "Total Triples",
    "M4_simple_density": "Graph Density",
    "M5_r_to_e_ratio": "Relation-to-Entity Ratio",
    "M6_t_to_e_ratio": "Triple-to-Entity Ratio",
    # Degree distribution
    "M7_avg_total_degree": "Average Total Degree",
    "M7_avg_in_degree": "Average In-Degree",
    "M7_avg_out_degree": "Average Out-Degree",
    "M8_median_total": "Median Total Degree",
    "M9_std_total_degree": "Std. of Total Degree",
    "M10_skewness": "Degree Skewness",
    "M11_degree_kurtosis": "Degree Kurtosis",
    "M13_low_deg_1": "Low-Degree Fraction (d \u2264 1)",
    "M13_low_deg_2": "Low-Degree Fraction (d \u2264 2)",
    "M13_low_deg_3": "Low-Degree Fraction (d \u2264 3)",
    "M13_low_deg_5": "Low-Degree Fraction (d \u2264 5)",
    "M13_low_deg_10": "Low-Degree Fraction (d \u2264 10)",
    # Relational diversity
    "M17_relation_entropy": "Relation Entropy",
    "M18_normalised_entropy": "Normalised Relation Entropy",
    "M19_relation_gini": "Relation Gini Coefficient",
    "M20_top1_conc": "Top-1 Relation Concentration",
    "M20_top2_conc": "Top-2 Relation Concentration",
    "M20_top3_conc": "Top-3 Relation Concentration",
    "M20_top5_conc": "Top-5 Relation Concentration",
    "M20_top10_conc": "Top-10 Relation Concentration",
    "M22_avg_rel_diversity": "Avg. Relational Diversity",
    "M22_median_rel_diversity": "Median Relational Diversity",
    "M23_avg_norm_diversity": "Avg. Normalised Diversity",
    # Context uniqueness
    "M25_distinct_fraction": "Distinct Context Fraction",
    "M25_unique_fraction": "Unique Context Fraction",
    "M26_compression_ratio": "Compression Ratio",
    "M27_largest_class": "Largest Equivalence Class",
    "M28_avg_class_size": "Avg. Equivalence Class Size",
    "M29_class_entropy": "Context Class Entropy",
    "M30_largest_class_fraction": "Largest Class Fraction",
    # Path richness
    "M31_avg_paths": "Avg. Paths per Entity",
    "M32_coverage_1": "Path Coverage (\u2265 1)",
    "M32_coverage_2": "Path Coverage (\u2265 2)",
    "M32_coverage_4": "Path Coverage (\u2265 4)",
    "M32_coverage_8": "Path Coverage (\u2265 8)",
    "M33_avg_path_length": "Avg. Path Length",
    "M34_avg_path_diversity": "Avg. Path Diversity",
    "M35_avg_rel_in_paths": "Avg. Relation Types in Paths",
    "M36_reach_2hop": "2-Hop Reachability",
    "M36_reach_3hop": "3-Hop Reachability",
    "M36_reach_5hop": "5-Hop Reachability",
    "M36_reach_10hop": "10-Hop Reachability",
    # Connectivity
    "M38_lcc_fraction": "LCC Fraction",
    "M40_lscc_fraction": "LSCC Fraction",
    "M41_diameter": "Diameter (approx.)",
    "M42_avg_shortest_path": "Avg. Shortest Path Length",
    "M43_avg_clustering": "Avg. Clustering Coefficient",
    "M44_transitivity": "Transitivity",
    "M45_untyped_reciprocity": "Reciprocity",
    "M46_assortativity": "Degree Assortativity",
    # Relation-level
    "M47_mean_rel_avg_degree": "Mean Relation Avg. Degree",
    "M50_rel_graph_density": "Relation Graph Density",
    "M51_mean_rel_recip": "Mean Relation Reciprocity",
    # Information-theoretic
    "M57_degree_entropy": "Degree Entropy",
    "M58_H_t_given_r": "H(Tail | Relation)",
    "M59_H_h_given_r": "H(Head | Relation)",
    "M60_H_r_given_ht": "H(Relation | Head, Tail)",
    # Multi-relational
    "M61_avg_multiplicity": "Avg. Edge Multiplicity",
    "M62_metapaths_2_count": "2-Step Meta-Path Count",
    "M62_metapaths_2_ratio": "2-Step Meta-Path Ratio",
    "M62_metapaths_3_count": "3-Step Meta-Path Count",
    "M62_metapaths_3_ratio": "3-Step Meta-Path Ratio",
    "M63_avg_symmetry": "Avg. Relation Symmetry",
    "M63_weighted_symmetry": "Weighted Relation Symmetry",
    # Performance metrics
    "LP_MRR": "Link Prediction MRR",
    "LP_H10": "Link Prediction Hits@10",
    "LP_H1": "Link Prediction Hits@1",
    "Effi_LP": "Efficiency (MRR / Params)",
    "RP_MRR": "Relation Prediction MRR",
    "RP_H10": "Relation Prediction Hits@10",
    "H10": "Hits@10 (Inductive)",
    # Inductive structural metrics
    "num_relations_inf": "Num. Relations (Inference)",
    "num_entities_inf": "Num. Entities (Inference)",
    "num_triples_inf": "Num. Triples (Inference)",
    "avg_degree_inf": "Avg. Degree (Inference)",
    "distinct_context_frac_inf": "Distinct Context Fraction (Inference)",
    "unique_context_frac_inf": "Unique Context Fraction (Inference)",
    "avg_degree_train": "Avg. Degree (Train)",
    "distinct_context_frac_train": "Distinct Context Fraction (Train)",
    "unique_context_frac_train": "Unique Context Fraction (Train)",
}

# ──────────────────────────────────────────────────────────────────────────────
# METRIC_DIRECTIONS: expected relationship with performance
# ──────────────────────────────────────────────────────────────────────────────
METRIC_DIRECTIONS = {
    "M2_num_relations": "higher_better",
    "M5_r_to_e_ratio": "higher_better",
    "M6_t_to_e_ratio": "higher_better",
    "M7_avg_total_degree": "higher_better",
    "M8_median_total": "higher_better",
    "M10_skewness": "lower_better",
    "M13_low_deg_1": "lower_better",
    "M13_low_deg_2": "lower_better",
    "M13_low_deg_5": "lower_better",
    "M17_relation_entropy": "higher_better",
    "M18_normalised_entropy": "higher_better",
    "M19_relation_gini": "lower_better",
    "M20_top2_conc": "lower_better",
    "M22_avg_rel_diversity": "higher_better",
    "M23_avg_norm_diversity": "higher_better",
    "M25_distinct_fraction": "higher_better",
    "M25_unique_fraction": "higher_better",
    "M27_largest_class": "lower_better",
    "M28_avg_class_size": "lower_better",
    "M30_largest_class_fraction": "lower_better",
    "M31_avg_paths": "higher_better",
    "M32_coverage_8": "higher_better",
    "M33_avg_path_length": "higher_better",
    "M34_avg_path_diversity": "higher_better",
    "M35_avg_rel_in_paths": "higher_better",
    "M36_reach_5hop": "higher_better",
    "M38_lcc_fraction": "higher_better",
    "M43_avg_clustering": "higher_better",
    "M44_transitivity": "higher_better",
    "M45_untyped_reciprocity": "higher_better",
    "M46_assortativity": "ambiguous",
    "M50_rel_graph_density": "higher_better",
    "M57_degree_entropy": "higher_better",
    "M58_H_t_given_r": "ambiguous",
    "M62_metapaths_2_count": "higher_better",
    "M62_metapaths_2_ratio": "higher_better",
}

KEY_STRUCTURAL_METRICS = [
    # Relational diversity
    "M2_num_relations", "M17_relation_entropy", "M18_normalised_entropy",
    "M19_relation_gini", "M20_top2_conc",
    # Degree
    "M7_avg_total_degree", "M8_median_total", "M13_low_deg_5",
    # Context uniqueness
    "M25_unique_fraction", "M27_largest_class", "M28_avg_class_size",
    # Path richness
    "M31_avg_paths", "M33_avg_path_length", "M34_avg_path_diversity",
    # Connectivity
    "M38_lcc_fraction", "M43_avg_clustering",
    # Meta-structure
    "M50_rel_graph_density", "M62_metapaths_2_ratio",
    # Information-theoretic
    "M57_degree_entropy", "M58_H_t_given_r", "M60_H_r_given_ht",
]

PERF_METRICS = ["LP_MRR", "LP_H10", "Effi_LP"]

# configs/structure/performance.yaml
INDUCTIVE_PERF = {split: {"H10": h10, "parent_dataset": split.rsplit("_", 1)[0]}
                  for split, h10 in performance.inductive_h10().items()}

INDUCTIVE_STRUCT_METRICS = [
    "num_relations_inf",
    "num_entities_inf",
    "num_triples_inf",
    "avg_degree_inf",
    "distinct_context_frac_inf",
    "unique_context_frac_inf",
]

PROFILE_METRICS = [
    "M2_num_relations", "M7_avg_total_degree", "M17_relation_entropy",
    "M18_normalised_entropy", "M19_relation_gini", "M25_unique_fraction",
    "M28_avg_class_size", "M31_avg_paths", "M34_avg_path_diversity",
    "M43_avg_clustering", "M50_rel_graph_density", "M62_metapaths_2_ratio",
]

CASE_STUDY_DATASETS = ["FB15k-237", "WN18RR", "CoDEx-Large"]

CASE_STUDY_METRICS = [
    "M1_num_entities", "M2_num_relations", "M3_train_triples",
    "M5_r_to_e_ratio", "M7_avg_total_degree", "M8_median_total",
    "M13_low_deg_5", "M17_relation_entropy", "M18_normalised_entropy",
    "M19_relation_gini", "M20_top2_conc",
    "M22_avg_rel_diversity", "M25_unique_fraction",
    "M27_largest_class", "M28_avg_class_size",
    "M31_avg_paths", "M33_avg_path_length", "M34_avg_path_diversity",
    "M43_avg_clustering", "M44_transitivity",
    "M45_untyped_reciprocity", "M50_rel_graph_density",
    "M62_metapaths_2_ratio",
]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def dn(key: str) -> str:
    """Return the human-readable display name for *key*, falling back to key."""
    return DISPLAY_NAMES.get(key, key)


def save_fig(fig, base_path: str) -> None:
    """Save *fig* as both PNG (300 dpi) and PDF (vector)."""
    fig.savefig(f"{base_path}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{base_path}.pdf", bbox_inches="tight")
    plt.close(fig)
    log.info("  saved %s.{png,pdf}", base_path)


def effect_size_label(r: float) -> str:
    """Cohen (1988) effect-size category for a correlation coefficient."""
    ar = abs(r)
    if ar >= 0.5:
        return "large"
    if ar >= 0.3:
        return "medium"
    if ar >= 0.1:
        return "small"
    return "negligible"


def bootstrap_pearson_ci(
    x: np.ndarray,
    y: np.ndarray,
    n_resamples: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 42,
):
    """Percentile bootstrap CI for Pearson r (paired resampling)."""
    rng = np.random.RandomState(seed)
    n = len(x)
    boot_r = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.choice(n, size=n, replace=True)
        if np.std(x[idx]) == 0 or np.std(y[idx]) == 0:
            boot_r[i] = np.nan
        else:
            boot_r[i] = pearsonr(x[idx], y[idx])[0]
    boot_r = boot_r[~np.isnan(boot_r)]
    alpha = 1 - confidence_level
    lo = np.percentile(boot_r, 100 * alpha / 2)
    hi = np.percentile(boot_r, 100 * (1 - alpha / 2))
    return lo, hi


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_transductive_data(output_dir: str = "output") -> pd.DataFrame:
    """Load summary_metrics.csv and rename columns to spec shorthand."""
    path = os.path.join(output_dir, "summary_metrics.csv")
    df = pd.read_csv(path, index_col="dataset")
    df.rename(columns=COLUMN_MAP, inplace=True)
    log.info("Loaded transductive data: %d datasets, %d columns", len(df), len(df.columns))
    return df


def load_inductive_data(output_dir: str = "output") -> pd.DataFrame:
    """Load inductive_metrics.csv, rename columns, join performance data."""
    path = os.path.join(output_dir, "inductive_splits", "inductive_metrics.csv")
    df = pd.read_csv(path, index_col="split")

    # Normalise split names to uppercase V (FB15k-237_v1 -> FB15k-237_V1)
    df.index = df.index.str.replace(r"_v(\d)", r"_V\1", regex=True)

    df.rename(columns=INDUCTIVE_COLUMN_MAP, inplace=True)

    # Join performance data
    perf_df = pd.DataFrame.from_dict(INDUCTIVE_PERF, orient="index")
    df = df.join(perf_df)

    log.info("Loaded inductive data: %d splits, %d columns", len(df), len(df.columns))
    return df


# ──────────────────────────────────────────────────────────────────────────────
# PART 1: Rank Concordance Analysis
# ──────────────────────────────────────────────────────────────────────────────

def part1_rank_concordance(analysis_df: pd.DataFrame, out_dir: str):
    """Methods 1A–1D: rank-based concordance analysis."""
    log.info("PART 1: Rank Concordance Analysis")

    # 1A – performance rankings
    perf_ranks = pd.DataFrame(index=analysis_df.index)
    for pm in PERF_METRICS:
        valid = analysis_df[pm].dropna()
        perf_ranks[f"{dn(pm)} rank"] = valid.rank(ascending=False)

    # 1B – structural metric rankings (with direction)
    struct_ranks = pd.DataFrame(index=analysis_df.index)
    for metric, direction in METRIC_DIRECTIONS.items():
        if metric not in analysis_df.columns:
            continue
        valid = analysis_df[metric].dropna()
        if direction == "higher_better":
            struct_ranks[f"{dn(metric)} rank"] = valid.rank(ascending=False)
        elif direction == "lower_better":
            struct_ranks[f"{dn(metric)} rank"] = valid.rank(ascending=True)
        else:
            struct_ranks[f"{dn(metric)} rank (desc)"] = valid.rank(ascending=False)
            struct_ranks[f"{dn(metric)} rank (asc)"] = valid.rank(ascending=True)

    # 1C – Kendall tau + Spearman rho for all (structural, performance) pairs
    concordance_results = []
    for struct_metric, direction in METRIC_DIRECTIONS.items():
        if struct_metric not in analysis_df.columns or direction == "ambiguous":
            continue
        struct_vals = analysis_df[struct_metric].values.astype(float)
        if direction == "lower_better":
            struct_vals = -struct_vals

        for pm in PERF_METRICS:
            perf_vals = analysis_df[pm].dropna()
            common_idx = perf_vals.index.intersection(analysis_df.index)
            if len(common_idx) < 3:
                continue
            mask = analysis_df.index.isin(common_idx)
            sv = struct_vals[mask]
            pv = perf_vals.loc[common_idx].values.astype(float)

            tau, tau_p = kendalltau(sv, pv)
            rho, rho_p = spearmanr(sv, pv)

            concordance_results.append({
                "structural_metric": struct_metric,
                "structural_metric_name": dn(struct_metric),
                "performance_metric": pm,
                "performance_metric_name": dn(pm),
                "kendall_tau": round(tau, 4),
                "kendall_p": round(tau_p, 4),
                "spearman_rho": round(rho, 4),
                "spearman_p": round(rho_p, 4),
                "n": len(common_idx),
            })

    concordance_df = pd.DataFrame(concordance_results)
    concordance_df.to_csv(os.path.join(out_dir, "concordance_results.csv"), index=False)
    log.info("  concordance_results.csv: %d rows", len(concordance_df))

    # 1D – perfect / strong concordances
    perfect = concordance_df[concordance_df["kendall_tau"] == 1.0].copy()
    strong = concordance_df[concordance_df["kendall_tau"] >= 0.67].copy()
    strong.to_csv(os.path.join(out_dir, "perfect_concordances.csv"), index=False)
    log.info("  perfect concordances (tau=1.0): %d | strong (tau>=0.67): %d",
             len(perfect), len(strong))

    return concordance_df


# ──────────────────────────────────────────────────────────────────────────────
# PART 2: Correlation Matrix with Effect Sizes
# ──────────────────────────────────────────────────────────────────────────────

def part2_correlation_matrix(analysis_df: pd.DataFrame, out_dir: str):
    """Methods 2A–2B: pairwise correlations and bootstrap CIs."""
    log.info("PART 2: Correlation Matrix with Effect Sizes")

    # 2A – pairwise correlations
    correlation_results = []
    for sm in KEY_STRUCTURAL_METRICS:
        if sm not in analysis_df.columns:
            continue
        for pm in PERF_METRICS:
            valid_idx = analysis_df[[sm, pm]].dropna().index
            if len(valid_idx) < 3:
                continue
            x = analysis_df.loc[valid_idx, sm].values.astype(float)
            y = analysis_df.loc[valid_idx, pm].values.astype(float)

            r_p, p_p = pearsonr(x, y)
            r_s, p_s = spearmanr(x, y)
            tau, p_t = kendalltau(x, y)

            signs = [np.sign(r_p), np.sign(r_s), np.sign(tau)]
            sign_agreement = len(set(signs)) == 1

            correlation_results.append({
                "structural_metric": sm,
                "structural_metric_name": dn(sm),
                "performance_metric": pm,
                "performance_metric_name": dn(pm),
                "pearson_r": round(r_p, 4),
                "pearson_p": round(p_p, 4),
                "spearman_rho": round(r_s, 4),
                "spearman_p": round(p_s, 4),
                "kendall_tau": round(tau, 4),
                "kendall_p": round(p_t, 4),
                "effect_size": effect_size_label(r_p),
                "sign_agreement": sign_agreement,
                "n": len(valid_idx),
            })

    corr_df = pd.DataFrame(correlation_results)
    corr_df.to_csv(os.path.join(out_dir, "correlation_matrix.csv"), index=False)
    log.info("  correlation_matrix.csv: %d rows", len(corr_df))

    # 2B – bootstrap CIs for top-10 strongest correlations (by |pearson_r|)
    corr_df["abs_pearson"] = corr_df["pearson_r"].abs()
    top = corr_df.nlargest(10, "abs_pearson")
    bootstrap_results = []

    for _, row in top.iterrows():
        sm, pm = row["structural_metric"], row["performance_metric"]
        valid_idx = analysis_df[[sm, pm]].dropna().index
        x = analysis_df.loc[valid_idx, sm].values.astype(float)
        y = analysis_df.loc[valid_idx, pm].values.astype(float)

        ci_lo, ci_hi = bootstrap_pearson_ci(x, y)
        bootstrap_results.append({
            "structural_metric": sm,
            "structural_metric_name": dn(sm),
            "performance_metric": pm,
            "performance_metric_name": dn(pm),
            "pearson_r": row["pearson_r"],
            "ci_lower_95": round(ci_lo, 4),
            "ci_upper_95": round(ci_hi, 4),
        })

    bootstrap_df = pd.DataFrame(bootstrap_results)
    bootstrap_df.to_csv(os.path.join(out_dir, "bootstrap_confidence_intervals.csv"), index=False)
    log.info("  bootstrap_confidence_intervals.csv: %d rows", len(bootstrap_df))

    corr_df.drop(columns=["abs_pearson"], inplace=True)
    return corr_df


# ──────────────────────────────────────────────────────────────────────────────
# PART 3: Normalised Profile Comparison
# ──────────────────────────────────────────────────────────────────────────────

def part3_normalised_profiles(
    analysis_df: pd.DataFrame, out_dir: str, plot_dir: str
):
    """Methods 3A–3C: z-score profiles, radar chart, heatmap."""
    log.info("PART 3: Normalised Profile Comparison")

    available = [m for m in PROFILE_METRICS if m in analysis_df.columns]
    profile_df = analysis_df[available].copy()

    # 3A – z-score normalisation
    zscore_df = profile_df.apply(zscore, nan_policy="omit")
    zscore_df.index = analysis_df.index

    # Save with display-name columns
    zscore_out = zscore_df.copy()
    zscore_out.columns = [dn(c) for c in zscore_out.columns]
    zscore_out.to_csv(os.path.join(out_dir, "normalised_profiles.csv"))
    log.info("  normalised_profiles.csv")

    # 3B – radar chart
    _radar_chart(zscore_df, analysis_df["LP_MRR"], plot_dir)

    # 3C – heatmap
    _heatmap_profiles(zscore_df, analysis_df["LP_MRR"], plot_dir)


def _radar_chart(
    zscore_df: pd.DataFrame,
    performance: pd.Series,
    plot_dir: str,
):
    labels = [dn(c) for c in zscore_df.columns]
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    cmap = plt.cm.RdYlGn
    perf_norm = (performance - performance.min()) / (performance.max() - performance.min())

    for dataset in zscore_df.index:
        values = zscore_df.loc[dataset].tolist()
        values += values[:1]
        colour = cmap(perf_norm[dataset])
        ax.plot(angles, values, "o-", linewidth=2, label=dataset, color=colour)
        ax.fill(angles, values, alpha=0.1, color=colour)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=7, wrap=True)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.12), fontsize=9)
    ax.set_title(
        "Structural Profiles by Dataset\n(colour intensity proportional to LP MRR)",
        size=13, y=1.08,
    )
    plt.tight_layout()
    save_fig(fig, os.path.join(plot_dir, "radar_profiles"))


def _heatmap_profiles(
    zscore_df: pd.DataFrame,
    performance: pd.Series,
    plot_dir: str,
):
    order = performance.sort_values(ascending=False).index
    ordered = zscore_df.loc[order]
    display_cols = [dn(c) for c in ordered.columns]
    ordered_display = ordered.copy()
    ordered_display.columns = display_cols

    fig, ax = plt.subplots(figsize=(max(14, len(display_cols) * 1.1), 4))
    sns.heatmap(
        ordered_display, annot=True, fmt=".2f", cmap="RdYlGn", center=0,
        linewidths=0.5, ax=ax, cbar_kws={"label": "z-score"},
    )
    ax.set_title(
        "Structural Properties (z-scored) \u2014 Datasets ordered by Link Prediction MRR \u2193"
    )
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    plt.tight_layout()
    save_fig(fig, os.path.join(plot_dir, "heatmap_profiles"))


# ──────────────────────────────────────────────────────────────────────────────
# PART 4: Scatter Plots
# ──────────────────────────────────────────────────────────────────────────────

def part4_scatter_plots(
    analysis_df: pd.DataFrame, out_dir: str, plot_dir: str
):
    """Methods 4A–4B: annotated scatter plots and small-multiple grid."""
    log.info("PART 4: Scatter Plots")

    # 4A – individual scatter plots
    scatter_pairs = [
        ("M25_unique_fraction", "LP_MRR"),
        ("M18_normalised_entropy", "LP_MRR"),
        ("M19_relation_gini", "LP_MRR"),
        ("M7_avg_total_degree", "LP_MRR"),
        ("M34_avg_path_diversity", "LP_MRR"),
        ("M28_avg_class_size", "LP_MRR"),
        ("M50_rel_graph_density", "LP_MRR"),
        ("M25_unique_fraction", "Effi_LP"),
        ("M18_normalised_entropy", "LP_H10"),
    ]

    for sm, pm in scatter_pairs:
        if sm not in analysis_df.columns:
            continue
        safe = f"{sm}_vs_{pm}".replace("/", "_")
        _annotated_scatter(
            analysis_df, sm, pm,
            os.path.join(plot_dir, f"scatter_{safe}"),
        )

    # 4B – small-multiple grid
    avail = [m for m in KEY_STRUCTURAL_METRICS if m in analysis_df.columns]
    _scatter_grid(analysis_df, avail, "LP_MRR",
                  os.path.join(plot_dir, "scatter_grid_LP_MRR"))


def _annotated_scatter(
    df: pd.DataFrame,
    sm: str, pm: str,
    base_path: str,
):
    valid_idx = df[[sm, pm]].dropna().index
    x = df.loc[valid_idx, sm].values.astype(float)
    y = df.loc[valid_idx, pm].values.astype(float)
    labels = valid_idx.tolist()

    r_p, _ = pearsonr(x, y)
    r_s, _ = spearmanr(x, y)
    slope, intercept, *_ = linregress(x, y)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y, s=120, zorder=5, edgecolors="black", linewidth=0.5)

    for i, label in enumerate(labels):
        ax.annotate(
            label, (x[i], y[i]),
            textcoords="offset points", xytext=(8, 8),
            fontsize=9, fontstyle="italic",
        )

    x_line = np.linspace(x.min() - 0.05 * abs(x.min()), x.max() + 0.05 * abs(x.max()), 100)
    ax.plot(x_line, slope * x_line + intercept, "--", color="gray", alpha=0.7, linewidth=1)

    textstr = (
        f"Pearson r = {r_p:.3f}\n"
        f"Spearman \u03c1 = {r_s:.3f}\n"
        f"n = {len(x)}"
    )
    ax.text(
        0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    ax.set_xlabel(dn(sm), fontsize=12)
    ax.set_ylabel(dn(pm), fontsize=12)
    ax.set_title(f"{dn(pm)} vs. {dn(sm)}", fontsize=13)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_fig(fig, base_path)


def _scatter_grid(
    df: pd.DataFrame,
    struct_metrics: list,
    perf_metric: str,
    base_path: str,
    ncols: int = 4,
):
    n = len(struct_metrics)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(3.5 * ncols, 3 * nrows),
    )
    axes = axes.flatten()

    for i, sm in enumerate(struct_metrics):
        ax = axes[i]
        valid_idx = df[[sm, perf_metric]].dropna().index
        x = df.loc[valid_idx, sm].values.astype(float)
        y = df.loc[valid_idx, perf_metric].values.astype(float)

        ax.scatter(x, y, s=60, edgecolors="black", linewidth=0.5)
        for j, label in enumerate(valid_idx):
            ax.annotate(
                label[:6], (x[j], y[j]),
                fontsize=6, textcoords="offset points", xytext=(3, 3),
            )

        title = dn(sm)
        if len(x) >= 3:
            rho, _ = spearmanr(x, y)
            title += f"\n\u03c1 = {rho:.2f}"
        ax.set_title(title, fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.2)

    for i in range(n, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(f"Structural Metrics vs. {dn(perf_metric)}", fontsize=14, y=1.02)
    plt.tight_layout()
    save_fig(fig, base_path)


# ──────────────────────────────────────────────────────────────────────────────
# PART 5: Inductive Split Analysis
# ──────────────────────────────────────────────────────────────────────────────

def part5_inductive_analysis(
    inductive_df: pd.DataFrame, out_dir: str, plot_dir: str
):
    """Methods 5A–5C: inductive correlations and scatter plots."""
    log.info("PART 5: Inductive Split Analysis")

    available = [m for m in INDUCTIVE_STRUCT_METRICS if m in inductive_df.columns]

    # 5A – correlation analysis (n=8)
    inductive_corr = []
    for sm in available:
        x = inductive_df[sm].values.astype(float)
        y = inductive_df["H10"].values.astype(float)
        r_p, p_p = pearsonr(x, y)
        r_s, p_s = spearmanr(x, y)
        tau, p_t = kendalltau(x, y)
        inductive_corr.append({
            "structural_metric": sm,
            "structural_metric_name": dn(sm),
            "pearson_r": round(r_p, 4),
            "pearson_p": round(p_p, 4),
            "spearman_rho": round(r_s, 4),
            "spearman_p": round(p_s, 4),
            "kendall_tau": round(tau, 4),
            "kendall_p": round(p_t, 4),
            "effect_size": effect_size_label(r_p),
            "n": 8,
        })

    inductive_corr_df = pd.DataFrame(inductive_corr)
    inductive_corr_df.to_csv(
        os.path.join(out_dir, "inductive_correlation_results.csv"), index=False,
    )
    log.info("  inductive_correlation_results.csv: %d rows", len(inductive_corr_df))

    # 5B – within-family correlations
    within_results = []
    for family in ["FB15k-237", "WN18RR"]:
        fdata = inductive_df[inductive_df["parent_dataset"] == family]
        for sm in available:
            x = fdata[sm].values.astype(float)
            y = fdata["H10"].values.astype(float)
            if len(x) >= 3:
                rho, p = spearmanr(x, y)
                tau, p_t = kendalltau(x, y)
            else:
                rho, p, tau, p_t = np.nan, np.nan, np.nan, np.nan
            within_results.append({
                "family": family,
                "structural_metric": sm,
                "structural_metric_name": dn(sm),
                "spearman_rho": round(rho, 4),
                "spearman_p": round(p, 4),
                "kendall_tau": round(tau, 4),
                "kendall_p": round(p_t, 4),
                "n": len(x),
            })

    within_df = pd.DataFrame(within_results)
    within_df.to_csv(os.path.join(out_dir, "inductive_within_family.csv"), index=False)
    log.info("  inductive_within_family.csv: %d rows", len(within_df))

    # 5C – scatter plots
    for sm in available:
        safe = sm.replace("/", "_")
        _inductive_scatter(
            inductive_df, sm,
            os.path.join(plot_dir, f"inductive_scatter_{safe}"),
        )


def _inductive_scatter(
    df: pd.DataFrame, sm: str, base_path: str,
):
    fig, ax = plt.subplots(figsize=(7, 5))
    colours = {"FB15k-237": "tab:blue", "WN18RR": "tab:orange"}

    for family, colour in colours.items():
        mask = df["parent_dataset"] == family
        fdata = df[mask]
        x = fdata[sm].values.astype(float)
        y = fdata["H10"].values.astype(float)
        ax.scatter(x, y, s=100, c=colour, label=family, edgecolors="black", linewidth=0.5)
        for idx, (xi, yi) in zip(fdata.index, zip(x, y)):
            version_label = idx.split("_")[-1]
            ax.annotate(
                version_label, (xi, yi),
                textcoords="offset points", xytext=(5, 5), fontsize=9,
            )

    x_all = df[sm].values.astype(float)
    y_all = df["H10"].values.astype(float)
    rho, _ = spearmanr(x_all, y_all)

    ax.text(
        0.05, 0.95,
        f"Spearman \u03c1 = {rho:.3f} (n = 8)",
        transform=ax.transAxes, fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    ax.set_xlabel(dn(sm), fontsize=12)
    ax.set_ylabel(dn("H10"), fontsize=12)
    ax.set_title(f"{dn('H10')} vs. {dn(sm)}", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_fig(fig, base_path)


# ──────────────────────────────────────────────────────────────────────────────
# PART 6: Case Study Data
# ──────────────────────────────────────────────────────────────────────────────

def part6_case_studies(analysis_df: pd.DataFrame, out_dir: str):
    """Methods 6A–6C: comparative tables and delta analysis."""
    log.info("PART 6: Case Study Data")

    avail_metrics = [m for m in CASE_STUDY_METRICS if m in analysis_df.columns]
    perf_cols = [p for p in PERF_METRICS if p in analysis_df.columns]
    all_cols = avail_metrics + perf_cols

    # 6A – comparative summary table
    case_table = analysis_df.loc[CASE_STUDY_DATASETS, all_cols].T.copy()
    case_table.index = [dn(c) for c in case_table.index]
    case_table.to_csv(os.path.join(out_dir, "case_study_comparison.csv"))
    log.info("  case_study_comparison.csv")

    # 6B – delta analysis
    pairs = [
        ("FB15k-237", "WN18RR"),
        ("FB15k-237", "CoDEx-Large"),
        ("CoDEx-Large", "WN18RR"),
    ]
    delta_rows = []
    for d1, d2 in pairs:
        for metric in avail_metrics + perf_cols:
            v1 = analysis_df.loc[d1, metric]
            v2 = analysis_df.loc[d2, metric]
            if pd.isna(v1) or pd.isna(v2) or v2 == 0:
                pct_diff = None
            else:
                pct_diff = ((v1 - v2) / abs(v2)) * 100

            delta_rows.append({
                "comparison": f"{d1} vs {d2}",
                "metric": metric,
                "metric_name": dn(metric),
                "value_1": v1,
                "value_2": v2,
                "absolute_diff": (v1 - v2) if not (pd.isna(v1) or pd.isna(v2)) else None,
                "pct_diff": round(pct_diff, 1) if pct_diff is not None else None,
            })

    delta_df = pd.DataFrame(delta_rows)
    delta_df.to_csv(os.path.join(out_dir, "case_study_deltas.csv"), index=False)
    log.info("  case_study_deltas.csv: %d rows", len(delta_df))

    # 6C – top-5 structural differences per pair
    for pair_name in ["FB15k-237 vs WN18RR", "FB15k-237 vs CoDEx-Large",
                       "CoDEx-Large vs WN18RR"]:
        pair_data = delta_df[delta_df["comparison"] == pair_name].dropna(subset=["pct_diff"])
        top_diffs = pair_data.reindex(
            pair_data["pct_diff"].abs().sort_values(ascending=False).index
        ).head(5)
        fname = f"case_study_top_diffs_{pair_name.replace(' ', '_')}.csv"
        top_diffs.to_csv(os.path.join(out_dir, fname), index=False)
        log.info("  %s", fname)

    return delta_df


# ──────────────────────────────────────────────────────────────────────────────
# PART 7: Summary Outputs for Paper Drafting
# ──────────────────────────────────────────────────────────────────────────────

def part7_summary_outputs(
    analysis_df: pd.DataFrame,
    concordance_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    delta_df: pd.DataFrame,
    out_dir: str,
):
    """Methods 7A–7D: paper-ready tables, findings JSON, guidelines."""
    log.info("PART 7: Summary Outputs for Paper Drafting")

    # 7A – master correlation summary table  (rows = display names)
    rho_pivot = corr_df.pivot_table(
        index="structural_metric_name",
        columns="performance_metric_name",
        values="spearman_rho",
    )
    tau_pivot = corr_df.pivot_table(
        index="structural_metric_name",
        columns="performance_metric_name",
        values="kendall_tau",
    )
    combined = rho_pivot.copy()
    for col in combined.columns:
        combined[col] = (
            rho_pivot[col].apply(lambda x: f"{x:.2f}")
            + " ("
            + tau_pivot[col].apply(lambda x: f"{x:.2f}")
            + ")"
        )
    combined.to_csv(os.path.join(out_dir, "paper_table_correlations.csv"))
    log.info("  paper_table_correlations.csv")

    # 7B – ranking agreement table
    mrr_ranking = analysis_df["LP_MRR"].rank(ascending=False).astype(int)
    ranking_rows = []
    for sm in KEY_STRUCTURAL_METRICS:
        if sm not in analysis_df.columns:
            continue
        direction = METRIC_DIRECTIONS.get(sm, "higher_better")
        vals = analysis_df[sm].dropna()
        if direction == "lower_better":
            sm_rank = vals.rank(ascending=True).astype(int)
        else:
            sm_rank = vals.rank(ascending=False).astype(int)

        matches = (sm_rank == mrr_ranking).all()

        tau_val = None
        match = concordance_df[
            (concordance_df["structural_metric"] == sm)
            & (concordance_df["performance_metric"] == "LP_MRR")
        ]
        if len(match) > 0:
            tau_val = match["kendall_tau"].values[0]

        ranking_rows.append({
            "structural_metric_name": dn(sm),
            "metric_rank": sm_rank.to_dict(),
            "mrr_rank": mrr_ranking.to_dict(),
            "matches_MRR_ranking": matches,
            "kendall_tau": tau_val,
        })

    ranking_df = pd.DataFrame(ranking_rows)
    ranking_df.to_csv(os.path.join(out_dir, "paper_table_rankings.csv"), index=False)
    log.info("  paper_table_rankings.csv")

    # 7C – key findings JSON
    findings = {
        "methodology_note": {
            "n_transductive": 4,
            "n_inductive": 8,
            "primary_method": "rank concordance (Kendall tau)",
            "supplementary_methods": [
                "Pearson r", "Spearman rho", "bootstrap CI", "visual profiles",
            ],
            "limitation": (
                "With n=4 transductive datasets, no correlation reaches p<0.05. "
                "Results are reported as effect sizes, not significance tests."
            ),
        },
        "strongest_correlations_transductive": [],
        "perfect_rank_concordances": [],
        "weakest_correlations": [],
        "case_study_explanations": {},
    }

    top = corr_df.nlargest(10, "kendall_tau")
    for _, row in top.iterrows():
        findings["strongest_correlations_transductive"].append({
            "structural_metric": row["structural_metric"],
            "display_name": row["structural_metric_name"],
            "performance_metric": row["performance_metric"],
            "performance_display_name": row["performance_metric_name"],
            "kendall_tau": float(row["kendall_tau"]),
            "spearman_rho": float(row["spearman_rho"]),
            "pearson_r": float(row["pearson_r"]),
        })

    perfect = concordance_df[concordance_df["kendall_tau"] == 1.0]
    for _, row in perfect.iterrows():
        findings["perfect_rank_concordances"].append({
            "structural_metric": row["structural_metric"],
            "display_name": row["structural_metric_name"],
            "performance_metric": row["performance_metric"],
            "performance_display_name": row["performance_metric_name"],
        })

    bottom = corr_df.nsmallest(5, "kendall_tau")
    for _, row in bottom.iterrows():
        findings["weakest_correlations"].append({
            "structural_metric": row["structural_metric"],
            "display_name": row["structural_metric_name"],
            "performance_metric": row["performance_metric"],
            "performance_display_name": row["performance_metric_name"],
            "kendall_tau": float(row["kendall_tau"]),
        })

    for d in CASE_STUDY_DATASETS:
        mrr_val = analysis_df.loc[d, "LP_MRR"]
        top_for_d = []
        for pair in ["FB15k-237 vs WN18RR", "FB15k-237 vs CoDEx-Large",
                       "CoDEx-Large vs WN18RR"]:
            if d in pair:
                sub = delta_df[delta_df["comparison"] == pair].dropna(subset=["pct_diff"])
                for _, r in sub.nlargest(3, "pct_diff").iterrows():
                    top_for_d.append({
                        "comparison": pair,
                        "metric_name": r["metric_name"],
                        "pct_diff": float(r["pct_diff"]),
                    })
        findings["case_study_explanations"][d] = {
            "LP_MRR": float(mrr_val) if not pd.isna(mrr_val) else None,
            "notable_differences": top_for_d,
        }

    with open(os.path.join(out_dir, "key_findings.json"), "w") as f:
        json.dump(findings, f, indent=2, default=str)
    log.info("  key_findings.json")

    # 7D – practical guidelines
    good = ["FB15k-237", "CoDEx-Large"]
    poor = ["WN18RR", "YAGO3-10"]
    guidelines = []
    for sm in KEY_STRUCTURAL_METRICS:
        if sm not in analysis_df.columns:
            continue
        good_vals = analysis_df.loc[good, sm].dropna()
        poor_vals = analysis_df.loc[poor, sm].dropna()
        if len(good_vals) == 0 or len(poor_vals) == 0:
            continue

        good_mean = good_vals.mean()
        poor_mean = poor_vals.mean()
        good_min = good_vals.min()
        good_max = good_vals.max()
        poor_min = poor_vals.min()
        poor_max = poor_vals.max()

        if good_mean > poor_mean:
            threshold = f">{poor_max:.4f}" if good_min > poor_max else "overlapping"
            direction = "higher is better"
        else:
            threshold = f"<{poor_min:.4f}" if good_max < poor_min else "overlapping"
            direction = "lower is better"

        separable = (good_min > poor_max) or (good_max < poor_min)

        guidelines.append({
            "metric_name": dn(sm),
            "metric_key": sm,
            "direction": direction,
            "good_performance_range": f"[{good_min:.4f}, {good_max:.4f}]",
            "poor_performance_range": f"[{poor_min:.4f}, {poor_max:.4f}]",
            "separable": separable,
            "threshold_region": threshold,
        })

    guidelines_df = pd.DataFrame(guidelines)
    guidelines_df.to_csv(os.path.join(out_dir, "practical_guidelines.csv"), index=False)
    log.info("  practical_guidelines.csv: %d rows", len(guidelines_df))


# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Correlation analysis pipeline")
    parser.add_argument("--output-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "structure"),
                        help="Base output directory (default: <repo>/results/structure)")
    args = parser.parse_args()

    base = args.output_dir
    corr_dir = os.path.join(base, "correlation_analysis")
    plot_dir = os.path.join(base, "plots")
    os.makedirs(corr_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # Load data
    analysis_df = load_transductive_data(base)
    inductive_df = load_inductive_data(base)

    # Run all 7 parts
    concordance_df = part1_rank_concordance(analysis_df, corr_dir)
    corr_df = part2_correlation_matrix(analysis_df, corr_dir)
    part3_normalised_profiles(analysis_df, corr_dir, plot_dir)
    part4_scatter_plots(analysis_df, corr_dir, plot_dir)
    part5_inductive_analysis(inductive_df, corr_dir, plot_dir)
    delta_df = part6_case_studies(analysis_df, corr_dir)
    part7_summary_outputs(analysis_df, concordance_df, corr_df, delta_df, corr_dir)

    log.info("=== Correlation analysis complete ===")
    log.info("CSV / JSON outputs: %s", corr_dir)
    log.info("Plot outputs:       %s", plot_dir)


if __name__ == "__main__":
    main()
