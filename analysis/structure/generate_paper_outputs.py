#!/usr/bin/env python3
"""
Generate the three CSV files required for paper Sections 5 and 7.

Reads from <output-dir>/summary_metrics.json and <output-dir>/inductive_splits/inductive_metrics.csv,
then writes:
  <output-dir>/paper_outputs/summary_metrics_for_paper.csv
  <output-dir>/paper_outputs/correlation_table_for_paper.csv
  <output-dir>/paper_outputs/inductive_metrics_for_paper.csv
(default output-dir: <repo>/results/structure)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau

import performance

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "results" / "structure"
PAPER_DIR = OUTPUT_DIR / "paper_outputs"

DATASETS = ["FB15k-237", "WN18RR", "CoDEx-Large", "YAGO3-10"]

# ── Mapping from paper column names → keys in summary_metrics.json ──

COLUMN_MAP = {
    "num_entities":                "M1_num_entities",
    "num_relations":               "M2_num_relations",
    "train_triples":               "M3_num_train",
    "valid_triples":               "M3_num_valid",
    "test_triples":                "M3_num_test",
    "full_density":                "M4_full_density",
    "simple_density":              "M4_simple_density",
    "r_to_e_ratio":                "M5_relation_to_entity_ratio",
    "t_to_e_ratio":                "M6_triple_to_entity_ratio",
    "avg_in_degree":               "M7_avg_in_degree",
    "avg_out_degree":              "M7_avg_out_degree",
    "avg_total_degree":            "M7_avg_total_degree",
    "median_in_degree":            "M8_median_in_degree",
    "median_out_degree":           "M8_median_out_degree",
    "median_total_degree":         "M8_median_total_degree",
    "std_total_degree":            "M9_std_total_degree",
    "degree_skewness":             "M10_degree_skewness",
    "degree_kurtosis":             "M11_degree_kurtosis",
    "min_degree":                  "M12_min_degree",
    "max_degree":                  "M12_max_degree",
    "low_deg_1":                   "M13_low_degree_leq_1",
    "low_deg_2":                   "M13_low_degree_leq_2",
    "low_deg_3":                   "M13_low_degree_leq_3",
    "low_deg_5":                   "M13_low_degree_leq_5",
    "low_deg_10":                  "M13_low_degree_leq_10",
    "powerlaw_alpha":              "M14_alpha",
    "powerlaw_xmin":               "M14_xmin",
    "in_out_degree_pearson":       "M15_pearson_corr",
    "in_out_degree_spearman":      "M15_spearman_corr",
    "relation_entropy":            "M17_relation_entropy",
    "normalised_relation_entropy": "M18_normalised_relation_entropy",
    "relation_gini":               "M19_relation_gini",
    "top1_concentration":          "M20_top1_concentration",
    "top2_concentration":          "M20_top2_concentration",
    "top3_concentration":          "M20_top3_concentration",
    "top5_concentration":          "M20_top5_concentration",
    "top10_concentration":         "M20_top10_concentration",
    "rare_relations_01pct":        "M21_rare_relations_0.1pct",
    "rare_relations_05pct":        "M21_rare_relations_0.5pct",
    "rare_relations_1pct":         "M21_rare_relations_1.0pct",
    "avg_rel_diversity_per_entity":    "M22_avg_rel_diversity",
    "median_rel_diversity_per_entity": "M22_median_rel_diversity",
    "std_rel_diversity_per_entity":    "M22_std_rel_diversity",
    "avg_norm_rel_diversity":          "M23_avg_normalised_diversity",
    "distinct_context_fraction":       "M25_distinct_fraction",
    "unique_context_fraction":         "M25_unique_fraction",
    "largest_equiv_class":             "M27_largest_class_size",
    "avg_equiv_class_size":            "M28_avg_class_size",
    "equiv_class_entropy":             "M29_class_entropy",
    "largest_class_fraction":          "M30_largest_class_fraction",
    "avg_paths_per_entity":            "M31_avg_paths",
    "median_paths_per_entity":         "M31_median_paths",
    "std_paths_per_entity":            "M31_std_paths",
    "path_coverage_1":                 "M32_coverage_geq_1",
    "path_coverage_2":                 "M32_coverage_geq_2",
    "path_coverage_4":                 "M32_coverage_geq_4",
    "path_coverage_8":                 "M32_coverage_geq_8",
    "avg_path_length":                 "M33_avg_path_length",
    "median_path_length":              "M33_median_path_length",
    "avg_path_diversity":              "M34_avg_path_diversity",
    "avg_rel_types_in_paths":          "M35_avg_rel_types_in_paths",
    "reach_2hop":                      "M36_reachability_2hop",
    "reach_3hop":                      "M36_reachability_3hop",
    "reach_5hop":                      "M36_reachability_5hop",
    "reach_10hop":                     "M36_reachability_10hop",
    "num_connected_components":        "M37_num_connected_components",
    "lcc_size":                        "M38_lcc_size",
    "lcc_fraction":                    "M38_lcc_fraction",
    "num_scc":                         "M39_num_scc",
    "lscc_size":                       "M40_lscc_size",
    "lscc_fraction":                   "M40_lscc_fraction",
    "approx_diameter":                 "M41_diameter",
    "avg_shortest_path":               "M42_avg_shortest_path",
    "std_shortest_path":               "M42_std_shortest_path",
    "avg_clustering_coefficient":      "M43_avg_clustering",
    "transitivity":                    "M44_transitivity",
    "untyped_reciprocity":             "M45_untyped_reciprocity",
    "typed_reciprocity":               "M45_typed_reciprocity",
    "degree_assortativity":            "M46_assortativity",
    "mean_rel_avg_degree":             "M47_mean_rel_avg_degree",
    "std_rel_avg_degree":              "M47_std_rel_avg_degree",
    "cooccurrence_density":            "M48_frac_nonzero_offdiag",
    "cooccurrence_entropy":            "M49_cooccurrence_entropy",
    "rel_graph_density":               "M50_relation_graph_density",
    "degree_entropy":                  "M57_degree_entropy",
    "H_t_given_r":                     "M58_H_tail_given_rel",
    "H_h_given_r":                     "M59_H_head_given_rel",
    "H_r_given_ht":                    "M60_H_rel_given_head_tail",
    "avg_multiplicity":                "M61_avg_multiplicity",
    "max_multiplicity":                "M61_max_multiplicity",
    "fraction_multi_relation_pairs":   "M61_fraction_multi",
    "metapaths_2_count":               "M62_num_metapaths_2",
    "metapaths_2_ratio":               "M62_metapath_2_ratio",
    "metapaths_3_count":               "M62_num_metapaths_3",
    "metapaths_3_ratio":               "M62_metapath_3_ratio",
    "avg_symmetry":                    "M63_avg_symmetry",
    "weighted_symmetry":               "M63_weighted_symmetry",
}

INTEGER_COLUMNS = {
    "num_entities", "num_relations", "train_triples", "valid_triples",
    "test_triples", "min_degree", "max_degree", "rare_relations_01pct",
    "rare_relations_05pct", "rare_relations_1pct", "largest_equiv_class",
    "num_connected_components", "lcc_size", "num_scc", "lscc_size",
    "approx_diameter", "max_multiplicity", "metapaths_2_count",
    "metapaths_3_count",
}


def generate_file1(raw: dict[str, dict]) -> pd.DataFrame:
    """FILE 1: summary_metrics_for_paper.csv"""
    paper_col_order = ["dataset"] + list(COLUMN_MAP.keys())
    rows = []
    for ds in DATASETS:
        ds_data = raw[ds]
        row = {"dataset": ds}
        for paper_col, json_key in COLUMN_MAP.items():
            val = ds_data.get(json_key)
            if val is None:
                row[paper_col] = ""
            elif paper_col in INTEGER_COLUMNS:
                row[paper_col] = int(val)
            else:
                row[paper_col] = val
        rows.append(row)

    df = pd.DataFrame(rows, columns=paper_col_order)
    out_path = PAPER_DIR / "summary_metrics_for_paper.csv"
    df.to_csv(out_path, index=False)
    print(f"[FILE 1] Wrote {len(df)} rows × {len(df.columns)} cols → {out_path}")
    return df


# ── FILE 2: Correlation table ──

# configs/structure/performance.yaml
PERF_DATA = {m: {ds: v[m] for ds, v in performance.transductive().items()} for m in ("LP_MRR", "LP_H10", "Effi_LP")}

STRUCTURAL_METRICS_SPEC = [
    ("M2_num_relations",       "num_relations",               "Number of Relations",                "Basic"),
    ("M5_r_to_e_ratio",        "r_to_e_ratio",                "Relation-to-Entity Ratio",           "Basic"),
    ("M6_t_to_e_ratio",        "t_to_e_ratio",                "Triple-to-Entity Ratio",             "Basic"),
    ("M7_avg_total_degree",    "avg_total_degree",            "Avg. Total Degree",                  "Degree"),
    ("M8_median_total",        "median_total_degree",         "Median Total Degree",                "Degree"),
    ("M10_skewness",           "degree_skewness",             "Degree Skewness",                    "Degree"),
    ("M13_low_deg_5",          "low_deg_5",                   "Low-Degree Fraction (d≤5)",          "Degree"),
    ("M17_relation_entropy",   "relation_entropy",            "Relation Entropy (bits)",            "Relational Diversity"),
    ("M18_normalised_entropy", "normalised_relation_entropy", "Normalised Relation Entropy",        "Relational Diversity"),
    ("M19_relation_gini",      "relation_gini",               "Relation Gini Coefficient",          "Relational Diversity"),
    ("M20_top2_conc",          "top2_concentration",          "Top-2 Relation Concentration",       "Relational Diversity"),
    ("M22_avg_rel_diversity",  "avg_rel_diversity_per_entity","Avg. Relation Diversity per Entity", "Relational Diversity"),
    ("M25_unique_fraction",    "unique_context_fraction",     "Unique Context Fraction",            "Context Uniqueness"),
    ("M27_largest_class",      "largest_equiv_class",         "Largest Equivalence Class",          "Context Uniqueness"),
    ("M28_avg_class_size",     "avg_equiv_class_size",        "Avg. Equivalence Class Size",        "Context Uniqueness"),
    ("M31_avg_paths",          "avg_paths_per_entity",        "Avg. Paths per Entity",              "Path Richness"),
    ("M33_avg_path_length",    "avg_path_length",             "Avg. Path Length (nodes)",            "Path Richness"),
    ("M34_avg_path_diversity", "avg_path_diversity",          "Avg. Path Diversity",                "Path Richness"),
    ("M35_avg_rel_in_paths",   "avg_rel_types_in_paths",     "Avg. Relation Types in Paths",       "Path Richness"),
    ("M38_lcc_fraction",       "lcc_fraction",                "LCC Fraction",                       "Connectivity"),
    ("M43_avg_clustering",     "avg_clustering_coefficient",  "Avg. Clustering Coefficient",        "Connectivity"),
    ("M44_transitivity",       "transitivity",                "Transitivity",                       "Connectivity"),
    ("M45_untyped_reciprocity","untyped_reciprocity",         "Reciprocity (untyped)",              "Connectivity"),
    ("M46_assortativity",      "degree_assortativity",        "Degree Assortativity",               "Connectivity"),
    ("M50_rel_graph_density",  "rel_graph_density",           "Relation Graph Density",             "Relation Structure"),
    ("M57_degree_entropy",     "degree_entropy",              "Degree Distribution Entropy",        "Information-Theoretic"),
    ("M58_H_t_given_r",        "H_t_given_r",                "H(Tail|Relation)",                   "Information-Theoretic"),
    ("M60_H_r_given_ht",       "H_r_given_ht",               "H(Relation|Head,Tail)",              "Information-Theoretic"),
    ("M62_metapaths_2_ratio",  "metapaths_2_ratio",           "Meta-path Coverage (len 2)",         "Multi-Relational"),
]

PERF_METRICS_SPEC = [
    ("LP_MRR",  "Link Prediction MRR"),
    ("LP_H10",  "Link Prediction Hits@10"),
    ("Effi_LP", "Efficiency (MRR / #Params)"),
]


def generate_file2(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """FILE 2: correlation_table_for_paper.csv"""
    perf_df = pd.DataFrame(PERF_DATA)
    perf_df.index.name = "dataset"

    rows = []
    for metric_id, col_name, display_name, category in STRUCTURAL_METRICS_SPEC:
        if col_name not in metrics_df.columns:
            print(f"  WARNING: column '{col_name}' not found, skipping {metric_id}")
            continue

        for perf_id, perf_display in PERF_METRICS_SPEC:
            common_idx = metrics_df.index.intersection(perf_df.index)
            x = metrics_df.loc[common_idx, col_name].astype(float)
            y = perf_df.loc[common_idx, perf_id].astype(float)

            valid = x.notna() & y.notna()
            xv, yv = x[valid].values, y[valid].values
            n = len(xv)
            if n < 3:
                continue

            r_p, p_p = pearsonr(xv, yv)
            r_s, p_s = spearmanr(xv, yv)
            tau, p_t = kendalltau(xv, yv)

            abs_r = abs(r_p)
            if abs_r >= 0.5:
                effect = "large"
            elif abs_r >= 0.3:
                effect = "medium"
            elif abs_r >= 0.1:
                effect = "small"
            else:
                effect = "negligible"

            signs = [np.sign(r_p), np.sign(r_s), np.sign(tau)]
            nonzero = [s for s in signs if s != 0]
            sign_agree = len(set(nonzero)) <= 1 if nonzero else True

            rows.append({
                "structural_metric":              metric_id,
                "structural_metric_display_name": display_name,
                "structural_metric_category":     category,
                "performance_metric":             perf_id,
                "performance_metric_display_name": perf_display,
                "n":                              n,
                "pearson_r":                      round(r_p, 4),
                "pearson_p":                      round(p_p, 4),
                "spearman_rho":                   round(r_s, 4),
                "spearman_p":                     round(p_s, 4),
                "kendall_tau":                    round(tau, 4),
                "kendall_p":                      round(p_t, 4),
                "effect_size_category":           effect,
                "sign_agreement":                 sign_agree,
            })

    df = pd.DataFrame(rows)
    out_path = PAPER_DIR / "correlation_table_for_paper.csv"
    df.to_csv(out_path, index=False)
    print(f"[FILE 2] Wrote {len(df)} rows → {out_path}")
    return df


# ── FILE 3: Inductive splits ──

# configs/structure/performance.yaml, keyed like inductive_metrics.csv ("FB15k-237_v1")
H10_VALUES = {k.replace("_V", "_v"): v for k, v in performance.inductive_h10().items()}


def _compute_normalised_entropy(triples_array: np.ndarray, num_relations: int) -> float:
    """Compute normalised relation entropy from a (N, 3) array of (h, r, t) triples."""
    if num_relations <= 1:
        return 1.0
    rels = triples_array[:, 1]
    counts = np.bincount(rels, minlength=num_relations).astype(float)
    counts = counts[counts > 0]
    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log2(probs))
    max_entropy = np.log2(num_relations)
    return entropy / max_entropy if max_entropy > 0 else 0.0


def _try_compute_inductive_entropy() -> dict[str, tuple[float, float]]:
    """
    Attempt to load PyKEEN inductive datasets and compute normalised
    relation entropy for each split's train and inference graphs.
    Returns {split_name: (entropy_train, entropy_infer)} or empty dict on failure.
    """
    try:
        from pykeen.datasets.inductive.ilp_teru import (
            InductiveFB15k237,
            InductiveWN18RR,
        )
    except ImportError:
        print("  PyKEEN inductive datasets not available; skipping entropy computation")
        return {}

    result = {}
    configs = [
        ("FB15k-237", InductiveFB15k237, ["v1", "v2", "v3", "v4"]),
        ("WN18RR", InductiveWN18RR, ["v1", "v2", "v3", "v4"]),
    ]
    for parent, cls, versions in configs:
        for ver in versions:
            split_name = f"{parent}_{ver}"
            try:
                ds = cls(version=ver)
                train_triples = ds.transductive_training.mapped_triples.numpy()
                infer_triples = ds.inductive_inference.mapped_triples.numpy()
                n_rel_train = int(train_triples[:, 1].max()) + 1
                n_rel_infer = int(infer_triples[:, 1].max()) + 1
                ent_train = _compute_normalised_entropy(train_triples, n_rel_train)
                ent_infer = _compute_normalised_entropy(infer_triples, n_rel_infer)
                result[split_name] = (ent_train, ent_infer)
                print(f"    {split_name}: norm_entropy train={ent_train:.4f}, infer={ent_infer:.4f}")
            except Exception as e:
                print(f"    {split_name}: failed ({e})")
    return result


def generate_file3() -> pd.DataFrame:
    """FILE 3: inductive_metrics_for_paper.csv"""
    ind_path = OUTPUT_DIR / "inductive_splits" / "inductive_metrics.csv"
    ind = pd.read_csv(ind_path)

    print("  Attempting to compute normalised relation entropy for inductive splits...")
    entropy_map = _try_compute_inductive_entropy()

    rows = []
    for _, r in ind.iterrows():
        split = r["split"]
        parts = split.rsplit("_", 1)
        parent_dataset = parts[0]
        split_version = parts[1].upper()

        ent_train, ent_infer = entropy_map.get(split, ("", ""))

        rows.append({
            "split":                          split.replace("_v", "_V"),
            "parent_dataset":                 parent_dataset,
            "split_version":                  split_version,
            "H10":                            H10_VALUES.get(split, ""),
            "num_train_entities":             int(r["M55_train_num_entities"]),
            "num_train_triples":              int(r["M55_train_num_triples"]),
            "num_infer_entities":             int(r["M55_inf_num_entities"]),
            "num_infer_triples":              int(r["M55_inf_num_triples"]),
            "num_relations_train":            int(r["M55_train_num_relations"]),
            "num_relations_infer":            int(r["M55_inf_num_relations"]),
            "relation_overlap":               r["M53_relation_overlap_frac_of_inf"],
            "avg_degree_train":               round(r["M55_train_avg_degree"], 4),
            "avg_degree_infer":               round(r["M55_inf_avg_degree"], 4),
            "degree_ratio":                   round(r["M55_inf_avg_degree"] / r["M55_train_avg_degree"], 4)
                                              if r["M55_train_avg_degree"] > 0 else "",
            "unique_context_fraction_train":  round(r["M56_train_unique_context_fraction"], 4),
            "unique_context_fraction_infer":  round(r["M56_inf_unique_context_fraction"], 4),
            "normalised_entropy_train":       round(ent_train, 4) if isinstance(ent_train, float) else "",
            "normalised_entropy_infer":       round(ent_infer, 4) if isinstance(ent_infer, float) else "",
        })

    df = pd.DataFrame(rows)
    out_path = PAPER_DIR / "inductive_metrics_for_paper.csv"
    df.to_csv(out_path, index=False)
    print(f"[FILE 3] Wrote {len(df)} rows × {len(df.columns)} cols → {out_path}")
    return df


# ── Verification ──

def verify(file1: pd.DataFrame, file2: pd.DataFrame, file3: pd.DataFrame):
    print("\n" + "=" * 60)
    print("VERIFICATION CHECKLIST")
    print("=" * 60)

    # File 1
    print("\n--- File 1: summary_metrics_for_paper.csv ---")
    assert len(file1) == 4, f"Expected 4 rows, got {len(file1)}"
    print(f"  [OK] 4 rows: {list(file1['dataset'])}")

    ucf = file1.set_index("dataset")["unique_context_fraction"]
    print(f"  unique_context_fraction: FB={ucf['FB15k-237']:.4f} (≈0.92), "
          f"WN={ucf['WN18RR']:.4f} (≈0.08), "
          f"CoDEx={ucf['CoDEx-Large']:.4f} (≈0.464), "
          f"YAGO={ucf['YAGO3-10']:.4f} (≈0.198)")

    atd = file1.set_index("dataset")["avg_total_degree"]
    print(f"  avg_total_degree: FB={atd['FB15k-237']:.2f} (≈37.5), "
          f"WN={atd['WN18RR']:.2f} (≈4.28), "
          f"CoDEx={atd['CoDEx-Large']:.2f} (≈14.14), "
          f"YAGO={atd['YAGO3-10']:.2f} (≈17.52)")

    # File 2
    print("\n--- File 2: correlation_table_for_paper.csv ---")
    assert len(file2) == 87, f"Expected 87 rows, got {len(file2)}"
    print(f"  [OK] 87 rows (29 metrics × 3 perf metrics)")

    m18_mrr = file2[(file2["structural_metric"] == "M18_normalised_entropy") &
                     (file2["performance_metric"] == "LP_MRR")]
    if len(m18_mrr) == 1:
        kt = m18_mrr.iloc[0]["kendall_tau"]
        print(f"  M18 vs LP_MRR kendall_tau = {kt} (expected 1.0)")

    m31_mrr = file2[(file2["structural_metric"] == "M31_avg_paths") &
                     (file2["performance_metric"] == "LP_MRR")]
    if len(m31_mrr) == 1:
        kt = m31_mrr.iloc[0]["kendall_tau"]
        print(f"  M31 vs LP_MRR kendall_tau = {kt} (expected 1.0)")

    nan_count = file2[["pearson_r", "spearman_rho", "kendall_tau"]].isna().sum().sum()
    print(f"  NaN values in correlation columns: {nan_count} (expected 0)")

    # File 3
    print("\n--- File 3: inductive_metrics_for_paper.csv ---")
    assert len(file3) == 8, f"Expected 8 rows, got {len(file3)}"
    print(f"  [OK] 8 rows: {list(file3['split'])}")

    h10_check = file3.set_index("split")["H10"]
    expected_h10 = performance.inductive_h10()
    h10_ok = all(h10_check[k] == v for k, v in expected_h10.items())
    print(f"  H10 values match expected: {h10_ok}")

    wn_rels = file3[file3["parent_dataset"] == "WN18RR"].set_index("split")["num_relations_infer"]
    print(f"  WN18RR num_relations_infer: {list(wn_rels.values)} (expected [9, 10, 11, 9])")

    print("\n" + "=" * 60)
    print(f"DONE — all files written to {PAPER_DIR}/")
    print("=" * 60)


def main():
    global OUTPUT_DIR, PAPER_DIR
    parser = argparse.ArgumentParser(description="Paper CSVs of the structure analysis")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="default: <repo>/results/structure")
    args = parser.parse_args()
    OUTPUT_DIR = Path(args.output_dir)
    PAPER_DIR = OUTPUT_DIR / "paper_outputs"
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    # Load raw data
    with open(OUTPUT_DIR / "summary_metrics.json") as f:
        raw = json.load(f)

    # FILE 1
    print("\n>>> Generating FILE 1: summary_metrics_for_paper.csv")
    file1 = generate_file1(raw)

    # FILE 2 (uses file1 as input)
    print("\n>>> Generating FILE 2: correlation_table_for_paper.csv")
    metrics_df = file1.set_index("dataset")
    for col in metrics_df.columns:
        metrics_df[col] = pd.to_numeric(metrics_df[col], errors="coerce")
    file2 = generate_file2(metrics_df)

    # FILE 3
    print("\n>>> Generating FILE 3: inductive_metrics_for_paper.csv")
    file3 = generate_file3()

    # Verify
    verify(file1, file2, file3)


if __name__ == "__main__":
    main()
