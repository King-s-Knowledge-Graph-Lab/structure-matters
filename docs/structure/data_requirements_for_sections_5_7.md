# Data Requirements for Paper Sections 5 and 7

## Overview

This document specifies the exact CSV files needed to write Sections 5 and 7 of the TGDK paper. There are **3 files** to generate. All should be saved as CSV with the exact column names specified.

---

## FILE 1: `summary_metrics_for_paper.csv`

**Purpose**: Provides the raw structural metric values per dataset. Used to write the tables in Section 5.

**Format**: One row per dataset (4 rows), one column per metric.

**Required columns** (use these exact column names):

```
dataset,
num_entities, num_relations, train_triples, valid_triples, test_triples,
full_density, simple_density,
r_to_e_ratio, t_to_e_ratio,
avg_in_degree, avg_out_degree, avg_total_degree,
median_in_degree, median_out_degree, median_total_degree,
std_total_degree, degree_skewness, degree_kurtosis,
min_degree, max_degree,
low_deg_1, low_deg_2, low_deg_3, low_deg_5, low_deg_10,
powerlaw_alpha, powerlaw_xmin,
in_out_degree_pearson, in_out_degree_spearman,
relation_entropy, normalised_relation_entropy,
relation_gini,
top1_concentration, top2_concentration, top3_concentration, top5_concentration, top10_concentration,
rare_relations_01pct, rare_relations_05pct, rare_relations_1pct,
avg_rel_diversity_per_entity, median_rel_diversity_per_entity, std_rel_diversity_per_entity,
avg_norm_rel_diversity,
distinct_context_fraction, unique_context_fraction,
largest_equiv_class, avg_equiv_class_size, equiv_class_entropy,
largest_class_fraction,
avg_paths_per_entity, median_paths_per_entity, std_paths_per_entity,
path_coverage_1, path_coverage_2, path_coverage_4, path_coverage_8,
avg_path_length, median_path_length,
avg_path_diversity,
avg_rel_types_in_paths,
reach_2hop, reach_3hop, reach_5hop, reach_10hop,
num_connected_components, lcc_size, lcc_fraction,
num_scc, lscc_size, lscc_fraction,
approx_diameter, avg_shortest_path, std_shortest_path,
avg_clustering_coefficient, transitivity,
untyped_reciprocity, typed_reciprocity,
degree_assortativity,
mean_rel_avg_degree, std_rel_avg_degree,
cooccurrence_density, cooccurrence_entropy,
rel_graph_density,
degree_entropy,
H_t_given_r, H_h_given_r, H_r_given_ht,
avg_multiplicity, max_multiplicity, fraction_multi_relation_pairs,
metapaths_2_count, metapaths_2_ratio,
metapaths_3_count, metapaths_3_ratio,
avg_symmetry, weighted_symmetry
```

**Expected rows**:
```
FB15k-237, ...
WN18RR, ...
CoDEx-Large, ...
YAGO3-10, ...
```

**Notes**:
- Use full numeric precision (at least 4 decimal places for floats).
- For integer metrics (num_entities, num_relations, etc.), use integers without decimals.
- If a metric was not computed or is not applicable, use an empty cell (not "N/A" or "None").

---

## FILE 2: `correlation_table_for_paper.csv`

**Purpose**: The full correlation matrix between structural metrics and performance metrics. This becomes the main table in Section 7.

**Format**: One row per (structural_metric, performance_metric) combination. Include ALL combinations, not just the top ones.

**Required columns**:

```
structural_metric,
structural_metric_display_name,
structural_metric_category,
performance_metric,
performance_metric_display_name,
n,
pearson_r,
pearson_p,
spearman_rho,
spearman_p,
kendall_tau,
kendall_p,
effect_size_category,
sign_agreement
```

**Column definitions**:

| Column | Description | Example |
|---|---|---|
| `structural_metric` | The metric ID from the specification | `M18_normalised_entropy` |
| `structural_metric_display_name` | Human-readable name for the paper | `Normalised Relation Entropy` |
| `structural_metric_category` | Which category the metric belongs to (for grouping in the table) | One of: `Basic`, `Degree`, `Relational Diversity`, `Context Uniqueness`, `Path Richness`, `Connectivity`, `Relation Structure`, `Information-Theoretic`, `Multi-Relational` |
| `performance_metric` | The performance metric ID | `LP_MRR` |
| `performance_metric_display_name` | Human-readable name | `Link Prediction MRR` |
| `n` | Number of datasets with valid values for both metrics | `4` |
| `pearson_r` | Pearson correlation coefficient | `0.9187` |
| `pearson_p` | Pearson two-sided p-value | `0.0813` |
| `spearman_rho` | Spearman rank correlation | `1.0` |
| `spearman_p` | Spearman two-sided p-value | `0.0833` |
| `kendall_tau` | Kendall's tau-b | `1.0` |
| `kendall_p` | Kendall two-sided p-value | `0.0833` |
| `effect_size_category` | Cohen's convention on |pearson_r| | One of: `large` (≥0.5), `medium` (≥0.3), `small` (≥0.1), `negligible` (<0.1) |
| `sign_agreement` | Whether all three correlations agree in sign | `True` or `False` |

**Required structural metrics to include** (these are the ones that will appear in the paper table — compute correlations for ALL of these against ALL three performance metrics):

| Metric ID | Display Name | Category |
|---|---|---|
| `M2_num_relations` | Number of Relations | Basic |
| `M5_r_to_e_ratio` | Relation-to-Entity Ratio | Basic |
| `M6_t_to_e_ratio` | Triple-to-Entity Ratio | Basic |
| `M7_avg_total_degree` | Avg. Total Degree | Degree |
| `M8_median_total` | Median Total Degree | Degree |
| `M10_skewness` | Degree Skewness | Degree |
| `M13_low_deg_5` | Low-Degree Fraction (d≤5) | Degree |
| `M17_relation_entropy` | Relation Entropy (bits) | Relational Diversity |
| `M18_normalised_entropy` | Normalised Relation Entropy | Relational Diversity |
| `M19_relation_gini` | Relation Gini Coefficient | Relational Diversity |
| `M20_top2_conc` | Top-2 Relation Concentration | Relational Diversity |
| `M22_avg_rel_diversity` | Avg. Relation Diversity per Entity | Relational Diversity |
| `M25_unique_fraction` | Unique Context Fraction | Context Uniqueness |
| `M27_largest_class` | Largest Equivalence Class | Context Uniqueness |
| `M28_avg_class_size` | Avg. Equivalence Class Size | Context Uniqueness |
| `M31_avg_paths` | Avg. Paths per Entity | Path Richness |
| `M33_avg_path_length` | Avg. Path Length (nodes) | Path Richness |
| `M34_avg_path_diversity` | Avg. Path Diversity | Path Richness |
| `M35_avg_rel_in_paths` | Avg. Relation Types in Paths | Path Richness |
| `M38_lcc_fraction` | LCC Fraction | Connectivity |
| `M43_avg_clustering` | Avg. Clustering Coefficient | Connectivity |
| `M44_transitivity` | Transitivity | Connectivity |
| `M45_untyped_reciprocity` | Reciprocity (untyped) | Connectivity |
| `M46_assortativity` | Degree Assortativity | Connectivity |
| `M50_rel_graph_density` | Relation Graph Density | Relation Structure |
| `M57_degree_entropy` | Degree Distribution Entropy | Information-Theoretic |
| `M58_H_t_given_r` | H(Tail\|Relation) | Information-Theoretic |
| `M60_H_r_given_ht` | H(Relation\|Head,Tail) | Information-Theoretic |
| `M62_metapaths_2_ratio` | Meta-path Coverage (length 2) | Multi-Relational |

**Required performance metrics**:

| Performance Metric ID | Display Name |
|---|---|
| `LP_MRR` | Link Prediction MRR |
| `LP_H10` | Link Prediction Hits@10 |
| `Effi_LP` | Efficiency (MRR / #Params) |

**Total rows expected**: 29 structural metrics × 3 performance metrics = **87 rows**.

**Direction handling**: When computing correlations, use the RAW metric values (do NOT flip signs for "lower is better" metrics). The sign of the correlation itself tells us the direction. For example, Gini should correlate negatively with MRR (higher Gini = worse performance), so pearson_r should be negative. The `sign_agreement` column checks if all three correlation measures agree in sign.

**Computation code**:

```python
from scipy.stats import pearsonr, spearmanr, kendalltau
import pandas as pd
import numpy as np

# Load your summary_metrics_for_paper.csv
metrics_df = pd.read_csv("summary_metrics_for_paper.csv", index_col="dataset")

# Performance data
perf_data = {
    "LP_MRR": {"FB15k-237": 0.216, "WN18RR": 0.069, "CoDEx-Large": 0.144, "YAGO3-10": 0.060},
    "LP_H10": {"FB15k-237": 0.350, "WN18RR": 0.124, "CoDEx-Large": 0.240, "YAGO3-10": 0.093},
    "Effi_LP": {"FB15k-237": 1.03, "WN18RR": 0.10, "CoDEx-Large": 0.21, "YAGO3-10": 0.25},
}
perf_df = pd.DataFrame(perf_data)

# Structural metrics to include (map metric_id -> column_name in metrics_df, display_name, category)
structural_metrics_spec = [
    ("M2_num_relations", "num_relations", "Number of Relations", "Basic"),
    ("M5_r_to_e_ratio", "r_to_e_ratio", "Relation-to-Entity Ratio", "Basic"),
    ("M6_t_to_e_ratio", "t_to_e_ratio", "Triple-to-Entity Ratio", "Basic"),
    ("M7_avg_total_degree", "avg_total_degree", "Avg. Total Degree", "Degree"),
    ("M8_median_total", "median_total_degree", "Median Total Degree", "Degree"),
    ("M10_skewness", "degree_skewness", "Degree Skewness", "Degree"),
    ("M13_low_deg_5", "low_deg_5", "Low-Degree Fraction (d≤5)", "Degree"),
    ("M17_relation_entropy", "relation_entropy", "Relation Entropy (bits)", "Relational Diversity"),
    ("M18_normalised_entropy", "normalised_relation_entropy", "Normalised Relation Entropy", "Relational Diversity"),
    ("M19_relation_gini", "relation_gini", "Relation Gini Coefficient", "Relational Diversity"),
    ("M20_top2_conc", "top2_concentration", "Top-2 Relation Concentration", "Relational Diversity"),
    ("M22_avg_rel_diversity", "avg_rel_diversity_per_entity", "Avg. Relation Diversity per Entity", "Relational Diversity"),
    ("M25_unique_fraction", "unique_context_fraction", "Unique Context Fraction", "Context Uniqueness"),
    ("M27_largest_class", "largest_equiv_class", "Largest Equivalence Class", "Context Uniqueness"),
    ("M28_avg_class_size", "avg_equiv_class_size", "Avg. Equivalence Class Size", "Context Uniqueness"),
    ("M31_avg_paths", "avg_paths_per_entity", "Avg. Paths per Entity", "Path Richness"),
    ("M33_avg_path_length", "avg_path_length", "Avg. Path Length (nodes)", "Path Richness"),
    ("M34_avg_path_diversity", "avg_path_diversity", "Avg. Path Diversity", "Path Richness"),
    ("M35_avg_rel_in_paths", "avg_rel_types_in_paths", "Avg. Relation Types in Paths", "Path Richness"),
    ("M38_lcc_fraction", "lcc_fraction", "LCC Fraction", "Connectivity"),
    ("M43_avg_clustering", "avg_clustering_coefficient", "Avg. Clustering Coefficient", "Connectivity"),
    ("M44_transitivity", "transitivity", "Transitivity", "Connectivity"),
    ("M45_untyped_reciprocity", "untyped_reciprocity", "Reciprocity (untyped)", "Connectivity"),
    ("M46_assortativity", "degree_assortativity", "Degree Assortativity", "Connectivity"),
    ("M50_rel_graph_density", "rel_graph_density", "Relation Graph Density", "Relation Structure"),
    ("M57_degree_entropy", "degree_entropy", "Degree Distribution Entropy", "Information-Theoretic"),
    ("M58_H_t_given_r", "H_t_given_r", "H(Tail|Relation)", "Information-Theoretic"),
    ("M60_H_r_given_ht", "H_r_given_ht", "H(Relation|Head,Tail)", "Information-Theoretic"),
    ("M62_metapaths_2_ratio", "metapaths_2_ratio", "Meta-path Coverage (len 2)", "Multi-Relational"),
]

perf_metrics_spec = [
    ("LP_MRR", "Link Prediction MRR"),
    ("LP_H10", "Link Prediction Hits@10"),
    ("Effi_LP", "Efficiency (MRR / #Params)"),
]

rows = []
for metric_id, col_name, display_name, category in structural_metrics_spec:
    if col_name not in metrics_df.columns:
        print(f"WARNING: {col_name} not found in metrics_df. Skipping {metric_id}.")
        continue
    
    for perf_id, perf_display in perf_metrics_spec:
        # Get aligned values
        common_idx = metrics_df.index.intersection(perf_df.index)
        x = metrics_df.loc[common_idx, col_name].astype(float)
        y = perf_df.loc[common_idx, perf_id].astype(float)
        
        # Drop any NaN pairs
        valid_mask = x.notna() & y.notna()
        x_valid = x[valid_mask].values
        y_valid = y[valid_mask].values
        n = len(x_valid)
        
        if n < 3:
            continue
        
        r_p, p_p = pearsonr(x_valid, y_valid)
        r_s, p_s = spearmanr(x_valid, y_valid)
        tau, p_t = kendalltau(x_valid, y_valid)
        
        # Effect size (Cohen 1988)
        abs_r = abs(r_p)
        if abs_r >= 0.5: effect = "large"
        elif abs_r >= 0.3: effect = "medium"
        elif abs_r >= 0.1: effect = "small"
        else: effect = "negligible"
        
        # Sign agreement
        signs = [np.sign(r_p), np.sign(r_s), np.sign(tau)]
        # Handle zero correlations
        nonzero_signs = [s for s in signs if s != 0]
        sign_agree = len(set(nonzero_signs)) <= 1 if nonzero_signs else True
        
        rows.append({
            "structural_metric": metric_id,
            "structural_metric_display_name": display_name,
            "structural_metric_category": category,
            "performance_metric": perf_id,
            "performance_metric_display_name": perf_display,
            "n": n,
            "pearson_r": round(r_p, 4),
            "pearson_p": round(p_p, 4),
            "spearman_rho": round(r_s, 4),
            "spearman_p": round(p_s, 4),
            "kendall_tau": round(tau, 4),
            "kendall_p": round(p_t, 4),
            "effect_size_category": effect,
            "sign_agreement": sign_agree,
        })

corr_table = pd.DataFrame(rows)
corr_table.to_csv("correlation_table_for_paper.csv", index=False)
print(f"Generated {len(corr_table)} rows")
```

---

## FILE 3: `inductive_metrics_for_paper.csv`

**Purpose**: Structural metrics and performance for the 8 inductive splits. Used for the inductive analysis in Section 7.

**Format**: One row per inductive split (8 rows).

**Required columns**:

```
split,
parent_dataset,
split_version,
H10,
num_train_entities, num_train_triples,
num_infer_entities, num_infer_triples,
num_relations_train, num_relations_infer,
relation_overlap,
avg_degree_train, avg_degree_infer, degree_ratio,
unique_context_fraction_train, unique_context_fraction_infer,
normalised_entropy_train, normalised_entropy_infer
```

**Expected rows**:
```
FB15k-237_V1, FB15k-237, V1, 0.847, ...
FB15k-237_V2, FB15k-237, V2, 0.961, ...
FB15k-237_V3, FB15k-237, V3, 0.954, ...
FB15k-237_V4, FB15k-237, V4, 0.959, ...
WN18RR_V1, WN18RR, V1, 0.576, ...
WN18RR_V2, WN18RR, V2, 0.594, ...
WN18RR_V3, WN18RR, V3, 0.651, ...
WN18RR_V4, WN18RR, V4, 0.562, ...
```

**Notes**: If you did not compute all inductive split metrics, include whatever you have. The most critical columns are: `num_relations_infer`, `unique_context_fraction_infer`, `avg_degree_infer`, and `H10`. With those I can write the inductive analysis.

---

## VERIFICATION CHECKLIST

Before submitting the files, verify:

1. **File 1 (summary_metrics_for_paper.csv)**:
   - [ ] Has exactly 4 rows (FB15k-237, WN18RR, CoDEx-Large, YAGO3-10)
   - [ ] All numeric columns parse as numbers (no stray text)
   - [ ] The `unique_context_fraction` values roughly match the original paper appendix (FB15k-237 ≈ 0.92, WN18RR ≈ 0.08, CoDEx-Large ≈ 0.464, YAGO3-10 ≈ 0.198)
   - [ ] `avg_total_degree` roughly matches the appendix (FB15k-237 ≈ 37.5, WN18RR ≈ 4.28, CoDEx-Large ≈ 14.14, YAGO3-10 ≈ 17.52)

2. **File 2 (correlation_table_for_paper.csv)**:
   - [ ] Has 87 rows (29 metrics × 3 performance metrics)
   - [ ] All Kendall tau values for M18_normalised_entropy vs LP_MRR = 1.0 (matching key_findings.json)
   - [ ] All Kendall tau values for M31_avg_paths vs LP_MRR = 1.0 (matching key_findings.json)
   - [ ] No NaN values in correlation columns
   - [ ] `sign_agreement` is True for the strongest correlations

3. **File 3 (inductive_metrics_for_paper.csv)**:
   - [ ] Has 8 rows
   - [ ] H10 values match: FB15k-237 V1=0.847, V2=0.961, V3=0.954, V4=0.959; WN18RR V1=0.576, V2=0.594, V3=0.651, V4=0.562
   - [ ] `num_relations_infer` for WN18RR splits should be 9, 10, 11, 9 (matching original paper)
