# Linking Graph Structural Properties to Model Performance: Analysis Specification

## Document Purpose

This document provides exhaustive instructions for correlating the structural properties computed in the metrics specification (v2) with the entity-agnostic framework's performance. The analysis supports **Section 7 ("Linking Performance to Graph Structure")** of the TGDK paper.

**Critical constraint**: With only 4 transductive datasets and 8 inductive splits, classical large-sample inferential statistics (e.g., testing whether a Pearson correlation is "statistically significant") are of limited value. The paper must acknowledge this honestly. The analysis strategy therefore combines multiple complementary approaches: rank-based analysis, normalised comparison profiles, effect-size reporting, visual evidence, and structured qualitative case studies. Together these build a *convergent evidence* argument rather than relying on any single statistical test.

---

## Academic References

| Reference Key | Citation | Used For |
|---|---|---|
| [Spearman1904] | Spearman, C. (1904). The proof and measurement of association between two things. *The American Journal of Psychology*, 15(1), 72–101. | Spearman rank correlation |
| [Pearson1895] | Pearson, K. (1895). Notes on regression and inheritance in the case of two parents. *Proceedings of the Royal Society of London*, 58, 240–242. | Pearson correlation |
| [Kendall1938] | Kendall, M. G. (1938). A new measure of rank correlation. *Biometrika*, 30(1/2), 81–93. | Kendall's τ |
| [Cohen1988] | Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Lawrence Erlbaum. | Effect size conventions |
| [Efron1993] | Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall. | Bootstrap confidence intervals |
| [Breiman2001] | Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32. | Feature importance (if using inductive data) |
| [Fisher1921] | Fisher, R. A. (1921). On the probable error of a coefficient of correlation deduced from a small sample. *Metron*, 1, 3–32. | Fisher z-transformation for small samples |
| [Freedman1981] | Freedman, D., & Diaconis, P. (1981). On the histogram as a density estimator: L2 theory. *Zeitschrift für Wahrscheinlichkeitstheorie*, 57(4), 453–476. | Histogram binning |
| [Tukey1977] | Tukey, J. W. (1977). *Exploratory Data Analysis*. Addison-Wesley. | EDA philosophy, box plots, residual analysis |
| [Wickham2010] | Wickham, H. (2010). A layered grammar of graphics. *Journal of Computational and Graphical Statistics*, 19(1), 3–28. | Visualisation design principles |
| [Cleveland1993] | Cleveland, W. S. (1993). *Visualizing Data*. Hobart Press. | Small-multiple plots, visual correlation |
| [Sullivan2012] | Sullivan, G. M., & Feinn, R. (2012). Using effect size—or why the P value is not enough. *Journal of Graduate Medical Education*, 4(3), 279–282. | Reporting effect sizes over p-values |

### Required Libraries

```
pip install numpy scipy pandas matplotlib seaborn scikit-learn pingouin bootstrapped
```

| Library | Used For | Why |
|---|---|---|
| `scipy.stats.spearmanr` | Spearman rank correlation | Standard implementation with exact p-values for small n |
| `scipy.stats.pearsonr` | Pearson correlation | Returns statistic + p-value |
| `scipy.stats.kendalltau` | Kendall's τ | More robust than Spearman for small samples; handles ties |
| `scipy.stats.zscore` | Z-score normalisation | Standard implementation |
| `pingouin.corr` | Correlation with confidence intervals | Computes CI via bootstrapping, appropriate for n=4 |
| `sklearn.preprocessing.MinMaxScaler` | Min-max normalisation for radar charts | Standard scaler |
| `numpy`, `pandas` | Data manipulation | Foundation |
| `matplotlib`, `seaborn` | Visualisation | Publication-quality plots |

---

## Input Data

### Structural Metrics (from previous specification)

Load the computed metrics:

```python
import pandas as pd
import numpy as np

# The summary_metrics.csv has one row per dataset, columns for all scalar metrics
metrics_df = pd.read_csv("output/summary_metrics.csv", index_col="dataset")

# Expected datasets as rows:
# FB15k-237, WN18RR, CoDEx-Large, YAGO3-10
```

### Performance Data

```python
# Transductive performance (from the original paper)
transductive_perf = pd.DataFrame({
    "dataset": ["FB15k-237", "WN18RR", "CoDEx-Large", "YAGO3-10"],
    "LP_MRR":  [0.216, 0.069, 0.144, 0.060],
    "LP_H1":   [0.146, None,  0.100, None],
    "LP_H3":   [0.236, None,  0.159, None],
    "LP_H5":   [0.282, None,  0.193, None],
    "LP_H10":  [0.350, 0.124, 0.240, 0.093],
    "RP_MRR":  [0.972, 0.874, None,  None],
    "RP_H10":  [0.998, 0.999, None,  None],
    "Effi_LP": [1.03,  0.10,  0.21,  0.25],
    "params_M_LP": [0.21, 0.67, 0.68, 0.24],
}).set_index("dataset")

# Inductive performance (Hits@10, configs/structure/performance.yaml)
inductive_perf = pd.DataFrame({
    "split": [
        "FB15k-237_V1", "FB15k-237_V2", "FB15k-237_V3", "FB15k-237_V4",
        "WN18RR_V1", "WN18RR_V2", "WN18RR_V3", "WN18RR_V4",
    ],
    "H10": [0.847, 0.961, 0.954, 0.959, 0.576, 0.594, 0.651, 0.562],
    "parent_dataset": [
        "FB15k-237", "FB15k-237", "FB15k-237", "FB15k-237",
        "WN18RR", "WN18RR", "WN18RR", "WN18RR",
    ],
}).set_index("split")
```

### Merging

```python
# Merge structural metrics with transductive performance
analysis_df = metrics_df.join(transductive_perf)

# For inductive splits, load their separate metrics
inductive_metrics_df = pd.read_csv("output/inductive_splits/all_splits_metrics.csv", index_col="split")
inductive_analysis_df = inductive_metrics_df.join(inductive_perf)
```

---

## Analysis Methods

The analysis is organised into 7 parts, each producing specific outputs that will feed into the paper. The methods escalate from simple to complex.

---

### PART 1: Rank Concordance Analysis (Primary Method for n=4)

**Why this is primary**: With only 4 datasets, rank-based methods are the most honest and robust approach. If every time the framework performs well, the structural metric also ranks high, that is strong evidence of association regardless of the exact functional form.

#### Method 1A: Dataset Performance Ranking

- **Definition**: Rank the 4 datasets by each performance metric (LP_MRR, LP_H10, RP_MRR, Effi_LP). Rank 1 = best.
- **Reference**: Ordinal ranking is the foundation of non-parametric statistics [Spearman1904].
- **Computation**:
  ```python
  perf_metrics = ["LP_MRR", "LP_H10", "Effi_LP"]
  
  perf_ranks = pd.DataFrame(index=analysis_df.index)
  for pm in perf_metrics:
      valid = analysis_df[pm].dropna()
      perf_ranks[f"{pm}_rank"] = valid.rank(ascending=False)  # Rank 1 = best
  ```
- **Output**: Table of performance rankings (4 datasets × n performance metrics).

#### Method 1B: Structural Metric Ranking

- **Definition**: For each structural metric, rank the 4 datasets. For metrics where higher is expected to be better (e.g., relation entropy, context uniqueness), rank descending. For metrics where lower is expected to be better (e.g., Gini, largest equivalence class), rank ascending.
- **Implementation**:
  ```python
  # Define expected direction for each metric
  # "higher_better" = higher metric value should correlate with higher performance
  # "lower_better" = lower metric value should correlate with higher performance
  metric_directions = {
      "M2_num_relations": "higher_better",
      "M5_r_to_e_ratio": "higher_better",
      "M6_t_to_e_ratio": "higher_better",
      "M7_avg_total_degree": "higher_better",
      "M8_median_total": "higher_better",
      "M10_skewness": "lower_better",        # less skew = more uniform = better
      "M13_low_deg_1": "lower_better",        # fewer low-degree entities = better
      "M13_low_deg_2": "lower_better",
      "M13_low_deg_5": "lower_better",
      "M17_relation_entropy": "higher_better",
      "M18_normalised_entropy": "higher_better",
      "M19_relation_gini": "lower_better",     # less inequality = better
      "M20_top2_conc": "lower_better",         # less concentration = better
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
      "M46_assortativity": "ambiguous",        # no clear a priori expectation
      "M50_rel_graph_density": "higher_better",
      "M57_degree_entropy": "higher_better",
      "M58_H_t_given_r": "ambiguous",          # complex relationship
      "M62_metapaths_2_count": "higher_better",
      "M62_metapaths_2_ratio": "higher_better",
  }
  
  struct_ranks = pd.DataFrame(index=analysis_df.index)
  for metric, direction in metric_directions.items():
      if metric not in analysis_df.columns:
          continue
      valid = analysis_df[metric].dropna()
      if direction == "higher_better":
          struct_ranks[f"{metric}_rank"] = valid.rank(ascending=False)
      elif direction == "lower_better":
          struct_ranks[f"{metric}_rank"] = valid.rank(ascending=True)
      else:
          # For ambiguous metrics, rank both ways and report both
          struct_ranks[f"{metric}_rank_desc"] = valid.rank(ascending=False)
          struct_ranks[f"{metric}_rank_asc"] = valid.rank(ascending=True)
  ```
- **Output**: Table of structural metric rankings.

#### Method 1C: Rank Concordance Table

- **Definition**: For each (structural metric, performance metric) pair, compute the number of concordant rank positions. Two datasets are concordant if the one that ranks higher on the structural metric also ranks higher on the performance metric. Report as a fraction of all (4 choose 2) = 6 pairwise comparisons.
  
  This is directly related to Kendall's τ: τ = (concordant - discordant) / (n choose 2).
  
- **Reference**: [Kendall1938]. Kendall's τ is more interpretable than Spearman's ρ for small samples because it directly counts concordant vs. discordant pairs.
- **Library**: **MUST use** `scipy.stats.kendalltau`.
- **Computation**:
  ```python
  from scipy.stats import kendalltau, spearmanr
  
  concordance_results = []
  
  for struct_metric in metric_directions.keys():
      if struct_metric not in analysis_df.columns:
          continue
      direction = metric_directions[struct_metric]
      if direction == "ambiguous":
          continue
      
      struct_vals = analysis_df[struct_metric].values
      if direction == "lower_better":
          struct_vals = -struct_vals  # Flip so higher = better for correlation
      
      for perf_metric in perf_metrics:
          perf_vals = analysis_df[perf_metric].dropna()
          common_idx = perf_vals.index.intersection(analysis_df.index)
          if len(common_idx) < 3:
              continue
          
          sv = struct_vals[analysis_df.index.isin(common_idx)]
          pv = perf_vals[common_idx].values
          
          tau, tau_p = kendalltau(sv, pv)
          rho, rho_p = spearmanr(sv, pv)
          
          concordance_results.append({
              "structural_metric": struct_metric,
              "performance_metric": perf_metric,
              "kendall_tau": tau,
              "kendall_p": tau_p,
              "spearman_rho": rho,
              "spearman_p": rho_p,
              "n": len(common_idx),
          })
  
  concordance_df = pd.DataFrame(concordance_results)
  ```

**IMPORTANT NOTE ON P-VALUES**: With n=4, even a perfect correlation (τ=1.0) yields p ≈ 0.083 for Kendall's τ (and p ≈ 0.083 for Spearman's ρ). This means **no correlation can reach conventional p < 0.05 significance with 4 data points**. The paper MUST:
  1. Report effect sizes (τ, ρ) as the primary result, not p-values [Sullivan2012].
  2. Acknowledge the small sample limitation explicitly.
  3. Use the p-values only to distinguish perfect/near-perfect concordance from random patterns.
  4. Build the argument through *convergent evidence* across multiple metrics and analyses.

- **Output**: `concordance_results.csv` — one row per (structural, performance) pair with τ, ρ, and p-values.

#### Method 1D: Perfect Concordance Identification

- **Definition**: Identify which structural metrics produce τ = 1.0 (perfect concordance) or τ ≥ 0.67 (at most one discordant pair out of 6) with each performance metric. These are the "headline" findings.
- **Computation**:
  ```python
  perfect = concordance_df[concordance_df["kendall_tau"] == 1.0]
  strong = concordance_df[concordance_df["kendall_tau"] >= 0.67]
  ```
- **Output**: Filtered tables highlighting the strongest associations.

---

### PART 2: Correlation Matrix with Effect Sizes

**Why**: While rank concordance is primary, Pearson and Spearman correlations on the raw values provide complementary information about the *strength* and *form* of associations. With n=4 we report these as effect sizes [Cohen1988], not hypothesis tests.

#### Method 2A: Pairwise Correlation Computation

- **Definition**: For every (structural metric, performance metric) pair, compute:
  1. Pearson r [Pearson1895] — measures linear association
  2. Spearman ρ [Spearman1904] — measures monotonic association
  3. Kendall τ [Kendall1938] — measures ordinal association (from Part 1)
  
  Report all three. Agreement across all three strengthens the finding.
  
- **Reference**: [Cohen1988] for effect size interpretation: |r| ≥ 0.5 is "large", |r| ≥ 0.3 is "medium".
- **Library**: **MUST use** `scipy.stats.pearsonr`, `scipy.stats.spearmanr`, `scipy.stats.kendalltau`.
- **Computation**:
  ```python
  from scipy.stats import pearsonr, spearmanr, kendalltau
  
  # Select metrics to correlate — use a curated subset of the most relevant ones
  # to avoid an overwhelming table. Group by category.
  key_structural_metrics = [
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
  
  correlation_results = []
  
  for sm in key_structural_metrics:
      if sm not in analysis_df.columns:
          continue
      for pm in ["LP_MRR", "LP_H10", "Effi_LP"]:
          valid_idx = analysis_df[[sm, pm]].dropna().index
          if len(valid_idx) < 3:
              continue
          x = analysis_df.loc[valid_idx, sm].values.astype(float)
          y = analysis_df.loc[valid_idx, pm].values.astype(float)
          
          r_pearson, p_pearson = pearsonr(x, y)
          r_spearman, p_spearman = spearmanr(x, y)
          tau, p_tau = kendalltau(x, y)
          
          # Effect size category (Cohen 1988)
          abs_r = abs(r_pearson)
          if abs_r >= 0.5:
              effect = "large"
          elif abs_r >= 0.3:
              effect = "medium"
          elif abs_r >= 0.1:
              effect = "small"
          else:
              effect = "negligible"
          
          # Agreement: do all three correlation types agree in sign?
          signs = [np.sign(r_pearson), np.sign(r_spearman), np.sign(tau)]
          sign_agreement = len(set(signs)) == 1
          
          correlation_results.append({
              "structural_metric": sm,
              "performance_metric": pm,
              "pearson_r": round(r_pearson, 4),
              "pearson_p": round(p_pearson, 4),
              "spearman_rho": round(r_spearman, 4),
              "spearman_p": round(p_spearman, 4),
              "kendall_tau": round(tau, 4),
              "kendall_p": round(p_tau, 4),
              "effect_size": effect,
              "sign_agreement": sign_agreement,
              "n": len(valid_idx),
          })
  
  corr_df = pd.DataFrame(correlation_results)
  ```
- **Output**: `correlation_matrix.csv`

#### Method 2B: Bootstrap Confidence Intervals (for Key Metrics)

- **Definition**: For the strongest correlations identified in 2A, compute 95% bootstrap confidence intervals for Pearson r. With n=4, analytic CIs are unreliable, so bootstrap (with 10,000 resamples) is preferred [Efron1993].
  
  Note: With n=4, the bootstrap CI will be very wide. This is honest and appropriate — it communicates the uncertainty correctly.
  
- **Reference**: [Efron1993]. The BCa (bias-corrected and accelerated) interval is recommended for small samples.
- **Library**: Use `scipy.stats.bootstrap` (available in scipy ≥1.9) or the `bootstrapped` package.
- **Computation**:
  ```python
  from scipy.stats import bootstrap
  
  def pearson_statistic(x, y, axis):
      """Pearson r for use with scipy.stats.bootstrap."""
      # x and y are 2D arrays with shape (1, n_resamples)
      r = np.array([pearsonr(x[0], y[0])[0]])
      return r
  
  bootstrap_results = []
  
  # Run only for the top-k strongest correlations from Method 2A
  top_correlations = corr_df.nlargest(10, "pearson_r", keep="all")
  
  for _, row in top_correlations.iterrows():
      sm = row["structural_metric"]
      pm = row["performance_metric"]
      valid_idx = analysis_df[[sm, pm]].dropna().index
      x = analysis_df.loc[valid_idx, sm].values.astype(float)
      y = analysis_df.loc[valid_idx, pm].values.astype(float)
      
      # BCa bootstrap CI
      res = bootstrap(
          (x, y),
          statistic=lambda x, y: pearsonr(x, y)[0],
          n_resamples=10000,
          method='BCa',
          confidence_level=0.95,
          random_state=42,
          paired=True,
      )
      
      bootstrap_results.append({
          "structural_metric": sm,
          "performance_metric": pm,
          "pearson_r": row["pearson_r"],
          "ci_lower": round(res.confidence_interval.low, 4),
          "ci_upper": round(res.confidence_interval.high, 4),
      })
  
  bootstrap_df = pd.DataFrame(bootstrap_results)
  ```
- **Output**: `bootstrap_confidence_intervals.csv`

---

### PART 3: Normalised Profile Comparison (Visual Method)

**Why**: With 4 datasets, the most effective communication tool is visual comparison. Normalised profiles let you see at a glance which datasets share structural characteristics and whether these align with performance.

#### Method 3A: Z-Score Normalisation Table

- **Definition**: For each structural metric, z-score normalise across the 4 datasets: z = (x − μ) / σ. This puts all metrics on a comparable scale.
- **Reference**: Standard statistical normalisation.
- **Library**: `scipy.stats.zscore`
- **Computation**:
  ```python
  from scipy.stats import zscore
  
  # Select key metrics for the profile
  profile_metrics = [
      "M2_num_relations", "M7_avg_total_degree", "M17_relation_entropy",
      "M18_normalised_entropy", "M19_relation_gini", "M25_unique_fraction",
      "M28_avg_class_size", "M31_avg_paths", "M34_avg_path_diversity",
      "M43_avg_clustering", "M50_rel_graph_density", "M62_metapaths_2_ratio",
  ]
  
  profile_df = analysis_df[profile_metrics].copy()
  
  # Z-score normalise each column
  zscore_df = profile_df.apply(zscore)
  zscore_df.index = analysis_df.index
  ```
- **Output**: `normalised_profiles.csv`

#### Method 3B: Radar (Spider) Chart

- **Definition**: For each dataset, plot a radar chart of the z-scored structural metrics. Overlay the LP_MRR value (or use colour intensity) to visually communicate which structural profiles correspond to high performance.
- **Reference**: Standard multivariate visualisation [Cleveland1993].
- **Computation**:
  ```python
  import matplotlib.pyplot as plt
  import numpy as np
  
  def radar_chart(zscore_df, performance_series, output_path):
      """
      Create a radar chart comparing dataset structural profiles.
      
      Args:
          zscore_df: DataFrame with z-scored metrics (rows=datasets, cols=metrics)
          performance_series: Series with performance metric per dataset
          output_path: path to save the figure
      """
      labels = zscore_df.columns.tolist()
      num_vars = len(labels)
      
      angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
      angles += angles[:1]  # Close the polygon
      
      fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
      
      # Color datasets by performance (darker = better)
      cmap = plt.cm.RdYlGn
      perf_norm = (performance_series - performance_series.min()) / \
                  (performance_series.max() - performance_series.min())
      
      for dataset in zscore_df.index:
          values = zscore_df.loc[dataset].tolist()
          values += values[:1]
          color = cmap(perf_norm[dataset])
          ax.plot(angles, values, 'o-', linewidth=2, label=dataset, color=color)
          ax.fill(angles, values, alpha=0.1, color=color)
      
      ax.set_xticks(angles[:-1])
      ax.set_xticklabels([m.replace("M", "").split("_", 1)[1] for m in labels],
                          size=8, wrap=True)
      ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
      ax.set_title("Structural Profiles by Dataset\n(colour intensity ∝ LP MRR)", 
                    size=14, y=1.08)
      
      plt.tight_layout()
      plt.savefig(output_path, dpi=300, bbox_inches='tight')
      plt.close()
  
  radar_chart(zscore_df, transductive_perf["LP_MRR"], "output/plots/radar_profiles.png")
  radar_chart(zscore_df, transductive_perf["LP_MRR"], "output/plots/radar_profiles.pdf")
  ```
- **Output**: `radar_profiles.png` and `.pdf`

#### Method 3C: Heatmap of Normalised Metrics vs. Performance

- **Definition**: A clustered heatmap where rows are datasets (ordered by LP_MRR), columns are structural metrics (grouped by category), and cells show z-scores. Add a performance annotation column.
- **Reference**: [Cleveland1993], [Wickham2010].
- **Library**: `seaborn.clustermap` or `seaborn.heatmap`.
- **Computation**:
  ```python
  import seaborn as sns
  
  # Order rows by LP_MRR descending
  ordered = zscore_df.loc[transductive_perf["LP_MRR"].sort_values(ascending=False).index]
  
  fig, ax = plt.subplots(figsize=(14, 4))
  sns.heatmap(ordered, annot=True, fmt=".2f", cmap="RdYlGn", center=0,
              linewidths=0.5, ax=ax, cbar_kws={"label": "z-score"})
  ax.set_title("Structural Properties (z-scored) — Datasets ordered by LP MRR ↓")
  plt.tight_layout()
  plt.savefig("output/plots/heatmap_profiles.png", dpi=300, bbox_inches='tight')
  plt.savefig("output/plots/heatmap_profiles.pdf", dpi=300, bbox_inches='tight')
  plt.close()
  ```
- **Output**: `heatmap_profiles.png` and `.pdf`

---

### PART 4: Scatter Plots with Annotations (Key Individual Relationships)

**Why**: For the most important structural metrics, individual scatter plots show the relationship directly. With n=4, every data point is visible and labelled.

#### Method 4A: Annotated Scatter Plots

- **Definition**: For each key (structural metric, performance metric) pair, create a scatter plot with:
  1. Each point labelled with the dataset name
  2. A linear trend line (OLS fit) with equation
  3. The Pearson r and Spearman ρ annotated on the plot
  4. No "significance stars" — report effect size only [Sullivan2012]
  
- **Reference**: [Tukey1977] for EDA philosophy; [Sullivan2012] for effect-size reporting.
- **Computation**:
  ```python
  from scipy.stats import pearsonr, spearmanr, linregress
  
  def annotated_scatter(analysis_df, struct_metric, perf_metric, output_path,
                         struct_label=None, perf_label=None):
      """
      Create an annotated scatter plot for one (structural, performance) pair.
      """
      valid_idx = analysis_df[[struct_metric, perf_metric]].dropna().index
      x = analysis_df.loc[valid_idx, struct_metric].values.astype(float)
      y = analysis_df.loc[valid_idx, perf_metric].values.astype(float)
      labels = valid_idx.tolist()
      
      r_p, _ = pearsonr(x, y)
      r_s, _ = spearmanr(x, y)
      slope, intercept, _, _, _ = linregress(x, y)
      
      fig, ax = plt.subplots(figsize=(7, 5))
      ax.scatter(x, y, s=120, zorder=5, edgecolors='black', linewidth=0.5)
      
      # Annotate each point with dataset name
      for i, label in enumerate(labels):
          ax.annotate(label, (x[i], y[i]), textcoords="offset points",
                      xytext=(8, 8), fontsize=9, fontstyle='italic')
      
      # Trend line
      x_line = np.linspace(x.min() * 0.9, x.max() * 1.1, 100)
      y_line = slope * x_line + intercept
      ax.plot(x_line, y_line, '--', color='gray', alpha=0.7, linewidth=1)
      
      # Annotation box with correlation values
      textstr = f"Pearson r = {r_p:.3f}\nSpearman ρ = {r_s:.3f}\nn = {len(x)}"
      props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
      ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
              verticalalignment='top', bbox=props)
      
      xl = struct_label or struct_metric
      yl = perf_label or perf_metric
      ax.set_xlabel(xl, fontsize=12)
      ax.set_ylabel(yl, fontsize=12)
      ax.set_title(f"{yl} vs. {xl}", fontsize=13)
      ax.grid(True, alpha=0.3)
      
      plt.tight_layout()
      plt.savefig(output_path, dpi=300, bbox_inches='tight')
      plt.close()
  
  # Generate scatter plots for the most important relationships
  scatter_pairs = [
      ("M25_unique_fraction", "LP_MRR", "Unique Context Fraction", "Link Prediction MRR"),
      ("M18_normalised_entropy", "LP_MRR", "Normalised Relation Entropy", "Link Prediction MRR"),
      ("M19_relation_gini", "LP_MRR", "Relation Gini Coefficient", "Link Prediction MRR"),
      ("M7_avg_total_degree", "LP_MRR", "Average Total Degree", "Link Prediction MRR"),
      ("M34_avg_path_diversity", "LP_MRR", "Average Path Diversity", "Link Prediction MRR"),
      ("M28_avg_class_size", "LP_MRR", "Average Context Class Size", "Link Prediction MRR"),
      ("M50_rel_graph_density", "LP_MRR", "Relation Graph Density", "Link Prediction MRR"),
      ("M25_unique_fraction", "Effi_LP", "Unique Context Fraction", "Efficiency (MRR/Params)"),
      ("M18_normalised_entropy", "LP_H10", "Normalised Relation Entropy", "Link Prediction Hits@10"),
  ]
  
  for sm, pm, sl, pl in scatter_pairs:
      safe_name = f"{sm}_vs_{pm}".replace("/", "_")
      annotated_scatter(analysis_df, sm, pm, f"output/plots/scatter_{safe_name}.png", sl, pl)
      annotated_scatter(analysis_df, sm, pm, f"output/plots/scatter_{safe_name}.pdf", sl, pl)
  ```
- **Output**: One PNG + PDF per scatter plot pair.

#### Method 4B: Small-Multiple Scatter Grid

- **Definition**: A grid of mini scatter plots showing many structural metrics against LP_MRR simultaneously. This gives a panoramic view.
- **Reference**: [Cleveland1993] for small-multiple design.
- **Computation**:
  ```python
  def scatter_grid(analysis_df, struct_metrics, perf_metric, output_path,
                    ncols=4, figsize_per_cell=(3.5, 3)):
      """Grid of scatter plots: each cell is one structural metric vs perf_metric."""
      n = len(struct_metrics)
      nrows = int(np.ceil(n / ncols))
      fig, axes = plt.subplots(nrows, ncols,
                               figsize=(figsize_per_cell[0]*ncols, figsize_per_cell[1]*nrows))
      axes = axes.flatten()
      
      for i, sm in enumerate(struct_metrics):
          ax = axes[i]
          valid_idx = analysis_df[[sm, perf_metric]].dropna().index
          x = analysis_df.loc[valid_idx, sm].values.astype(float)
          y = analysis_df.loc[valid_idx, perf_metric].values.astype(float)
          
          ax.scatter(x, y, s=60, edgecolors='black', linewidth=0.5)
          for j, label in enumerate(valid_idx):
              ax.annotate(label[:6], (x[j], y[j]), fontsize=6,
                          textcoords="offset points", xytext=(3, 3))
          
          # Compute and display Spearman rho
          if len(x) >= 3:
              rho, _ = spearmanr(x, y)
              ax.set_title(f"{sm.split('_', 1)[1][:20]}\nρ={rho:.2f}", fontsize=8)
          else:
              ax.set_title(sm.split('_', 1)[1][:20], fontsize=8)
          
          ax.tick_params(labelsize=7)
          ax.grid(True, alpha=0.2)
      
      # Hide unused axes
      for i in range(n, len(axes)):
          axes[i].set_visible(False)
      
      fig.suptitle(f"Structural Metrics vs. {perf_metric}", fontsize=14, y=1.02)
      plt.tight_layout()
      plt.savefig(output_path, dpi=300, bbox_inches='tight')
      plt.close()
  
  scatter_grid(analysis_df, key_structural_metrics, "LP_MRR",
               "output/plots/scatter_grid_LP_MRR.png")
  scatter_grid(analysis_df, key_structural_metrics, "LP_MRR",
               "output/plots/scatter_grid_LP_MRR.pdf")
  ```
- **Output**: `scatter_grid_LP_MRR.png` and `.pdf`

---

### PART 5: Inductive Split Analysis (Larger n)

**Why**: The 8 inductive splits (FB15k-237 V1–V4, WN18RR V1–V4) provide more data points. While these splits come from only 2 parent datasets, they have different graph structures (entity counts, degree distributions, relation counts), so within-family variation is meaningful.

#### Method 5A: Inductive Correlation Analysis

- **Definition**: Correlate structural metrics of each inductive split's inference graph with performance (Hits@10).
- **Computation**:
  ```python
  # Merge inductive metrics with performance
  # inductive_analysis_df has 8 rows (one per split)
  
  inductive_corr_results = []
  
  inductive_struct_metrics = [
      "M54_num_infer_relations", "M55_avg_degree_infer",
      "M56_distinct_fraction", "M56_unique_fraction",
      # Add other metrics computed for inductive splits
  ]
  
  for sm in inductive_struct_metrics:
      if sm not in inductive_analysis_df.columns:
          continue
      x = inductive_analysis_df[sm].values.astype(float)
      y = inductive_analysis_df["H10"].values.astype(float)
      
      r_p, p_p = pearsonr(x, y)
      r_s, p_s = spearmanr(x, y)
      tau, p_t = kendalltau(x, y)
      
      inductive_corr_results.append({
          "structural_metric": sm,
          "pearson_r": round(r_p, 4),
          "pearson_p": round(p_p, 4),
          "spearman_rho": round(r_s, 4),
          "spearman_p": round(p_s, 4),
          "kendall_tau": round(tau, 4),
          "kendall_p": round(p_t, 4),
          "n": 8,
      })
  
  inductive_corr_df = pd.DataFrame(inductive_corr_results)
  ```
- **Output**: `inductive_correlation_results.csv`

#### Method 5B: Within-Family vs. Between-Family Decomposition

- **Definition**: Decompose the variance in performance into "between-family" (FB15k-237 vs. WN18RR) and "within-family" (V1 vs. V2 vs. V3 vs. V4) components. This shows whether structural variation within a dataset family predicts performance differences.
- **Computation**:
  ```python
  # Add family label
  inductive_analysis_df["family"] = inductive_analysis_df["parent_dataset"]
  
  # Between-family: compare FB15k-237 mean H10 vs WN18RR mean H10
  family_means = inductive_analysis_df.groupby("family")["H10"].mean()
  
  # Within-family correlations
  for family in ["FB15k-237", "WN18RR"]:
      family_data = inductive_analysis_df[inductive_analysis_df["family"] == family]
      for sm in inductive_struct_metrics:
          if sm not in family_data.columns:
              continue
          x = family_data[sm].values.astype(float)
          y = family_data["H10"].values.astype(float)
          if len(x) >= 3:
              rho, p = spearmanr(x, y)
              print(f"{family} | {sm}: ρ = {rho:.3f}, p = {p:.3f}")
  ```
- **Output**: `inductive_within_family_correlations.csv`

#### Method 5C: Inductive Scatter Plots

- **Definition**: Scatter plots for inductive splits, colour-coded by parent dataset family.
- **Computation**:
  ```python
  def inductive_scatter(inductive_df, struct_metric, output_path, struct_label=None):
      fig, ax = plt.subplots(figsize=(7, 5))
      
      colors = {"FB15k-237": "tab:blue", "WN18RR": "tab:orange"}
      
      for family, color in colors.items():
          mask = inductive_df["parent_dataset"] == family
          fdata = inductive_df[mask]
          x = fdata[struct_metric].values.astype(float)
          y = fdata["H10"].values.astype(float)
          ax.scatter(x, y, s=100, c=color, label=family, edgecolors='black', linewidth=0.5)
          for idx, (xi, yi) in zip(fdata.index, zip(x, y)):
              ax.annotate(idx.split("_")[-1], (xi, yi),  # e.g., "V1"
                          textcoords="offset points", xytext=(5, 5), fontsize=9)
      
      # Overall correlation
      x_all = inductive_df[struct_metric].values.astype(float)
      y_all = inductive_df["H10"].values.astype(float)
      rho, _ = spearmanr(x_all, y_all)
      
      ax.text(0.05, 0.95, f"Spearman ρ = {rho:.3f} (n=8)",
              transform=ax.transAxes, fontsize=10,
              verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
      
      ax.set_xlabel(struct_label or struct_metric, fontsize=12)
      ax.set_ylabel("Hits@10 (Inductive)", fontsize=12)
      ax.legend()
      ax.grid(True, alpha=0.3)
      plt.tight_layout()
      plt.savefig(output_path, dpi=300, bbox_inches='tight')
      plt.close()
  
  inductive_scatter(inductive_analysis_df, "M54_num_infer_relations",
                     "output/plots/inductive_scatter_num_relations.png",
                     "Number of Unique Relations in Inference Graph")
  inductive_scatter(inductive_analysis_df, "M54_num_infer_relations",
                     "output/plots/inductive_scatter_num_relations.pdf",
                     "Number of Unique Relations in Inference Graph")
  ```
- **Output**: Inductive scatter plots (PNG + PDF).

---

### PART 6: Structured Case Study Data

**Why**: Section 7.2 of the paper requires case studies of 3 datasets (FB15k-237 as high-performer, WN18RR as low-performer, CoDEx-Large as moderate). This part prepares the data for those case studies.

#### Method 6A: Comparative Summary Table

- **Definition**: A side-by-side table of the 3 case study datasets showing all key structural metrics and performance.
- **Computation**:
  ```python
  case_study_datasets = ["FB15k-237", "WN18RR", "CoDEx-Large"]
  
  case_study_metrics = [
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
  
  case_study_table = analysis_df.loc[case_study_datasets, 
                                      case_study_metrics + ["LP_MRR", "LP_H10", "Effi_LP"]].T
  case_study_table.to_csv("output/case_study_comparison.csv")
  ```
- **Output**: `case_study_comparison.csv`

#### Method 6B: Delta Analysis (Pairwise Differences)

- **Definition**: For each pair of case study datasets, compute the percentage difference in each metric and in performance. This directly answers "FB15k-237 has X% more relational diversity than WN18RR, and Y% higher MRR."
- **Computation**:
  ```python
  pairs = [
      ("FB15k-237", "WN18RR"),
      ("FB15k-237", "CoDEx-Large"),
      ("CoDEx-Large", "WN18RR"),
  ]
  
  delta_results = []
  for d1, d2 in pairs:
      for metric in case_study_metrics + ["LP_MRR", "LP_H10"]:
          v1 = analysis_df.loc[d1, metric]
          v2 = analysis_df.loc[d2, metric]
          if pd.isna(v1) or pd.isna(v2) or v2 == 0:
              pct_diff = None
          else:
              pct_diff = ((v1 - v2) / abs(v2)) * 100
          
          delta_results.append({
              "comparison": f"{d1} vs {d2}",
              "metric": metric,
              "value_1": v1,
              "value_2": v2,
              "absolute_diff": v1 - v2 if not pd.isna(v1) and not pd.isna(v2) else None,
              "pct_diff": round(pct_diff, 1) if pct_diff is not None else None,
          })
  
  delta_df = pd.DataFrame(delta_results)
  delta_df.to_csv("output/case_study_deltas.csv", index=False)
  ```
- **Output**: `case_study_deltas.csv`

#### Method 6C: Structural Advantage Narrative Data

- **Definition**: For each case study pair, identify the top-5 structural metrics with the largest percentage differences. These become the "explanatory factors" in the narrative.
- **Computation**:
  ```python
  for pair_name in ["FB15k-237 vs WN18RR", "FB15k-237 vs CoDEx-Large", "CoDEx-Large vs WN18RR"]:
      pair_data = delta_df[delta_df["comparison"] == pair_name].dropna(subset=["pct_diff"])
      top_diffs = pair_data.reindex(pair_data["pct_diff"].abs().sort_values(ascending=False).index).head(5)
      print(f"\n=== {pair_name} ===")
      print(top_diffs[["metric", "value_1", "value_2", "pct_diff"]].to_string())
      top_diffs.to_csv(f"output/case_study_top_diffs_{pair_name.replace(' ', '_')}.csv", index=False)
  ```
- **Output**: Top-5 difference tables per pair.

---

### PART 7: Summary Outputs for Paper Drafting

**Why**: The outputs from Parts 1–6 must be consolidated into formats that directly support writing the paper sections.

#### Output 7A: Master Correlation Summary Table (for paper Table N)

- **Definition**: A table intended for direct inclusion in the paper. Rows = key structural metrics (grouped by category). Columns = performance metrics. Cells = Spearman ρ values, with the Kendall τ in parentheses. Highlight cells with |ρ| ≥ 0.8.
- **Computation**:
  ```python
  # Pivot the correlation results into a table format
  summary_for_paper = corr_df.pivot_table(
      index="structural_metric",
      columns="performance_metric",
      values="spearman_rho"
  )
  
  # Add Kendall tau in parentheses
  tau_table = corr_df.pivot_table(
      index="structural_metric",
      columns="performance_metric",
      values="kendall_tau"
  )
  
  # Combine into formatted strings
  combined = summary_for_paper.copy()
  for col in combined.columns:
      combined[col] = summary_for_paper[col].apply(lambda x: f"{x:.2f}") + \
                      " (" + tau_table[col].apply(lambda x: f"{x:.2f}") + ")"
  
  combined.to_csv("output/paper_table_correlations.csv")
  ```
- **Output**: `paper_table_correlations.csv` — directly usable as a LaTeX table source.

#### Output 7B: Ranking Agreement Table (for paper Table N+1)

- **Definition**: A compact table showing, for each structural metric, the dataset ranking it induces and whether this matches the LP_MRR ranking. Perfect matches are marked ✓.
- **Computation**:
  ```python
  mrr_ranking = transductive_perf["LP_MRR"].rank(ascending=False).astype(int)
  
  ranking_comparison = pd.DataFrame(index=key_structural_metrics)
  ranking_comparison["LP_MRR_rank"] = [mrr_ranking.to_dict()] * len(key_structural_metrics)
  
  for sm in key_structural_metrics:
      if sm not in analysis_df.columns:
          continue
      direction = metric_directions.get(sm, "higher_better")
      vals = analysis_df[sm].dropna()
      if direction == "lower_better":
          sm_rank = vals.rank(ascending=True).astype(int)
      else:
          sm_rank = vals.rank(ascending=False).astype(int)
      
      ranking_comparison.loc[sm, "metric_rank"] = str(sm_rank.to_dict())
      ranking_comparison.loc[sm, "matches_MRR"] = (sm_rank == mrr_ranking).all()
      ranking_comparison.loc[sm, "kendall_tau"] = concordance_df[
          (concordance_df["structural_metric"] == sm) & 
          (concordance_df["performance_metric"] == "LP_MRR")
      ]["kendall_tau"].values[0] if len(concordance_df[
          (concordance_df["structural_metric"] == sm) & 
          (concordance_df["performance_metric"] == "LP_MRR")
      ]) > 0 else None
  
  ranking_comparison.to_csv("output/paper_table_rankings.csv")
  ```
- **Output**: `paper_table_rankings.csv`

#### Output 7C: Key Findings JSON (for paper drafting)

- **Definition**: A structured JSON file summarising the headline findings. This is the primary input for drafting Section 7.
- **Computation**:
  ```python
  import json
  
  findings = {
      "methodology_note": {
          "n_transductive": 4,
          "n_inductive": 8,
          "primary_method": "rank concordance (Kendall tau)",
          "supplementary_methods": ["Pearson r", "Spearman rho", "bootstrap CI", "visual profiles"],
          "limitation": "With n=4 transductive datasets, no correlation reaches p<0.05. "
                        "Results are reported as effect sizes, not significance tests."
      },
      "strongest_correlations_transductive": [],
      "perfect_rank_concordances": [],
      "weakest_correlations": [],
      "inductive_findings": [],
      "case_study_explanations": {},
      "practical_guidelines": [],
  }
  
  # Populate strongest correlations
  top = corr_df.nlargest(10, "kendall_tau")
  for _, row in top.iterrows():
      findings["strongest_correlations_transductive"].append({
          "structural_metric": row["structural_metric"],
          "performance_metric": row["performance_metric"],
          "kendall_tau": row["kendall_tau"],
          "spearman_rho": row["spearman_rho"],
          "pearson_r": row["pearson_r"],
          "interpretation": ""  # To be filled during paper drafting
      })
  
  # Perfect concordances
  perfect = concordance_df[concordance_df["kendall_tau"] == 1.0]
  for _, row in perfect.iterrows():
      findings["perfect_rank_concordances"].append({
          "structural_metric": row["structural_metric"],
          "performance_metric": row["performance_metric"],
      })
  
  # Weakest / negative correlations
  bottom = corr_df.nsmallest(5, "kendall_tau")
  for _, row in bottom.iterrows():
      findings["weakest_correlations"].append({
          "structural_metric": row["structural_metric"],
          "performance_metric": row["performance_metric"],
          "kendall_tau": row["kendall_tau"],
      })
  
  # Case study data
  for d in ["FB15k-237", "WN18RR", "CoDEx-Large"]:
      findings["case_study_explanations"][d] = {
          "LP_MRR": float(transductive_perf.loc[d, "LP_MRR"]),
          "key_structural_advantages": [],  # To be populated from delta analysis
          "key_structural_disadvantages": [],
      }
  
  with open("output/key_findings.json", "w") as f:
      json.dump(findings, f, indent=2, default=str)
  ```
- **Output**: `key_findings.json`

#### Output 7D: Practical Guidelines Table (for Section 7.5)

- **Definition**: Based on the correlation analysis, produce a table that translates findings into actionable guidance: "Use the entity-agnostic approach when..."
- **Computation**:
  ```python
  # Identify thresholds from the data
  # The "good" datasets: FB15k-237, CoDEx-Large (LP_MRR > 0.1)
  # The "poor" datasets: WN18RR, YAGO3-10 (LP_MRR < 0.1)
  
  good = ["FB15k-237", "CoDEx-Large"]
  poor = ["WN18RR", "YAGO3-10"]
  
  guidelines = []
  for sm in key_structural_metrics:
      if sm not in analysis_df.columns:
          continue
      good_vals = analysis_df.loc[good, sm].dropna()
      poor_vals = analysis_df.loc[poor, sm].dropna()
      
      if len(good_vals) == 0 or len(poor_vals) == 0:
          continue
      
      good_mean = good_vals.mean()
      poor_mean = poor_vals.mean()
      good_min = good_vals.min()
      poor_max = poor_vals.max()
      
      # The "threshold" is between poor_max and good_min
      if good_mean > poor_mean:
          threshold_region = f">{poor_max:.3f}" if good_min > poor_max else "overlapping"
          direction = "higher is better"
      else:
          threshold_region = f"<{poor_min:.3f}" if good_max < poor_min else "overlapping"
          direction = "lower is better"
      
      guidelines.append({
          "metric": sm,
          "direction": direction,
          "good_performance_range": f"[{good_vals.min():.4f}, {good_vals.max():.4f}]",
          "poor_performance_range": f"[{poor_vals.min():.4f}, {poor_vals.max():.4f}]",
          "separable": good_min > poor_max or good_max < poor_min,
          "threshold_region": threshold_region,
      })
  
  guidelines_df = pd.DataFrame(guidelines)
  guidelines_df.to_csv("output/practical_guidelines.csv", index=False)
  ```
- **Output**: `practical_guidelines.csv`

---

## Complete Output File Structure

```
output/
├── correlation_analysis/
│   ├── concordance_results.csv              # Part 1: Kendall tau + Spearman for all pairs
│   ├── perfect_concordances.csv             # Part 1D: Metrics with tau = 1.0
│   ├── correlation_matrix.csv               # Part 2A: Pearson + Spearman + Kendall for key metrics
│   ├── bootstrap_confidence_intervals.csv   # Part 2B: Bootstrap CIs for top correlations
│   ├── normalised_profiles.csv              # Part 3A: Z-scored profiles
│   ├── inductive_correlation_results.csv    # Part 5A: Inductive split correlations
│   ├── inductive_within_family.csv          # Part 5B: Within-family correlations
│   ├── case_study_comparison.csv            # Part 6A: Side-by-side case studies
│   ├── case_study_deltas.csv               # Part 6B: Percentage differences
│   ├── case_study_top_diffs_*.csv          # Part 6C: Top-5 differences per pair
│   ├── paper_table_correlations.csv         # Part 7A: Table for the paper
│   ├── paper_table_rankings.csv             # Part 7B: Ranking agreement table
│   ├── key_findings.json                    # Part 7C: Structured findings for drafting
│   └── practical_guidelines.csv             # Part 7D: When to use entity-agnostic methods
├── plots/
│   ├── radar_profiles.png / .pdf            # Part 3B: Radar chart
│   ├── heatmap_profiles.png / .pdf          # Part 3C: Heatmap
│   ├── scatter_*.png / .pdf                 # Part 4A: Individual scatter plots
│   ├── scatter_grid_LP_MRR.png / .pdf       # Part 4B: Small-multiple grid
│   └── inductive_scatter_*.png / .pdf       # Part 5C: Inductive scatter plots
└── README.md                                # Summary of all outputs and how to interpret them
```

---

## Methodological Notes for Paper Writing

The following points MUST be addressed in Section 7 of the paper to ensure methodological rigour:

1. **Small sample caveat**: "With four benchmark datasets in the transductive setting, standard significance thresholds (p < 0.05) are unattainable. We therefore report correlation coefficients as effect sizes [Cohen1988] and supplement quantitative analysis with visual profiles and structured case studies. We note that even a perfect rank concordance (Kendall τ = 1.0) across four datasets yields p ≈ 0.083."

2. **Multiple comparison note**: "Given the number of structural metrics examined, we do not apply multiple comparison corrections. Instead, we focus on metrics where all three correlation measures (Pearson, Spearman, Kendall) agree in sign and magnitude, providing convergent evidence."

3. **Causality disclaimer**: "Correlations between structural properties and model performance do not establish causal relationships. Multiple structural properties are themselves correlated (e.g., relational diversity correlates with context uniqueness), making it difficult to isolate individual effects. Nevertheless, the consistent pattern across multiple metrics provides converging evidence."

4. **Inductive advantage**: "The inductive evaluation, with eight benchmark splits, provides a somewhat richer basis for quantitative correlation analysis. However, these splits derive from only two parent datasets (FB15k-237 and WN18RR), and within-family variation is limited."

5. **Effect size interpretation**: Following [Cohen1988] and [Sullivan2012]: |r| ≥ 0.1 small, |r| ≥ 0.3 medium, |r| ≥ 0.5 large. With n=4, only large effects are meaningful.

6. **Reproducibility**: "All structural metrics were computed using established library implementations (scipy, NetworkX, powerlaw). Correlation analyses used scipy.stats with fixed random seeds for bootstrap procedures."
