# Knowledge Graph Structural Properties

A high-performance, checkpoint-resumable pipeline for computing **64 structural metrics** (M1--M64) across Knowledge Graph benchmark datasets. Designed for an empirical study analysing entity-agnostic KG embedding methods, correlating graph structural properties with model performance (MRR, Hits@K).

---

## Overview

This pipeline computes a comprehensive suite of graph-theoretic, information-theoretic, and relational metrics across four transductive KG benchmarks and eight inductive (Teru/GrAIL) splits. It exploits **multiprocessing** to parallelise heavy computations (path mining, reachability estimation, diameter approximation) across all available CPU cores, and provides **checkpoint-based resumption** so long-running jobs can be stopped and restarted without losing progress.

### Datasets

| Dataset | Entities | Relations | Train Triples |
|---|---:|---:|---:|
| FB15k-237 | 14,505 | 237 | 272,115 |
| WN18RR | 40,559 | 11 | 86,835 |
| CoDEx-Large | 77,951 | 69 | 551,193 |
| YAGO3-10 | 123,143 | 37 | 1,079,040 |

Plus **8 inductive splits** (FB15k-237 V1--V4, WN18RR V1--V4) from Teru et al. (2020).

### Key Results (sample)

| Dataset | Unique Context Fraction | Relation Entropy | Avg Clustering | Typed Reciprocity | LP MRR |
|---|---:|---:|---:|---:|---:|
| FB15k-237 | 0.891 | 6.55 | 0.190 | 0.131 | 0.216 |
| WN18RR | 0.054 | 2.23 | 0.062 | 0.342 | 0.069 |
| CoDEx-Large | 0.375 | 3.99 | 0.090 | 0.026 | 0.144 |
| YAGO3-10 | 0.148 | 2.90 | 0.126 | 0.030 | 0.060 |

---

## Metric Categories

The 64 metrics span 10 categories:

| # | Category | Metrics | Description |
|---|---|---|---|
| 1 | **Basic Graph Statistics** | M1--M6 | Entity/relation counts, density, ratios |
| 2 | **Degree Distribution** | M7--M15 | Mean/median/skewness/kurtosis, power-law exponent, in/out correlation |
| 3 | **Relational Diversity** | M16--M23 | Relation entropy, Gini coefficient, concentration, per-entity diversity |
| 4 | **Context Uniqueness** | M24--M30 | Relational context vectors, equivalence classes, disambiguation ceiling |
| 5 | **Path Availability** | M31--M36 | Random-walk path mining, coverage, length, diversity, k-hop reachability |
| 6 | **Connectivity** | M37--M46 | Components, diameter, shortest paths, clustering, reciprocity, assortativity |
| 7 | **Relation-Level** | M47--M51 | Per-relation degree, co-occurrence matrix, relation-specific reciprocity |
| 8 | **Inductive Splits** | M52--M56 | Entity/relation overlap, degree comparison, context uniqueness across splits |
| 9 | **Information-Theoretic** | M57--M60 | Degree entropy, conditional entropies H(T\|R), H(H\|R), H(R\|H,T) |
| 10 | **Multi-Relational** | M61--M64 | Relation multiplicity, meta-path diversity, symmetry scores |

---

## Project Structure

```
analysis/structure/
  compute_metrics.py        # Main orchestrator (CLI entry point)
  graph_utils.py            # Graph construction, parallel path mining, helpers
  checkpoint_manager.py     # Checkpoint/resume state management
  correlation_analysis.py   # Second stage: structure vs performance (reads results/structure/, writes results/structure/correlation_analysis/)
  generate_paper_outputs.py # Third stage: the three CSVs of results/structure/paper_outputs/ (summary, correlation table, inductive metrics)
  performance.py            # reads the reported PathE results from configs/structure/performance.yaml
  README.md
  metrics/
    __init__.py
    basic_stats.py          # Category 1:  M1--M6
    degree_dist.py          # Category 2:  M7--M15
    relational_diversity.py # Category 3:  M16--M23
    context_uniqueness.py   # Category 4:  M24--M30
    path_metrics.py         # Category 5:  M31--M36
    connectivity.py         # Category 6:  M37--M46
    relation_level.py       # Category 7:  M47--M51
    inductive.py            # Category 8:  M52--M56
    info_theoretic.py       # Category 9:  M57--M60
    multi_relational.py     # Category 10: M61--M64
    unit_tests.py           # Synthetic-graph validation tests

docs/structure/
  graph_metrics_specification_v2.md          # specification of M1-M64
  correlation_analysis_specification.md      # specification of the correlation stage (n = 4 caveat included)
  data_requirements_for_sections_5_7.md      # what the paper sections need from this pipeline

results/structure/          # Generated results (after a run); the large folders are not committed
    summary_metrics.csv
    summary_metrics.json
    checkpoint_state.json
    computation.log
    degree_distributions/   # Per-entity degree CSVs
    relation_distributions/ # Per-relation frequency and detail CSVs
    context_analysis/       # Equivalence class CSVs
    path_analysis/          # Per-entity path summaries + checkpoints
    relation_cooccurrence/  # R x R co-occurrence matrices (.npy)
    inductive_splits/       # Inductive split metrics CSV
    plots/                  # Comparative visualisations
```

---

## Installation

**Python 3.10+** is required.

Install the repository's root `requirements.txt`. The pipeline itself needs:

Dependencies:

| Library | Purpose |
|---|---|
| `pykeen` | Dataset loading (transductive + inductive splits) |
| `networkx` | Graph construction and analysis (components, clustering, diameter, ...) |
| `numpy`, `scipy` | Vectorised computation, entropy, correlation, skewness/kurtosis |
| `pandas` | Data wrangling and CSV output |
| `powerlaw` | Power-law fitting via Clauset et al. (2009) MLE method |
| `matplotlib`, `seaborn` | Comparative plots |
| `tqdm` | Progress bars |

---

## Usage

### Full run (all datasets, all metrics)

Run from `analysis/structure/` (outputs go to `<repo>/results/structure/` wherever you run from):

```bash
python compute_metrics.py
python correlation_analysis.py
python generate_paper_outputs.py
```

### Customise datasets or categories

```bash
# Only FB15k-237 and WN18RR
python compute_metrics.py --datasets "FB15k-237,WN18RR"

# Only degree distribution and connectivity metrics
python compute_metrics.py --categories "degree_dist,connectivity"
```

### Control parallelism

```bash
# Use 8 workers instead of the default 16
python compute_metrics.py --workers 8
```

### Skip optional steps

```bash
# Skip inductive splits, plots, or unit-test pre-validation
python compute_metrics.py --skip-inductive --skip-plots --skip-unit-tests
```

### Resume after interruption

Simply re-run the same command. Completed (dataset, category) pairs are detected from the checkpoint file and skipped automatically. Path mining resumes at the entity level.

```bash
# Ctrl+C during a long run, then:
python compute_metrics.py   # picks up where it left off
```

### Full CLI reference

```
usage: compute_metrics.py [-h] [--datasets DATASETS] [--categories CATEGORIES]
                          [--workers WORKERS] [--output-dir OUTPUT_DIR]
                          [--skip-unit-tests] [--skip-plots] [--skip-inductive]

options:
  --datasets DATASETS       Comma-separated dataset names (default: all four)
  --categories CATEGORIES   Comma-separated category names (default: all)
  --workers WORKERS         Number of worker processes (default: 16)
  --output-dir OUTPUT_DIR   Output directory (default: <repo>/results/structure)
  --skip-unit-tests         Skip synthetic-graph validation before running
  --skip-plots              Skip plot generation at the end
  --skip-inductive          Skip inductive split metrics (Category 8)
```

---

## Architecture

### Execution Flow

```
                    ┌──────────────────────────────┐
                    │   Unit Tests (synthetic KG)  │
                    └──────────────┬───────────────┘
                                   ▼
              ┌─────────────────────────────────────────┐
              │  For each dataset (sequential, memory)  │◄── checkpoint
              │                                         │    resume
              │  1. Load from PyKEEN                    │
              │  2. Build MultiDiGraph + undirected      │
              │  3. Compute shared data (vectorised)    │
              │                                         │
              │  ┌───────────────────────────────────┐  │
              │  │ Fast categories (single-process)  │  │
              │  │ Cat 1-4, 9-10: ~seconds each      │  │
              │  └───────────────────────────────────┘  │
              │                                         │
              │  ┌───────────────────────────────────┐  │
              │  │ Heavy categories (multiprocessing) │  │
              │  │ Cat 5: path mining (16 workers)   │  │
              │  │ Cat 6: connectivity (parallel BFS) │  │
              │  │ Cat 7: relation-level metrics     │  │
              │  └───────────────────────────────────┘  │
              │                                         │
              │  Save checkpoint ──► next dataset        │
              └─────────────────────────────────────────┘
                                   ▼
              ┌─────────────────────────────────────────┐
              │  Inductive Splits (Cat 8, V1--V4)       │
              │  FB15k-237 + WN18RR via PyKEEN          │
              └─────────────────────────────────────────┘
                                   ▼
              ┌─────────────────────────────────────────┐
              │  Aggregate ──► summary CSV/JSON + plots  │
              └─────────────────────────────────────────┘
```

### Parallelisation Strategy

| Operation | Method | Workers | Speedup |
|---|---|---:|---|
| Path mining (M31--M35) | `multiprocessing.Pool` + `imap_unordered` | 16 | Near-linear with cores |
| Reachability (M36) | Parallel pair sampling | 8 | ~8x |
| Diameter approx (M41) | Parallel sampled BFS | 8 | ~8x |
| Shortest path avg (M42) | Parallel pair sampling | 8 | ~8x |
| Clustering (M43) | Sampled for graphs > 20K nodes | 1 | N/A (sampling reduces work) |

### Checkpoint System

State is tracked at two granularities:

- **Category level**: `results/structure/checkpoint_state.json` maps each `(dataset, category)` to `completed` or `in_progress`. On restart, completed pairs are skipped.
- **Entity level** (path mining only): `results/structure/path_analysis/{dataset}_paths_checkpoint.pkl` saves mined paths every 5,000 entities. On restart, already-mined entities are loaded and skipped.

---

## Output Files

After a full run, `results/structure/` contains (committed: the summary files, `relation_distributions/`, `inductive_splits/`, `results/`, `correlation_analysis/`, `paper_outputs/` and `plots/*.pdf`; the rest is regenerated):

| Path | Description |
|---|---|
| `summary_metrics.csv` | One row per dataset, all 127 scalar metric columns + performance data |
| `summary_metrics.json` | Same data in JSON format |
| `degree_distributions/{dataset}_degrees.csv` | Per-entity in/out/total degree |
| `relation_distributions/{dataset}_relation_freq.csv` | Per-relation frequency and fraction |
| `relation_distributions/{dataset}_relation_details.csv` | Per-relation avg degree, reciprocity |
| `context_analysis/{dataset}_context_classes.csv` | Equivalence class sizes with example entity IDs |
| `path_analysis/{dataset}_paths_per_entity.csv` | Per-entity path count and avg length |
| `relation_cooccurrence/{dataset}_cooccurrence.npy` | R x R co-occurrence matrix (NumPy) |
| `inductive_splits/inductive_metrics.csv` | M52--M56 for all 8 inductive splits |
| `results/{dataset}__{category}.json` | Per-dataset, per-category raw metric values (e.g. `FB15k-237__path_metrics.json`) |
| `correlation_analysis/*.csv`, `key_findings.json` | Output of `correlation_analysis.py`: correlation matrix, rank concordance, bootstrap CIs, case studies, inductive within-family analysis, practical guidelines |
| `plots/*.pdf`, `plots/*.png` | Degree distributions, relation frequency, context classes, key metrics, correlation plots |
| `computation.log` | Full timestamped log |

`paper_outputs/` (from `generate_paper_outputs.py`): `summary_metrics_for_paper.csv`, `correlation_table_for_paper.csv`, `inductive_metrics_for_paper.csv`.

---

## Relation to the TGDK paper "Structure Matters"

This folder produced the **dataset-level** structural analysis of the paper: Section 6 (structural characterisation of the four KGs), Appendices C–E (full metric tables, correlation tables, inductive-split metrics) and the inductive-split table. The main values it reports, as they stand in the paper: unique-context fraction 0.891 / 0.375 / 0.054 / 0.148 and mean mined-path length 8.621 / 3.241 / 12.996 / 5.279 for FB15k-237 / CoDEx-Large / WN18RR / YAGO3-10 (`results/structure/results/*__context_uniqueness.json`, `*__path_metrics.json`). The unique-context fractions were recomputed independently from the PathE path datasets (`analysis/query_level/context_ceiling.py`, which only counts entities with identical in/out relation-count vectors) and agree to three decimals.

Things to know before re-running or reusing it:

- **Path mining here is a replication budget, not the model's.** `graph_utils.mine_paths_for_entity` uses 8 random walks of at most 20 nodes per entity, seed 42 (`mine_paths_parallel`, `base_seed=42`). The PathE models of the paper were trained on 20 walks of at most 10 steps (`20_10` path files, produced by the PathE repository) and sample 2–8 paths per entity. The "20 nodes / 8 paths" of Section 6 therefore describe this analysis.
- **Performance values are the paper's reported numbers, not computed here.** They are kept in one file, `configs/structure/performance.yaml` (link and relation prediction, efficiency, parameters per KG; inductive Hits@10 per split: FB15k-237 V1–V4 0.847 / 0.961 / 0.954 / 0.959, WN18RR 0.576 / 0.594 / 0.651 / 0.562), which `compute_metrics.py`, `generate_paper_outputs.py` and `correlation_analysis.py` read. The "LP MRR" column of the sample table above is the paper's value.
- **Status of the correlation stage in the paper.** After the review (September 2026) the dataset-level correlations between the 29 selected metrics and performance were moved to the appendix and labelled exploratory: with four KGs every rank ordering has probability 1/24, so one or two perfect concordances among 29 metrics are expected by chance, and the two found (normalised relation entropy, average paths per entity) cannot be told from noise. Section 7 of the paper now rests on a **query-level** analysis (per-query ranks of PathE, an entity-based reference model and a popularity baseline, 118K test queries), produced by `analysis/query_level/`; the numeric thresholds once derived here (H_n(R) > 0.65, u > 0.40) are no longer in the paper. `correlation_analysis.py` and its outputs remain the source of Appendix E and are still valid as descriptive statistics.
- **Environment used for the stored outputs:** Python 3.10, numpy 2.2.6, pandas 2.3.3, scipy 1.15.3. In the environment of the root `requirements.txt` (numpy 1.24.1, pandas 1.5.3, powerlaw 2.0.0) a full `compute_metrics.py` run reproduces every value of `summary_metrics.csv` (largest relative difference 2e-10 in the per-KG JSON files; `relation_distributions/*.csv` list tied relations in another order), `generate_paper_outputs.py` and `correlation_analysis.py` reproduce the stored CSV and JSON outputs byte for byte (the PDF plots differ only in their metadata), and the unit tests pass. Two settings matter: **powerlaw 2.0.0** (with 1.5 the WN18RR power-law fit selects x_min 9 instead of 27) and **`--workers 16`**, the default (every chunk of entities gets its own seed and the chunk size depends on the number of workers, so another value mines other paths and changes M31–M35 by up to 0.3%).

## Academic References

The metrics in this pipeline are grounded in the following works:

| Metric(s) | Reference |
|---|---|
| Entropy (M17, M18, M29, M57--M60) | Shannon (1948), Cover & Thomas (2006) |
| Gini coefficient (M19) | Gini (1912) |
| Power-law fitting (M14) | Clauset, Shalizi & Newman (2009) |
| Degree assortativity (M46) | Newman (2002) |
| Clustering, components, diameter (M37--M44) | Newman (2003), Watts & Strogatz (1998) |
| Reciprocity (M45, M51) | Garlaschelli & Loffredo (2004) |
| Meta-paths (M62) | Sun & Han (2012) |
| Relational context (M24--M30) | Galkin et al. (2022) |
| Inductive splits (M52--M56) | Teru, Denis & Hamilton (2020) |
| Skewness, kurtosis (M10--M11) | Fisher (1930) |
| Correlations (M15) | Pearson (1895), Spearman (1904) |

All metrics prefer established library implementations (`scipy.stats`, `networkx`, `powerlaw`) over hand-rolled code. Where custom implementations are necessary (Gini, reciprocity, context uniqueness), they are validated against synthetic graphs with known properties via the built-in unit tests.

---

## License

This project is provided for academic research purposes.
