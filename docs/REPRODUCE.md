# Reproducing the article

Every item of the article, the command that produces it from the released artifacts, and the check
that was made (gates G1–G7, October 2026, environment of `requirements.txt`). Run from the repository
root with the package installed (`README.md`, Installation) and the artifacts fetched
(`scripts/00_setup/fetch_artifacts.sh`, `docs/ARTIFACTS.md`).

| Paper item | Command (below) | Output | Check |
|---|---|---|---|
| Dataset statistics (Section 4.1), inductive split statistics (App. A) | [structure](#structure-analysis-section-6-app-a-ce) | `results/structure/summary_metrics.csv`, `inductive_splits/` | G4: all values reproduced |
| Relation prediction (Section 5.1), PathE row | [relation prediction](#relation-prediction-section-51) | `test_mrr`, `test_hits10` | G5: 0.9722 / 0.9977 and 0.873 / 0.999, within the run-to-run spread |
| Link prediction (Section 5.2), PathE row | [link prediction](#link-prediction-section-52) | `results/link_prediction/ranks/pathe_<kg>.csv` | G1: MRR and Hits@10 identical to the rank files |
| Inductive link prediction (Section 5.3), PathE row | [inductive](#inductive-link-prediction-section-53) | `test_link_hits10`, `results/inductive/test_hits.csv` | G6: identical (eval seeds 0–4, CPU) |
| Parameter efficiency (Section 5.4) | [parameters](#parameter-counts-section-54-app-g) | per-module counts | 210,971 / 669,803 / 679,563 / 235,727 |
| Component ablation (Section 5.5, App. F) | [ablations](#component-ablation-section-55-app-f) | MRR per ablation | `docs/ARTIFACTS.md`: 5 of 6 within 0.002 of the article |
| Path-count sensitivity (App. G) | [path count](#path-count-sensitivity-app-g) | `results/figures/pathe_ablation_metrics.pdf` | figure from the stored values; their checkpoints UNCONFIRMED |
| Section 6 tables; App. C, D, E | [structure](#structure-analysis-section-6-app-a-ce) | `results/structure/` | G4 |
| Section 7: figure, tables and numbers in the text | [Section 7](#query-level-analysis-section-7) | `results/query_level/`, printed statistics | G2, G3: identical outputs |
| Positional-embedding figure (Section 7) | [positional](#positional-embedding-figure) | `results/figures/positional_embeddings.pdf` | from the stored tables; two of four tables of unknown origin |
| Hyperparameter tables (App. B) | `configs/` | one YAML per released model | differences stated in the header of each YAML |
| App. H, Wikidata relation frequencies | none | | no code or data found |
| Rows of the baselines from the literature | none | | copied from the cited papers |

## Link prediction (Section 5.2)

```bash
python analysis/query_level/dump_ranks.py --dataset fb15k237    --checkpoint artifacts/checkpoints/link_prediction/fb15k237.ckpt    --path_setup 20_10 --ppe 4 --nhead 2 --seed 7  --out ranks_fb15k237.csv
python analysis/query_level/dump_ranks.py --dataset codex-large --checkpoint artifacts/checkpoints/link_prediction/codex-large.ckpt --path_setup 20_10 --ppe 8 --nhead 2 --seed 7  --out ranks_codex-large.csv
python analysis/query_level/dump_ranks.py --dataset wn18rr      --checkpoint artifacts/checkpoints/link_prediction/wn18rr.ckpt      --path_setup 20_10 --ppe 2 --nhead 4 --seed 46 --out ranks_wn18rr.csv
python analysis/query_level/dump_ranks.py --dataset yago        --checkpoint artifacts/checkpoints/link_prediction/yago.ckpt        --path_setup 20_10 --ppe 2 --nhead 4 --seed 7  --out ranks_yago.csv
```

Each writes `<out>` (one row per test triple and side) and `<out>_summary.json`. Expected MRR / Hits@10:
0.2200 / 0.3533, 0.1475 / 0.2425, 0.0689 / 0.1238, 0.0606 / 0.0927 (article: 0.216 / 0.350, 0.144 / 0.240,
0.069 / 0.124, 0.060 / 0.093). `--limit 500` evaluates the first 500 test triples only (a minute on CPU);
the full CoDEx-Large set takes hours on CPU and minutes on a GPU. The package's own evaluation gives the
same metrics without per-query output:

```bash
python -m pathe.runner full_eval pathe --config configs/link_prediction/<kg>.yaml \
    --checkpoint artifacts/checkpoints/link_prediction/<kg>.ckpt --device cuda --num_devices 1
```

Training: `python -m pathe.runner train pathe --config configs/link_prediction/<kg>.yaml --device cuda --num_devices 1 --num_workers 8`.
The configs are the training commands of the released checkpoints, with the exceptions stated in
their headers.

## Relation prediction (Section 5.1)

```bash
python -m pathe.runner test pathe --config configs/relation_prediction/fb15k237.yaml --device cuda --num_devices 1
python -m pathe.runner test pathe --config configs/relation_prediction/wn18rr.yaml   --device cuda --num_devices 1
```

Expected `test_mrr` / `test_hits10`: 0.972 / 0.998 (FB15k-237), 0.873–0.875 / 0.999 (WN18RR; article
0.874 / 0.999). Training: `train` instead of `test`, same configs.

## Inductive link prediction (Section 5.3)

```bash
for kg in fb15k237 wn18rr; do for v in v1 v2 v3 v4; do for s in 0 1 2 3 4; do
  python -m pathe.runner test pathe --config configs/inductive/${kg}_$v.yaml --device cpu --eval_seed $s
done; done; done
```

Read `test_link_hits10` and `test_link_mrr`: each test triple is ranked against 50 head and 50 tail
negatives from the inference graph. Expected Hits@10, mean of eval seeds 0–4 (the article's values): 0.847 / 0.961 / 0.954 /
0.959 (FB15k-237 v1–v4), 0.576 / 0.594 / 0.651 / 0.562 (WN18RR v1–v4). With `--eval_seed 0`:
0.83415 / 0.96234 / 0.95549 / 0.96173 and 0.57713 / 0.59751 / 0.64463 / 0.55563. Every row (Hits@1/3/10,
MRR) is in `results/inductive/test_hits.csv`. Training: `train` instead of `test`. The path datasets are
built with `scripts/01_data/build_inductive.sh <kg>`.

## Parameter counts (Section 5.4, App. G)

```bash
python analysis/figures/count_params.py           # the four link prediction checkpoints
python analysis/figures/count_params.py <ckpt>    # any checkpoint
```

Totals 210,971 / 669,803 / 679,563 / 235,727 (FB15k-237 / WN18RR / CoDEx-Large / YAGO3-10), with the
per-module breakdown of App. G.

## Component ablation (Section 5.5, App. F)

```bash
python analysis/query_level/dump_ranks.py --dataset fb15k237 --checkpoint artifacts/checkpoints/ablations/fb15k237_no_aggregator.ckpt \
    --path_setup 20_10 --ppe 4 --nhead 2 --seed 7 --out ablation_fb15k237_no_aggregator.csv
```

and likewise for `{fb15k237,codex-large}_{no_aggregator,single_path,no_positional}` with `--ppe` = `max_ppt / 2`
of the config (4, 1, 4). Results and the article's values: `docs/ARTIFACTS.md`, "Evaluation of the ablation
checkpoints". The two `no_positional` models were trained with an unrecorded code change, so their
configs do not retrain them.

## Path-count sensitivity (App. G)

```bash
python analysis/figures/path_count_sensitivity.py      # results/figures/pathe_ablation_metrics.pdf from results/figures/fb_cola_ablation.csv
```

The figure is drawn from the stored values. The checkpoints with 2, 4, 8 and 16 paths per triple are
released (`ablations/*_single_path`, `*_ppt4`, `*_ppt16`, `link_prediction/codex-large`) and can be
evaluated as above with `--ppe` 1, 2, 4 or 8; which run gave which point of the figure is UNCONFIRMED.

## Structure analysis (Section 6, App. A, C–E)

```bash
cd analysis/structure
python compute_metrics.py --workers 16     # all metrics, four KGs and eight inductive splits; resumable
python correlation_analysis.py             # App. E
python generate_paper_outputs.py           # results/structure/paper_outputs/
```

The graphs are loaded through PyKEEN; no artifact is needed. `--workers 16` is required to reproduce the
stored values exactly (the mined paths depend on the number of workers) and `powerlaw` must be 2.0.0
(`requirements.txt`). Unique-context fraction 0.891 / 0.375 / 0.054 / 0.148 and mean path length
8.621 / 3.241 / 12.996 / 5.279 (FB15k-237 / CoDEx-Large / WN18RR / YAGO3-10). Details:
`analysis/structure/README.md`.

## Query-level analysis (Section 7)

Inputs: the per-query rank files of PathE, ComplEx and the popularity baseline in
`results/link_prediction/ranks/`. To regenerate them:

```bash
# PathE: the four dump_ranks.py commands of "Link prediction", with --out results/link_prediction/ranks/pathe_<kg>.csv
python analysis/query_level/popularity_baseline.py <kg>        # popularity_<kg>.csv; MRR 0.2349 / 0.1135 / 0.0257 / 0.0474
python analysis/query_level/align_ids.py --out /tmp/idmaps --check results/link_prediction/idmaps   # all maps "ok"
python analysis/query_level/complex_ranks.py <kg>              # complex_<kg>.csv; MRR 0.3476 / 0.2939 / 0.5090 / 0.5525
```

(`<kg>` = `fb15k237`, `codex-large`, `wn18rr`, `yago`; MRR in that order. `align_ids.py` needs LibKGE's
CoDEx-L id files for the CoDEx-Large maps, see `scripts/07_complex_baseline/setup_libkge.sh`; the stored
maps in `results/link_prediction/idmaps/` are used otherwise.) Then:

```bash
python analysis/query_level/context_ceiling.py <kg>     # results/query_level/ceiling/ceiling_<kg>.{csv,json}: context classes, MRR bound
python analysis/query_level/query_level.py              # results/query_level/query_level_4kg.json: pooled regression
python analysis/query_level/ea_vs_eb.py                 # PathE vs ComplEx vs popularity: retained share, Spearman correlations
python analysis/query_level/make_fig_degree.py          # results/query_level/mrr_by_degree.pdf and fig_degree_data.csv
python analysis/query_level/within_kg.py <kg>           # per-KG covariates of the rank
python analysis/query_level/degree_by_class.py          # MRR by context-class size at fixed degree
python analysis/query_level/compare_baseline.py         # PathE vs popularity
```

Checked: `fig_degree_data.csv` and `query_level_4kg.json` are byte-identical to the committed files;
ComplEx retains 63 / 50 / 14 / 11 % and Spearman(rank, answer degree) is −0.62 / −0.74 / −0.42 / −0.56.
The ComplEx model for CoDEx-Large was trained with `scripts/07_complex_baseline/` (published
configuration of the CoDEx authors); the other three are the public LibKGE checkpoints.

## Positional-embedding figure

```bash
python analysis/figures/positional_pca.py    # results/figures/positional_embeddings.pdf
```

Drawn from the stored tables `results/figures/positional_embeddings/<kg>_pos.csv`. The CoDEx-Large and
WN18RR tables are those of the released checkpoints; the origin of the FB15k-237 and YAGO3-10 tables is
unknown.

## Tests

```bash
python -m unittest discover -s tests
```
