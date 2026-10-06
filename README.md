# PathE

PathE is a family of path-based, **fully entity-agnostic** knowledge graph embedding models. A PathE
model still produces one embedding per entity, but no trainable parameter is indexed by an entity
identifier: the embedding of an entity is computed on the fly from (i) its *relational context* (the
relations it takes part in) and (ii) a small set of *entity-relation paths* that pass through it. The
parameter count therefore scales with the number of relation types, not with the number of entities
(0.2M to 0.7M parameters on the standard benchmarks).

This repository holds the code, configurations and results of the TGDK article "Structure Matters:
An Empirical Study of Path-Based Entity-Agnostic Knowledge Graph Embeddings", which evaluates PathE on
FB15k-237, WN18RR, CoDEx-Large and YAGO3-10 (link prediction, relation prediction, inductive link
prediction) and relates its performance to structural properties of the graphs.

- `docs/REPRODUCE.md`: every table, figure and number of the article, with the command that produces it.
- `docs/ARTIFACTS.md`: the released checkpoints and path datasets.

## Repository layout

```
src/pathe/                 the model package
  runner.py                entry point: train | resume | test | full_eval
  pathe_trainer.py         datasets, model and Lightning trainer for runner.py
  pathe_full_eval.py       filtered ranking of the test set against all entities (full_eval)
  pather_models.py         PathEModel, node projectors
  aggregation.py           aggregators of the per-path views of an entity
  wrappers.py              LightningModule: losses, optimiser, metrics
  pathdata.py, path_lib.py, data_utils.py, triple_lib.py, corruption.py   datasets, path indexing, negatives
  config.py, paths.py      --config YAML files; locations from configs/paths.yaml
  build/                   path-dataset builders (transductive, inductive)
configs/                   one YAML per reported model; paths.yaml = the only machine-dependent file
scripts/00_setup/          fetch / pack the released artifacts
scripts/01_data/           build the path datasets
scripts/04_inductive/, scripts/06_hparam_search/   launch loops and grids of the original runs
scripts/07_complex_baseline/   ComplEx on CoDEx-Large with LibKGE (Section 7 reference model)
analysis/query_level/      per-query ranks and the analysis of Section 7
analysis/structure/        structural metrics of the graphs (Section 6, App. C–E)
analysis/figures/          parameter counts, path-count and positional-embedding figures
results/                   committed outputs: rank files, id maps, tables, figures
tests/                     unit tests (python -m unittest discover -s tests)
artifacts/                 not in git: data, checkpoints, runs, environments (configs/paths.yaml)
```

## Installation

Python 3.10:

```bash
python3.10 -m venv venv && venv/bin/pip install -r requirements.txt
venv/bin/pip install --no-deps -e .      # the pathe package and the `pathe` command (= python -m pathe.runner)
```

The pins are those of the environment that reproduces the reported numbers. `torch` 2.1.1 comes from PyPI
(CUDA 12.1 build) and `torch-scatter` from the PyG wheel index named in `requirements.txt`; for another CUDA
version, install the matching builds of both. `wandb` is optional (`--wandb_project`). The ComplEx baseline
needs LibKGE in its own environment (`scripts/07_complex_baseline/setup_libkge.sh`). All commands below are
run from the repository root.

## Checkpoints and path datasets

```bash
scripts/00_setup/fetch_artifacts.sh https://github.com/King-s-Knowledge-Graph-Lab/structure-matters/releases/download/v1.0
```

downloads the released checkpoints and path datasets, checks their md5 and unpacks them into
`checkpoint_root` and `data_root`; `docs/ARTIFACTS.md` lists every file and the paper item it belongs to.

## Quick start

```bash
# test a released model (relation prediction, FB15k-237: test MRR 0.972)
python -m pathe.runner test pathe --config configs/relation_prediction/fb15k237.yaml --device cuda --num_devices 1

# filtered link prediction ranks of a released model, one row per test query (FB15k-237: MRR 0.2200, Hits@10 0.3533)
python analysis/query_level/dump_ranks.py --dataset fb15k237 \
    --checkpoint artifacts/checkpoints/link_prediction/fb15k237.ckpt \
    --path_setup 20_10 --ppe 4 --nhead 2 --seed 7 --out ranks_fb15k237.csv

# train a model with the arguments of a released one
python -m pathe.runner train pathe --config configs/link_prediction/fb15k237.yaml --device cuda --num_devices 1 --num_workers 8
```

## Data

### Path datasets

`scripts/01_data/build_transductive.sh <kg>` and `scripts/01_data/build_inductive.sh <kg>` mine the
path datasets into `data_root` (`configs/paths.yaml`), with a seed (`--seed`, default 42). The released
`20_10` path files were mined without a seed, so a rebuild gives other paths: the released files, not a
rebuild, reproduce the reported numbers exactly.

The builder loads the knowledge graph through PyKEEN (splits and PyKEEN ids), then for each split writes
the relational context of every entity, mines `num_paths` random walks of at most `num_steps` steps per
entity, and saves under `<data_root>/path_datasets/<kg>/<split>/`:

| File | Content |
|---|---|
| `triples.pt` | `IntTensor [n, 3]` of `(head, relation, tail)` ids |
| `id2entity.json`, `id2relation.json` | id → label maps from PyKEEN |
| `relational_context.csv` | `node_id,direction,edges`: for each entity two rows (`in`, `out`) listing the relation ids of all its incoming / outgoing training triples, with repetitions |
| `rel_neighbors.csv` | relation-level neighbourhood, transductive datasets only (not used by PathE) |
| `<num_paths>_<num_steps>_er_paths.csv` | `relation_path,entity_path`: one walk per row, space-separated ids; the entity path has one element more than the relation path |

The file-name prefix is the **path setup** (`--path_setup`): `20_10` for link prediction and the
inductive models, `20_20` for relation prediction. In the transductive setting only the files of the `train`
folder are used to build entity representations, in all phases: validation and test triples are never
used as context. The inductive splits (GraIL v1–v4) live in
`<data_root>/path_datasets/<kg>/v<k>/{train,val,test,inference}/`; validation and test score the triples
of `val/` and `test/` (which hold only `triples.pt`) in the context of the inference graph only
(`inference/triples.pt`, `20_10_er_paths_inference.csv`, `relational_context_inference.csv`).

### What the model sees

- **Relational context.** `path_lib.encode_relcontext_freqs` turns `relational_context.csv` into two count
  matrices (incoming, outgoing) of shape `[num_entities + 2, num_relations + 2]` (the offset 2 is for
  `PAD`, `MSK`). These counts are the only entity-specific input of the model: two entities with the same
  counts receive the same initial representation.
- **Paths.** Every mined path is indexed by its first (`out`) and last (`in`) entity. For an entity, up to
  `ppe` incoming and `ppe` outgoing paths are sampled and joined pairwise into one sequence that passes
  through the entity; entities without mined paths fall back to their training triples as length-1 paths.
  Each position gets an *entity-focused positional index*: 1 at the entity, growing with the distance from
  it (0 is padding).
- **Path budget.** `--max_ppt` is the number of paths per triple; every entity gets `ppe = max_ppt // 2`.
  Path sampling is seeded per entity, so `--seed` is part of the evaluation setting.

## Model (`pather_models.PathEModel`)

1. **Node projector** (`--node_projector`): maps the incoming and outgoing relation counts of each entity
   on a path to a `d_model` vector and fuses the two with a 2-layer MLP. `GCN`: attention over relation
   types computed from the counts, applied to learned relation features. `dummy`: dropout (0.5) and a
   linear layer over the counts.
2. **Path encoder**: entity and relation vectors are interleaved (`e r e r … e`), the entity-focused
   positional embeddings are added, and the sequence goes through a Transformer encoder (`--nhead`,
   `--num_encoder_layers`, `--dim_feedforward`, `--dropout`). The output at the entity's position is that
   path's view of the entity.
3. **Aggregator** (`--ent_aggregation`): fuses the path views of an entity into one embedding: `avg`,
   `transformer` (a learned summary token, `--num_agg_layers`, `--num_agg_heads`), `recurrent` (BiLSTM),
   `single_query`, `laf`. With `--context_heads 0` (all released link prediction models) head and tail are
   aggregated independently; with `--context_heads > 0` each attends to the other's path views.
4. **Heads**: relation prediction, a linear layer over `[h; t]`; link prediction, an MLP over `[h; r; t]`
   with a separate relation table.

Losses: cross-entropy for relation prediction (`--label_smoothing`, optional `--class_weigths`); for link
prediction `--lp_loss_fn ce|bce|nssa` over each positive and its `--num_negatives` corruptions. The total
is `loss_weight · L_lp + (1 − loss_weight) · L_rp` (`--loss_weight`, default 1.0); `--num_negatives 0`
trains relation prediction only.

## Training

```bash
python -m pathe.runner train pathe --config configs/<task>/<model>.yaml --device cuda --num_devices 1 --num_workers 8
```

A config holds the training arguments of one released model (keys are the long options of `runner.py`;
`{data_root}`, `{runs_root}`, … are filled in from `configs/paths.yaml`); options given on the command line
override it. Without a config, pass the arguments directly (`python -m pathe.runner train pathe --help`).

- Runs are written to `<log_dir>/<expname>/version_<k>/`: TensorBoard logs, the best checkpoint, and
  `config.yaml` with every argument and the git commit, which `--config` reads back. New checkpoints also
  store their hyperparameters; of the released ones only the inductive WN18RR v2–v4 do.
- `--wandb_project <name>` adds a Weights & Biases logger.
- `batch_size` counts triples including negatives and is rounded to a multiple of `num_negatives + 1`.
- Validation and `test` rank each positive against `--val_num_negatives` sampled negatives. For the
  transductive KGs the `valid_*` / `test_*` link metrics are therefore not comparable with the literature;
  the reported transductive numbers are full rankings (next section). The reported inductive numbers are
  these metrics, with 50 head and 50 tail negatives from the inference graph, seeded by `--eval_seed`.
  In the output of `test`, the link prediction metrics are `test_link_*`; `test_mrr` and
  `test_hits*` belong to the relation prediction head.
- `resume` and `test` re-open an existing run (`--version`); `test` uses the single checkpoint of
  `version_<k>/checkpoints/`.

## Evaluation

Transductive link prediction is reported as a filtered ranking against all entities, on both sides,
with the realistic rank (mean of the optimistic and pessimistic rank). Two equivalent ways:

```bash
# per-query ranks (CPU or GPU); exact for --context_heads 0
python analysis/query_level/dump_ranks.py --dataset <kg> --checkpoint <ckpt> --path_setup 20_10 \
    --ppe <max_ppt/2> --nhead <nhead> --seed <seed> --out <csv>

# the original evaluation of the package: metrics only, one row appended to <results_root>/link_prediction/full_eval/<kg>.csv
python -m pathe.runner full_eval pathe --config configs/link_prediction/wn18rr.yaml \
    --checkpoint artifacts/checkpoints/link_prediction/wn18rr.ckpt --device cuda --num_devices 1
```

`full_eval` options: `--eval_type Full|Partial` (`--eval_num_negatives`), `--trim_paths` (one path per
entity), `--override_pos`, `--relation_head`, `--dataset`, `--results_dir`.

Settings that reproduce the article with the released checkpoints (`dump_ranks.py`, full test sets):

| KG | `--path_setup` | `--ppe` | `--nhead` | `--seed` | MRR / Hits@10 | Article | Queries |
|---|---|---|---|---|---|---|---|
| FB15k-237 | `20_10` | 4 | 2 | 7 | 0.2200 / 0.3533 | 0.216 / 0.350 | 40,876 |
| CoDEx-Large | `20_10` | 8 | 2 | 7 | 0.1475 / 0.2425 | 0.144 / 0.240 | 61,244 |
| WN18RR | `20_10` | 2 | 4 | 46 | 0.0689 / 0.1238 | 0.069 / 0.124 | 5,848 |
| YAGO3-10 | `20_10` | 2 | 4 | 7 | 0.0606 / 0.0927 | 0.060 / 0.093 | 9,964 |

The per-query rank files are `results/link_prediction/ranks/pathe_<kg>.csv`; a re-run gives the same
metrics and a few differently broken ties. The checkpoints store no
hyperparameters, and `nhead` and `num_agg_heads` change no tensor shape, so a wrong value loads without
error and changes the results: always evaluate with the values above or those of the config.

## Reproducing the article

`docs/REPRODUCE.md` maps every item of the article to a command, its output and the check that was made.
The values of the baselines taken from the literature (RotatE, EARL, NodePiece, Neural LP, DRUM, RuleN,
GraIL, NodePiece + CompGCN) are copied from the cited papers; nothing here produces them.

## Licence

MIT, see `LICENSE`.
