# Artifacts: checkpoints and path datasets

The trained models and the path datasets are not in git. They are published as nine archives,
listed with their md5 in `scripts/00_setup/artifacts.txt`; three further checkpoints are the public
LibKGE ComplEx models, downloaded from LibKGE (`scripts/00_setup/artifacts_external.txt`).

The archives are attached to the release `v1.0` of this repository:

```bash
RELEASE=https://github.com/King-s-Knowledge-Graph-Lab/structure-matters/releases/download/v1.0
scripts/00_setup/fetch_artifacts.sh $RELEASE                       # everything (568 MB)
scripts/00_setup/fetch_artifacts.sh $RELEASE pathe-checkpoints.tar.gz paths-fb15k237.tar.gz
PATHE_ARTIFACTS=$RELEASE scripts/00_setup/fetch_artifacts.sh        # same, location from the environment
```

The location may also be a local folder holding the archives. The script checks every md5, unpacks
into `data_root` and `checkpoint_root` of `configs/paths.yaml`, and links each checkpoint into the run
folder of its config (`scripts/00_setup/link_runs.py`), so that

```bash
python -m pathe.runner test pathe --config configs/relation_prediction/fb15k237.yaml --device cuda
```

tests the released model. Downloads are kept in `artifacts/downloads/` (`$PATHE_DOWNLOADS`) and are
not fetched again.

The knowledge graphs themselves are downloaded by PyKEEN on first use. The structure analysis
(Section 6, `analysis/structure/`) needs none of the archives.

## Archives

| Archive | Size | md5 | Unpacks into | Needed for |
|---|---|---|---|---|
| `pathe-checkpoints.tar.gz` | 72.0 MB | `1f4598a006aba0f45975458f24e90677` | `checkpoint_root` | every PathE result (23 checkpoints, below) |
| `complex-codex-l.tar.gz` | 73.4 MB | `f0ff1d74aeee420173d87a58fceae9fe` | `checkpoint_root` | Section 7: ComplEx on CoDEx-Large |
| `complex-entity-matrices.tar.gz` | 91.7 MB | `3269d876fe4f80623d63f406fe3ed8ff` | `data_root` | Section 7: `align_ids.py` (gate G2) only |
| `paths-fb15k237.tar.gz` | 14.1 MB | `160823512f147a7acd677f06b75b6248` | `data_root` | link prediction, ablations, path count (FB15k-237) |
| `paths-codex-large.tar.gz` | 29.2 MB | `55d7ff5c65123b4618e9c56bc8a1f775` | `data_root` | link prediction, ablations, path count (CoDEx-Large) |
| `paths-wn18rr.tar.gz` | 13.5 MB | `6a5cabafebd526e88e7f6519d781610b` | `data_root` | link prediction (WN18RR) |
| `paths-yago.tar.gz` | 49.6 MB | `0b12ffbb8d529e7309e1a2a47aa39d74` | `data_root` | link prediction (YAGO3-10) |
| `paths-relation-prediction.tar.gz` | 11.5 MB | `379cbaacd2e5c6ba2db535b1105b9716` | `data_root` | relation prediction (with `paths-fb15k237` / `paths-wn18rr`) |
| `paths-inductive.tar.gz` | 12.1 MB | `91a6cc36311906e2bcb71815671e777f` | `data_root` | inductive link prediction |

Contents:

- `paths-<kg>`: `path_datasets/<kg>/{train,val,test}/` with `20_10_er_paths.csv` (20 walks of up to
  10 steps per entity), `relational_context.csv`, `rel_neighbors.csv`, `triples.pt`,
  `id2entity.json`, `id2relation.json`. These are the files the reported models were trained and
  evaluated on; a rebuild with `scripts/01_data/build_transductive.sh` gives other walks (the
  released ones were mined without a seed).
- `paths-relation-prediction`: `path_datasets/{fb15k237,wn18rr}/{train,val,test}/20_20_er_paths.csv`.
- `paths-inductive`: `path_datasets/{fb15k237,wn18rr}/v1..v4/{train,val,test,inference}/`, the
  GraIL splits: `triples.pt` in every part, `id2entity.json` and `id2relation.json` in `train/` and
  `inference/`, the training graph's `20_10_er_paths.csv` and `relational_context.csv`, and the
  inference graph's `20_10_er_paths_inference.csv` and `relational_context_inference.csv`. These are
  the files the inductive models are trained and evaluated on.
- `complex-entity-matrices`: `path_datasets/{fb15k237,wn18rr,yago}/train/<prefix>-complexpretrained.pt`
  and `<prefix>-complex_labels.json`, the entity embeddings of the public LibKGE models in PyKEEN id
  order, from which `align_ids.py` rebuilds `results/link_prediction/idmaps/`.

The archives are built by `scripts/00_setup/pack_artifacts.sh` and are byte-reproducible.

## Checkpoints

Paths are relative to `checkpoint_root`. Each PathE checkpoint is linked to the config of the same
name under `configs/`, which holds the arguments of its training run.

| File | Size | md5 | Paper item |
|---|---|---|---|
| `link_prediction/fb15k237.ckpt` | 2.6 MB | `9bcaf5ac213ab368315ee4ef5d79af66` | link prediction (Section 5.2), parameter counts (Section 5.4), Section 7 |
| `link_prediction/codex-large.ckpt` | 8.2 MB | `aa06deaae8e5d583a2083e79cce5ca6e` | link prediction, Sections 5.4 and 7; also `configs/ablations/codex-large_ppt8.yaml` |
| `link_prediction/wn18rr.ckpt` | 8.1 MB | `f11781bb48842d582b836bb828d30995` | link prediction, Sections 5.4 and 7 |
| `link_prediction/yago.ckpt` | 2.9 MB | `3843bd9bb782c840f9bf8a04ce09e90a` | link prediction, Sections 5.4 and 7 (epoch 10 of its run) |
| `relation_prediction/fb15k237.ckpt` | 10.2 MB | `93a0877f8b872e8be273cd2ae7692945` | relation prediction (Section 5.1) |
| `relation_prediction/wn18rr.ckpt` | 0.7 MB | `70a01208f4432ac35ca7523664e8e165` | relation prediction (Section 5.1) |
| `inductive/fb15k237_v1.ckpt` | 2.0 MB | `00109e154a9359d6e1781f8d3a56751a` | inductive link prediction (Section 5.3) |
| `inductive/fb15k237_v2.ckpt` | 2.1 MB | `3cb8075ad86a134f279c4708ba7ab97d` | inductive link prediction |
| `inductive/fb15k237_v3.ckpt` | 2.2 MB | `0d94e69691381b02cf79ec49691798a3` | inductive link prediction |
| `inductive/fb15k237_v4.ckpt` | 2.2 MB | `ee0da0b8f907db08acf624fac229ed11` | inductive link prediction |
| `inductive/wn18rr_v1.ckpt` | 1.2 MB | `7f061f17cda99ec8686b925c6301bdeb` | inductive link prediction |
| `inductive/wn18rr_v2.ckpt` | 1.2 MB | `5f41101ea78d6b922fc1a975c7f7126f` | inductive link prediction |
| `inductive/wn18rr_v3.ckpt` | 1.2 MB | `fe2bd2f61819e0cb1152b09a1321e971` | inductive link prediction |
| `inductive/wn18rr_v4.ckpt` | 1.2 MB | `dd726851dd6e8d4a646dccfd989acb34` | inductive link prediction |
| `ablations/fb15k237_no_aggregator.ckpt` | 2.3 MB | `dd4078509fbe6685fdcc9a8d9d491f2d` | component ablation (Section 5.5, App. F) |
| `ablations/fb15k237_single_path.ckpt` | 2.6 MB | `3690a26606ab3a9e6f6db97e239a708e` | component ablation (same value as the 2-path point of App. G) |
| `ablations/fb15k237_no_positional.ckpt` | 2.6 MB | `0ec424a58242cfb9051000c5da7ff090` | component ablation |
| `ablations/fb15k237_ppt4.ckpt` | 2.6 MB | `eeb3fd39e0fc8032e30b8e315dd286b6` | path count (App. G), UNCONFIRMED |
| `ablations/fb15k237_ppt16.ckpt` | 2.6 MB | `6f3544f2ca3f83b77d5839ec4d939e74` | path count (App. G), UNCONFIRMED |
| `ablations/codex-large_no_aggregator.ckpt` | 5.8 MB | `29601cb46eafc0a9c42d76823528d6b5` | component ablation |
| `ablations/codex-large_single_path.ckpt` | 7.0 MB | `e9e85623a00a9dfdb530ae0c5158364a` | component ablation (same value as the 2-path point of App. G) |
| `ablations/codex-large_no_positional.ckpt` | 8.2 MB | `7f4591019e5aa86ead7b017ca681057c` | component ablation |
| `ablations/codex-large_ppt4.ckpt` | 2.7 MB | `a37ae1abf283e7114e15eaa3dab32ebb` | path count (App. G), UNCONFIRMED |
| `complex/codex-l-complex.pt` | 80.0 MB | `bca5d676e84645d5b53fafeaa91dced7` | Section 7: ComplEx on CoDEx-Large, trained with `scripts/07_complex_baseline/` (test MRR 0.2939) |

Public LibKGE checkpoints (`artifacts_external.txt`; Section 7, ComplEx on the other three KGs):

| File | Size | md5 | Source |
|---|---|---|---|
| `complex/fb15k-237-complex.pt` | 31.3 MB | `2b2ed335b790269f893f905b9350739b` | <http://web.informatik.uni-mannheim.de/pi1/iclr2020-models/fb15k-237-complex.pt> |
| `complex/wnrr-complex.pt` | 42.4 MB | `6b4c3a339b8a50dce56b727c1e7440bd` | <http://web.informatik.uni-mannheim.de/pi1/iclr2020-models/wnrr-complex.pt> |
| `complex/yago3-10-complex.pt` | 126.8 MB | `6e71e7debf9414d658af350839e5969e` | <http://web.informatik.uni-mannheim.de/pi1/libkge-models/yago3-10-complex.pt> |

### Released files and the training originals

The released checkpoints are copies of the training checkpoints made by
`scripts/00_setup/clean_checkpoint.py`: the absolute paths stored by the training job (the
`ModelCheckpoint` state of PyTorch Lightning) are reduced to their last component, and the user and
host fields of LibKGE's trace are emptied. Weights, optimizer state, scores and epochs are unchanged
(checked tensor by tensor). md5 of the originals, which other documents may cite:

| File | md5 of the training original |
|---|---|
| `link_prediction/fb15k237.ckpt` | `4a5a747cee58daeec559214653ca5a7c` |
| `link_prediction/codex-large.ckpt` | `bded3e2539cb59b0ee0fec2715de1d75` |
| `link_prediction/wn18rr.ckpt` | `1d1c4a96da073f508735f87435a76bca` |
| `link_prediction/yago.ckpt` | `da814bd87fb1206d23adc4da26285ec3` |
| `relation_prediction/fb15k237.ckpt` | `88b83e772192480868ba7c1f97a0d5c0` |
| `relation_prediction/wn18rr.ckpt` | `ccad66bb1233b4691db316feececb7da` |
| `inductive/fb15k237_v1.ckpt` | `0034d466705a18bcb4f9a9ae43286eff` |
| `inductive/fb15k237_v2.ckpt` | `49f33f806d37006c7bf0cae239a203a2` |
| `inductive/fb15k237_v3.ckpt` | `337492cf6ffecf5d50ebf63cea16294f` |
| `inductive/fb15k237_v4.ckpt` | `043ff56d919cb5e3466cfc8d944a07aa` |
| `inductive/wn18rr_v1.ckpt` | `b8c395af7f6ad5c373d206731c919609` |
| `inductive/wn18rr_v2.ckpt` | `f6754c8f1a6c71367bbae871a2fe8fb4` |
| `inductive/wn18rr_v3.ckpt` | `787b6b6e396a4484cfa28bc70a92d29b` |
| `inductive/wn18rr_v4.ckpt` | `bb285a4d1094173321ae35730458df1a` |
| `ablations/fb15k237_no_aggregator.ckpt` | `e5ac74cc4942c31fffbee87da3870d04` |
| `ablations/fb15k237_single_path.ckpt` | `6e3aa19c86d55d8c44e7e7e53fce09ee` |
| `ablations/fb15k237_no_positional.ckpt` | `07ff51480987ab151d8770891729598c` |
| `ablations/fb15k237_ppt4.ckpt` | `54e6f8bd5066dc8eec7c7e7fdf3f913c` |
| `ablations/fb15k237_ppt16.ckpt` | `e33c9ec6d511d0cc90aeb3976b3416cd` |
| `ablations/codex-large_no_aggregator.ckpt` | `7ce7e506f8f5a1017a2ae629185bd6ae` |
| `ablations/codex-large_single_path.ckpt` | `a2bf8d8a5acb0a88828c6c2b84b917f0` |
| `ablations/codex-large_no_positional.ckpt` | `b00937ac3503f333eb3660f14be38e7d` |
| `ablations/codex-large_ppt4.ckpt` | `0bc4273515ebfd5c54684a1578f271ff` |
| `complex/codex-l-complex.pt` | `d023a21d63c96795c05fccaf3f099be8` |

## Evaluation of the ablation checkpoints

`analysis/query_level/dump_ranks.py`, full test sets, paths per entity = `max_ppt / 2` of the
config, `--nhead 2`, `--seed 7`, GPU (2026-10-05). The paper's values come from `full_eval` logs;
which paths those evaluations drew is not recorded, and for the models with few paths the MRR moves
by up to 0.016 with the path sample (seeds 7, 42, 46 and 0, last column).

| Config | Paper MRR | MRR | Hits@10 | MRR over 4 seeds |
|---|---|---|---|---|
| `fb15k237_no_aggregator` | 0.215 | 0.2145 | 0.3580 | |
| `fb15k237_single_path` | 0.184 | 0.1841 | 0.3267 | |
| `fb15k237_no_positional` | 0.191 | 0.1891 | 0.3185 | |
| `fb15k237_ppt4` | 0.2017 (App. G) | 0.2053 | 0.3359 | 0.2037–0.2055 |
| `fb15k237_ppt16` | 0.2146 (App. G) | 0.2145 | 0.3621 | |
| `codex-large_no_aggregator` | 0.132 | 0.1312 | 0.2300 | |
| `codex-large_single_path` | 0.105 | 0.1036 | 0.1879 | 0.1001–0.1092 |
| `codex-large_no_positional` | 0.118 | 0.1138 | 0.2039 | |
| `codex-large_ppt4` | 0.1252 (App. G) | 0.1230 | 0.2158 | 0.1183–0.1341 |

## Not released

- The `20_20` path files of CoDEx-Large and YAGO3-10, and other path settings: no reported model
  uses them.
- The W&B run records and TensorBoard logs of the training runs.
- The CoDEx-Large 16-path run (ended after 4 epochs) and the other checkpoints of the authors'
  experiments that no reported number comes from.
