# ComplEx on CoDEx-L (entity-based reference for the query-level analysis of Section 7)

Why: the ComplEx checkpoint that the CoDEx authors released for CoDEx-L is no longer downloadable (dead Dropbox link), and ComplEx is the entity-based reference used on the other three KGs (public LibKGE checkpoints). This folder trains the same model with **the configuration released by the CoDEx authors**.

Files
- `configs/baselines/complex_codex-l_published.yaml`: verbatim copy of `models/link-prediction/codex-l/complex/config.yaml` from github.com/tsafavi/codex (reciprocal-relations ComplEx, dimension 128, 1vsAll, batch 1,024, Adagrad, lr 0.265, at most 400 epochs, early stopping on the validation MRR). Published test MRR: 0.294.
- `configs/baselines/complex_codex-l.yaml`: the same, with three changes only: `dataset.name: codex-l`, `job.device: cuda`, `train.subbatch_auto_tune: true` (halves the sub-batch automatically if the 8 GB of the A2000 are not enough; results are unaffected).
- `make_libkge_dataset.py`: writes CoDEx-L in LibKGE's format from PathE's `path_datasets/codex-large` (77,951 entities, 69 relations, 551,193 / 30,622 / 30,622 triples), so that entities, splits and labels are exactly those PathE was evaluated on. Nothing is downloaded.
- `setup_libkge.sh`: creates `venv_kge/`, installs torch 2.1.1+cu118, clones LibKGE at commit `9bca31f`, applies `libkge_optional_search.patch`, installs it, builds the dataset and checks the GPU. **Does not train.**
- `libkge_optional_search.patch`: makes the imports of LibKGE's hyper-parameter-search backends (`ax`, `hpbandster`) optional; the training does not use them.
- `train.sh`: starts the training, or resumes it if the run folder already exists.

Everything is written under `artifacts/libkge/` (`LIBKGE_ROOT`); the dataset is read from `artifacts/data/path_datasets/codex-large` (`PATHE_DATA`).

Steps
1. `bash scripts/07_complex_baseline/setup_libkge.sh` (needs internet for pip and the LibKGE clone; about 10 minutes).
2. `nohup bash scripts/07_complex_baseline/train.sh > train.log 2>&1 &` and `tail -f train.log`. Validation runs every 5 epochs; training stops early when the validation MRR has not improved for 5 validations.
3. When it ends, check the last `mean_reciprocal_rank_filtered` lines in `train.log`: a validation MRR around 0.29 to 0.30 means the run matches the published model. Our run: test MRR 0.2939.
4. Copy `artifacts/libkge/run/checkpoint_best.pt` to `artifacts/checkpoints/complex/codex-l-complex.pt`. `analysis/query_level/align_ids.py` reads `artifacts/libkge/kge/data/codex-l/{entity_ids,relation_ids}.del` for the CoDEx-Large id maps, and `analysis/query_level/complex_ranks.py codex-large` computes the ranks on PathE's test queries in minutes.

The released checkpoints (ours and the three public LibKGE ones, `fb15k-237-complex.pt`, `wnrr-complex.pt`, `yago3-10-complex.pt`) are listed in `docs/ARTIFACTS.md`; with them, steps 1–3 are not needed for the analysis (the `.del` files are, see `setup_libkge.sh`, or the stored maps in `results/link_prediction/idmaps/`).

If `pip install -e kge` fails on one of LibKGE's hyper-parameter-search dependencies (`ax-platform`, `botorch`, `hpbandster`): `pip install -e kge --no-deps` followed by `pip install numpy pandas pyyaml path numba "sqlalchemy<2.0.0" igraph ConfigSpace torchviz ax-platform hpbandster`.
