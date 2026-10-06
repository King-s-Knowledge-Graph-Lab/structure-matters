"""
Rebuild the entity / relation id maps between PathE's datasets (PyKEEN ids) and the LibKGE ComplEx
checkpoints used as the entity-based reference of Section 7.  Reconstructed on 2026-10-04 from the
method of September 2026; the maps it writes are byte-identical to the ones used for the paper
(`results/link_prediction/idmaps/entmap_<kg>.npy`, `relmap_codex-large.npy`).

Entity map (FB15k-237, WN18RR, YAGO3-10): the extracted matrices
`data/path_datasets/<kg>/train/<prefix>-complexpretrained.pt` (one row per PyKEEN entity id, rows copied
from the LibKGE checkpoint) are matched row-by-row, as exact float32 byte strings, to the entity embedder of
the checkpoint.  Every row matches exactly one checkpoint row.  The checkpoints hold a few more entities
than PathE's datasets (e.g. 14,541 vs 14,505 on FB15k-237): those are entities without training triples
in PyKEEN's split and are simply never candidates.

Relation map: for CoDEx-Large the LibKGE dataset was built by us from PathE's own id files, so
`kge/data/codex-l/relation_ids.del` (and `entity_ids.del`) give the maps exactly.  For the three public
checkpoints no id files exist, and `complex_ranks.py` infers the relation permutation from the training
triples (linear assignment on the mean ComplEx score per relation).

usage:  python analysis/query_level/align_ids.py [--pathe-data DIR] [--ckpt-dir DIR] [--codex-ids DIR]
                                                [--out DIR] [--check DIR]
defaults (configs/paths.yaml): --pathe-data <data_root>/path_datasets, --ckpt-dir <checkpoint_root>/complex,
--codex-ids <libkge_root>/kge/data/codex-l (skipped if absent), --out <results_root>/link_prediction/idmaps.
Gate G2: --out <tmp dir> --check results/link_prediction/idmaps (all maps must be reported "ok").
"""
import argparse, json, os, sys, numpy as np, torch
from pathe.paths import load_paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from libkge_checkpoint import load  # unpickler that stubs the kge.* classes stored in LibKGE checkpoints

KGS = {"fb15k237": ("fb15k-237", "fb15k-237-complex.pt"),
       "wn18rr": ("wnrr", "wnrr-complex.pt"),
       "yago": ("yago3-10", "yago3-10-complex.pt")}

def embedder(sd, what):
    keys = [k for k in sd if what in k and k.endswith("weight") and "_base_model" not in k]
    assert len(keys) == 1, keys
    return sd[keys[0]].float()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pathe-data", default=None); ap.add_argument("--ckpt-dir", default=None)
    ap.add_argument("--codex-ids", default=None); ap.add_argument("--out", default=None); ap.add_argument("--check", default=None)
    a = ap.parse_args()
    paths = load_paths()
    a.pathe_data = a.pathe_data or str(paths["data_root"] / "path_datasets")
    a.ckpt_dir = a.ckpt_dir or str(paths["checkpoint_root"] / "complex")
    if a.codex_ids is None and (paths["libkge_root"] / "kge" / "data" / "codex-l").is_dir():
        a.codex_ids = str(paths["libkge_root"] / "kge" / "data" / "codex-l")
    a.out = a.out or str(paths["results_root"] / "link_prediction" / "idmaps")
    os.makedirs(a.out, exist_ok=True)
    for kg, (prefix, ck) in KGS.items():
        X = torch.load(f"{a.pathe_data}/{kg}/train/{prefix}-complexpretrained.pt", map_location="cpu").float().contiguous().numpy()
        E = embedder(load(f"{a.ckpt_dir}/{ck}")["model"][0], "_entity_embedder").contiguous().numpy()
        assert X.shape[1] == E.shape[1], (X.shape, E.shape)
        index = {}
        for j, row in enumerate(E): index.setdefault(row.tobytes(), []).append(j)
        ent_map = -np.ones(X.shape[0], dtype=np.int64); dup = 0
        for i, row in enumerate(X):
            js = index.get(row.tobytes(), [])
            if len(js) == 1: ent_map[i] = js[0]
            elif len(js) > 1: dup += 1
        matched = int((ent_map >= 0).sum()); one_to_one = len(set(ent_map[ent_map >= 0])) == matched
        print(f"{kg}: ours {X.shape[0]} libkge {E.shape[0]} | matched {matched} | ambiguous {dup} | one-to-one {one_to_one}")
        assert matched == X.shape[0] and one_to_one and dup == 0
        np.save(f"{a.out}/entmap_{kg}.npy", ent_map)
    if a.codex_ids:
        root = f"{a.pathe_data}/codex-large/train"
        id2e = json.load(open(f"{root}/id2entity.json")); id2r = json.load(open(f"{root}/id2relation.json"))
        for name, id2x, fn in (("entmap", id2e, "entity_ids.del"), ("relmap", id2r, "relation_ids.del")):
            lib = {}
            for line in open(f"{a.codex_ids}/{fn}"):
                i, s = line.rstrip("\n").split("\t"); lib[s] = int(i)
            m = np.array([lib[id2x[str(i)]] for i in range(len(id2x))], dtype=np.int64)
            assert len(set(m)) == len(m) == len(lib), (len(set(m)), len(m), len(lib))
            np.save(f"{a.out}/{name}_codex-large.npy", m); print(f"codex-large {name}: {len(m)} ids, exact")
    if a.check:
        for f in sorted(os.listdir(a.out)):
            if f.endswith(".npy") and os.path.exists(f"{a.check}/{f}"):
                same = np.array_equal(np.load(f"{a.out}/{f}"), np.load(f"{a.check}/{f}"))
                print(("ok   " if same else "DIFF ") + f)

if __name__ == "__main__":
    main()
