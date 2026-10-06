"""
Write CoDEx-L in LibKGE's raw format (train.txt / valid.txt / test.txt, tab-separated labels) from the files
PathE was trained and evaluated on, so that the entity set, the splits and the labels are identical.

usage: python make_libkge_dataset.py <data_root>/path_datasets/codex-large <kge>/data/codex-l
then:  cd <kge>/data && python preprocess/preprocess_default.py codex-l
(setup_libkge.sh runs both)
"""
import sys, os, json, torch

src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)
id2e = json.load(open(os.path.join(src, "train", "id2entity.json")))
id2r = json.load(open(os.path.join(src, "train", "id2relation.json")))
ent = {int(k): v for k, v in id2e.items()}
rel = {int(k): v for k, v in id2r.items()}
assert len(set(ent.values())) == len(ent) and len(set(rel.values())) == len(rel), "labels must be unique"
for split, out in (("train", "train.txt"), ("val", "valid.txt"), ("test", "test.txt")):
    t = torch.load(os.path.join(src, split, "triples.pt")).long().tolist()
    with open(os.path.join(dst, out), "w") as f:
        for h, r, o in t:
            f.write(f"{ent[h]}\t{rel[r]}\t{ent[o]}\n")
    print(f"{out}: {len(t)} triples")
print(f"entities {len(ent)}, relations {len(rel)} -> {dst}")
