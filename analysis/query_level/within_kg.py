"""Step 5 (first pass): within-KG association between PathE's per-query rank and properties of the target entity.

usage: python analysis/query_level/within_kg.py <kg>
"""
import argparse, numpy as np, pandas as pd, torch
from scipy.stats import spearmanr
from common import KGS, add_root_args, roots, dataset_dir, ranks_file, ceiling_file
ap = add_root_args(argparse.ArgumentParser()); ap.add_argument("kg", choices=KGS)
args = ap.parse_args(); data_root, results_root, _ = roots(args)
name = args.kg; root = dataset_dir(data_root, name); ranks = ranks_file(results_root, "pathe", name)
tr = torch.load(f"{root}/train/triples.pt").long().numpy()
N = int(max(tr[:, 0].max(), tr[:, 2].max())) + 1
r = pd.read_csv(ranks); c = pd.read_csv(ceiling_file(results_root, name))
d = r.merge(c, on=["triple_idx", "side"])
N = max(N, int(d.target.max()) + 1)
deg = np.bincount(tr[:, 0], minlength=N) + np.bincount(tr[:, 2], minlength=N)
# number of distinct (direction, relation) types around the entity
types = np.zeros(N)
for col, off in ((0, 0), (2, 1000000)):
    u = np.unique(np.stack([tr[:, col], tr[:, 1] + off], 1), axis=0)
    types += np.bincount(u[:, 0], minlength=N)
d["deg"], d["reltypes"], d["rr"] = deg[d.target], types[d.target], 1.0 / d["rank"]
print(f"== {name}: queries {len(d)}  MRR {d.rr.mean():.4f}  Hits@10 {(d['rank']<=10).mean():.4f}")
d["kbin"] = pd.cut(d.k, [0, 1, 2, 5, 20, 100, 1e9], labels=["1", "2", "3-5", "6-20", "21-100", ">100"])
print(d.groupby("kbin", observed=True).agg(n=("rr", "size"), mrr=("rr", "mean"), hits10=("rank", lambda x: (x <= 10).mean()), med_deg=("deg", "median")).round(4))
d["dbin"] = pd.qcut(d.deg, 5, duplicates="drop")
print(d.groupby("dbin", observed=True).agg(n=("rr", "size"), mrr=("rr", "mean"), uniq=("k", lambda x: (x == 1).mean())).round(4))
for v in ["k", "deg", "reltypes"]:
    rho, p = spearmanr(d[v], d["rank"]); print(f"spearman(rank, {v}) = {rho:+.3f}  p={p:.1e}")
# partial association of rank with uniqueness within degree strata
d["uniq"] = (d.k == 1).astype(int)
d["dstr"] = pd.qcut(d.deg, 10, duplicates="drop")
g = d.groupby(["dstr", "uniq"], observed=True).rr.mean().unstack()
g["n_uniq"] = d.groupby("dstr", observed=True).uniq.sum(); g["n"] = d.groupby("dstr", observed=True).size()
print("MRR by degree decile, non-unique (0) vs unique (1) target context:"); print(g.round(4))
