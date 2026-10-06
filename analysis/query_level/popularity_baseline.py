"""Training-free popularity baseline: rank candidate answers of (h,r,?) by how often they are a tail of r in train (heads analogously).

usage: python analysis/query_level/popularity_baseline.py <kg> [--out CSV]   (default: results/link_prediction/ranks/popularity_<kg>.csv)
"""
import argparse, collections, numpy as np, pandas as pd, torch
from common import KGS, add_root_args, roots, dataset_dir, ranks_file
ap = add_root_args(argparse.ArgumentParser()); ap.add_argument("kg", choices=KGS); ap.add_argument("--out", default=None)
args = ap.parse_args(); data_root, results_root, _ = roots(args)
name = args.kg; root = dataset_dir(data_root, name); out = args.out or ranks_file(results_root, "popularity", name)
tr, va, te = [torch.load(f"{root}/{s}/triples.pt").long().numpy() for s in ("train","val","test")]
allt = np.concatenate([tr, va, te]); N = int(allt[:, [0, 2]].max()) + 1; R = int(allt[:, 1].max()) + 1
tailc = np.zeros((R, N), dtype=np.float32); headc = np.zeros((R, N), dtype=np.float32)
np.add.at(tailc, (tr[:, 1], tr[:, 2]), 1); np.add.at(headc, (tr[:, 1], tr[:, 0]), 1)
deg = np.bincount(tr[:, 0], minlength=N) + np.bincount(tr[:, 2], minlength=N)
tf, hf = collections.defaultdict(list), collections.defaultdict(list)
for h, r, t in allt: tf[(h, r)].append(t); hf[(r, t)].append(h)
rows = []
for i, (h, r, t) in enumerate(te):
    for side, true, s, filt in (("tail", t, tailc[r] + 1e-6 * deg, tf[(h, r)]), ("head", h, headc[r] + 1e-6 * deg, hf[(r, t)])):
        st = s[true]; keep = np.ones(N, bool); keep[filt] = False; neg = s[keep]
        rows.append((i, side, (neg > st).sum() + 1 + (neg == st).sum() / 2.0))
d = pd.DataFrame(rows, columns=["triple_idx", "side", "rank"]); d.to_csv(out, index=False)
print(name, "frequency baseline  MRR", round((1 / d["rank"]).mean(), 4), " Hits@10", round((d["rank"] <= 10).mean(), 4), d.assign(rr=1/d["rank"]).groupby("side").rr.mean().round(4).to_dict())
