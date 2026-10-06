"""
Query-level analysis: how PathE's rank on a test query relates to the relational evidence available
for the answer entity, within each KG and across KGs (direct standardisation over evidence strata).

usage: python analysis/query_level/query_level.py [--kgs KG [KG ...]] [--out JSON]
inputs: results/link_prediction/ranks/pathe_<kg>.csv, results/query_level/ceiling/ceiling_<kg>.csv
default output: results/query_level/query_level_4kg.json
"""
import argparse, sys, json, collections
import numpy as np, pandas as pd, torch
from scipy.stats import spearmanr, rankdata
from common import KGS, add_root_args, roots, dataset_dir, ranks_file, ceiling_file, query_level_dir

BINS = [0, 2, 5, 10, 20, 50, 100, 10**9]
LABELS = ["1-2", "3-5", "6-10", "11-20", "21-50", "51-100", ">100"]


def load(name, root, ranks, ceiling):
    tr = torch.load(f"{root}/train/triples.pt").long().numpy()
    r = pd.read_csv(ranks)
    c = pd.read_csv(ceiling)[["triple_idx", "side", "target", "k"]]
    d = r.merge(c, on=["triple_idx", "side"])
    N = int(max(tr[:, [0, 2]].max(), d.target.max(), d["head"].max(), d["tail"].max())) + 1
    deg = np.bincount(tr[:, 0], minlength=N) + np.bincount(tr[:, 2], minlength=N)
    types = np.zeros(N)
    for col, off in ((0, 0), (2, 10**6)):
        u = np.unique(np.stack([tr[:, col], tr[:, 1] + off], 1), axis=0)
        types += np.bincount(u[:, 0], minlength=N)
    relfreq = np.bincount(tr[:, 1])
    d["kg"] = name
    d["deg"] = deg[d.target]
    d["types"] = types[d.target]
    anchor = np.where(d.side == "tail", d["head"], d["tail"])
    d["anchor_deg"] = deg[anchor]
    d["relfreq"] = relfreq[d.relation]
    d["rr"] = 1.0 / d["rank"]
    d["h10"] = (d["rank"] <= 10).astype(float)
    d["logrank"] = np.log(d["rank"])
    d["bin"] = pd.cut(d.deg.clip(lower=1), BINS, labels=LABELS)
    d["unseen"] = (d.deg == 0)
    stats = dict(kg=name, entities=int((deg > 0).sum()), median_entity_degree=float(np.median(deg[deg > 0])),
                 share_entities_deg_le5=float((deg[deg > 0] <= 5).mean()))
    return d, stats


def partial_spearman(d, x, y, controls):
    """Spearman partial correlation of x and y given controls (rank-transform, residualise by OLS)."""
    R = np.column_stack([rankdata(d[c]) for c in controls] + [np.ones(len(d))])
    res = []
    for v in (x, y):
        rv = rankdata(d[v]); beta, *_ = np.linalg.lstsq(R, rv, rcond=None); res.append(rv - R @ beta)
    return float(np.corrcoef(res[0], res[1])[0, 1])


def std_ols(d, y, xs):
    X = np.column_stack([(np.log1p(d[c]) - np.log1p(d[c]).mean()) / (np.log1p(d[c]).std() + 1e-12) for c in xs] + [np.ones(len(d))])
    yy = (d[y] - d[y].mean()) / d[y].std()
    beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
    r2 = 1 - ((yy - X @ beta) ** 2).sum() / (yy ** 2).sum()
    return dict(zip(xs, np.round(beta[:-1], 3))), round(float(r2), 3)


ap = add_root_args(argparse.ArgumentParser())
ap.add_argument("--kgs", nargs="+", choices=KGS, default=KGS); ap.add_argument("--out", default=None)
args = ap.parse_args(); data_root, results_root, _ = roots(args)
frames, kgstats = [], []
for name in args.kgs:
    d, s = load(name, dataset_dir(data_root, name), ranks_file(results_root, "pathe", name), ceiling_file(results_root, name))
    frames.append(d); kgstats.append(s)
D = pd.concat(frames, ignore_index=True)
out = {"kg": kgstats, "per_kg": {}}

print("\n=== 1. Overall, and evidence available for the answer entity")
for name, d in D.groupby("kg", sort=False):
    print(f"{name:12s} queries {len(d):6d}  MRR {d.rr.mean():.4f}  Hits@10 {d.h10.mean():.4f}  | answer degree: median {d.deg.median():.0f}, "
          f"share <=5: {(d.deg <= 5).mean():.3f}, share <=10: {(d.deg <= 10).mean():.3f} | answer unique context {(d.k == 1).mean():.3f}")

print("\n=== 2. MRR (and share of queries) by degree of the answer entity in the training graph")
t = D.groupby(["kg", "bin"], observed=True, sort=False).agg(n=("rr", "size"), mrr=("rr", "mean"), h10=("h10", "mean")).reset_index()
t["share"] = t.n / t.groupby("kg").n.transform("sum")
piv = t.pivot(index="bin", columns="kg", values=["mrr", "share"]).reindex(LABELS).round(3)
print(piv.to_string())
out["by_degree_bin"] = t.round(4).to_dict(orient="records")

print("\n=== 3. Within-KG association (Spearman with rank; negative = more of x, better rank)")
for name, d in D.groupby("kg", sort=False):
    row = {}
    for v in ["deg", "types", "k", "anchor_deg", "relfreq"]:
        row[v] = round(float(spearmanr(d[v], d["rank"])[0]), 3)
    row["k | deg"] = round(partial_spearman(d, "k", "rank", ["deg"]), 3)
    row["types | deg"] = round(partial_spearman(d, "types", "rank", ["deg"]), 3)
    row["deg | k,types"] = round(partial_spearman(d, "deg", "rank", ["k", "types"]), 3)
    beta, r2 = std_ols(d, "logrank", ["deg", "types", "k", "anchor_deg", "relfreq"])
    print(f"{name:12s} {row}   | std. OLS on log-rank: {beta}  R2={r2}")
    out["per_kg"][name] = {"spearman": row, "ols_beta": {k: float(v) for k, v in beta.items()}, "ols_r2": r2}

print("\n=== 4. Direct standardisation: MRR each KG would have with the answer-degree distribution of another KG")
names = list(D.kg.unique())
cell = D.groupby(["kg", "bin"], observed=True).rr.mean().unstack("bin").reindex(columns=LABELS)
share = D.groupby(["kg", "bin"], observed=True).size().unstack("bin").reindex(columns=LABELS).fillna(0)
share = share.div(share.sum(1), axis=0)
res = pd.DataFrame(index=names, columns=names, dtype=float)
for a in names:          # rates of a
    for b in names:      # distribution of b
        w = share.loc[b].copy(); ok = cell.loc[a].notna()
        res.loc[a, b] = float((cell.loc[a][ok] * w[ok]).sum() / w[ok].sum())
print("rows: KG whose per-stratum MRR is used; columns: KG whose distribution of answer degree is used")
print(res.round(3).to_string())
out["standardised_mrr"] = res.round(4).to_dict()

print("\n=== 5. Pooled model: how much of the between-KG gap is accounted for by query-level evidence")
ref = names[0]
for label, xs in (("KG only", []), ("KG + degree", ["deg"]), ("KG + degree, types, class size, anchor degree, rel. freq.", ["deg", "types", "k", "anchor_deg", "relfreq"])):
    cols = [np.log1p(D[c]) for c in xs] + [(D.kg == n).astype(float) for n in names[1:]] + [np.ones(len(D))]
    X = np.column_stack(cols); y = D.logrank.values
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r2 = 1 - ((y - X @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    gaps = {n: round(float(b), 3) for n, b in zip(names[1:], beta[len(xs):-1])}
    print(f"{label:60s} R2={r2:.3f}  log-rank gap vs {ref}: {gaps}")
    out.setdefault("pooled", []).append({"model": label, "r2": round(float(r2), 4), "gap_vs_" + ref: gaps})
json.dump(out, open(args.out or query_level_dir(results_root) / "query_level_4kg.json", "w"), indent=1)
