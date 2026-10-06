"""PathE (entity-agnostic) vs ComplEx (entity-based, public LibKGE checkpoints) on the same test queries.

usage: python analysis/query_level/ea_vs_eb.py   (reads results/link_prediction/ranks/ and results/query_level/ceiling/)
"""
import argparse
import numpy as np, pandas as pd, torch
from scipy.stats import spearmanr, rankdata
from common import KGS, add_root_args, roots, dataset_dir, ranks_file, ceiling_file
def pspear(d, x, y, c):
    Rm = np.column_stack([rankdata(d[c]), np.ones(len(d))]); res = []
    for v in (x, y):
        rv = rankdata(d[v]); b, *_ = np.linalg.lstsq(Rm, rv, rcond=None); res.append(rv - Rm @ b)
    return np.corrcoef(res[0], res[1])[0, 1]
args = add_root_args(argparse.ArgumentParser()).parse_args(); data_root, results_root, _ = roots(args)
for name in KGS:
    pr = ranks_file(results_root, "pathe", name)
    root=dataset_dir(data_root, name); tr=torch.load(f"{root}/train/triples.pt").long().numpy()
    d=pd.read_csv(pr)[["triple_idx","side","rank"]].rename(columns={"rank":"pathe"})
    d=d.merge(pd.read_csv(ranks_file(results_root, "complex", name))[["triple_idx","side","rank"]].rename(columns={"rank":"complex"}),on=["triple_idx","side"])
    d=d.merge(pd.read_csv(ranks_file(results_root, "popularity", name)).rename(columns={"rank":"pop"}),on=["triple_idx","side"])
    d=d.merge(pd.read_csv(ceiling_file(results_root, name))[["triple_idx","side","target","k"]],on=["triple_idx","side"])
    N=int(max(tr[:,[0,2]].max(), d.target.max()))+1; deg=np.bincount(tr[:,0],minlength=N)+np.bincount(tr[:,2],minlength=N); d["deg"]=deg[d.target]
    for m in ("pathe","complex","pop"): d["rr_"+m]=1/d[m]
    print(f"\n===== {name}  (n={len(d)})  MRR PathE {d.rr_pathe.mean():.3f} | ComplEx {d.rr_complex.mean():.3f} | popularity {d.rr_pop.mean():.3f}")
    print("Spearman(rank, answer degree): PathE %.3f  ComplEx %.3f | Spearman(rank, class size k): PathE %.3f ComplEx %.3f | partial k given degree: PathE %.3f ComplEx %.3f" % (
        spearmanr(d.pathe,d.deg)[0], spearmanr(d["complex"],d.deg)[0], spearmanr(d.pathe,d.k)[0], spearmanr(d["complex"],d.k)[0], pspear(d,"k","pathe","deg"), pspear(d,"k","complex","deg")))
    print("Spearman between rankings: PathE-popularity %.3f  ComplEx-popularity %.3f  PathE-ComplEx %.3f" % (spearmanr(d.pathe,d["pop"])[0], spearmanr(d["complex"],d["pop"])[0], spearmanr(d.pathe,d["complex"])[0]))
    d["dbin"]=pd.cut(d.deg.clip(lower=1),[0,2,5,10,20,50,100,1e9],labels=["1-2","3-5","6-10","11-20","21-50","51-100",">100"])
    t=d.groupby("dbin",observed=True).agg(n=("pathe","size"),PathE=("rr_pathe","mean"),ComplEx=("rr_complex","mean"),popularity=("rr_pop","mean"))
    t["retained"]=t.PathE/t.ComplEx; print(t.round(3).to_string())
    d["d4"]=pd.cut(d.deg.clip(lower=1),[0,5,20,100,1e9],labels=["1-5","6-20","21-100",">100"]); d["kb"]=pd.cut(d.k,[0,1,10,100,1e9],labels=["unique","2-10","11-100",">100"])
    for m in ("pathe","complex"):
        g=d.groupby(["d4","kb"],observed=True)["rr_"+m].mean().unstack().round(3); print(f"MRR of {m} by answer degree (rows) x context-class size (cols):"); print(g.to_string())
    print("by side:", d.groupby("side")[["rr_pathe","rr_complex","rr_pop"]].mean().round(3).to_dict())
