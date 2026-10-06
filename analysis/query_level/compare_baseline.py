"""PathE vs the popularity baseline on the same test queries.

usage: python analysis/query_level/compare_baseline.py
"""
import argparse
import numpy as np, pandas as pd, torch
from scipy.stats import spearmanr
from common import add_root_args, roots, dataset_dir, ranks_file
args = add_root_args(argparse.ArgumentParser()).parse_args(); data_root, results_root, _ = roots(args)
for name in ["fb15k237", "wn18rr", "yago"]:
    ranks = ranks_file(results_root, "pathe", name)
    root=dataset_dir(data_root, name); tr=torch.load(f"{root}/train/triples.pt").long().numpy()
    p=pd.read_csv(ranks); f=pd.read_csv(ranks_file(results_root, "popularity", name)).rename(columns={"rank":"frank"})
    d=p.merge(f,on=["triple_idx","side"]); N=int(max(tr[:,[0,2]].max(), d["head"].max(), d["tail"].max()))+1
    deg=np.bincount(tr[:,0],minlength=N)+np.bincount(tr[:,2],minlength=N)
    tgt=np.where(d.side=="tail",d["tail"],d["head"]); d["deg"]=deg[tgt]
    d["bin"]=pd.cut(d.deg.clip(lower=1),[0,5,20,100,1e9],labels=["1-5","6-20","21-100",">100"])
    d["rr"]=1/d["rank"]; d["frr"]=1/d.frank
    print(f"== {name}: PathE MRR {d.rr.mean():.4f} H@1 {(d['rank']<=1).mean():.4f} | freq baseline MRR {d.frr.mean():.4f} H@1 {(d.frank<=1).mean():.4f} | spearman(rank_PathE, rank_freq) {spearmanr(d['rank'],d.frank)[0]:.3f}")
    print(f"   PathE better on {(d['rank']<d.frank).mean():.3f} of queries, worse on {(d['rank']>d.frank).mean():.3f}; best-of-both MRR {np.maximum(d.rr,d.frr).mean():.4f}")
    print(d.groupby("bin",observed=True).agg(n=("rr","size"),pathe=("rr","mean"),freq=("frr","mean")).round(4).to_string())
