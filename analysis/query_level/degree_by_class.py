"""PathE's MRR by answer degree and by size of the answer's context class.

usage: python analysis/query_level/degree_by_class.py
"""
import argparse, numpy as np, pandas as pd, torch
from common import add_root_args, roots, dataset_dir, ranks_file, ceiling_file
args = add_root_args(argparse.ArgumentParser()).parse_args(); data_root, results_root, _ = roots(args)
for name in ["wn18rr", "yago", "fb15k237"]:
    ranks = ranks_file(results_root, "pathe", name)
    root=dataset_dir(data_root, name); tr=torch.load(f"{root}/train/triples.pt").long().numpy()
    d=pd.read_csv(ranks).merge(pd.read_csv(ceiling_file(results_root, name))[["triple_idx","side","target","k"]],on=["triple_idx","side"])
    N=int(max(tr[:,[0,2]].max(), d.target.max()))+1; deg=np.bincount(tr[:,0],minlength=N)+np.bincount(tr[:,2],minlength=N)
    d["deg"]=deg[d.target]; d["rr"]=1/d["rank"]
    d["dbin"]=pd.cut(d.deg.clip(lower=1),[0,5,20,100,1e9],labels=["1-5","6-20","21-100",">100"])
    d["kbin"]=pd.cut(d.k,[0,1,10,100,1e9],labels=["unique","2-10","11-100",">100"])
    print("==",name,"MRR by answer degree (rows) and size of its context class (columns); n in brackets")
    m=d.groupby(["dbin","kbin"],observed=True).rr.agg(["mean","size"]).reset_index()
    m["cell"]=m["mean"].round(3).astype(str)+" ("+m["size"].astype(str)+")"
    print(m.pivot(index="dbin",columns="kbin",values="cell").to_string())
    print(d.groupby("side").rr.mean().round(4).to_dict())
