"""
Per-query filtered ranks of a public LibKGE ComplEx checkpoint on OUR test triples (PyKEEN ids), same rank
definition as dump_ranks.py (realistic rank, filtered with train+val+test, both sides).
LibKGE ids are rebuilt from the order of first appearance in the training file (subject, then object per line;
relations in order of first appearance), which is how LibKGE's preprocessing assigns them. The mapping is
validated by the resulting MRR (a wrong mapping gives an MRR near zero).
The entity maps (and the CoDEx-Large relation map) are those of align_ids.py, in results/link_prediction/idmaps/.
usage: python analysis/query_level/complex_ranks.py <kg> [--checkpoint PT] [--out CSV]
defaults: <checkpoint_root>/complex/<released file of the kg>, results/link_prediction/ranks/complex_<kg>.csv
"""
import argparse, sys, collections, json, numpy as np, pandas as pd, torch
from libkge_checkpoint import load
from common import KGS, COMPLEX_CHECKPOINTS, add_root_args, roots, dataset_dir, ranks_file, idmaps_dir
ap = add_root_args(argparse.ArgumentParser()); ap.add_argument("kg", choices=KGS)
ap.add_argument("--checkpoint", default=None); ap.add_argument("--out", default=None)
args = ap.parse_args(); data_root, results_root, paths = roots(args)
name = args.kg
ckpt = args.checkpoint or str(paths["checkpoint_root"] / "complex" / COMPLEX_CHECKPOINTS[name])
maps = idmaps_dir(results_root); out = args.out or ranks_file(results_root, "complex", name)
root = dataset_dir(data_root, name)
tr, va, te = [torch.load(f"{root}/{s}/triples.pt").long().numpy() for s in ("train", "val", "test")]
allt = np.concatenate([tr, va, te]); N = int(allt[:, [0, 2]].max()) + 1; R = int(allt[:, 1].max()) + 1

# entity mapping: rows of the user's extracted matrix (PyKEEN ids) matched to rows of the checkpoint
ent_map = np.load(f"{maps}/entmap_{name}.npy"); ne = int((ent_map >= 0).sum())
ck = load(ckpt); sd = ck["model"][0]
key = lambda s: [k for k in sd if s in k and "_base_model" not in k][0]
E_lib, R_lib = sd[key("_entity_embedder")].float(), sd[key("_relation_embedder")].float()
reciprocal = R_lib.shape[0] == 2 * R
print(name, "entities ours/libkge", N, E_lib.shape[0], "| relations", R, R_lib.shape[0], "| reciprocal", reciprocal, "| mapped entities", ne)
known = ent_map >= 0
import os
if os.path.exists(f"{maps}/relmap_{name}.npy"):   # exact mapping from LibKGE's relation_ids.del
    rel_map = np.load(f"{maps}/relmap_{name}.npy"); print("exact relation mapping loaded")
else:  # infer a one-to-one relation mapping from the training triples (public checkpoints without id files)
    E0 = torch.zeros((N, E_lib.shape[1])); E0[known] = E_lib[ent_map[known]]
    x_re, x_im = E0.chunk(2, dim=1); rel_map = -np.ones(R, dtype=np.int64); rng = np.random.default_rng(0)
    Rre, Rim = R_lib[:R].chunk(2, dim=1)
    S = np.full((R, R), -1e9)
    for r in range(R):
        T = tr[tr[:, 1] == r]
        if len(T) == 0: continue
        T = T[rng.choice(len(T), min(len(T), 5000), replace=False)]
        hr, hi, tr_, ti = x_re[T[:, 0]], x_im[T[:, 0]], x_re[T[:, 2]], x_im[T[:, 2]]
        A = (hr * tr_ + hi * ti).mean(0); B = (hr * ti - hi * tr_).mean(0)
        v = (Rre @ A + Rim @ B).numpy(); S[r] = (v - v.mean()) / (v.std() + 1e-9)
    from scipy.optimize import linear_sum_assignment
    ri, ci = linear_sum_assignment(-S); rel_map[ri] = ci
    print("assignment agrees with row-wise argmax for", int((S.argmax(1) == rel_map).sum()), "of", R, "relations")
    print("relation mapping is a permutation:", len(set(rel_map[rel_map >= 0])) == (rel_map >= 0).sum(), "| mapped", int((rel_map >= 0).sum()), "of", R)
E = torch.zeros((N, E_lib.shape[1])); E[known] = E_lib[ent_map[known]]
e_re, e_im = E.chunk(2, dim=1)

def rel(r, inverse=False):
    v = R_lib[rel_map[r] + (R if inverse else 0)]; return v.chunk(2)

tf, hf = collections.defaultdict(list), collections.defaultdict(list)
for h, r, t in allt: tf[(h, r)].append(t); hf[(r, t)].append(h)
rows = []
with torch.no_grad():
    for i, (h, r, t) in enumerate(te):
        p_re, p_im = rel(r)
        # tail side: Re(<e_h, r, conj(e_o)>) for all o
        a_re = e_re[h] * p_re - e_im[h] * p_im; a_im = e_re[h] * p_im + e_im[h] * p_re
        s_tail = e_re @ a_re + e_im @ a_im
        if reciprocal:   # head side scored as (t, r^-1, ?)
            q_re, q_im = rel(r, True)
            b_re = e_re[t] * q_re - e_im[t] * q_im; b_im = e_re[t] * q_im + e_im[t] * q_re
            s_head = e_re @ b_re + e_im @ b_im
        else:            # Re(<e_s, r, conj(e_t)>) for all s
            c_re = p_re * e_re[t] + p_im * e_im[t]; c_im = p_re * e_im[t] - p_im * e_re[t]
            s_head = e_re @ c_re + e_im @ c_im
        for side, true, s, filt in (("tail", t, s_tail, tf[(h, r)]), ("head", h, s_head, hf[(r, t)])):
            st = s[true].item(); keep = torch.ones(N, dtype=torch.bool); keep[filt] = False; keep &= torch.from_numpy(known)
            neg = s[keep]
            rows.append((i, side, h, r, t, (neg > st).sum().item() + 1 + (neg == st).sum().item() / 2.0))
d = pd.DataFrame(rows, columns=["triple_idx", "side", "head", "relation", "tail", "rank"])
d.to_csv(out, index=False)
rr = 1 / d["rank"]
print(json.dumps({"kg": name, "mrr": round(rr.mean(), 4), "hits1": round((d["rank"] <= 1).mean(), 4), "hits10": round((d["rank"] <= 10).mean(), 4),
                  "mrr_head": round(rr[d.side == "head"].mean(), 4), "mrr_tail": round(rr[d.side == "tail"].mean(), 4)}))
