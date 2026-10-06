"""
Step 4: analytic ceiling of a relational-context-only model (the PathE node projector without paths).

Entities whose (incoming, outgoing) relation-count vectors are identical receive identical node-projector
outputs, so a model that sees only the relational context must give them the same score. For a test query
whose answer lies in a tie group of k candidates (after filtering), the best such a model can do is:
  expected reciprocal rank under random tie-breaking   H_k / k
  reciprocal of the 'realistic' rank used in the repo   2 / (k + 1)
  Hits@10                                               min(1, 10 / k)
Averaged over all test queries (both sides) this is an upper bound for any context-only model.

usage: python analysis/query_level/context_ceiling.py <kg> [--out-dir DIR]   (default: results/query_level/ceiling/)
writes ceiling_<kg>.csv (per query) and ceiling_<kg>.json (summary)
"""
import argparse, json, os, collections
import numpy as np, pandas as pd, torch
from common import KGS, add_root_args, roots, dataset_dir, ceiling_file

ap = add_root_args(argparse.ArgumentParser()); ap.add_argument("kg", choices=KGS); ap.add_argument("--out-dir", default=None)
args = ap.parse_args(); data_root, results_root, _ = roots(args)
name = args.kg; root = dataset_dir(data_root, name)
out_csv, out_json = ceiling_file(results_root, name), ceiling_file(results_root, name, "json")
if args.out_dir:
    out_csv, out_json = os.path.join(args.out_dir, out_csv.name), os.path.join(args.out_dir, out_json.name)
os.makedirs(os.path.dirname(out_csv), exist_ok=True)
tr, va, te = [torch.load(f"{root}/{s}/triples.pt").long() for s in ("train", "val", "test")]
allt = torch.cat([tr, va, te])
N = int(torch.stack([allt[:, 0], allt[:, 2]]).max()) + 1
R = int(allt[:, 1].max()) + 1

# context signature from the training graph only (same file the model reads)
rc = pd.read_csv(f"{root}/train/relational_context.csv")
sig = collections.defaultdict(lambda: [(), ()])
for node, direction, edges in zip(rc.node_id, rc.direction, rc.edges):
    if isinstance(edges, str) and edges.strip():
        vals, cnt = np.unique(np.array(edges.split(), dtype=np.int64), return_counts=True)
        sig[int(node)][0 if direction == "in" else 1] = tuple(zip(vals.tolist(), cnt.tolist()))
keys = [tuple(sig[e]) if e in sig else ((), ()) for e in range(N)]
cls_of = {}
cls = np.array([cls_of.setdefault(k, len(cls_of)) for k in keys])
size = np.bincount(cls)
members = collections.defaultdict(set)
for e, c in enumerate(cls):
    members[c].add(e)

n_train_ents = len(set(tr[:, 0].tolist()) | set(tr[:, 2].tolist()))
stats = {
    "dataset": name, "entities": N, "relations": R, "entities_in_train": n_train_ents,
    "distinct_contexts": int(len(size)),
    "unique_context_fraction": float((size[cls] == 1).mean()),
    "largest_class": int(size.max()), "mean_class_size": float(N / len(size)),
}

tailf, headf = collections.defaultdict(set), collections.defaultdict(set)
for h, r, t in allt.tolist():
    tailf[(h, r)].add(t); headf[(r, t)].add(h)

H = np.concatenate([[0.0], np.cumsum(1.0 / np.arange(1, size.max() + 2))])
rows = []
for i, (h, r, t) in enumerate(te.tolist()):
    for side, true, filt in (("tail", t, tailf[(h, r)]), ("head", h, headf[(r, t)])):
        c = cls[true]
        k = size[c] - len(members[c] & filt) + 1 if size[c] > 1 else 1
        rows.append((i, side, true, int(size[c]), int(k)))
q = pd.DataFrame(rows, columns=["triple_idx", "side", "target", "class_size", "k"])
q["rr_random"] = H[q.k.values] / q.k.values
q["rr_realistic"] = 2.0 / (q.k.values + 1)
q["hits10"] = np.minimum(1.0, 10.0 / q.k.values)
q.to_csv(out_csv, index=False)
stats.update({
    "test_queries": int(len(q)),
    "queries_with_unique_target": float((q.k == 1).mean()),
    "median_k": float(q.k.median()),
    "ceiling_mrr_random_tiebreak": float(q.rr_random.mean()),
    "ceiling_mrr_realistic_rank": float(q.rr_realistic.mean()),
    "ceiling_hits10": float(q.hits10.mean()),
    "ceiling_mrr_tail": float(q[q.side == "tail"].rr_random.mean()),
    "ceiling_mrr_head": float(q[q.side == "head"].rr_random.mean()),
})
print(json.dumps(stats, indent=1))
json.dump(stats, open(out_json, "w"), indent=1)
