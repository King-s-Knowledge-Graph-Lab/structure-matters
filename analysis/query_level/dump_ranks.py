"""
Per-triple filtered ranks for a trained PathE link prediction checkpoint.

Evaluation only (no training). Mirrors pathe/pathe_full_eval.py (EVAL_TYPE="Full"):
same entity dataset, same path sampling (seed), same filtering, same "realistic"
rank (mean of optimistic and pessimistic). Differences: (i) runs on CPU or GPU,
(ii) aggregates the path embeddings of every entity once (valid because the
final models use context_heads=0, i.e. the aggregator never looks at the other
entity of the triple), (iii) writes one row per (test triple, side).

Usage (pathe installed; data under data_root of configs/paths.yaml unless --data-root is given):
  python analysis/query_level/dump_ranks.py --dataset fb15k237 \
      --checkpoint artifacts/checkpoints/link_prediction/fb15k237.ckpt \
      --path_setup 20_10 --ppe 4 --nhead 2 --seed 7 --out ranks_fb15k237.csv
"""
import argparse, os, sys, time, json
import torch

from pathe.paths import load_paths

p = argparse.ArgumentParser()
p.add_argument("--data-root", default=None, help="default: data_root of configs/paths.yaml")
p.add_argument("--dataset", required=True)
p.add_argument("--checkpoint", required=True)
p.add_argument("--path_setup", default="20_20")
p.add_argument("--ppe", type=int, required=True, help="paths per entity (max_ppt // 2)")
p.add_argument("--nhead", type=int, required=True)
p.add_argument("--num_agg_heads", type=int, default=1)
p.add_argument("--seed", type=int, default=46)
p.add_argument("--batch_entities", type=int, default=256)
p.add_argument("--limit", type=int, default=None, help="only the first k test triples (smoke test)")
p.add_argument("--out", required=True)
p.add_argument("--emb_cache", default=None)
p.add_argument("--getitem", choices=["combined", "separate"], default="combined",
               help="how an entity's paths are built (pathdata.EntityMultiPathDataset.__getitem__)")
args = p.parse_args()

CUDA = torch.cuda.is_available()
dev = torch.device("cuda" if CUDA else "cpu")

try:
    import torch_scatter  # only needed by the LAF aggregator, which the final models do not use
except ImportError:
    import types
    sys.modules["torch_scatter"] = types.ModuleType("torch_scatter")
from functools import partial
from pathe import triple_lib, data_utils as du
from pathe.pather_models import PathEModel
from pathe.pathdata import EntityMultiPathDataset
from pathe.path_lib import encode_relcontext_freqs
from pathe.data_utils import collate_multipaths, load_triple_tensors
from pathe.utils import set_random_seed

set_random_seed(args.seed, "cuda" if CUDA else "cpu")
data_root = args.data_root or str(load_paths()["data_root"])
root = os.path.join(data_root, "path_datasets", args.dataset)
tr_dir, va_dir, te_dir = [os.path.join(root, s) + "/" for s in ("train", "val", "test")]
train, val, test = load_triple_tensors(tr_dir, va_dir, te_dir)
paths, relcon, _ = du.load_unrolled_setup(tr_dir, args.path_setup)
head_filter, tail_filter = triple_lib.make_head_tail_dicts(train, val, test)
ents = triple_lib.get_unique_entities(train, val, test)
N = int(ents.max()) + 1
assert N == ents.numel(), "entity ids are expected to be contiguous"

# ---- hyper-parameters that can be read from the weights
sd = torch.load(args.checkpoint, map_location="cpu")["state_dict"]
sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
d_model = sd["rel_embeddings.weight"].shape[1]
n_enc = len({k.split(".")[2] for k in sd if k.startswith("encoder.layers.")})
d_ff = sd["encoder.layers.0.linear1.weight"].shape[0]
has_agg = any(k.startswith("aggregator.") for k in sd)
n_agg = len({k.split(".")[4] for k in sd if k.startswith("aggregator.aggregator.sa_encoder.layers.")})
projector = "dummy" if any("incoming_gcn.model" in k for k in sd) else "GCN"
max_seqlen = sd["pos_embeddings.weight"].shape[0]
assert not any("contextualiser" in k for k in sd), "pair-aware contextualiser found: per-entity aggregation is not valid"
cfg = dict(d_model=d_model, nhead=args.nhead, num_encoder_layers=n_enc, dim_feedforward=d_ff,
           ent_aggregation="transformer" if has_agg else "avg", num_agg_layers=max(n_agg, 1),
           num_agg_heads=args.num_agg_heads, context_heads=0, node_projector=projector,
           max_seqlen=max_seqlen)
print("config:", cfg, "| entities", N, "| test triples", len(test), flush=True)

t0 = time.time()
if args.getitem == "separate":
    EntityMultiPathDataset.__getitem__ = EntityMultiPathDataset._getitem_separate
ds = EntityMultiPathDataset(path_store=paths, relcontext_store=relcon, triple_store=train,
                            unique_entities=ents, paths_per_entity=args.ppe, seed=args.seed)
relgraph = encode_relcontext_freqs(relcontext=relcon, num_entities=N,
                                   num_relations=ds.vocab_size - 2, offset=2)
model = PathEModel(vocab_size=ds.vocab_size, padding_idx=ds.tokens_to_idxs["PAD"],
                   relcontext_graph=relgraph, **cfg)
missing, unexpected = model.load_state_dict(sd, strict=False)
print("missing keys:", missing, "| unexpected keys:", unexpected, flush=True)
assert not [k for k in missing if "index_tensor" not in k and "encodings" not in k], missing
model.to(dev).eval()
print(f"dataset+model ready in {time.time()-t0:.0f}s", flush=True)

# ---- 1. one embedding per entity
if args.emb_cache and os.path.exists(args.emb_cache):
    A = torch.load(args.emb_cache)
else:
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_entities, shuffle=False, num_workers=0,
        collate_fn=partial(collate_multipaths, padding_idx=ds.tokens_to_idxs["PAD"]))
    A = torch.zeros((N, d_model))
    n_paths = torch.zeros(N, dtype=torch.long)
    with torch.no_grad():
        for b, batch in enumerate(loader):
            ni = batch["net_input"]
            out = model.embed_nodes(ni["ent_paths"].to(dev), ni["rel_paths"].to(dev),
                                    (ni["head_idxs"] * 2).to(dev), ni["pos"].to(dev),
                                    batch["ppt"].to(dev))
            ids = batch["id"].long()
            if cfg["ent_aggregation"] == "avg":
                A[ids] = out.cpu()
            else:
                padded = torch.nn.utils.rnn.pad_sequence(list(out), batch_first=True)
                A[ids] = model.aggregator.aggregator(padded).cpu()
            n_paths[ids] = batch["ppt"].long()
            if b % 20 == 0:
                print(f"  embedded {min((b+1)*args.batch_entities, N)}/{N}  ({time.time()-t0:.0f}s)", flush=True)
    if args.emb_cache:
        torch.save(A, args.emb_cache)
        torch.save(n_paths, args.emb_cache + ".npaths")
print("NaN rows in embedding table:", int(torch.isnan(A).any(dim=1).sum()), flush=True)

# ---- 2. filtered ranking of every test triple, both sides
with torch.no_grad():
    A = A.to(dev)
    W1, b1 = model.link_predict_head1.weight, model.link_predict_head1.bias
    W2, b2 = model.link_predict_head2.weight, model.link_predict_head2.bias
    Wh, Wr, Wt = W1[:, :d_model], W1[:, d_model:2*d_model], W1[:, 2*d_model:]
    HA, TA = A @ Wh.T, A @ Wt.T                       # (N, d)
    RR = model.pred_link_embeddings.weight @ Wr.T     # (R, d)
    rows = []
    T = test if args.limit is None else test[:args.limit]
    for i, (h, r, t) in enumerate(T.tolist()):
        for side in ("tail", "head"):
            if side == "tail":
                hidden = torch.relu(HA[h] + RR[r] + b1 + TA)
                true, filt = t, tail_filter[(h, r)]
            else:
                hidden = torch.relu(HA + RR[r] + b1 + TA[t])
                true, filt = h, head_filter[(r, t)]
            s = (hidden @ W2.T).squeeze(1) + b2
            s_true = s[true].item()
            keep = torch.ones(N, dtype=torch.bool, device=dev)
            keep[torch.tensor(list(filt), device=dev)] = False   # removes the true entity too
            neg = s[keep]
            greater, equal = int((neg > s_true).sum()), int((neg == s_true).sum())
            rank = greater + 1 + equal / 2.0                     # realistic rank
            rows.append((i, side, h, r, t, rank, int(keep.sum()), equal))
        if i % 2000 == 0:
            print(f"  ranked {i}/{len(T)}  ({time.time()-t0:.0f}s)", flush=True)

import csv
with open(args.out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["triple_idx", "side", "head", "relation", "tail", "rank", "n_candidates", "n_ties"])
    w.writerows(rows)
rk = torch.tensor([x[5] for x in rows])
summary = {"dataset": args.dataset, "checkpoint": args.checkpoint, "n_queries": len(rows),
           "mrr": float((1 / rk).mean()), **{f"hits{k}": float((rk <= k).float().mean()) for k in (1, 3, 5, 10)},
           "config": cfg, "ppe": args.ppe, "path_setup": args.path_setup, "seed": args.seed}
print(json.dumps(summary, indent=1))
with open(args.out.replace(".csv", "_summary.json"), "w") as f:
    json.dump(summary, f, indent=1)
