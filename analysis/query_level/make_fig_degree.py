"""Figure for Section 7: MRR by training degree of the answer entity, PathE vs ComplEx vs popularity, per KG.

usage: python analysis/query_level/make_fig_degree.py [--out-dir DIR]
writes fig_degree_data.csv and mrr_by_degree.{pdf,png} (default: results/query_level/)
"""
import argparse, os
import numpy as np, pandas as pd, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import common

ap = common.add_root_args(argparse.ArgumentParser()); ap.add_argument("--out-dir", default=None)
args = ap.parse_args(); data_root, results_root, _ = common.roots(args)
out_dir = args.out_dir or common.query_level_dir(results_root); os.makedirs(out_dir, exist_ok=True)
KGS = [(common.TITLES[k], k, common.ranks_file(results_root, "pathe", k)) for k in common.KGS]
BINS = [0, 2, 5, 10, 20, 50, 100, 10**9]; LABELS = ["1–2", "3–5", "6–10", "11–20", "21–50", "51–100", ">100"]
SERIES = [("PathE", "pathe", "#2a78d6", "o", "-"), ("ComplEx", "complex", "#eb6834", "s", "--"), ("popularity", "pop", "#1baf7a", "^", ":")]
rows = []
for title, name, pr in KGS:
    tr = torch.load(f"{common.dataset_dir(data_root, name)}/train/triples.pt").long().numpy()
    d = pd.read_csv(pr)[["triple_idx", "side", "head", "tail", "rank"]].rename(columns={"rank": "pathe"})
    d = d.merge(pd.read_csv(common.ranks_file(results_root, "complex", name))[["triple_idx", "side", "rank"]].rename(columns={"rank": "complex"}), on=["triple_idx", "side"])
    d = d.merge(pd.read_csv(common.ranks_file(results_root, "popularity", name)).rename(columns={"rank": "pop"}), on=["triple_idx", "side"])
    N = int(max(tr[:, [0, 2]].max(), d["head"].max(), d["tail"].max())) + 1
    deg = np.bincount(tr[:, 0], minlength=N) + np.bincount(tr[:, 2], minlength=N)
    d["deg"] = deg[np.where(d.side == "tail", d["tail"], d["head"])]
    d["bin"] = pd.cut(d.deg.clip(lower=1), BINS, labels=LABELS)
    g = d.groupby("bin", observed=True)
    for lab in LABELS:
        sub = d[d.bin == lab]
        rows.append(dict(kg=title, bin=lab, n=len(sub), share=len(sub) / len(d), **{m: (1 / sub[m]).mean() for _, m, *_ in SERIES}))
tab = pd.DataFrame(rows); tab.to_csv(os.path.join(out_dir, "fig_degree_data.csv"), index=False)

plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#888", "xtick.color": "#444", "ytick.color": "#444"})
fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.1), sharex="col", gridspec_kw=dict(height_ratios=[3, 1], hspace=0.12, wspace=0.28))
x = np.arange(len(LABELS))
for j, (title, name, _) in enumerate(KGS):
    t = tab[tab.kg == title]; ax = axes[0, j]; bx = axes[1, j]
    for lab, m, col, mk, ls in SERIES:
        ax.plot(x, t[m].values, ls, color=col, marker=mk, ms=3.5, lw=1.3, label=lab, markeredgecolor="white", markeredgewidth=0.5)
    ax.set_title(title, fontsize=8.5, pad=4); ax.set_ylim(0, 0.9); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8]); ax.grid(axis="y", color="#e5e5e5", lw=0.6)
    if j == 0: ax.set_ylabel("MRR")
    else: ax.tick_params(labelleft=False)
    bx.bar(x, 100 * t["share"].values, color="#bbbbbb", width=0.7); bx.set_ylim(0, 60); bx.set_yticks([0, 30, 60]); bx.grid(axis="y", color="#e5e5e5", lw=0.6)
    bx.set_xticks(x); bx.set_xticklabels(LABELS, rotation=45, ha="right", fontsize=7)
    if j == 0: bx.set_ylabel("% queries", fontsize=7.5)
    else: bx.tick_params(labelleft=False)
fig.text(0.5, -0.02, "training degree of the answer entity", ha="center", fontsize=8)
axes[0, 0].legend(frameon=False, fontsize=7.5, loc="upper left", handlelength=2.2)
fig.savefig(os.path.join(out_dir, "mrr_by_degree.pdf"), bbox_inches="tight"); fig.savefig(os.path.join(out_dir, "mrr_by_degree.png"), dpi=180, bbox_inches="tight")
print(tab.pivot(index="bin", columns="kg", values="pathe").reindex(LABELS).round(3))
