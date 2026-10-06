"""
Figure of Section 7.5: first principal component of the learned positional embeddings of the entity
positions in a path, per KG (from experiments/article/embed_viz.ipynb, the call that made
positional_embeddings.pdf: normalise=True, n_components=1, trim_to=11).

Inputs: results/figures/positional_embeddings/<kg>_pos.csv, the pos_embeddings.weight tables (100 x d)
of the models (exported from the .pt files of the notebook, float32, exact).
usage: python analysis/figures/positional_pca.py [--out PDF]   (default: results/figures/positional_embeddings.pdf)
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

RESULTS = Path(__file__).resolve().parents[2] / "results" / "figures"


def visualise_positional_entembeddings_datasets(embedding_dict,
                                                out,
                                                normalise=False,
                                                n_components=2,
                                                cmap='inferno',
                                                trim_to=None):
    """
    Visualises positional embeddings after dimensionality reduction.
    """
    # relevant_indices = torch.arange(1, 6)
    relevant_indices = torch.arange(1, 40)  # FIXME

    reducer = PCA(n_components=n_components)
    fig, axs = plt.subplots(len(embedding_dict)//2, len(embedding_dict)//2,
                            sharex=True, sharey=True,
                            figsize=(6, len(embedding_dict)-2))

    for i, (dataset, embedding_table) in enumerate(embedding_dict.items()):
        print(f"Processing {dataset}...")
        # Isolate relevant embeddings
        ent_embeddings = embedding_table[relevant_indices]
        ent_embeddings = ent_embeddings[0::2]

        if normalise:  # normalization (if requested)
            ent_embeddings = (ent_embeddings - ent_embeddings.mean(dim=0)) / ent_embeddings.std(dim=0)

        # Dimensionality Reduction
        if n_components > 0:
            reduced_ent_embeddings = reducer.fit_transform(ent_embeddings.numpy())
            # Print the explained variance ratio of the PCA
            print(f"Explained variance ratio: {reducer.explained_variance_ratio_}")
        else:  # just sum over the embeddings to get a single value
            reduced_ent_embeddings = ent_embeddings.sum(dim=1).numpy().reshape(-1, 1)
            print(reduced_ent_embeddings.shape)

        if trim_to is not None:
            reduced_ent_embeddings = reduced_ent_embeddings[:trim_to]

        # Plot entity embeddings (left) as heatmap and for rel (right) as subplots
        img1 = axs[i%2, i//2].imshow(reduced_ent_embeddings.T, cmap=cmap, aspect="auto")
        axs[i%2, i//2].set_ylabel(dataset, fontsize=8)
        axs[i%2, i//2].set_yticks([])
        axs[i%2, i//2].set_xticks(range(reduced_ent_embeddings.shape[0]))
        # adjust xtick font size to 8
        for tick in axs[i%2, i//2].xaxis.get_major_ticks():
            tick.label1.set_fontsize(8)

        if i != len(embedding_dict) - 1:
            axs[i%2, i//2].set_xlabel("")
            axs[i%2, i//2].set_xlabel("")

    # use "Relative Position (entities)" for both x-axes
    axs[1, 0].set_xlabel("relative position", fontsize=8)
    axs[1, 1].set_xlabel("relative position", fontsize=8)
    # Use a horizontal colorbar at the top of the plot that spans the entire width
    cbar = fig.colorbar(img1, ax=axs.ravel().tolist(), orientation='horizontal')
    cbar.ax.tick_params(labelsize=8)

    fig.savefig(out, bbox_inches='tight')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default=str(RESULTS / "positional_embeddings"))
    ap.add_argument("--out", default=str(RESULTS / "positional_embeddings.pdf"))
    args = ap.parse_args()
    load = lambda kg: torch.from_numpy(np.loadtxt(f"{args.inputs}/{kg}_pos.csv", delimiter=",", dtype=np.float32))
    embedding_dict = {
        "WN18RR": load("wn18rr"),
        "FB15k-237": load("fb15k237"),
        "CoDEx-L": load("codex-large"),
        "YAGO": load("yago"),
    }
    visualise_positional_entembeddings_datasets(embedding_dict, args.out,
                                                normalise=True,
                                                n_components=1,
                                                cmap='inferno',
                                                trim_to=11)


if __name__ == "__main__":
    main()
