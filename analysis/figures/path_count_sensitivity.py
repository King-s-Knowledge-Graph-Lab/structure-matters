"""
Figure of Appendix G: link prediction metrics as a function of the number of paths per triple (ppt),
FB15k-237 and CoDEx-Large (from experiments/article/embed_viz.ipynb, "Ablation plots").

Input: results/figures/fb_cola_ablation.csv (ppt, MRR, h1, h3, h5, h10, dataset).
usage: python analysis/figures/path_count_sensitivity.py [--out PDF]   (default: results/figures/pathe_ablation_metrics.pdf)
"""
import argparse
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path(__file__).resolve().parents[2] / "results" / "figures"


def plot_ablation_metrics(df, dataset, axis):
    df_dataset = df[df['dataset'] == dataset]
    axis.plot(df_dataset['ppt'], df_dataset['MRR'],
              marker='o', linestyle='-', label='MRR')
    axis.plot(df_dataset['ppt'], df_dataset['h1'],
              marker='o', linestyle='-', label='h1')
    axis.plot(df_dataset['ppt'], df_dataset['h3'],
              marker='o', linestyle='-', label='h3')
    axis.plot(df_dataset['ppt'], df_dataset['h5'],
              marker='o', linestyle='-', label='h5')
    axis.plot(df_dataset['ppt'], df_dataset['h10'],
              marker='o', linestyle='-', label='h10')
    axis.set_xlabel('paths per triple (ppt)')
    axis.set_ylabel('MRR and Hits@k')
    axis.set_title(f'{dataset}')
    axis.legend()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(RESULTS / "fb_cola_ablation.csv"))
    ap.add_argument("--out", default=str(RESULTS / "pathe_ablation_metrics.pdf"))
    args = ap.parse_args()
    df = pd.read_csv(args.input)

    # For each dataset (FB15k-237 and CoDeX-Large), separately,
    # plot the MRR, h1, h3, h5, h10 as a function of the number of PPTs
    datasets = df['dataset'].unique()
    fig, axis = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
    for i, dataset in enumerate(datasets):
        plot_ablation_metrics(df, dataset, axis[i])
        axis[i].set_xticks([2, 4, 8, 16])
    # remove label legend from the right plot
    axis[1].legend().remove()
    fig.savefig(args.out, bbox_inches='tight')


if __name__ == "__main__":
    main()
