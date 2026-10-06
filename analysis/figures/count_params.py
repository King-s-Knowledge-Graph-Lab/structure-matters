"""
Per-module parameter counts of PathE checkpoints (Section 5.4 parameter efficiency, Appendix G, config tables).

The totals of the paper count every entry of the checkpoint's state_dict, including the integer buffer
rel_index_tensor (one entry per relation); "float" excludes integer buffers.
usage: python analysis/figures/count_params.py CKPT [CKPT ...]
       python analysis/figures/count_params.py      (the four link prediction checkpoints under <checkpoint_root>/link_prediction/)
"""
import argparse
import collections
from pathlib import Path

import torch


def count(ckpt):
    sd = torch.load(ckpt, map_location="cpu")
    sd = sd.get("state_dict", sd)
    modules = collections.OrderedDict()
    for k, v in sd.items():
        name = k[len("model."):] if k.startswith("model.") else k
        modules[name.split(".")[0]] = modules.get(name.split(".")[0], 0) + v.numel()
    total = sum(v.numel() for v in sd.values())
    floats = sum(v.numel() for v in sd.values() if v.dtype.is_floating_point)
    return modules, total, floats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="*")
    args = ap.parse_args()
    ckpts = args.checkpoints
    if not ckpts:
        from pathe.paths import load_paths
        root = load_paths()["checkpoint_root"] / "link_prediction"
        ckpts = [root / f"{kg}.ckpt" for kg in ("fb15k237", "wn18rr", "codex-large", "yago")]
    for ckpt in ckpts:
        modules, total, floats = count(ckpt)
        print(f"== {Path(ckpt).name}: total {total:,} (float {floats:,})")
        for name, n in modules.items():
            print(f"   {name:28s} {n:>10,}")


if __name__ == "__main__":
    main()
