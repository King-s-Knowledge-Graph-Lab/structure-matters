"""
Make the released checkpoints testable with ``runner.py test pathe --config <file>``.

``runner.py test`` reads the single checkpoint of ``<log_dir>/<expname>/version_<N>/checkpoints/``.
For every config of configs/{link_prediction,relation_prediction,inductive,ablations}/ whose
checkpoint is under <checkpoint_root>, this links it as
``<log_dir>/<expname>/version_0/checkpoints/<name>.ckpt``. Existing run folders that hold another
checkpoint are left alone.

    PYTHONPATH=src python scripts/00_setup/link_runs.py
"""
import glob
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from pathe.paths import load_paths, repo_root  # noqa: E402

GROUPS = ("link_prediction", "relation_prediction", "inductive", "ablations")
# configs whose model is released under another name
SAME_AS = {"ablations/codex-large_ppt8": "link_prediction/codex-large"}


def main():
    roots = {k: str(v) for k, v in load_paths().items()}
    ckpt_root = Path(roots["checkpoint_root"])
    linked = skipped = missing = 0
    for group in GROUPS:
        for config in sorted((repo_root() / "configs" / group).glob("*.yaml")):
            name = f"{group}/{config.stem}"
            ckpt = ckpt_root / f"{SAME_AS.get(name, name)}.ckpt"
            if not ckpt.is_file():
                missing += 1
                continue
            with open(config) as f:
                c = yaml.safe_load(f)
            run_dir = Path(c["log_dir"].format(**roots)) / c["expname"]
            ckpt_dir = run_dir / "version_0" / "checkpoints"
            link = ckpt_dir / f"{config.stem}.ckpt"
            present = [Path(p) for p in glob.glob(str(run_dir / "version_*" / "checkpoints" / "*.ckpt"))]
            if link in present and link.resolve() == ckpt.resolve():
                linked += 1
                continue
            if present:
                print(f"skipped {name}: {run_dir} already holds {len(present)} checkpoint(s)")
                skipped += 1
                continue
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            os.symlink(ckpt.resolve(), link)
            linked += 1
    print(f"{linked} configs linked to a released checkpoint, {skipped} skipped, "
          f"{missing} without a released checkpoint")


if __name__ == "__main__":
    main()
