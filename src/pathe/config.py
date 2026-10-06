"""
Experiment configurations: YAML files whose keys are the long options of
``runner.py`` (as written on the command line, without the leading ``--``).

String values may refer to the locations of ``configs/paths.yaml`` as
``{data_root}``, ``{runs_root}``, ``{checkpoint_root}`` or ``{results_root}``.
"""
import argparse
import subprocess

import yaml

from .paths import load_paths, repo_root


def option_dests(parser: argparse.ArgumentParser) -> dict:
    """Map every long option of ``parser`` (without ``--``) to its destination."""
    return {opt[2:]: action.dest for action in parser._actions
            for opt in action.option_strings if opt.startswith("--")}


def load_config(path, parser: argparse.ArgumentParser) -> dict:
    """Read ``path`` and return ``{dest: value}`` for ``parser.set_defaults``."""
    with open(path) as f:
        config = yaml.safe_load(f) or {}
    roots = {k: str(v) for k, v in load_paths().items()}
    dests = option_dests(parser)
    unknown = [k for k in config if k not in dests or k == "config"]
    if unknown:
        raise ValueError(f"{path}: unknown options {unknown}")
    return {dests[k]: (v.format(**roots) if isinstance(v, str) else v)
            for k, v in config.items()}


def git_commit() -> str:
    """The commit of the repository, with ``-dirty`` if tracked files differ."""
    try:
        root = str(repo_root())
        commit = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", root, "status", "--porcelain",
            "--untracked-files=no"], capture_output=True, text=True,
            check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return commit + ("-dirty" if dirty else "")


def dump_config(args: dict, parser: argparse.ArgumentParser, path):
    """Write the options in ``args`` as a config that ``load_config`` reads."""
    names = {}
    for name, dest in option_dests(parser).items():
        names.setdefault(dest, name)
    config = {names[d]: (v.replace("{", "{{").replace("}", "}}")
                         if isinstance(v, str) else v)
              for d, v in args.items() if d in names and d != "config"}
    with open(path, "w") as f:
        f.write(f"# Arguments of this run (runner.py --config), "
                f"commit {git_commit()}\n")
        yaml.safe_dump(config, f, sort_keys=False)
