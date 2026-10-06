"""
Write a release copy of a checkpoint without the locations and accounts of the machine it was
trained on: every absolute path (string or dict key) is reduced to its last component, and the
``username`` and ``hostname`` fields of LibKGE's trace entries are emptied. Tensors and all other
values are unchanged, which is checked after writing.

    python scripts/00_setup/clean_checkpoint.py IN OUT

Works for PyTorch Lightning checkpoints (.ckpt) and LibKGE checkpoints (.pt); the latter are read
without LibKGE installed.
"""
import argparse
import os
import sys
import types

import torch

ACCOUNT_KEYS = ("username", "hostname")


def stub_libkge():
    """Let LibKGE checkpoints unpickle (and pickle again) without the ``kge`` package."""
    try:
        import kge.config  # noqa: F401
        return
    except ImportError:
        pass
    kge, config = types.ModuleType("kge"), types.ModuleType("kge.config")

    class Config:
        pass

    Config.__module__, Config.__qualname__ = "kge.config", "Config"
    config.Config, kge.config = Config, config
    sys.modules.setdefault("kge", kge)
    sys.modules.setdefault("kge.config", config)


def is_path(s):
    return isinstance(s, str) and s.startswith("/") and len(s) > 1


def clean(o, changes):
    if isinstance(o, torch.Tensor):
        return o
    if isinstance(o, dict):
        out = type(o)()
        for k, v in o.items():
            if is_path(k):
                k, changes[0] = os.path.basename(k.rstrip("/")), changes[0] + 1
            if k in ACCOUNT_KEYS and isinstance(v, str) and v:
                v, changes[0] = "", changes[0] + 1
            out[k] = clean(v, changes)
        return out
    if isinstance(o, (list, tuple)):
        return type(o)(clean(v, changes) for v in o)
    if is_path(o):
        changes[0] += 1
        return os.path.basename(o.rstrip("/"))
    if hasattr(o, "__dict__") and type(o).__module__.startswith("kge"):
        o.__dict__ = clean(vars(o), changes)
        return o
    return o


def same_str(a, b):
    return a == b or (is_path(a) and b == os.path.basename(a.rstrip("/")))


def same(a, b, key=None):
    """Equal up to the cleaned strings: tensors bit for bit, everything else by value."""
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and len(a) == len(b) and all(
            (same_str(ka, kb) if isinstance(ka, str) else ka == kb) and same(va, vb, ka)
            for (ka, va), (kb, vb) in zip(a.items(), b.items()))
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, str):
        return isinstance(b, str) and (same_str(a, b) or (key in ACCOUNT_KEYS and b == ""))
    if hasattr(a, "__dict__") and type(a).__module__.startswith("kge"):
        return same(vars(a), vars(b))
    return a == b or (a != a and b != b)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src")
    p.add_argument("dst")
    a = p.parse_args()
    stub_libkge()
    original = torch.load(a.src, map_location="cpu")
    changes = [0]
    cleaned = clean(torch.load(a.src, map_location="cpu"), changes)
    torch.save(cleaned, a.dst)
    if not same(original, torch.load(a.dst, map_location="cpu")):
        os.remove(a.dst)
        sys.exit(f"{a.src}: the cleaned copy differs in more than the cleaned strings")
    print(f"{a.dst}: {changes[0]} values cleaned")


if __name__ == "__main__":
    main()
