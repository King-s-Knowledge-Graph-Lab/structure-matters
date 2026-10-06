"""Every experiment config uses options of runner.py only and holds no machine settings."""
import os
import tempfile
import unittest
from pathlib import Path

import yaml

from pathe.config import load_config, dump_config
from pathe.runner import build_parser, parse_args

ROOT = Path(__file__).resolve().parents[1]
GROUPS = ("link_prediction", "relation_prediction", "inductive", "ablations")
MACHINE = {"device", "num_devices", "num_workers", "wandb_project"}


class TestConfigs(unittest.TestCase):

    def configs(self):
        files = sorted(p for g in GROUPS for p in (ROOT / "configs" / g).glob("*.yaml"))
        self.assertTrue(files)
        return files

    def test_options_exist(self):
        parser = build_parser()
        for path in self.configs():
            with self.subTest(config=path.name):
                load_config(path, parser)

    def test_no_machine_settings(self):
        for path in self.configs():
            with self.subTest(config=path.name):
                config = yaml.safe_load(path.read_text())
                self.assertFalse(MACHINE & set(config))
                for value in config.values():
                    if isinstance(value, str) and "/" in value:
                        self.assertTrue(value.startswith("{"), value)

    def test_dump_round_trip(self):
        parser = build_parser()
        for path in self.configs():
            with self.subTest(config=path.name), tempfile.TemporaryDirectory() as tmp:
                args = vars(parse_args(["train", "pathe", "--config", str(path)]))
                dumped = os.path.join(tmp, "config.yaml")
                dump_config(args, parser, dumped)
                reloaded = vars(parse_args(["train", "pathe", "--config", dumped]))
                args.pop("config"), reloaded.pop("config")
                self.assertEqual(args, reloaded)


if __name__ == "__main__":
    unittest.main()
