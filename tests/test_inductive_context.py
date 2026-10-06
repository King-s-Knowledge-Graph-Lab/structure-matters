"""The inductive validation and test see the inference graph only (GraIL protocol)."""
import os
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

import pandas as pd
import torch

from pathe import data_utils as du
from pathe import pathe_trainer
from pathe.corruption import SeededLinkCorruptor
from pathe.runner import parse_args

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = sorted((ROOT / "configs" / "inductive").glob("*.yaml"))


def inductive_args(config, *extra):
    """The arguments of a config, or None if its path dataset is missing."""
    try:
        return parse_args(["test", "pathe", "--config", str(config), *extra])
    except SystemExit:
        return None


def available(args):
    return args is not None and all(os.path.isdir(p) for p in (
        args.train_paths, args.valid_paths, args.test_paths, args.inference_paths))


def walk_steps(paths_file):
    steps = set()
    for rels, ents in pd.read_csv(paths_file)[["relation_path", "entity_path"]]\
            .itertuples(index=False):
        rels, ents = str(rels).split(), str(ents).split()
        steps.update((int(ents[i]), int(r), int(ents[i + 1]))
                     for i, r in enumerate(rels))
    return steps


def relational_context(triples):
    context = {}
    for h, r, t in triples.tolist():
        context.setdefault((h, "out"), Counter())[r] += 1
        context.setdefault((t, "in"), Counter())[r] += 1
    return context


class TestInferenceContext(unittest.TestCase):

    def test_loaded_context_is_inference_graph(self):
        checked = 0
        for config in CONFIGS:
            args = inductive_args(config)
            if not available(args):
                continue
            checked += 1
            with self.subTest(config=config.name):
                _, val, test, inference = du.load_triple_tensors_inductive(
                    args.train_paths, args.valid_paths, args.test_paths,
                    args.inference_paths)
                paths_file, relcontext_file = du.load_inference_setup(
                    args.inference_paths, args.path_setup)
                steps = walk_steps(paths_file)
                heldout = set(map(tuple, torch.cat((val, test)).tolist()))
                self.assertEqual(len(steps & heldout), 0,
                                 "held-out triples are walk steps")
                self.assertEqual(
                    len(steps - set(map(tuple, inference.tolist()))), 0,
                    "walk steps that are not inference triples")
                loaded = {(int(n), d): Counter(int(r) for r in str(e).split())
                          for n, d, e in pd.read_csv(relcontext_file)
                          .itertuples(index=False)}
                self.assertEqual(loaded, relational_context(inference))
        if not checked:
            self.skipTest("No inductive path dataset available")

    def test_check_rejects_context_with_heldout_triples(self):
        args = inductive_args(CONFIGS[0])
        if not available(args):
            self.skipTest(f"{CONFIGS[0].name}: path dataset not available")
        _, val, test, inference = du.load_triple_tensors_inductive(
            args.train_paths, args.valid_paths, args.test_paths,
            args.inference_paths)
        heldout = torch.cat((val, test))
        paths_file, relcontext_file = du.load_inference_setup(
            args.inference_paths, args.path_setup)
        du.check_inference_context(paths_file, relcontext_file, inference, heldout)
        h, r, t = test[0].tolist()
        walks = pd.read_csv(paths_file)
        leaky_walk = pd.DataFrame({"relation_path": [str(r)],
                                   "entity_path": [f"{h} {t}"]})
        with tempfile.TemporaryDirectory() as tmp:
            leaky = os.path.join(tmp, "leaky_er_paths.csv")
            pd.concat((walks, leaky_walk)).to_csv(leaky, index=False)
            with self.assertRaises(ValueError):
                du.check_inference_context(leaky, relcontext_file, inference, heldout)
            context = pd.read_csv(relcontext_file)
            context.loc[0, "edges"] = f"{context.loc[0, 'edges']} {r}"
            leaky_context = os.path.join(tmp, "relational_context.csv")
            context.to_csv(leaky_context, index=False)
            with self.assertRaises(ValueError):
                du.check_inference_context(paths_file, leaky_context, inference, heldout)

    def test_versions_hold_different_triples(self):
        """Guards the version argument of the builder: v1-v4 of a KG are different splits."""
        by_kg = {}
        for config in CONFIGS:
            args = inductive_args(config)
            if available(args):
                by_kg.setdefault(config.stem.rsplit("_", 1)[0], []).append(args)
        if not by_kg:
            self.skipTest("No inductive path dataset available")
        for kg, versions in by_kg.items():
            for part in ("train_paths", "valid_paths", "test_paths", "inference_paths"):
                triples = [torch.load(os.path.join(getattr(a, part), "triples.pt"))
                           for a in versions]
                for i in range(len(triples)):
                    for j in range(i + 1, len(triples)):
                        with self.subTest(kg=kg, part=part, pair=(i, j)):
                            self.assertFalse(
                                triples[i].shape == triples[j].shape
                                and torch.equal(triples[i], triples[j]),
                                f"{kg}: two versions have the same {part} triples")


class StopBeforeModel(Exception):
    pass


class TestTrainerWiring(unittest.TestCase):

    def test_validation_and_test_sets(self):
        config = ROOT / "configs" / "inductive" / "fb15k237_v1.yaml"
        args = inductive_args(config, "--num_negatives", "2", "--device", "cpu")
        if not available(args):
            self.skipTest("FB15k-237 v1 inductive path dataset not available")
        args.val_batch_size = args.batch_size
        created = []

        class Recorder(pathe_trainer.TripleEntityMultiPathDataset):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                created.append(kwargs)

        def stop(*unused, **kwunused):
            raise StopBeforeModel

        with mock.patch.object(pathe_trainer, "TripleEntityMultiPathDataset",
                               Recorder), \
                mock.patch.object(pathe_trainer, "PathEModel", stop):
            with self.assertRaises(StopBeforeModel):
                pathe_trainer.create_and_run_training_exp(args)

        _, val, test, inference = du.load_triple_tensors_inductive(
            args.train_paths, args.valid_paths, args.test_paths,
            args.inference_paths)
        paths_file, relcontext_file = du.load_inference_setup(
            args.inference_paths, args.path_setup)
        inference_entities = set(inference[:, [0, 2]].flatten().tolist())
        for kwargs, scored in zip(created[1:], (val, test)):
            self.assertEqual(kwargs["path_store"], paths_file)
            self.assertEqual(kwargs["relcontext_store"], relcontext_file)
            self.assertTrue(torch.equal(kwargs["triple_store"], scored))
            self.assertTrue(torch.equal(kwargs["context_triple_store"], inference))
            self.assertIsInstance(kwargs["triple_corruptor"], SeededLinkCorruptor)
            negatives = kwargs["triple_corruptor"].get_filtered_corrupted_triples(
                scored[:5], args.val_num_negatives)
            self.assertLessEqual(
                set(negatives[:, :, [0, 2]].flatten().tolist()), inference_entities)


class StopAfterWrapper(Exception):
    pass


def has_relational_context(projector, triples, offset=2):
    """The count rows of the projector are those of the given triples (entity and relation ids + offset)."""
    expected = {"in": torch.zeros_like(projector.incoming_gcn.encodings),
                "out": torch.zeros_like(projector.outgoing_gcn.encodings)}
    if expected["in"].shape[0] <= triples[:, [0, 2]].max() + offset:
        return False
    for (entity, direction), counts in relational_context(triples).items():
        for r, n in counts.items():
            expected[direction][entity + offset, r + offset] = n
    return (torch.equal(projector.incoming_gcn.encodings, expected["in"])
            and torch.equal(projector.outgoing_gcn.encodings, expected["out"]))


class TestRelationalContextSwitch(unittest.TestCase):

    def test_projector_uses_inference_graph_at_validation_and_test(self):
        config = ROOT / "configs" / "inductive" / "fb15k237_v1.yaml"
        args = inductive_args(config, "--num_negatives", "2", "--device", "cpu")
        if not available(args):
            self.skipTest("FB15k-237 v1 inductive path dataset not available")
        args.val_batch_size = args.batch_size

        class Recorder(pathe_trainer.PathEModelWrapper):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                raise StopAfterWrapper(self)

        with mock.patch.object(pathe_trainer, "PathEModelWrapper", Recorder):
            with self.assertRaises(StopAfterWrapper) as stopped:
                pathe_trainer.create_and_run_training_exp(args)
        wrapper = stopped.exception.args[0]
        projector = wrapper.model.rcontext_projector
        train, _, _, inference = du.load_triple_tensors_inductive(
            args.train_paths, args.valid_paths, args.test_paths,
            args.inference_paths)

        self.assertTrue(has_relational_context(projector, train))
        for start, end in ((wrapper.on_validation_start, wrapper.on_validation_end),
                           (wrapper.on_test_start, wrapper.on_test_end)):
            with self.subTest(hook=start.__name__):
                start()
                self.assertTrue(has_relational_context(projector, inference))
                end()
                self.assertTrue(has_relational_context(projector, train))


class TestSeededLinkCorruptor(unittest.TestCase):

    def test_filtered_distinct_and_seeded(self):
        triples = torch.tensor([[10, 0, 11], [12, 1, 10]])
        heads = {(0, 11): {10, 13}, (1, 10): {12}}
        tails = {(10, 0): {11, 14}, (12, 1): {10}}
        entities = torch.arange(10, 30)
        make = lambda seed: SeededLinkCorruptor(heads, tails, entities, seed)
        out = make((3, 1)).get_filtered_corrupted_triples(triples, 5)
        self.assertEqual(tuple(out.shape), (4, 6, 3))
        self.assertTrue(torch.equal(
            out, make((3, 1)).get_filtered_corrupted_triples(triples, 5)))
        self.assertFalse(torch.equal(
            out, make((3, 0)).get_filtered_corrupted_triples(triples, 5)))
        for block, (h, r, t), column, excluded in (
                (out[0], triples[0].tolist(), 2, tails[(10, 0)]),
                (out[1], triples[0].tolist(), 0, heads[(0, 11)]),
                (out[2], triples[1].tolist(), 2, tails[(12, 1)]),
                (out[3], triples[1].tolist(), 0, heads[(1, 10)])):
            self.assertEqual(block[0].tolist(), [h, r, t])
            corrupted = block[1:, column].tolist()
            self.assertEqual(len(set(corrupted)), 5)
            self.assertFalse(set(corrupted) & excluded)
            self.assertLessEqual(set(corrupted), set(range(10, 30)))
            kept = [c for c in range(3) if c != column]
            self.assertTrue((block[1:, kept] == torch.tensor([h, r, t])[kept]).all())


if __name__ == "__main__":
    unittest.main()
