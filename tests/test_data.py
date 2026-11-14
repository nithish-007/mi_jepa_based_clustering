import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.config import apply_overrides
from src.data.indexing import decode_sku_dirname, family_key
from src.data.sampler import PKBatchSampler
from src.data.splits import load_manifest, prepare_splits, select_records
from tests.fixtures import data_config, make_image_tree


class ConfigAndIndexingTests(unittest.TestCase):
    def test_overrides_parse_yaml_values(self):
        config = apply_overrides({"a": {"b": 1}}, ["a.b=2", "a.c=[1, 2]", "d=null"])
        self.assertEqual(config, {"a": {"b": 2, "c": [1, 2]}, "d": None})

    def test_cp437_directory_names_are_repaired(self):
        original = "可口可乐330ml"
        self.assertEqual(decode_sku_dirname(original.encode("utf-8").decode("cp437")), original)
        self.assertEqual(decode_sku_dirname("plain_name"), "plain_name")

    def test_family_key_groups_size_variants(self):
        self.assertEqual(family_key("cola_330ml"), family_key("cola_500ml"))
        self.assertNotEqual(family_key("cola_330ml"), family_key("sprite_330ml"))


class SplitTests(unittest.TestCase):
    def test_splits_are_sku_disjoint_nested_and_reproducible(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            tmp = Path(tmp)
            make_image_tree(tmp / "raw")
            summary = prepare_splits(data_config(tmp))
            manifest = load_manifest(tmp / "splits" / "manifest.csv")
            self.assertEqual(manifest.groupby("sku_id")["split"].nunique().max(), 1)
            self.assertNotIn("others", set(manifest["sku_id"]))
            self.assertEqual(sum(summary["skus"].values()), 24)
            small = set(select_records(manifest, "test", 2, 2)["image_path"])
            large = set(select_records(manifest, "test", 4, 3)["image_path"])
            self.assertTrue(small and small <= large)
            self.assertEqual(prepare_splits(data_config(tmp))["sha256"], summary["sha256"])


class SamplerTests(unittest.TestCase):
    def test_pk_batches_are_deterministic_per_epoch_and_use_siblings(self):
        labels = np.repeat(np.arange(10), 3)
        families = labels // 2
        sampler = PKBatchSampler(labels, families, 4, 2, batches_per_epoch=50, same_family_prob=1.0, seed=1)
        batches = list(sampler)
        self.assertEqual(len(batches), 50)
        self.assertTrue(all(len(b) == 8 and len(set(labels[b])) == 4 for b in batches))
        self.assertEqual(batches, list(sampler))
        sampler.set_epoch(1)
        self.assertNotEqual(batches, list(sampler))
        self.assertGreater(np.mean([len(set(families[b])) < 4 for b in batches]), 0.9)


if __name__ == "__main__":
    unittest.main()
