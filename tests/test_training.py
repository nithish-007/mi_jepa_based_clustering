"""End to end on synthetic data: train, resume, evaluate, export cluster folders."""
import json
import tempfile
import unittest
from pathlib import Path

import torch

from src.config import load_config
from src.data.splits import prepare_splits
from src.evaluation.pipeline import evaluate
from src.inference.run import run_clustering
from src.training.trainer import train
from tests.fixtures import data_config, make_image_tree, train_config


class EndToEndTests(unittest.TestCase):
    def test_train_resume_evaluate_and_cluster(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            self._run(Path(tmp))

    def _run(self, tmp: Path):
        make_image_tree(tmp / "raw")
        prepare_splits(data_config(tmp))
        config = train_config(tmp)

        first = train(config)
        directory = tmp / "checkpoints" / "tiny"
        names = sorted(p.name for p in directory.glob("*.ckpt"))
        self.assertEqual(len(names), 3, names)
        self.assertIn("last.ckpt", names)
        self.assertTrue(all(n == "last.ckpt" or (n.startswith("epoch_") and "-bcubed_f1_" in n) for n in names), names)
        state = torch.load(directory / "last.ckpt", map_location="cpu", weights_only=False)
        self.assertIn(state["calibration"]["threshold"], config["validation"]["clustering"]["threshold_grid"])
        self.assertEqual(state["mlflow_run_id"], first["mlflow_run_id"])

        config["trainer"]["max_epochs"] = 3
        second = train(config)
        self.assertEqual(second["mlflow_run_id"], first["mlflow_run_id"])
        self.assertEqual(torch.load(directory / "last.ckpt", map_location="cpu", weights_only=False)["epoch"], 3)
        self.assertEqual(len(list(directory.glob("*.ckpt"))), 3)

        eval_cfg = load_config("configs/evaluate.yaml")
        eval_cfg.update({
            "output_dir": str(tmp / "results"),
            "checkpoint_root": str(tmp / "checkpoints"),
            "data": {"root": str(tmp / "raw"), "manifest": str(tmp / "splits" / "manifest.csv")},
            "models": [{"name": "tiny", "checkpoint": "best"}],
            "corpora": [{"name": "test_all", "split": "test", "max_skus": None, "max_images_per_sku": None}],
            "embedding": {"batch_size": 8, "num_workers": 0},
            "mlflow": {"enabled": True, "experiment": "tests", "tracking_uri": config["mlflow"]["tracking_uri"]},
        })
        eval_cfg["calibration"]["max_images_per_sku"] = None
        eval_cfg["clustering"]["hdbscan"]["pca_dim"] = None
        summary = evaluate(eval_cfg)
        self.assertEqual(set(summary["method"]), {"graph", "graph_cosine", "hdbscan", "kmeans_oracle"})
        self.assertTrue(summary["bcubed_f1"].between(0, 1).all())
        self.assertTrue((tmp / "results" / "evaluation_tiny" / "calibration.json").is_file())

        cluster_cfg = load_config("configs/cluster.yaml")
        cluster_cfg.update({"checkpoint_root": str(tmp / "checkpoints"), "model": {"name": "tiny", "checkpoint": "best"}})
        cluster_cfg["input"].update({"manifest": str(tmp / "splits" / "manifest.csv"), "data_root": str(tmp / "raw"), "max_skus": None})
        cluster_cfg["embedding"].update({"batch_size": 8, "num_workers": 0})
        cluster_cfg["export"]["output_dir"] = str(tmp / "results")
        result = run_clustering(cluster_cfg)
        out = tmp / "results" / "cluster_tiny"
        self.assertTrue(result["calibration_source"].endswith("calibration.json"))
        written = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        folders = [p.name for p in out.iterdir() if p.is_dir()]
        self.assertEqual(written["clusters"], sum(name.isdigit() for name in folders))
        self.assertEqual(sum(len(list((out / f).iterdir())) for f in folders), written["images"])


if __name__ == "__main__":
    unittest.main()
