import itertools
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.evaluation.metrics import clustering_metrics, expand_unassigned, recall_at_k
from src.inference.clustering import candidate_edges, graph_cluster, knn
from src.inference.export import export_clusters, folder_names


def _brute_force(true, pred):
    n = len(true)
    precision = np.mean([np.mean(true[pred == pred[i]] == true[i]) for i in range(n)])
    recall = np.mean([np.mean(pred[true == true[i]] == pred[i]) for i in range(n)])
    pairs = list(itertools.combinations(range(n), 2))
    tp = sum(true[i] == true[j] and pred[i] == pred[j] for i, j in pairs)
    pp = sum(pred[i] == pred[j] for i, j in pairs)
    ap = sum(true[i] == true[j] for i, j in pairs)
    return precision, recall, tp / pp, tp / ap


class MetricTests(unittest.TestCase):
    def test_metrics_match_brute_force(self):
        rng = np.random.default_rng(0)
        true, pred = rng.integers(0, 5, 60), rng.integers(0, 7, 60)
        m = clustering_metrics(true, pred)
        expected = _brute_force(true, pred)
        for key, value in zip(("bcubed_precision", "bcubed_recall", "pairwise_precision", "pairwise_recall"), expected):
            self.assertAlmostEqual(m[key], value)

    def test_perfect_clustering_and_unassigned_singletons(self):
        true = np.array([0, 0, 1, 1, 2])
        perfect = clustering_metrics(true, np.array([5, 5, 7, 7, 9]))
        self.assertAlmostEqual(perfect["bcubed_f1"], 1.0)
        self.assertAlmostEqual(perfect["cluster_accuracy"], 1.0)
        self.assertEqual(list(expand_unassigned(np.array([0, -1, 0, -1]))), [0, 1, 0, 2])
        self.assertAlmostEqual(clustering_metrics(true, np.array([-1] * 5))["bcubed_precision"], 1.0)


class ClusteringTests(unittest.TestCase):
    def test_graph_clustering_recovers_separated_groups(self):
        rng = np.random.default_rng(0)
        labels = np.repeat(np.arange(4), 5)
        z = np.eye(4, 16)[labels] + 0.05 * rng.normal(size=(20, 16))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        neighbors, _ = knn(z, 6)
        self.assertEqual(recall_at_k(neighbors, labels)["recall_at_1"], 1.0)
        edges = candidate_edges(neighbors)
        probs = (z[edges[:, 0]] * z[edges[:, 1]]).sum(1)
        pred = graph_cluster(20, edges, probs, 0.5, lambda i, j: (z[i] * z[j]).sum(1))
        self.assertAlmostEqual(clustering_metrics(labels, pred)["bcubed_f1"], 1.0)

    def test_folders_are_numbered_by_size_and_overwrite_is_guarded(self):
        labels = np.array([3, 3, 1, 1, 1, 7, 9])
        self.assertEqual(list(folder_names(labels, "unassigned")), ["2", "2", "1", "1", "1", "unassigned", "unassigned"])
        self.assertEqual(list(folder_names(labels, "own_folder"))[-2:], ["3", "4"])
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            tmp = Path(tmp)
            paths = []
            for i in range(len(labels)):
                path = tmp / "in" / f"img{i}.jpg"
                path.parent.mkdir(exist_ok=True)
                path.write_bytes(b"x")
                paths.append(path)
            out = tmp / "results" / "cluster_demo"
            summary = export_clusters(paths, labels, {}, {"overwrite": True}, out, {"model": "demo"})
            self.assertEqual((summary["clusters"], summary["unassigned_images"]), (2, 2))
            self.assertEqual(len(list((out / "1").iterdir())), 3)
            export_clusters(paths, labels, {}, {"overwrite": True}, out, {"model": "demo"})
            (out / "summary.json").unlink()
            with self.assertRaises(FileExistsError):
                export_clusters(paths, labels, {}, {"overwrite": True}, out, {"model": "demo"})


if __name__ == "__main__":
    unittest.main()
