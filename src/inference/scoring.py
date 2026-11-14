"""Pair scoring and validation calibration.

P(same | i, j) = sigmoid(w_cos * cos(z_inv_i, z_inv_j) + w_energy * (-E(z_dis_i, z_dis_j)) + b)

The weights are fit with logistic regression on kNN candidate pairs of the
validation split, and the merge threshold is the one that maximises B-cubed F1
when the validation split itself is clustered. Test labels are never used.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.evaluation.metrics import clustering_metrics
from src.inference.clustering import candidate_edges, graph_cluster, knn


class PairScorer:
    """Scores arbitrary pairs from cached embeddings. Runs on CPU so union-find can call it cheaply."""

    def __init__(self, z_inv: np.ndarray, z_dis: np.ndarray | None = None, discrimination_head=None,
                 use_energy: bool = True, calibration: dict | None = None):
        self.z_inv = np.asarray(z_inv, dtype=np.float32)
        self.use_energy = bool(use_energy and z_dis is not None and discrimination_head is not None)
        if self.use_energy:
            self.z_dis = torch.as_tensor(np.asarray(z_dis, dtype=np.float32))
            self.head = copy.deepcopy(discrimination_head).cpu().float().eval()
        self.calibration = calibration

    @torch.inference_mode()
    def energy(self, i: np.ndarray, j: np.ndarray, chunk: int = 65536) -> np.ndarray:
        out = []
        for start in range(0, len(i), chunk):
            a = self.z_dis[i[start : start + chunk]]
            b = self.z_dis[j[start : start + chunk]]
            out.append(self.head.energy(a, b).numpy())
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)

    def features(self, i: np.ndarray, j: np.ndarray) -> np.ndarray:
        cos = (self.z_inv[i] * self.z_inv[j]).sum(axis=1)
        if not self.use_energy:
            return cos[:, None]
        return np.stack([cos, -self.energy(i, j)], axis=1)

    def prob(self, i: np.ndarray, j: np.ndarray) -> np.ndarray:
        if self.calibration is None:
            raise RuntimeError("PairScorer has no calibration. Fit one on validation first.")
        logits = self.features(i, j) @ np.asarray(self.calibration["weights"]) + self.calibration["bias"]
        return 1.0 / (1.0 + np.exp(-logits))


def fit_calibration(scorer: PairScorer, labels: np.ndarray, cfg: dict, seed: int = 0) -> tuple[dict, dict]:
    """Fit pair weights and the merge threshold on labelled validation embeddings."""
    neighbors, _ = knn(scorer.z_inv, cfg["k_neighbors"])
    edges = candidate_edges(neighbors, cfg.get("mutual", False))
    feats = scorer.features(edges[:, 0], edges[:, 1])
    same = (labels[edges[:, 0]] == labels[edges[:, 1]]).astype(int)
    if same.min() == same.max():
        raise RuntimeError("Validation candidate pairs are all positive or all negative; cannot calibrate.")
    model = LogisticRegression(class_weight="balanced", max_iter=1000).fit(feats, same)
    scorer.calibration = {
        "weights": model.coef_[0].tolist(),
        "bias": float(model.intercept_[0]),
        "features": ["cosine", "neg_energy"] if scorer.use_energy else ["cosine"],
    }
    probs = scorer.prob(edges[:, 0], edges[:, 1])
    sweep = []
    for threshold in cfg["threshold_grid"]:
        pred = graph_cluster(len(labels), edges, probs, float(threshold), scorer.prob,
                             cfg.get("linkage_check", "average"), int(cfg.get("max_check_pairs", 32)), seed)
        sweep.append({"threshold": float(threshold), **clustering_metrics(labels, pred)})
    best = max(sweep, key=lambda row: row["bcubed_f1"])
    scorer.calibration["threshold"] = best["threshold"]
    diagnostics = {
        "candidate_pairs": int(len(edges)),
        "candidate_positive_rate": float(same.mean()),
        "candidate_recall_of_positives": _candidate_recall(labels, edges),
        "pair_auc": float(roc_auc_score(same, probs)),
        "pair_auc_cosine_only": float(roc_auc_score(same, feats[:, 0])),
        "threshold_sweep": sweep,
        "best": best,
        "edge_probs": probs,
        "edge_same": same,
        "edge_features": feats,
        "neighbors": neighbors,
    }
    return scorer.calibration, diagnostics


def _candidate_recall(labels: np.ndarray, edges: np.ndarray) -> float:
    """Fraction of all same-product pairs that kNN proposes at all (the ceiling for pair recall)."""
    sizes = np.bincount(np.unique(labels, return_inverse=True)[1]).astype(np.float64)
    total = (sizes * (sizes - 1) / 2).sum()
    found = float((labels[edges[:, 0]] == labels[edges[:, 1]]).sum())
    return found / total if total else 1.0


def cluster_with_scorer(scorer: PairScorer, cfg: dict, seed: int = 0) -> tuple[np.ndarray, dict]:
    neighbors, _ = knn(scorer.z_inv, cfg["k_neighbors"])
    edges = candidate_edges(neighbors, cfg.get("mutual", False))
    probs = scorer.prob(edges[:, 0], edges[:, 1])
    labels = graph_cluster(len(scorer.z_inv), edges, probs, float(scorer.calibration["threshold"]), scorer.prob,
                           cfg.get("linkage_check", "average"), int(cfg.get("max_check_pairs", 32)), seed)
    return labels, {"edges": edges, "edge_probs": probs, "neighbors": neighbors}
