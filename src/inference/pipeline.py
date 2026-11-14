"""Clustering methods behind one interface, shared by evaluation and the folder export.

graph          kNN on z_inv + calibrated P(same) from cosine and energy + union-find  (proposed)
graph_cosine   same graph, but P(same) from cosine only                                 (ablation: no EBM)
hdbscan        density clustering on z_inv                                              (baseline)
kmeans_oracle  k-means given the true number of products                                (diagnostic upper reference)
"""
from __future__ import annotations

import numpy as np

from src.inference.clustering import hdbscan_cluster, kmeans_cluster
from src.inference.embedding import LoadedModel
from src.inference.scoring import PairScorer, cluster_with_scorer, fit_calibration

GRAPH_METHODS = ("graph", "graph_cosine")


def supports(method: str, model: LoadedModel) -> bool:
    return method != "graph" or model.has_energy


def make_scorer(method: str, model: LoadedModel, embeddings: dict, calibration: dict | None = None) -> PairScorer:
    use_energy = method == "graph"
    return PairScorer(embeddings["z_inv"], embeddings.get("z_dis"), model.net.discrimination_head if use_energy else None,
                      use_energy=use_energy, calibration=calibration)


def calibrate(method: str, model: LoadedModel, val_embeddings: dict, val_labels: np.ndarray, cfg: dict, seed: int):
    scorer = make_scorer(method, model, val_embeddings)
    return fit_calibration(scorer, val_labels, cfg, seed)


def predict(method: str, model: LoadedModel, embeddings: dict, cfg: dict, seed: int,
            calibration: dict | None = None, n_true: int | None = None) -> tuple[np.ndarray, dict]:
    """Cluster labels for one corpus. ``-1`` means unassigned (scored as a singleton)."""
    if method in GRAPH_METHODS:
        return cluster_with_scorer(make_scorer(method, model, embeddings, calibration), cfg, seed)
    if method == "hdbscan":
        return hdbscan_cluster(embeddings["z_inv"], cfg.get("hdbscan", {})), {}
    if method == "kmeans_oracle":
        if n_true is None:
            raise ValueError("kmeans_oracle needs the true number of products")
        return kmeans_cluster(embeddings["z_inv"], n_true, seed), {}
    raise ValueError(f"Unknown clustering method {method!r}")
