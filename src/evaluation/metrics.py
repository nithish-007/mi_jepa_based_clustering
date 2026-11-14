"""Clustering and retrieval metrics. All clustering metrics come from one contingency
table, so they run in O(N) even for the full test split."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

MAX_HUNGARIAN_CELLS = 50_000_000


def expand_unassigned(labels: np.ndarray) -> np.ndarray:
    """Give every ``-1`` (unassigned) item its own cluster id, which is how it is scored."""
    labels = np.asarray(labels).copy()
    noise = labels < 0
    if noise.any():
        start = labels.max() + 1 if (~noise).any() else 0
        labels[noise] = np.arange(start, start + noise.sum())
    return labels


def _pairs(sizes: np.ndarray) -> float:
    sizes = sizes.astype(np.float64)
    return float((sizes * (sizes - 1) / 2).sum())


def _f1(precision: float, recall: float) -> float:
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def _purity(cell_p: np.ndarray, counts: np.ndarray, n_pred: int, n: int) -> float:
    best = np.zeros(n_pred)
    np.maximum.at(best, cell_p, counts)
    return float(best.sum() / n)


def clustering_metrics(labels_true, labels_pred) -> dict:
    true = np.asarray(labels_true)
    pred = expand_unassigned(labels_pred)
    n = len(true)
    _, t = np.unique(true, return_inverse=True)
    _, p = np.unique(pred, return_inverse=True)
    n_true, n_pred = t.max() + 1, p.max() + 1
    cells, counts = np.unique(t.astype(np.int64) * n_pred + p, return_counts=True)
    cell_t, cell_p = cells // n_pred, cells % n_pred
    size_t, size_p = np.bincount(t), np.bincount(p)
    counts_f = counts.astype(np.float64)

    bcubed_precision = float((counts_f**2 / size_p[cell_p]).sum() / n)
    bcubed_recall = float((counts_f**2 / size_t[cell_t]).sum() / n)
    true_positive = _pairs(counts)
    predicted_pairs, actual_pairs = _pairs(size_p), _pairs(size_t)
    pair_precision = true_positive / predicted_pairs if predicted_pairs else 1.0
    pair_recall = true_positive / actual_pairs if actual_pairs else 1.0

    accuracy = None
    if n_true * n_pred <= MAX_HUNGARIAN_CELLS:
        table = coo_matrix((counts, (cell_p, cell_t)), shape=(n_pred, n_true)).toarray()
        rows, cols = linear_sum_assignment(table, maximize=True)
        accuracy = float(table[rows, cols].sum() / n)

    return {
        "bcubed_precision": bcubed_precision,
        "bcubed_recall": bcubed_recall,
        "bcubed_f1": _f1(bcubed_precision, bcubed_recall),
        "pairwise_precision": pair_precision,
        "pairwise_recall": pair_recall,
        "pairwise_f1": _f1(pair_precision, pair_recall),
        "nmi": float(normalized_mutual_info_score(true, pred)),
        "ari": float(adjusted_rand_score(true, pred)),
        "cluster_accuracy": accuracy,
        "purity": _purity(cell_p, counts_f, n_pred, n),
        "n_images": int(n),
        "n_true_clusters": int(n_true),
        "n_pred_clusters": int(n_pred),
        "n_pred_singletons": int((size_p == 1).sum()),
        "cluster_count_ratio": float(n_pred / n_true),
    }


def recall_at_k(neighbors: np.ndarray, labels: np.ndarray, ks=(1, 5)) -> dict:
    """Retrieval recall from a precomputed kNN table (self excluded). Queries whose product has
    no other image are skipped."""
    labels = np.asarray(labels)
    codes = np.unique(labels, return_inverse=True)[1]
    has_pair = np.bincount(codes)[codes] > 1
    hits = labels[neighbors] == labels[:, None]
    return {f"recall_at_{k}": float(hits[has_pair, :k].any(axis=1).mean()) for k in ks if k <= neighbors.shape[1]}
