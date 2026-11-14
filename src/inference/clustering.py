"""Open-world clustering: the number of products is never given.

graph method:
  1. kNN on z_inv proposes candidate pairs (cheap, high recall).
  2. A calibrated pair scorer turns (cosine, energy) into P(same product).
  3. Pairs above the threshold are merged in descending probability with union-find.
     With ``linkage_check: average`` a merge of two existing clusters is accepted only if
     sampled cross pairs also score above the threshold, which stops long chains.
  Items that never merge stay singletons.
"""
from __future__ import annotations

import faiss
import numpy as np
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import PCA


def knn(z: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Exact cosine kNN on L2-normalised rows. Returns ``(neighbors, similarities)`` without self."""
    z = np.ascontiguousarray(z, dtype=np.float32)
    k = min(int(k), len(z) - 1)
    index = faiss.IndexFlatIP(z.shape[1])
    index.add(z)
    sims, neighbors = index.search(z, k + 1)
    keep = neighbors != np.arange(len(z))[:, None]
    neighbors = np.stack([row[mask][:k] for row, mask in zip(neighbors, keep)])
    sims = np.stack([row[mask][:k] for row, mask in zip(sims, keep)])
    return neighbors, sims


def candidate_edges(neighbors: np.ndarray, mutual: bool = False) -> np.ndarray:
    """Unique undirected ``(i, j)`` pairs with ``i < j`` from a kNN table."""
    n, k = neighbors.shape
    src = np.repeat(np.arange(n), k)
    dst = neighbors.reshape(-1)
    edges = np.stack([np.minimum(src, dst), np.maximum(src, dst)], axis=1)
    edges, counts = np.unique(edges, axis=0, return_counts=True)
    if mutual:
        edges = edges[counts > 1]
    return edges


class _UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n)
        self.members = {i: [i] for i in range(n)}

    def find(self, i: int) -> int:
        root = i
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[i] != root:
            self.parent[i], i = root, self.parent[i]
        return root

    def union(self, a: int, b: int) -> None:
        if len(self.members[a]) < len(self.members[b]):
            a, b = b, a
        self.parent[b] = a
        self.members[a].extend(self.members.pop(b))


def graph_cluster(n: int, edges: np.ndarray, probs: np.ndarray, threshold: float, pair_prob=None,
                  linkage_check: str = "average", max_check_pairs: int = 32, seed: int = 0) -> np.ndarray:
    """Labels ``0..C-1`` (singletons included). ``pair_prob(i_idx, j_idx)`` scores arbitrary pairs."""
    rng = np.random.default_rng(seed)
    forest = _UnionFind(n)
    order = np.argsort(-probs, kind="stable")
    for e in order:
        if probs[e] < threshold:
            break
        a, b = forest.find(int(edges[e, 0])), forest.find(int(edges[e, 1]))
        if a == b:
            continue
        left, right = forest.members[a], forest.members[b]
        if linkage_check == "average" and pair_prob is not None and len(left) * len(right) > 1:
            m = min(max_check_pairs, len(left) * len(right))
            i = rng.choice(left, m)
            j = rng.choice(right, m)
            if pair_prob(i, j).mean() < threshold:
                continue
        forest.union(a, b)
    roots = np.array([forest.find(i) for i in range(n)])
    return np.unique(roots, return_inverse=True)[1]


def hdbscan_cluster(z: np.ndarray, cfg: dict) -> np.ndarray:
    """HDBSCAN baseline. Noise points (-1) are returned as -1 and scored as singletons."""
    x = np.asarray(z, dtype=np.float32)
    if cfg.get("pca_dim") and x.shape[1] > int(cfg["pca_dim"]):
        x = PCA(int(cfg["pca_dim"]), random_state=0).fit_transform(x)
    return HDBSCAN(min_cluster_size=int(cfg.get("min_cluster_size", 2)), min_samples=cfg.get("min_samples"),
                   metric="euclidean", copy=False).fit_predict(x)


def kmeans_cluster(z: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """Diagnostic only: uses the true number of products, which open-world clustering never has."""
    return KMeans(n_clusters=int(k), n_init=3, random_state=seed).fit_predict(np.asarray(z, dtype=np.float32))
