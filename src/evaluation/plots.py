from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def pair_diagnostics_figure(edge_features: np.ndarray, edge_same: np.ndarray, feature_names: list[str], sweep: list[dict],
                            title: str = "") -> plt.Figure:
    """Same/different histograms for each pair feature plus the threshold sweep on validation."""
    same = np.asarray(edge_same).astype(bool)
    panels = len(feature_names) + 1
    fig, axes = plt.subplots(1, panels, figsize=(5 * panels, 3.8))
    for axis, column, name in zip(axes, edge_features.T, feature_names):
        bins = np.linspace(column.min(), column.max(), 60)
        axis.hist(column[~same], bins=bins, alpha=0.6, density=True, label="different product")
        axis.hist(column[same], bins=bins, alpha=0.6, density=True, label="same product")
        axis.set_title(f"kNN candidate pairs: {name}")
        axis.legend()
    axis = axes[-1]
    thresholds = [row["threshold"] for row in sweep]
    for key in ("bcubed_precision", "bcubed_recall", "bcubed_f1"):
        axis.plot(thresholds, [row[key] for row in sweep], marker="o", label=key)
    axis.set_xlabel("merge probability threshold")
    axis.set_title("validation clustering vs threshold")
    axis.legend()
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def cluster_size_figure(labels_pred: np.ndarray, labels_true: np.ndarray | None = None, title: str = "") -> plt.Figure:
    fig, axis = plt.subplots(figsize=(6, 3.8))
    pred_sizes = np.bincount(np.unique(labels_pred, return_inverse=True)[1])
    bins = np.logspace(0, np.log10(max(pred_sizes.max(), 2)) + 0.1, 30)
    axis.hist(pred_sizes, bins=bins, alpha=0.6, label=f"predicted ({len(pred_sizes)} clusters)")
    if labels_true is not None:
        true_sizes = np.bincount(np.unique(labels_true, return_inverse=True)[1])
        axis.hist(true_sizes, bins=bins, alpha=0.6, label=f"true ({len(true_sizes)} products)")
    axis.set_xscale("log")
    axis.set_xlabel("images per cluster")
    axis.set_ylabel("clusters")
    axis.set_title(title or "cluster sizes")
    axis.legend()
    fig.tight_layout()
    return fig
