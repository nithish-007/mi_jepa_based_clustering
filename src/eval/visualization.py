"""
Visualization utilities for MI-JEPA evaluation.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image


def plot_tsne(
    features: np.ndarray,
    labels: np.ndarray,
    save_path: str,
    title: str = "t-SNE",
    max_points: int = 5000,
    perplexity: int = 30,
):
    """
    t-SNE visualization of embeddings colored by class.
    Subsamples if too many points for readability.
    """
    from sklearn.manifold import TSNE

    if len(features) > max_points:
        idx = np.random.choice(len(features), max_points, replace=False)
        features = features[idx]
        labels = labels[idx]

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    coords = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=(10, 10))
    scatter = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=labels, cmap="tab20", s=3, alpha=0.6,
    )
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved t-SNE plot to {save_path}")


def save_cluster_images(
    image_paths: list[str],
    cluster_labels: np.ndarray,
    true_labels: np.ndarray,
    output_dir: str,
    images_per_cluster: int = 16,
    grid_cols: int = 4,
):
    """
    Save grid images for each predicted cluster.

    Creates one image grid per cluster showing sample images with their
    true class labels, saved to output_dir/cluster_XX.png.

    Args:
        image_paths: List of image file paths (len N).
        cluster_labels: (N,) predicted cluster IDs from K-Means.
        true_labels: (N,) true class IDs.
        output_dir: Directory to save cluster grid images.
        images_per_cluster: Max images to show per cluster.
        grid_cols: Number of columns in the grid.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    unique_clusters = np.unique(cluster_labels)
    print(f"Saving cluster images for {len(unique_clusters)} clusters to {out}")

    for cid in unique_clusters:
        mask = cluster_labels == cid
        indices = np.where(mask)[0]

        # Subsample if too many
        if len(indices) > images_per_cluster:
            indices = np.random.choice(indices, images_per_cluster, replace=False)

        n = len(indices)
        grid_rows = (n + grid_cols - 1) // grid_cols

        fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(3 * grid_cols, 3 * grid_rows))
        if grid_rows == 1:
            axes = [axes] if grid_cols == 1 else [axes]
        axes = np.array(axes).reshape(-1)

        for ax in axes:
            ax.axis("off")

        for i, idx in enumerate(indices):
            img = Image.open(image_paths[idx]).convert("RGB")
            axes[i].imshow(img)
            axes[i].set_title(f"true={true_labels[idx]}", fontsize=7)

        fig.suptitle(f"Cluster {cid}  (n={int(mask.sum())})", fontsize=12, fontweight="bold")
        fig.tight_layout()
        fig.savefig(out / f"cluster_{cid:02d}.png", dpi=100, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved {len(unique_clusters)} cluster grids to {out}")


def plot_energy_distribution(
    energy_intra: np.ndarray,
    energy_inter: np.ndarray,
    save_path: str,
    title: str = "Energy Distribution",
):
    """Histogram of energy scores for intra vs inter bags."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(energy_intra.ravel(), bins=50, alpha=0.6, label="Intra (positive)", color="green")
    ax.hist(energy_inter.ravel(), bins=50, alpha=0.6, label="Inter (negative)", color="red")
    ax.set_xlabel("Energy")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_attention_weights(
    attention_weights: np.ndarray,
    save_path: str,
    title: str = "Attention Weights",
):
    """Heatmap of attention weights across bag instances."""
    fig, ax = plt.subplots(figsize=(6, 8))
    sns.heatmap(
        attention_weights, annot=True, fmt=".3f",
        cmap="YlOrRd", ax=ax, vmin=0, vmax=1,
    )
    ax.set_xlabel("Instance in Bag")
    ax.set_ylabel("Sample")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
