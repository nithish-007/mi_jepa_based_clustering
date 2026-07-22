"""
Clustering evaluation pipeline for MI-JEPA.
Supports image-level and product-level clustering with NMI, ARI, ACC metrics.
"""
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from torch.utils.data import DataLoader
from tqdm import tqdm


@torch.no_grad()
def extract_features(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device = torch.device("cuda"),
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract [CLS] embeddings from a frozen encoder.

    Returns:
        features: (N, D) numpy array of embeddings.
        labels: (N,) numpy array of integer class IDs.
    """
    model.eval()
    all_features = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Extracting features"):
        # Eval datasets may yield (image, label) or (image, label, index).
        images, class_ids = batch[0], batch[1]
        images = images.to(device)
        features = model(images)
        all_features.append(features.cpu().numpy())
        all_labels.append(class_ids.numpy())

    return np.concatenate(all_features), np.concatenate(all_labels)


def run_kmeans(features: np.ndarray, n_clusters: int, seed: int = 42) -> np.ndarray:
    """Run K-Means clustering."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    return kmeans.fit_predict(features)


def clustering_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Clustering accuracy via Hungarian matching.
    Finds the optimal 1-to-1 mapping between predicted clusters and true labels.
    """
    assert y_true.shape == y_pred.shape
    n = y_true.shape[0]

    # Build cost matrix
    true_labels = np.unique(y_true)
    pred_labels = np.unique(y_pred)
    n_true = len(true_labels)
    n_pred = len(pred_labels)

    # Map labels to contiguous indices
    true_map = {l: i for i, l in enumerate(true_labels)}
    pred_map = {l: i for i, l in enumerate(pred_labels)}

    size = max(n_true, n_pred)
    cost = np.zeros((size, size), dtype=np.int64)

    for t, p in zip(y_true, y_pred):
        cost[true_map[t], pred_map[p]] += 1

    # Hungarian matching (maximize matches = minimize -cost)
    row_ind, col_ind = linear_sum_assignment(-cost)
    return cost[row_ind, col_ind].sum() / n


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    """Compute NMI, ARI, and ACC."""
    return {
        "NMI": normalized_mutual_info_score(y_true, y_pred),
        "ARI": adjusted_rand_score(y_true, y_pred),
        "ACC": clustering_accuracy(y_true, y_pred),
    }


def product_level_features(
    features: np.ndarray,
    labels: np.ndarray,
    pooling: str = "mean",
    attention_pooling_fn=None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Aggregate image-level features to product-level.

    Args:
        features: (N, D) image embeddings.
        labels: (N,) class IDs.
        pooling: 'mean' or 'attention'.
        attention_pooling_fn: Callable for gated attention pooling
            (takes (1, N_p, D) tensor → (1, D) tensor). Used when pooling='attention'.

    Returns:
        prod_features: (C, D) one embedding per product class.
        prod_labels: (C,) class IDs.
    """
    unique_labels = np.unique(labels)
    prod_features = []
    prod_labels = []

    for cid in unique_labels:
        mask = labels == cid
        class_features = features[mask]  # (N_p, D)

        if pooling == "attention" and attention_pooling_fn is not None:
            feat_tensor = torch.from_numpy(class_features).unsqueeze(0).float()
            with torch.no_grad():
                agg, _ = attention_pooling_fn(feat_tensor)
            prod_features.append(agg.squeeze(0).cpu().numpy())
        else:
            prod_features.append(class_features.mean(axis=0))

        prod_labels.append(cid)

    return np.stack(prod_features), np.array(prod_labels)


def evaluate_clustering(
    features: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
    attention_pooling_fn=None,
) -> dict[str, dict[str, float]]:
    """
    Full evaluation at both image-level and product-level.

    Returns dict with keys 'image_level' and 'product_level',
    each containing NMI, ARI, ACC.
    """
    results = {}

    # Image-level
    pred_img = run_kmeans(features, n_clusters)
    results["image_level"] = compute_metrics(labels, pred_img)

    # Product-level (mean pooling)
    prod_feat_mean, prod_labels = product_level_features(features, labels, "mean")
    pred_prod_mean = run_kmeans(prod_feat_mean, n_clusters)
    results["product_level_mean"] = compute_metrics(prod_labels, pred_prod_mean)

    # Product-level (attention pooling) if available
    if attention_pooling_fn is not None:
        prod_feat_attn, _ = product_level_features(
            features, labels, "attention", attention_pooling_fn
        )
        pred_prod_attn = run_kmeans(prod_feat_attn, n_clusters)
        results["product_level_attn"] = compute_metrics(prod_labels, pred_prod_attn)

    return results
