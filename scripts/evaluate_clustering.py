"""
Evaluate clustering performance of a trained MI-JEPA model or baseline.
Supports both image-level and product-level evaluation.
Config-driven via configs/evaluation.yaml.
"""
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.rp2k_dataset import RP2KDataset
from src.data.transforms import get_eval_transform
from src.eval.clustering import evaluate_clustering, extract_features, run_kmeans
from src.eval.visualization import plot_tsne, save_cluster_images
from src.models.attention_pooling import GatedAttentionPooling
from src.models.encoders import ContextEncoder
from src.models.mi_jepa import MIJEPAModule


def load_config(path: str = "configs/evaluation.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_encoder(cfg) -> tuple[torch.nn.Module, GatedAttentionPooling | None]:
    """Load encoder (standalone or from MI-JEPA checkpoint)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = cfg["model"].get("checkpoint")

    if checkpoint:
        mi_jepa = MIJEPAModule.load_from_checkpoint(checkpoint, map_location=device)
        encoder = mi_jepa.context_encoder.to(device)
        attn_pool = mi_jepa.attention_pooling.to(device)
        return encoder, attn_pool

    encoder = ContextEncoder(
        model_name=cfg["model"]["name"],
        embed_dim=cfg["model"]["embed_dim"],
        pretrained=True,
    ).to(device)
    return encoder, None


def main():
    cfg = load_config()
    data_cfg = cfg["data"]
    vis_cfg = cfg["visualization"]
    out_cfg = cfg["output"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load model ---
    encoder, attn_pool = load_encoder(cfg)
    encoder.eval()

    # --- Dataset ---
    transform = get_eval_transform(data_cfg["img_size"])
    dataset = RP2KDataset(
        meta_csv=data_cfg["meta_csv"],
        split=data_cfg["split"],
        transform=transform,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=data_cfg["batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
    )
    n_clusters = dataset.num_classes
    print(f"Test set: {len(dataset)} images, {n_clusters} classes")

    # --- Extract features ---
    features, labels = extract_features(encoder, dataloader, device)
    print(f"Features shape: {features.shape}")

    # --- Evaluate ---
    attn_fn = None
    if attn_pool is not None:
        attn_pool.eval()
        attn_fn = lambda x: attn_pool(x.to(device))

    results = evaluate_clustering(features, labels, n_clusters, attn_fn)

    # --- Print results ---
    print("\n" + "=" * 60)
    for level, metrics in results.items():
        print(f"\n{level}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    print("=" * 60)

    # --- Save results ---
    output_dir = Path(out_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "clustering_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # --- t-SNE visualization ---
    if vis_cfg.get("plot_tsne", True):
        tsne_path = output_dir / "tsne.png"
        plot_tsne(features, labels, str(tsne_path), title=out_cfg["run_name"])

    # --- Cluster image visualization ---
    if vis_cfg.get("save_cluster_images", False):
        cluster_preds = run_kmeans(features, n_clusters)
        save_cluster_images(
            image_paths=dataset.image_paths,
            cluster_labels=cluster_preds,
            true_labels=labels,
            output_dir=vis_cfg["cluster_images_dir"],
            images_per_cluster=vis_cfg.get("images_per_cluster", 16),
            grid_cols=vis_cfg.get("grid_cols", 4),
        )


if __name__ == "__main__":
    main()
