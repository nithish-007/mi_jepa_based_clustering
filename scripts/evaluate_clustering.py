"""
Evaluate clustering of an MI-JEPA model trained on RPC.

Builds a held-out feature set from RPC `test2019/` bbox crops (each crop is one
product instance with a known category_id) and runs K-Means + NMI/ARI/ACC.

Config: `configs/evaluation.yaml`. If no checkpoint is set, evaluates a
pretrained ViT-S baseline (useful for the "before training" reference number).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.rpc_cached_dataset import RPCCachedCropEvalDataset
from src.data.transforms import get_eval_transform
from src.eval.clustering import compute_metrics, extract_features, run_kmeans
from src.eval.visualization import plot_tsne
from src.models.encoders import ContextEncoder
from src.models.mi_jepa import MIJEPAModule


# --------------------- Encoder loader --------------------- #


def load_encoder(cfg) -> torch.nn.Module:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = cfg["model"].get("checkpoint")
    if checkpoint:
        mi_jepa = MIJEPAModule.load_from_checkpoint(checkpoint, map_location=device)
        return mi_jepa.context_encoder.to(device).eval()
    return ContextEncoder(
        model_name=cfg["model"]["name"],
        embed_dim=cfg["model"].get("embed_dim"),
        pretrained=True,
    ).to(device).eval()


# --------------------- Main --------------------- #


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/evaluation.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    out_cfg = cfg["output"]
    vis_cfg = cfg["visualization"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    manifest = data_cfg["manifest_csv"]
    if not Path(manifest).exists():
        raise FileNotFoundError(
            f"Manifest not found at {manifest!r}. Run "
            "`python scripts/preprocess_crops.py` first."
        )

    transform = get_eval_transform(data_cfg["img_size"])
    dataset = RPCCachedCropEvalDataset(
        manifest,
        split=data_cfg["split"],
        transform=transform,
        samples_per_class=data_cfg["samples_per_class"],
        seed=42,
    )
    print(f"Eval set: {len(dataset)} crops from {dataset.num_categories} categories")

    loader = DataLoader(
        dataset,
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
        shuffle=False,
    )

    encoder = load_encoder(cfg)
    features, labels = extract_features(encoder, loader, device)
    print(f"Features: {features.shape}")

    n_clusters = int(len(np.unique(labels)))
    preds = run_kmeans(features, n_clusters)
    metrics = compute_metrics(labels, preds)
    results = {
        "image_level": {
            "NMI": metrics["NMI"],
            "ARI": metrics["ARI"],
            "ACC": metrics["ACC"],
            "num_crops": int(len(labels)),
            "num_categories": n_clusters,
        }
    }

    print("\n" + "=" * 60)
    for level, m in results.items():
        print(f"\n{level}:")
        for k, v in m.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print("=" * 60)

    output_dir = Path(out_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "clustering_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {output_dir/'clustering_results.json'}")

    if vis_cfg.get("plot_tsne", True):
        plot_tsne(features, labels, str(output_dir / "tsne.png"), title=out_cfg["run_name"])


if __name__ == "__main__":
    main()
