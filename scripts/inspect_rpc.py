"""
RPC sanity-probe / quality report.

Verifies:
1. RPC checkout images really are multi-class bags (≥5 distinct categories on average).
2. Bbox-cropped product images have median size ≥150x150.
3. A pretrained ViT-S baseline NMI/ARI/ACC on RPC bbox crops.

Outputs are printed to stdout and saved to `results/rpc_inspect/`.
"""
from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.rpc_parser import RPCIndex, get_or_build_index
from src.data.transforms import get_eval_transform
from src.eval.clustering import compute_metrics, run_kmeans
from src.models.encoders import ContextEncoder


# ---------- A small in-process dataset of cropped products ---------- #


class RPCCropProbeDataset(Dataset):
    """
    Yields individual product crops from RPC checkout images for the baseline
    clustering probe. One sample == one bbox cropped from one image.
    """

    def __init__(self, index: RPCIndex, split: str, transform, samples_per_class: int):
        self.index = index
        self.split = split
        self.transform = transform

        # Build a balanced list of (annotation) by class
        rng = random.Random(42)
        ann_by_cat = index.annotations_by_category[split]
        sampled: list = []
        for cat_id, anns in ann_by_cat.items():
            if len(anns) == 0:
                continue
            take = anns if len(anns) <= samples_per_class else rng.sample(
                anns, samples_per_class
            )
            sampled.extend(take)
        rng.shuffle(sampled)
        self.samples = sampled

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        ann = self.samples[idx]
        path = self.index.image_path(self.split, ann.image_id)
        img = Image.open(path).convert("RGB")
        x, y, w, h = ann.bbox
        x0, y0 = max(0, int(round(x))), max(0, int(round(y)))
        x1, y1 = min(img.width, int(round(x + w))), min(img.height, int(round(y + h)))
        if x1 <= x0 + 1 or y1 <= y0 + 1:
            # Degenerate bbox — return whole image
            crop = img
        else:
            crop = img.crop((x0, y0, x1, y1))
        return self.transform(crop), ann.category_id, idx


# ---------- Per-section probes ---------- #


def probe_bag_composition(index: RPCIndex, split: str) -> dict:
    """Report distinct-category / total-products counts per checkout image."""
    distinct = []
    totals = []
    for img_id, anns in index.ann_by_image[split].items():
        cats = {a.category_id for a in anns}
        distinct.append(len(cats))
        totals.append(len(anns))
    return {
        "split": split,
        "num_images": len(distinct),
        "products_per_image": {
            "min": int(min(totals)),
            "mean": float(np.mean(totals)),
            "max": int(max(totals)),
        },
        "distinct_cats_per_image": {
            "min": int(min(distinct)),
            "mean": float(np.mean(distinct)),
            "max": int(max(distinct)),
        },
        "pct_with_ge_4_distinct_cats": float(np.mean([d >= 4 for d in distinct]) * 100),
    }


def probe_crop_sizes(
    index: RPCIndex, split: str, sample_n: int = 5000, seed: int = 42
) -> dict:
    """Report bbox sizes (without loading images — purely from annotations)."""
    rng = random.Random(seed)
    all_anns = [a for anns in index.ann_by_image[split].values() for a in anns]
    if len(all_anns) > sample_n:
        all_anns = rng.sample(all_anns, sample_n)
    widths = np.array([a.bbox[2] for a in all_anns])
    heights = np.array([a.bbox[3] for a in all_anns])
    min_dims = np.minimum(widths, heights)
    return {
        "split": split,
        "sample_n": int(len(all_anns)),
        "width": {
            "median": float(np.median(widths)),
            "p10": float(np.percentile(widths, 10)),
            "p90": float(np.percentile(widths, 90)),
        },
        "height": {
            "median": float(np.median(heights)),
            "p10": float(np.percentile(heights, 10)),
            "p90": float(np.percentile(heights, 90)),
        },
        "min_side": {
            "median": float(np.median(min_dims)),
            "p10": float(np.percentile(min_dims, 10)),
            "p90": float(np.percentile(min_dims, 90)),
            "frac_lt_100": float(np.mean(min_dims < 100)),
            "frac_lt_150": float(np.mean(min_dims < 150)),
        },
    }


@torch.no_grad()
def probe_pretrained_clustering(
    index: RPCIndex,
    split: str,
    samples_per_class: int = 30,
    img_size: int = 224,
    batch_size: int = 128,
    num_workers: int = 4,
    device: str | torch.device = "cuda",
) -> dict:
    """Run pretrained ViT-S clustering on bbox crops."""
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    transform = get_eval_transform(img_size)
    dataset = RPCCropProbeDataset(index, split, transform, samples_per_class)
    loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True
    )

    encoder = ContextEncoder(
        model_name="vit_small_patch16_224", embed_dim=384, pretrained=True
    ).to(device).eval()

    feats: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for imgs, cats, _ in tqdm(loader, desc=f"Encoding {split} crops"):
        imgs = imgs.to(device, non_blocking=True)
        f = encoder(imgs).cpu().numpy()
        feats.append(f)
        labels.append(np.asarray(cats))
    features = np.concatenate(feats, axis=0)
    label_arr = np.concatenate(labels, axis=0)

    n_clusters = int(len(np.unique(label_arr)))
    preds = run_kmeans(features, n_clusters)
    metrics = compute_metrics(label_arr, preds)
    return {
        "split": split,
        "num_crops": int(len(label_arr)),
        "num_categories": n_clusters,
        "feature_dim": int(features.shape[1]),
        **metrics,
    }


# ---------- Main ---------- #


def _format(d: dict, indent: int = 2) -> str:
    import json
    return json.dumps(d, indent=indent, sort_keys=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rpc_root", default="retail_product_checkout", help="Path to RPC dataset root"
    )
    parser.add_argument(
        "--cache", default="dataset/rpc_index.pkl", help="Index cache path"
    )
    parser.add_argument(
        "--probe_split", default="val", choices=["val", "test"],
        help="Split to use for the pretrained-clustering probe",
    )
    parser.add_argument("--samples_per_class", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--skip_clustering", action="store_true",
        help="Skip the (slow) GPU clustering probe.",
    )
    parser.add_argument("--out_dir", default="results/rpc_inspect")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("RPC sanity probe")
    print("=" * 70)
    print(f"Loading / building index from {args.rpc_root} ...")
    index = get_or_build_index(args.rpc_root, args.cache)
    print(f"  Categories: {index.num_categories}")
    print(f"  Splits: train={len(index.images.get('train', {}))}, "
          f"val={len(index.images.get('val', {}))}, "
          f"test={len(index.images.get('test', {}))}")

    report: dict = {"rpc_root": args.rpc_root, "num_categories": index.num_categories}

    print("\n--- 1) Bag composition (checkout images) ---")
    for split in ("val", "test"):
        info = probe_bag_composition(index, split)
        report.setdefault("bag_composition", {})[split] = info
        print(_format(info))

    print("\n--- 2) Bbox crop sizes ---")
    for split in ("val", "test"):
        info = probe_crop_sizes(index, split)
        report.setdefault("crop_sizes", {})[split] = info
        print(_format(info))

    if not args.skip_clustering:
        print(f"\n--- 3) Pretrained ViT-S clustering on {args.probe_split} crops ---")
        info = probe_pretrained_clustering(
            index,
            args.probe_split,
            samples_per_class=args.samples_per_class,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        report["pretrained_clustering"] = info
        print(_format(info))

    out_path = out_dir / "rpc_inspect.json"
    import json
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nReport saved to {out_path}")


if __name__ == "__main__":
    main()
