"""
Dataset Curation Script for MI-JEPA.

Scans the train/ directory directly (RP2K test dir has extraction issues on Windows
due to Chinese folder names).

Balancing strategy:
  - Select classes whose total image count is in [min_images, max_images] (default 100–300)
  - Take top `num_classes` of those, sorted by count descending
  - Cap each class at `cap_per_class` images (default 100) for a perfectly balanced dataset
  - Split each class into train/val/test (70/15/15)

Result: 100 classes × 100 images = 10 000 images, 70 train / 15 val / 15 test per class.

Outputs: dataset/meta_100.csv, dataset/class_mapping.csv
         dataset/dataset_100/ (copied class folders for visual verification)
"""
import csv
import random
import shutil
from pathlib import Path

import yaml


def load_config(config_path: str = "configs/dataset.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["dataset"]


def main():
    cfg = load_config()
    train_dir = Path(cfg["train_dir"])
    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")

    print(f"Scanning: {train_dir}")

    # Scan all subdirectories and collect image paths
    class_data = {}
    for subdir in sorted(train_dir.iterdir()):
        if not subdir.is_dir():
            continue
        imgs = [
            f for f in subdir.iterdir()
            if f.suffix.lower() in (".jpg", ".jpeg", ".png")
        ]
        if imgs:
            class_data[subdir.name] = imgs

    print(f"Found {len(class_data)} classes with images")
    all_counts = sorted([len(v) for v in class_data.values()], reverse=True)
    print(f"Image count range: {min(all_counts)}–{max(all_counts)}")

    # Filter classes whose count falls within [min_images, max_images]
    eligible = {
        k: v for k, v in class_data.items()
        if cfg["min_images"] <= len(v) <= cfg["max_images"]
    }
    eligible_sorted = sorted(eligible.items(), key=lambda x: len(x[1]), reverse=True)
    print(f"Classes with {cfg['min_images']}–{cfg['max_images']} images: {len(eligible)}")

    if len(eligible_sorted) < cfg["num_classes"]:
        print(
            f"WARNING: Only {len(eligible_sorted)} eligible classes found, "
            f"requested {cfg['num_classes']}. Using all eligible."
        )
        selected = eligible_sorted
    else:
        selected = eligible_sorted[: cfg["num_classes"]]

    print(f"Selected: {len(selected)} classes")

    # Class ID mapping (sorted by name for reproducibility)
    selected_names = sorted([name for name, _ in selected])
    class_to_id = {name: idx for idx, name in enumerate(selected_names)}

    # Build rows: cap each class at cap_per_class images, then split 70/15/15
    rng = random.Random(cfg["seed"])
    rows = []
    split_stats = {"train": 0, "val": 0, "test": 0}
    per_class_counts = {}

    for class_name, img_paths in selected:
        class_id = class_to_id[class_name]

        # Sort deterministically, shuffle, then cap
        img_paths_sorted = sorted(img_paths, key=lambda p: p.name)
        shuffled = list(img_paths_sorted)
        rng.shuffle(shuffled)
        capped = shuffled[: cfg["cap_per_class"]]

        n = len(capped)
        n_train = max(int(n * cfg["train_ratio"]), 4)  # min 4 for intra-bags (K=4)
        n_val = max(int(n * cfg["val_ratio"]), 1)
        n_test = n - n_train - n_val
        if n_test < 1:
            n_train -= 1
            n_test = 1

        per_class_counts[class_name] = {"total": n, "train": n_train, "val": n_val, "test": n_test}

        for i, img_path in enumerate(capped):
            if i < n_train:
                split = "train"
            elif i < n_train + n_val:
                split = "val"
            else:
                split = "test"
            split_stats[split] += 1
            rows.append([img_path.as_posix(), class_name, class_id, split])

    # Statistics
    totals  = [v["total"]  for v in per_class_counts.values()]
    trains  = [v["train"]  for v in per_class_counts.values()]
    vals    = [v["val"]    for v in per_class_counts.values()]
    tests   = [v["test"]   for v in per_class_counts.values()]

    print(f"\n--- Balanced Dataset Statistics ---")
    print(f"Classes : {len(selected)}")
    print(f"Total images : {len(rows)}")
    print(f"  Train : {split_stats['train']}  ({split_stats['train']/len(rows)*100:.1f}%)")
    print(f"  Val   : {split_stats['val']}   ({split_stats['val']/len(rows)*100:.1f}%)")
    print(f"  Test  : {split_stats['test']}  ({split_stats['test']/len(rows)*100:.1f}%)")
    print(f"Per-class total — min: {min(totals)}, max: {max(totals)}, "
          f"mean: {sum(totals)/len(totals):.1f}")
    print(f"Per-class train — min: {min(trains)}, max: {max(trains)}, "
          f"mean: {sum(trains)/len(trains):.1f}")
    print(f"Per-class val   — min: {min(vals)},  max: {max(vals)},  "
          f"mean: {sum(vals)/len(vals):.1f}")
    print(f"Per-class test  — min: {min(tests)},  max: {max(tests)},  "
          f"mean: {sum(tests)/len(tests):.1f}")

    if min(trains) < 4:
        low = sum(1 for c in trains if c < 4)
        print(f"\nWARNING: {low} classes have <4 train images — intra-bags will repeat images.")

    # Save meta CSV
    output_path = Path(cfg["meta_csv"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "class_name", "class_id", "split"])
        writer.writerows(rows)
    print(f"\nSaved  → {output_path}")

    # Save class mapping
    mapping_path = Path(cfg["class_mapping_csv"])
    with open(mapping_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_name", "class_id"])
        for name in selected_names:
            writer.writerow([name, class_to_id[name]])
    print(f"Mapping → {mapping_path}")

    # Copy selected class folders to dataset_100/ for visual verification
    dataset_100_dir = Path(cfg["dataset_100_dir"])
    if dataset_100_dir.exists():
        print(f"\n{dataset_100_dir} already exists — skipping copy.")
    else:
        dataset_100_dir.mkdir(parents=True)
        print(f"\nCopying {len(selected_names)} class folders to {dataset_100_dir} ...")
        for class_name in selected_names:
            src = train_dir / class_name
            dst = dataset_100_dir / class_name
            shutil.copytree(src, dst)
        print(f"Done. {len(selected_names)} folders copied.")


if __name__ == "__main__":
    main()
