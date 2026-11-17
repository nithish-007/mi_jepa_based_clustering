"""SKU-disjoint train / val / test splits.

Every image of a product lands in exactly one split, so validation and test
measure clustering of products the model never saw. ``sku_order`` and
``image_order`` make subsets nested: the 20-SKU test subset is inside the
100-SKU one, and a 10-image cap is inside a 20-image cap.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import pandas as pd

from src.config import resolve_path
from src.data.indexing import IMAGE_EXTENSIONS, discover_images, family_key, file_md5, image_error

MANIFEST_COLUMNS = ["image_path", "sku_id", "family", "split", "sku_order", "image_order", "official_split", "raw_dirname"]
SPLITS = ("train", "val", "test")


def assign_skus(skus: list[str], fractions: dict, seed: int) -> dict[str, list[str]]:
    order = sorted(skus)
    random.Random(seed).shuffle(order)
    n_train = int(round(len(order) * float(fractions["train"])))
    n_val = int(round(len(order) * float(fractions["val"])))
    return {"train": order[:n_train], "val": order[n_train : n_train + n_val], "test": order[n_train + n_val :]}


def _drop_bad_files(frame: pd.DataFrame, root: Path, validate: bool, drop_duplicates: bool) -> tuple[pd.DataFrame, dict]:
    audit = {"unreadable": 0, "within_sku_duplicates": 0, "cross_sku_duplicates": 0}
    if validate:
        keep = []
        for count, relative in enumerate(frame["image_path"], start=1):
            if count % 20000 == 0:
                print(f"validated {count}/{len(frame)} images", flush=True)
            keep.append(image_error(root / relative) is None)
        audit["unreadable"] = int(len(keep) - sum(keep))
        frame = frame.loc[keep].copy()
    if drop_duplicates:
        frame = frame.copy()
        frame["md5"] = [file_md5(root / relative) for relative in frame["image_path"]]
        drop = []
        for _digest, part in frame.groupby("md5"):
            if len(part) < 2:
                continue
            extras = part.sort_values("image_path").index[1:]
            drop.extend(extras)
            key = "cross_sku_duplicates" if part["sku_id"].nunique() > 1 else "within_sku_duplicates"
            audit[key] += int(len(extras))
        frame = frame.drop(index=drop).drop(columns=["md5"])
    return frame.reset_index(drop=True), audit


def _dense_image_order(frame: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for _key, part in frame.groupby(["split", "sku_id"], sort=False):
        part = part.sort_values("image_path").copy()
        part["image_order"] = range(len(part))
        pieces.append(part)
    return pd.concat(pieces, ignore_index=True)


def prepare_splits(config: dict) -> dict:
    """Index ``raw_root``, drop unusable files, split by SKU, write the manifest and a summary."""
    root = resolve_path(config["raw_root"])
    destination = resolve_path(config["output_dir"])
    destination.mkdir(parents=True, exist_ok=True)
    seed = int(config["seed"])
    discovered = discover_images(root, config.get("image_extensions") or IMAGE_EXTENSIONS)
    if discovered.empty:
        raise RuntimeError(f"No images found under {root}/all/<train|test>/<sku>/")
    excluded = set(config.get("exclude_sku_ids") or [])
    n_discovered = len(discovered)
    n_excluded = int(discovered["sku_id"].isin(excluded).sum())
    shared = int(discovered.groupby("sku_id")["official_split"].nunique().gt(1).sum())
    frame = discovered.loc[~discovered["sku_id"].isin(excluded)]
    frame, audit = _drop_bad_files(frame, root, bool(config["validate_images"]), bool(config["drop_duplicate_bytes"]))

    assignment = assign_skus(frame["sku_id"].unique().tolist(), config["split_fractions"], seed)
    split_of, order_of = {}, {}
    for split, skus in assignment.items():
        for rank, sku in enumerate(skus):
            split_of[sku], order_of[sku] = split, rank
    frame = frame.copy()
    frame["split"] = frame["sku_id"].map(split_of)
    frame["sku_order"] = frame["sku_id"].map(order_of)
    frame["family"] = frame["sku_id"].map(family_key)
    frame = _dense_image_order(frame)
    manifest = frame[MANIFEST_COLUMNS].sort_values(["split", "sku_order", "image_order", "image_path"]).reset_index(drop=True)
    check_disjoint(manifest)

    manifest_path = destination / "manifest.csv"
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    summary = {
        "raw_root": str(config["raw_root"]),
        "manifest": str(config["output_dir"]) + "/manifest.csv",
        "sha256": digest,
        "seed": seed,
        "images_discovered": int(n_discovered),
        "images_in_excluded_skus": n_excluded,
        "excluded_sku_ids": sorted(excluded),
        "skus_in_both_official_folders": shared,
        "removed": audit,
        "skus": {s: int(manifest.loc[manifest["split"] == s, "sku_id"].nunique()) for s in SPLITS},
        "images": {s: int((manifest["split"] == s).sum()) for s in SPLITS},
        "images_per_sku": manifest.groupby("sku_id").size().describe().round(2).to_dict(),
    }
    (destination / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def check_disjoint(manifest: pd.DataFrame) -> None:
    owners = manifest.groupby("sku_id")["split"].nunique()
    leaked = owners[owners > 1]
    if len(leaked):
        raise RuntimeError(f"{len(leaked)} SKUs appear in more than one split")


def load_manifest(path: str | Path) -> pd.DataFrame:
    path = resolve_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{path} does not exist. Run: python scripts/prepare_data.py")
    manifest = pd.read_csv(path, encoding="utf-8-sig", dtype={"sku_id": str, "family": str})
    check_disjoint(manifest)
    return manifest


def select_records(manifest: pd.DataFrame, split: str, max_skus: int | None = None, max_images_per_sku: int | None = None) -> pd.DataFrame:
    """Nested subset of one split. Smaller settings are always contained in larger ones."""
    chosen = manifest.loc[manifest["split"] == split]
    if chosen.empty:
        raise RuntimeError(f"Manifest has no rows for split {split!r}")
    if max_skus is not None:
        chosen = chosen.loc[chosen["sku_order"] < int(max_skus)]
    if max_images_per_sku is not None:
        chosen = chosen.loc[chosen["image_order"] < int(max_images_per_sku)]
    return chosen.reset_index(drop=True)
