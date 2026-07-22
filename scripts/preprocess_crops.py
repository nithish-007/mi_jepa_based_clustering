"""
One-time bbox pre-extraction (parallel).

Iterates every annotation in train2019 / val2019 / test2019, crops the bbox from
the HD source image, optionally resizes (max long side), and saves as JPEG.

This replaces on-the-fly HD-load-and-crop and restores training throughput from
~0.13 it/s (IO-bound) to ~1.7 it/s (cached small JPEGs).

Output:
    dataset/rpc_crops/<split>/<category_id>/ann<ann_id>.jpg
    dataset/rpc_crops_manifest.csv  (ann_id, split, image_id, category_id, crop_path, crop_w, crop_h)

Parallelism: source images are split across N worker processes; each worker
crops the bboxes from images assigned to it and returns the manifest rows. This
turns the sequential ~30 it/s into ~800+ it/s on a 12-core CPU.

Run-once. Idempotent — skips crops whose target file already exists, so
re-running on top of a partial run is safe and cheap.
"""
from __future__ import annotations

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys

from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.rpc_parser import get_or_build_index


# Globals visible to worker processes (populated by _init_worker via fork-equivalent
# on Windows we use spawn + initializer to set them).
_INDEX = None
_OUT_DIR = None
_MAX_SIDE = 384
_QUALITY = 90
_PADDING = 0.0


def _init_worker(rpc_root: str, index_cache: str, out_dir: str,
                 max_side: int, quality: int, padding: float):
    global _INDEX, _OUT_DIR, _MAX_SIDE, _QUALITY, _PADDING
    from src.data.rpc_parser import get_or_build_index as _build  # noqa
    _INDEX = _build(rpc_root, index_cache)
    _OUT_DIR = Path(out_dir)
    _MAX_SIDE = max_side
    _QUALITY = quality
    _PADDING = padding


def _crop_with_padding(img: Image.Image, bbox, padding: float):
    x, y, w, h = bbox
    px, py = w * padding, h * padding
    x0 = max(0, int(round(x - px)))
    y0 = max(0, int(round(y - py)))
    x1 = min(img.width, int(round(x + w + px)))
    y1 = min(img.height, int(round(y + h + py)))
    if x1 <= x0 + 1 or y1 <= y0 + 1:
        return None
    return img.crop((x0, y0, x1, y1))


def _resize_max_side(img: Image.Image, max_side: int) -> Image.Image:
    long = max(img.size)
    if max_side <= 0 or long <= max_side:
        return img
    scale = max_side / long
    return img.resize(
        (max(1, int(img.width * scale)), max(1, int(img.height * scale))),
        Image.LANCZOS,
    )


def _process_image(args):
    """Worker: crop all bboxes from one source image."""
    split, image_id = args
    anns = _INDEX.ann_by_image[split].get(image_id, [])
    if not anns:
        return []

    src_path = _INDEX.image_path(split, image_id)
    try:
        src = Image.open(src_path).convert("RGB")
    except Exception:
        return []

    rows = []
    for a in anns:
        target = _OUT_DIR / split / str(a.category_id) / f"ann{a.ann_id}.jpg"
        if target.exists():
            try:
                with Image.open(target) as im:
                    cw, ch = im.size
                rows.append((a.ann_id, split, image_id, a.category_id,
                             target.as_posix(), cw, ch))
                continue
            except Exception:
                pass  # corrupt — re-crop below

        crop = _crop_with_padding(src, a.bbox, _PADDING)
        if crop is None:
            continue
        if _MAX_SIDE > 0:
            crop = _resize_max_side(crop, _MAX_SIDE)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            crop.save(target, "JPEG", quality=_QUALITY, optimize=True)
        except Exception:
            continue
        rows.append((a.ann_id, split, image_id, a.category_id,
                     target.as_posix(), crop.width, crop.height))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rpc_root", default="retail_product_checkout")
    ap.add_argument("--index_cache", default="dataset/rpc_index.pkl")
    ap.add_argument("--out_dir", default="dataset/rpc_crops")
    ap.add_argument("--manifest", default="dataset/rpc_crops_manifest.csv")
    ap.add_argument("--max_side", type=int, default=384)
    ap.add_argument("--quality", type=int, default=90)
    ap.add_argument("--padding", type=float, default=0.0)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                    choices=["train", "val", "test"])
    ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) - 1),
                    help="Number of worker processes (default: cpu_count-1)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    # Load index in the master too (for planning + dir pre-creation)
    index = get_or_build_index(args.rpc_root, args.index_cache)
    print(f"Loaded RPC index: {index.num_categories} categories")
    print(f"Workers: {args.workers}")

    # Discover existing manifest entries so we don't duplicate rows on re-run.
    seen: set[tuple[str, int]] = set()
    write_header = not manifest_path.exists()
    if not write_header:
        with open(manifest_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                seen.add((row["split"], int(row["ann_id"])))
        print(f"Manifest has {len(seen)} existing entries — they will be skipped.")

    # Pre-create per-category dirs so workers don't race on mkdir.
    for split in args.splits:
        sd = out_dir / split
        for cat in index.categories.keys():
            (sd / str(cat)).mkdir(parents=True, exist_ok=True)

    # Build the work list: one job per (split, image_id).
    jobs: list[tuple[str, int]] = []
    for split in args.splits:
        for image_id in index.ann_by_image[split].keys():
            jobs.append((split, image_id))
    print(f"Total source images to process: {len(jobs)}")
    total_anns = sum(
        len(index.ann_by_image[s][i]) for s, i in jobs
    )
    print(f"Total annotations to process: {total_anns}")

    fout = open(manifest_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(fout)
    if write_header:
        writer.writerow([
            "ann_id", "split", "image_id", "category_id",
            "crop_path", "crop_w", "crop_h",
        ])

    n_written = n_skip = 0
    pbar = tqdm(total=total_anns, desc="Cropping bboxes", smoothing=0.05)

    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(args.rpc_root, args.index_cache, str(out_dir),
                  args.max_side, args.quality, args.padding),
    ) as pool:
        futures = [pool.submit(_process_image, j) for j in jobs]
        for fut in as_completed(futures):
            rows = fut.result()
            for r in rows:
                key = (r[1], r[0])  # (split, ann_id)
                if key in seen:
                    n_skip += 1
                    continue
                writer.writerow(r)
                seen.add(key)
                n_written += 1
            pbar.update(len(rows))
            # Flush every ~5k rows so progress is durable
            if (n_written + n_skip) % 5000 < len(rows):
                fout.flush()

    pbar.close()
    fout.close()
    print(f"\nDone. New rows: {n_written}, skipped (already in manifest): {n_skip}")
    print(f"Manifest: {manifest_path}")
    print(f"Crops dir: {out_dir}")


if __name__ == "__main__":
    main()
