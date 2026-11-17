"""Index the raw RP2K image tree.

RP2K stores one directory per product under ``all/train`` and ``all/test``. The
directory names are UTF-8 bytes that were decoded as CP437 when the archive was
extracted. The repaired name equals ``instance_id`` in ``meta.csv`` and is used
as the SKU id. The official train/test folders share SKUs, so they are not used
as splits.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
_SIZE_TOKEN = re.compile(r"\d+(?:\.\d+)?(?:ml|l|g|kg|oz|mg)?", re.IGNORECASE)


def decode_sku_dirname(name: str) -> str:
    try:
        return name.encode("cp437").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return name


def family_key(sku_id: str) -> str:
    """Group size variants of one printed name, e.g. ``cola330ml`` and ``cola500ml``.

    Used only to sample harder negatives during training.
    """
    stripped = _SIZE_TOKEN.sub("", str(sku_id))
    key = re.sub(r"[\s_\-]+", "", stripped).casefold()
    return key or str(sku_id)


def discover_images(root: str | Path, extensions=IMAGE_EXTENSIONS) -> pd.DataFrame:
    """One row per image with ``image_path`` relative to ``root``."""
    root = Path(root)
    allowed = {ext.lower() for ext in extensions}
    rows = []
    for official in ("train", "test", "val"):
        base = root / "all" / official
        if not base.is_dir():
            continue
        for directory in sorted(base.iterdir()):
            if not directory.is_dir():
                continue
            sku_id = decode_sku_dirname(directory.name)
            for path in sorted(directory.iterdir()):
                if path.is_file() and path.suffix.lower() in allowed:
                    rows.append(
                        {
                            "image_path": path.relative_to(root).as_posix(),
                            "sku_id": sku_id,
                            "raw_dirname": directory.name,
                            "official_split": official,
                        }
                    )
    return pd.DataFrame(rows, columns=["image_path", "sku_id", "raw_dirname", "official_split"])


def list_image_files(directory: str | Path, extensions=IMAGE_EXTENSIONS) -> list[Path]:
    """Every image below ``directory``, recursively, in a stable order."""
    allowed = {ext.lower() for ext in extensions}
    return sorted(p for p in Path(directory).rglob("*") if p.is_file() and p.suffix.lower() in allowed)


def image_error(path: str | Path) -> str | None:
    try:
        with Image.open(path) as handle:
            handle.verify()
        with Image.open(path) as handle:
            handle.load()
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def file_md5(path: str | Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.md5()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()
