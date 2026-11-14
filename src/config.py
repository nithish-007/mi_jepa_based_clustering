"""YAML configuration loading.

Every script reads exactly one YAML file. Command-line arguments are optional
``dotted.key=value`` overrides for quick experiments, for example::

    python scripts/train.py trainer.max_epochs=2 data.sampler.batches_per_epoch=20
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(value: str | Path | None) -> Path | None:
    """Expand ``~`` and make relative paths relative to the project root."""
    if value is None or str(value) == "":
        return None
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _parse_value(text: str) -> Any:
    return yaml.safe_load(text)


def apply_overrides(config: dict, overrides: list[str]) -> dict:
    result = copy.deepcopy(config)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override {item!r} must look like key.subkey=value")
        key, raw = item.split("=", 1)
        node = result
        parts = key.split(".")
        for part in parts[:-1]:
            if not isinstance(node.get(part), dict):
                node[part] = {}
            node = node[part]
        node[parts[-1]] = _parse_value(raw)
    return result


def load_config(default_path: str | Path, argv: list[str] | None = None) -> dict:
    """Load ``default_path``, or a YAML path given as the first argument, then apply overrides."""
    argv = list(argv or [])
    path = resolve_path(default_path)
    if argv and argv[0].endswith((".yaml", ".yml")):
        path = resolve_path(argv.pop(0))
    with open(path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    config = apply_overrides(config, argv)
    config["_source"] = str(path)
    return config


def fingerprint(section: dict) -> str:
    """Stable hash of a config section. Used to refuse resuming a different architecture."""
    payload = json.dumps(section, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def save_yaml(config: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
