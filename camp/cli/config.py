"""Load CLI defaults from a YAML config file."""

from __future__ import annotations

from pathlib import Path

import yaml

DEFAULT_CONFIG_RELATIVE = Path("configs") / "default.yaml"

# Built-in fallbacks if a key is missing from the YAML (same numbers as before).
_FALLBACKS = {
    "common": {
        "class_name": None,
        "data_dir": "data",
        "device": "cuda",
        "encoder": "efficientnet-b3",
    },
    "segmentor": {
        "decoder_dropout": 0.4,
        "image_size": 512,
        "patch_size": 128,
        "epochs": 80,
        "batch_size": 8,
        "lr": 3e-4,
        "freeze_epochs": 30,
        "finetune_lr": 1e-4,
        "accumulation_steps": 2,
        "patience": 25,
        "split_ratio": 0.2,
        "output": None,
        "debug": False,
    },
    "recommender": {
        "embedding_dim": 2048,
        "num_heads": 16,
        "dropout": 0.10,
        "segmentor_weights": None,
        "epochs": 60,
        "lr": 3e-4,
        "batch_size": 4,
        "accumulation_steps": 4,
        "patience": 30,
        "finetune_epochs": 20,
        "finetune_lr": 5e-5,
        "image_size": 512,
        "patch_size": 128,
        "split_ratio": 0.2,
        "output": None,
        "debug": False,
    },
    "generate": {
        "num": 10,
        "output": None,
        "target_dir": None,
        "source_dir": None,
        "use_tto": True,
        "seed": None,
        "max_source": 50,
        "debug": False,
    },
}

__all__ = [
    "DEFAULT_CONFIG_RELATIVE",
    "load_cli_config",
    "resolve_config_path",
    "section_defaults",
]


def resolve_config_path(path, repo_root):
    """Resolve ``path`` against ``repo_root`` when it is relative."""
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path(repo_root) / candidate
    return candidate


def load_cli_config(path, repo_root):
    """Load the YAML config. Missing file → empty dict (fallbacks still apply)."""
    resolved = resolve_config_path(path, repo_root)
    if not resolved.is_file():
        return {}
    with open(resolved, encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("CLI config must be a mapping of sections, got %r" % type(data))
    return data


def section_defaults(data, section):
    """Merge ``common`` + ``section`` over built-in fallbacks.

    Section keys win over ``common``. Explicit ``null`` in YAML keeps the
    fallback only when the fallback is not itself ``None`` for required-ish
    strings we want absent — actually ``null`` means ``None`` and is kept.
    """
    merged = {}
    merged.update(_FALLBACKS.get("common", {}))
    merged.update(_FALLBACKS.get(section, {}))
    common = data.get("common") or {}
    if not isinstance(common, dict):
        raise ValueError("CLI config 'common' must be a mapping")
    body = data.get(section) or {}
    if not isinstance(body, dict):
        raise ValueError("CLI config %r must be a mapping" % section)
    for key, value in common.items():
        merged[key] = value
    for key, value in body.items():
        merged[key] = value
    return merged
