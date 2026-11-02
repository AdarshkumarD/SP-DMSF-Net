"""Small shared helpers: config loading, seeding, formatting, JSON dumping."""
import json
import os
import random

import numpy as np
import yaml


def load_config(path, overrides=None):
    """Load a YAML config and apply optional ``dotted.key=value`` overrides.

    Example override: ``training.epochs=5`` or ``augmentation.enabled=false``.
    Values are parsed with YAML, so ``true``/``5``/``0.001`` get proper types.
    """
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    for item in overrides or []:
        key, sep, val = item.partition("=")
        if not sep:
            raise ValueError(f"Override must look like key=value, got: {item!r}")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    validate_config(cfg)
    return cfg


def validate_config(cfg):
    data = cfg["data"]
    ch = data["in_channels"]
    if ch not in (3, 4):
        raise ValueError(f"data.in_channels must be 3 (RGB) or 4 (RGB+DEM), got {ch}")
    if ch == 4 and not data.get("dem_dir"):
        raise ValueError("data.in_channels=4 requires data.dem_dir to be set")
    if ch == 3 and data.get("dem_dir"):
        raise ValueError("data.in_channels=3 but data.dem_dir is set; set it to null")
    mode = cfg["training"]["loss"]["mode"].lower()
    if mode not in ("bce", "dice", "focal", "tversky", "combined"):
        raise ValueError(f"Unsupported training.loss.mode: {mode}")


def seed_everything(seed):
    import tensorflow as tf

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def format_time(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{int(h)}h {int(m)}m {int(s)}s"
    return f"{int(m)}m {int(s)}s"


def format_param_size(params):
    b = params * 4
    if b >= 1024 ** 2:
        return f"{params:,} ({b / 1024 ** 2:.2f} MB)"
    if b >= 1024:
        return f"{params:,} ({b / 1024:.2f} KB)"
    return f"{params:,} ({b} Bytes)"


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)
