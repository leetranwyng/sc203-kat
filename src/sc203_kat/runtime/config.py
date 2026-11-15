"""Lightweight config loading usable from Kaggle's kernel Python."""
import json
import os
from pathlib import Path

METHODS = ("apc_vis", "pcd", "sympl", "vst", "spatial_ssrl")

def project_root():
    override = os.environ.get("SC203_KAT_ROOT")
    if override:
        root = Path(override).resolve()
        if not (root / "pyproject.toml").is_file():
            raise ValueError("SC203_KAT_ROOT must point to a full repository checkout")
        return root
    for root in Path(__file__).resolve().parents:
        if (root / "pyproject.toml").is_file():
            return root
    raise RuntimeError("Use a repository checkout or set SC203_KAT_ROOT")

def merge(base, overlay):
    result = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = value
    return result

def load_config(path, root=None):
    root = Path(root) if root else project_root()
    path = Path(path)
    if not path.is_file():
        path = root / path
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    method = config.get("method")
    if method not in METHODS:
        raise ValueError("method must be one of: " + ", ".join(METHODS))
    defaults = root / "configs" / "methods" / (method + ".json")
    if defaults.is_file():
        config = merge(json.loads(defaults.read_text(encoding="utf-8")), config)
    if config.get("mode", "sanity") not in ("sanity", "full"):
        raise ValueError("mode must be sanity or full")
    config.setdefault("mode", "sanity")
    return config
