"""Загрузка config.yaml с наложением config.local.yaml."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def _merge(base, over):
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path=None):
    path = Path(path) if path else ROOT / "config.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    local = path.with_name("config.local.yaml")
    if local.exists():
        _merge(cfg, yaml.safe_load(local.read_text(encoding="utf-8")) or {})
    return cfg


def resolve(p):
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def data_path(cfg, key):
    if key in ("mo_names", "events", "national"):
        return resolve(cfg["data"][key])
    return resolve(cfg["data"]["dir"]) / cfg["data"][key]


def out_path(cfg, *parts):
    p = resolve(cfg["output_dir"]).joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
