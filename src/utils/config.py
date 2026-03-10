"""YAML config loading and merging helpers."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def resolve_config_path(path_value: str, base_dir: Path, repo_root: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    in_base = (base_dir / path).resolve()
    if in_base.exists():
        return in_base
    return (repo_root / path).resolve()


def load_composed_config(experiment_cfg_path: Path, repo_root: Path) -> Dict[str, Any]:
    experiment_cfg_path = experiment_cfg_path.resolve()
    base_dir = experiment_cfg_path.parent
    cfg = load_yaml(experiment_cfg_path)

    merged: Dict[str, Any] = {}
    dataset_config_ref = cfg.get("dataset_config")
    if dataset_config_ref:
        dataset_path = resolve_config_path(str(dataset_config_ref), base_dir=base_dir, repo_root=repo_root)
        merged["dataset"] = load_yaml(dataset_path)
    model_config_ref = cfg.get("model_config")
    if model_config_ref:
        model_path = resolve_config_path(str(model_config_ref), base_dir=base_dir, repo_root=repo_root)
        merged["model"] = load_yaml(model_path)

    merged = deep_merge(merged, cfg)
    return merged

